import torch
import time
import numpy as np
import lib_dataset
import nmf_v01
from anndata import AnnData

def prepare_data(vi_, xe_, holdouts, R):
    vi, xe = vi_.copy(), xe_.copy()
    vi.obsm['spatial'] = lib_dataset.batch_apply(R, vi.obsm['spatial'])
    gamma = lib_dataset.gen_gamma(vi.obsm['spatial'], xe.obsm['spatial'])
    vcols = set(vi.var_names)
    xcols = set(xe.var_names)
    common_genes = vcols.intersection(xcols)
    hcols = set(holdouts).intersection(common_genes)
    train_genes = common_genes - hcols
    test_genes = hcols - train_genes
    sorted_xe = xe[:, list(train_genes)].copy()
    vi_genes = list(train_genes) + list(test_genes)
    sorted_vi = vi[:, vi_genes]
    return sorted_vi, sorted_xe, gamma


def impute_work(vi, xe, gamma, hdim, num_epoch, plsf, seed, ent_offset, ent_div):
    model = nmf_v02.LowDimSparseGamma(sorted_xe, sorted_vi, gamma, n_factors = hdim, loss_type = "poisson", lambda_x = 1, lambda_v = 1, poi_reduction="sum", l2_w = 0.0001, device = device, dense_gamma = False, platform_scaling = plsf, entropy_w = lambda ep: np.exp((ep - ent_offset) / ent_div))
    _ = model.train(num_epochs = num_epoch + 1, lr=5e-2, print_freq=1000, callback = cb)
    full_pred = AnnData(X = nmf_v02.dcn(model.get_xenium_est()), obs = sorted_xe.obs, var = sorted_vi.var)
    return cb.data, full_pred[:, holdout_xe.var_names].copy()


def impute_genes(vi, xe, holdouts, hdim, R = None, ent_div = 1000, seed = 10042, lr = 1e-3, epochs = 5000, device = "cuda:0"):
    t = time.time()
    if R is None:
        R = np.eye(3)
    sorted_vi, sorted_xe, gamma = prepare_data(vi, xe, holdouts, R)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nmf_v01.LowDimWithScaling(sorted_xe,
                                      sorted_vi,
                                      gamma,
                                      n_factors = hdim,
                                      device = device,
                                      adaptive_entropy = True,
                                      adaptive_entropy_denominator = ent_div,
                                      poisson_sum = True,
                                      add_entropy = True,
                                      platform_scaling = True)
    _ = model.train(num_epochs = epochs + 1, lr=lr, print_freq = 500, loss_type='poisson')
    full_pred = AnnData(X = nmf_v01.dcn(model.get_xenium_est()), obs = sorted_xe.obs, var = sorted_vi.var)
    print("Training finished in (seconds):", time.time() - t)
    return full_pred[:, list(x for x in holdouts if x in full_pred.var_names)].copy()

def infer_latent_types(vi, xe, hdim, R = None, ent_div = 400, seed = 10042, lr = 1e-3, epochs = 5000, device = "cuda:0"):
    t = time.time()
    if R is None:
        R = np.eye(3)
    sorted_vi, sorted_xe, gamma = prepare_data(vi, xe, [], R)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nmf_v01.LowDimWithScaling(sorted_xe,
                                      sorted_vi,
                                      gamma,
                                      n_factors = hdim,
                                      device = device,
                                      adaptive_entropy = True,
                                      adaptive_entropy_denominator = ent_div,
                                      poisson_sum = True,
                                      add_entropy = True,
                                      platform_scaling = True)
    _ = model.train(num_epochs = epochs + 1, lr=lr, print_freq = 500, loss_type='poisson')
    full_pred = AnnData(X = nmf_v01.dcn(model.get_xenium_est()), obs = sorted_xe.obs, var = sorted_vi.var)
    print("Training finished in (seconds):", time.time() - t)
    ret_x = nmf_v01.dcn(model.Px_soft)
    ret_v = nmf_v01.dcn(torch.mm(model.gamma.T, torch.exp(model.scale_factor_visium) * model.Px_soft))
    ret_v = ret_v / (1e-10 + np.sum(ret_v, axis = 1, keepdims = True))
    Px = AnnData(X = ret_x, obs = sorted_xe.obs, obsm = sorted_xe.obsm)
    Pv = AnnData(X = ret_v, obs = sorted_vi.obs, obsm = sorted_vi.obsm)
    return Px, Pv
