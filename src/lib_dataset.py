import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
from scipy.spatial import KDTree

def apply(proj, p):
    return proj.dot(np.concatenate((p, [1])))[:2]
    
def batch_apply(proj, ps):
    ret = np.zeros_like(ps)
    for i in range(ps.shape[0]):
        ret[i] = apply(proj, ps[i])
    return ret

def sim_with_poisson(meta_src, expr_src, sparse = False):
    ret = meta_src.copy()
    ret.X = np.random.poisson(expr_src)
    if sparse:
        ret.X = csr_matrix(ret.X)
    ret.layers['src'] = expr_src.copy()
    return ret

def subset_to_region(adata, minx, maxx, miny, maxy):
    c = adata.obsm['spatial']
    return adata[(c[:, 0] >= minx) & (c[:, 0] <= maxx) & (c[:, 1] >= miny) & (c[:, 1] <= maxy)].copy()

def gen_random_nld_matrix(n_row, n_col, n_hd, sparsity_pct = 0):
    '''
        Build a random matrix that accepts a ROW-NORMALIZED low-dimensional non-negative matrix factorization (NMF).
        @param n_row, n_col: Desired shape of the matrix.
        @param n_hd: Desired number of hidden dimensions.
        @param sparsity_pct: Percentage of sparse elements.
        @return: full matrix, (mat1, mat2), such that mat1 . mat2 == full_mat, and all mats have row sum of 1.
    '''
    gx_init = np.random.rand(n_row, n_hd)
    gy_init = np.random.rand(n_hd, n_col)
    gx_mask = np.random.rand(n_row, n_hd)
    gy_mask = np.random.rand(n_hd, n_col)
    gx_init[gx_mask < sparsity_pct / 100] = 0
    gy_init[gy_mask < sparsity_pct / 100] = 0
    gx = gx_init / gx_init.sum(axis = 1, keepdims = True)
    gy = gy_init / gy_init.sum(axis = 1, keepdims = True)
    return gx, gy

def nmf_synth_dataset(nc, ng, nv, n_tar, h_dim, vi_scale, xe_scale, sparsity_pct = 0):
    '''
    Construct a fully synthetic dataset according to the joint NMF specification.
    @param nc, ng, nv, ntar: # Xenium cell, # total genes, # Visium spots (assuming equipartition for all cells in assignments), # targeted genes
    @param h_dim: # of hidden dimensions.
    @param vi_scale, xe_scale: # UMI/cell for Visium/Xenium experiments (With all genes observed!). 
           Note that UMI/cell(spot)/gene will be lower for Visium due to more genes
    @param sparsity_pct: percentage of zeros in Gx/Gy (Does NOT translate to sparsity in the final expression matrices).
    @return: The generated datasets (vi, xe, asgn) with metadata filled.
             vi.layer['src'], xe.layer['src'] contains the generating parameters. Further, the generated Gx/Gy is stored in xe.obsm['gx_src'] and vi.varm['gy_src'].
    '''
    gx, gy = gen_random_nld_matrix(nc, ng, h_dim, sparsity_pct)
    full_src = gx.dot(gy)
    assert (nc % nv == 0)
    vi_src = full_src.copy()
    vi_src = np.reshape(vi_src, (nv, nc // nv, ng)).sum(axis = 1) * vi_scale
    xe_src = full_src[:, :n_tar] * xe_scale  # no re-normalization; handle this later if we want
    # construct the metadata
    vi_meta = AnnData(np.zeros((nv, ng)))
    xe_meta = AnnData(np.zeros((nc, n_tar)))
    vi_meta.obsm['spatial'] = np.zeros((nv, 2))
    xe_meta.obsm['spatial'] = np.zeros((nc, 2))
    vi_meta.var_names = list(f"#{i}" for i in range(ng))
    xe_meta.var_names = vi_meta.var_names[:n_tar]
    xe_meta.obsm['gx_src'] = gx
    vi_meta.varm['gy_src'] = gy.T
    return sim_with_poisson(vi_meta, vi_src), sim_with_poisson(xe_meta, xe_src), np.array([i for i in range(nv) for _ in range(nc // nv)])

def gen_xe_assignment(vi_sp, xe_sp, max_radius = 100):
    '''
    For each Xenium cell, return its allocation to Visium spots. -1 denotes unassigned cell.
    '''
    ret = []
    kdt = KDTree(vi_sp)
    for i in range(xe_sp.shape[0]):
        d, vi_idx = kdt.query(xe_sp[i])
        if d > max_radius:
            ret.append(-1)
        else:
            ret.append(vi_idx)
    return np.array(ret)

def gen_gamma(vi_sp, xe_sp, max_radius = 100, exclusive = True):
    assert exclusive
    ret = np.zeros((xe_sp.shape[0], vi_sp.shape[0]))
    kdt = KDTree(vi_sp)
    for i in range(xe_sp.shape[0]):
        d, vi_idx = kdt.query(xe_sp[i])
        if d <= max_radius:
            ret[i, vi_idx] = 1
    return ret

def gen_flexible_gamma(vi_sp, xe_sp, max_radius = 100, max_k = 4):
    if max_k == 1:
        return gen_gamma(vi_sp, xe_sp, max_radius)
    ret = np.zeros((xe_sp.shape[0], vi_sp.shape[0]))
    kdt = KDTree(vi_sp)
    for i in range(xe_sp.shape[0]):
        ld, lidx = kdt.query(xe_sp[i], k = max_k)
        for d, idx in zip(ld, lidx):
            if d <= max_radius:
                ret[i, idx] = 1
    return ret


def aggr_pseudospot_expr(expr_mat, asgn, dim0 = None):
    '''
    Generate the aggregated pseudospot expression.
    '''
    if dim0 is None:
        ret = np.zeros((max(asgn) + 1, expr_mat.shape[1]))
    else:
        assert dim0 > max(asgn)
        ret = np.zeros((dim0, expr_mat.shape[1]))
    for idx, x in enumerate(asgn):
        if x >= 0:
            ret[x] += expr_mat[idx]
    return ret

def remove_unmatched(vi, xe, asgn, rematch_radius = 100):
    '''
    Remove unassigned xenium cells and visium spots with no cell matches, then rebuild the assignment vector.
    '''
    set_asgn = set(asgn)
    vi_match_flag = list(x in set_asgn for x in range(vi.shape[0]))
    vi_ret = vi[vi_match_flag].copy()
    xe_ret = xe[asgn != -1].copy()
    asgn_ret = gen_xe_assignment(vi_ret.obsm['spatial'], xe_ret.obsm['spatial'], rematch_radius)
    return vi_ret, xe_ret, asgn_ret
    