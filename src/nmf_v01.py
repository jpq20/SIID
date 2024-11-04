import torch
import numpy as np
from scipy.sparse import csr_matrix, issparse
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import scanpy as sc
from anndata import AnnData

# Detach and convert to numpy
def dcn(tensor):
    return tensor.detach().cpu().numpy()

# Sanity check functions
def has_nan(data):
    if isinstance(data, torch.Tensor):
        return torch.isnan(data).any().item()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).any()
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy ndarray.")

def has_inf(data):
    if isinstance(data, torch.Tensor):
        return torch.isinf(data).any().item()
    elif isinstance(data, np.ndarray):
        return np.isinf(data).any()
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy ndarray.")

def has_nan_or_inf(data):
    return has_nan(data) or has_inf(data)


# Convert numpy to sparse tensor
def convert_to_sparse_csr_tensor(array):
    # Step 1: Check if the array is already sparse (SciPy sparse matrix)
    if not issparse(array):
        # If not sparse, convert it to a sparse CSR matrix
        array = csr_matrix(array)

    # Step 2: Extract crow_indices, col_indices, and values from the sparse CSR matrix
    crow_indices = torch.tensor(array.indptr, dtype=torch.int64)
    col_indices = torch.tensor(array.indices, dtype=torch.int64)
    values = torch.tensor(array.data, dtype=torch.float32)

    # Step 3: Create the PyTorch sparse CSR tensor
    sparse_csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=array.shape)
    sparse_coo_tensor = sparse_csr_tensor.to_sparse()

    return sparse_coo_tensor


# Convert Anndata to two dimensional numpy array
def to_dense_mat(ad, eps):
    """
    This function converts the AnnData object to a dense matrix.
    Parameters
    ----------
    ad : AnnData
        The AnnData object
    eps : float
        The epsilon value for the matrix.
    Returns the dense matrix.
    """

    if isinstance(ad.X, csr_matrix):
        mat = ad.X.toarray()
    else:
        mat = ad.X
    return mat + eps

class LowDimWithScaling(torch.nn.Module):
    """
    This class finds out low-dimensional representation Given two count matrices.
    The main workhorse behind the jointly optimizing membership functions for the Xenium and visium
    datasets.
    This is adapted from Hongyu's code.
    Parameters
    ----------
    xe : AnnData
        The AnnData object for Xenium dataset
    vi : AnnData
        The AnnData object for Visium dataset
    gamma : float
        The mapping matrix that maps the Xenium cells to visium data, this is obtained from spatial
        mapping.
    n_factors : int
        The number of cell-types
    device : str
        The device on which the model is run.
    eps : float
        The epsilon value for the model.
    """
    def __init__(
            self,
            xe: AnnData,
            vi: AnnData,
            gamma: np.ndarray,
            n_factors: int,
            lambda_x: float = 1.0,
            lambda_v: float = 1.0,
            lambda_r: float = 1.0,
            l2_w: float = 1e-5,
            entropy_w: float = 1.0,
            adaptive_entropy: bool = False,
            adaptive_entropy_denominator: int = 500,
            device: str = 'gpu0',
            eps: float = 1e-6,
            init_scale_xenium: bool = False,
            platform_scaling: bool = False,
            add_entropy: bool = False,
            poisson_sum: bool = False
        ):
        super().__init__()
        self.xe = xe
        self.vi = vi
        self.gamma = convert_to_sparse_csr_tensor(gamma).to(device)
        self.n_factors = n_factors
        self.lambda_x = lambda_x
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
        self.l2_w = l2_w
        self.entropy_w = entropy_w
        self.device = device
        self.num_xenium_cells = self.xe.shape[0]
        self.num_visium_cells = self.vi.shape[0]
        self.num_xenium_genes = self.xe.shape[1]
        self.num_genes = self.vi.shape[1]
        self.add_entropy = add_entropy
        self.adaptive_entropy = adaptive_entropy
        self.adaptive_entropy_denominator = adaptive_entropy_denominator
        self.poisson_sum = poisson_sum

        # These are observed data
        self.X = torch.tensor(to_dense_mat(self.xe, 0), dtype=torch.float32).to(device)
        self.V = torch.tensor(to_dense_mat(self.vi, 0), dtype=torch.float32).to(device)

        # These are parameters to be learned
        self.Px = torch.nn.Parameter(
            torch.randn((self.num_xenium_cells, self.n_factors), dtype=torch.float32).to(device) * 0.01
        )
        # This is not needed if you consider mapping matrix is constant and
        # it's perfect
        # self.Pv = torch.nn.Parameter(
        #     torch.zeros((self.num_visium_cells, self.n_factors), dtype=torch.float32).to(device))# what's the equivalent ?
        self.Q = torch.nn.Parameter(
            torch.randn((self.n_factors, self.num_genes), dtype=torch.float32).to(device))
        # If we need scaling factor then
        # Each cell/spot has a separate scaling factor
        if init_scale_xenium:
            # adds eps to avoid log(0)
            self.scale_factor_xenium = torch.nn.Parameter(
                torch.tensor(np.log(self.xe.to_df().sum(1).values + eps), dtype=torch.float32).view(-1,1).to(device))
        else:
            self.scale_factor_xenium = torch.nn.Parameter(
                torch.ones((self.num_xenium_cells, 1), dtype=torch.float32).to(device))
        self.platform_scaling = platform_scaling
        self.platform_sf = torch.nn.Parameter(torch.randn((1, self.num_xenium_genes), dtype = torch.float32).to(device))
        self.scale_factor_visium = torch.nn.Parameter(
            torch.ones((self.num_xenium_cells, 1), dtype=torch.float32).to(device))

        # Initialize the learnable parameters Px, Pv, Q
        torch.nn.init.xavier_uniform_(self.Px)
        torch.nn.init.xavier_uniform_(self.Q)

    @property
    def Q_soft(self):
        return torch.nn.Softmax(dim=1)(self.Q)

    @property
    def Px_soft(self):
        return torch.nn.Softmax(dim=1)(self.Px)

    # Get the estimate of the Xenium counts
    def get_xenium_est(self, end_at=None):
        if end_at is None:
            end_at = self.num_genes

        #Q_T_soft = torch.nn.Softmax(dim=1)(self.Q_soft[:, :end_at])
        ret = torch.mm(self.Px_soft, self.Q_soft[:, :end_at])
        #ret = torch.mm(self.Px_soft, Q_T_soft)
        ret = torch.exp(self.scale_factor_xenium) * ret
        return ret

    # Get the estimate of the Visium counts
    def get_visium_est(self):
        scaled_Px = torch.exp(self.scale_factor_visium) * self.Px_soft
        combined_Px = torch.mm(self.gamma.T, scaled_Px)
        ret = torch.mm(combined_Px, self.Q_soft)
        if self.platform_scaling:
            ret[:, :self.xe.shape[1]] *= torch.exp(self.platform_sf)
        return ret

    def loss(self, verbose=False,loss_type='cosine',num_epoch=1):
        xe_est = self.get_xenium_est(end_at=self.num_xenium_genes)
        vi_est = self.get_visium_est()

        test_tensors = [xe_est, vi_est]
        for idx, t in enumerate(test_tensors):
            assert not has_nan(t), f"Training Failure: Tensor #{idx} contains NaN"
            assert not has_inf(t), f"Training Failure: Tensor #{idx} contains Inf"
        # Do cosine loss between X and xe_est
        if loss_type == 'cosine':
            x_loss = self.lambda_x * (2- \
                cosine_similarity(self.X, xe_est, dim=1).mean() -
                cosine_similarity(self.X, xe_est, dim=0).mean()
            )
            v_loss = self.lambda_v * (2- \
                cosine_similarity(self.V, vi_est, dim=1).mean() -
                cosine_similarity(self.V, vi_est, dim=0).mean()
            )
            expression_loss = (x_loss + v_loss)
        elif loss_type == 'kl':
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
            xe_norm = F.log_softmax(xe_est, dim=1)
            target_x = F.softmax(self.X, dim=1)
            vi_norm = F.log_softmax(vi_est, dim=1)
            target_v = F.softmax(self.V, dim=1)
            test_tensors = [xe_norm, target_x, vi_norm, target_v]
            for idx, t in enumerate(test_tensors):
                assert not has_nan(t), f"Training Failure: Tensor #{idx} contains NaN"
                assert not has_inf(t), f"Training Failure: Tensor #{idx} contains Inf"
            x_loss = kl_loss(xe_norm, target_x)
            v_loss = kl_loss(vi_norm, target_v)
            expression_loss = x_loss + v_loss
        elif loss_type == 'fro':
            x_loss = self.lambda_x * torch.norm(self.X - xe_est, p = 'fro')**2
            v_loss = self.lambda_v * torch.norm(self.V - vi_est, p = 'fro')**2
            expression_loss = x_loss + v_loss
        elif loss_type == 'poisson':
            # no need to add eps here, poisson_nll_loss has intrinstic eps term with log_input = False
            if self.poisson_sum:
                x_loss = self.lambda_x * F.poisson_nll_loss(xe_est, self.X, log_input=False, reduction='sum')
                v_loss = self.lambda_v * F.poisson_nll_loss(vi_est, self.V, log_input=False, reduction='sum')
            else:
                x_loss = self.lambda_x * F.poisson_nll_loss(xe_est, self.X, log_input=False)
                v_loss = self.lambda_v * F.poisson_nll_loss(vi_est, self.V, log_input=False)
            expression_loss = x_loss + v_loss
        # Regularization loss for Px, Pv, Q
        l2_reg_loss = self.l2_w * sum(torch.norm(t, p=2) ** 2 for t in [self.Q, self.Px, self.scale_factor_visium, self.scale_factor_xenium, self.platform_sf])
        entropy_loss = 0
        if self.add_entropy:
            if self.adaptive_entropy:
                if num_epoch % 100 == 0:
                    self.entropy_w = np.exp(num_epoch/self.adaptive_entropy_denominator)
                entropy_loss = (self.entropy_w)  * torch.mean(torch.sum(self.Px_soft * torch.log(self.Px_soft + 1e-12), dim=1))
            else:
                entropy_loss = (self.entropy_w)  * torch.mean(torch.sum(self.Px_soft * torch.log(self.Px_soft + 1e-12), dim=1))

        l2_reg_loss -= entropy_loss

        total_loss = expression_loss + l2_reg_loss

        if verbose:
            term_members = [dcn(x_loss), dcn(v_loss), dcn(l2_reg_loss)]
            term_names = ['x_loss', 'v_loss', 'l2_reg_loss']
            score_dict = dict(zip(term_names, term_members))
            clean_dict = {
                    k: score_dict[k] for k in score_dict if not np.isnan(score_dict[k])
                }
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))
        return (
            total_loss,
            x_loss,
            v_loss,
            l2_reg_loss,
            entropy_loss
        )

    def train(self, num_epochs=1000, lr=1e-3,verbose=False,loss_type='cosine',print_freq=100,warm_restart=False, callback = None):
        print(f"Training the model with {loss_type} loss")
        train_tensors = [self.Px, self.Q, self.scale_factor_xenium, self.scale_factor_visium, self.platform_sf]
        optimizer = torch.optim.Adam(train_tensors, lr=lr)
        if warm_restart:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
        keys = [
            "total_loss",
            "main_loss",
            "vg_reg",
            "kl_reg",
            "entropy_reg",
            "count_reg",
            "lambda_f_reg",
        ]
        values = [[] for i in range(len(keys))]
        training_history = {key: value for key, value in zip(keys, values)}

        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            if verbose and epoch % 100 == 0:
                run_loss = self.loss(verbose=True,loss_type=loss_type,num_epoch=epoch)
            else:
                run_loss = self.loss(loss_type=loss_type, num_epoch=epoch)

            run_loss[0].backward()
            optimizer.step()

            if warm_restart:
                scheduler.step(epoch)
            if epoch % print_freq == 0:
                if loss_type == 'cosine':
                    print(f"Epoch {epoch}, Loss: {run_loss[0].item()}, Xenium cosine similarity: {2 - run_loss[1].item()/self.lambda_x}, Visium cosine similarity: {2 - run_loss[2].item()/self.lambda_v}, Regularization {run_loss[3].item()}")
                else:
                    print(f"Epoch {epoch}, Loss: {run_loss[0].item()}, Xenium (KL/Fro) divergence: {run_loss[1].item()/self.lambda_x}, Visium (KL/Fro) divergence: {run_loss[2].item()/self.lambda_v}, Regularization {run_loss[3].item()}, Entropy {run_loss[4].item() if self.add_entropy else 0}")
            losses.append(run_loss[0].item())
            if callback is not None:
                callback(self, epoch, run_loss)

        # Return items of interest
        with torch.no_grad():
            return {
                "Px": dcn(self.Px_soft),
                "Q": dcn(self.Q_soft),
                "scale_factor_xenium": dcn(self.scale_factor_xenium),
                "scale_factor_visium": dcn(self.scale_factor_visium),
                "training_history": training_history,
                "losses": losses
            }


    def save(self, filename):
        # Save the model parameters in a config file
        # This is not a PyTorch model, but a simple class
        # with some parameters
        # Don't use torch.save for this
        # Use pickle
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    # load the model
    @staticmethod
    def load(filename):
        # Load the model parameters from a config file
        # This is not a PyTorch model, but a simple class
        # with some parameters
        # Don't use torch.load for this
        # Use pickle
        import pickle
        with open(filename, "rb") as f:
            obj = LowDimWithScaling.__new__(LowDimWithScaling)
            obj.__dict__.update(pickle.load(f))
        return obj
