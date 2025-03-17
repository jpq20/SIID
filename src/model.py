import torch
import numpy as np
from scipy.sparse import csr_matrix, issparse
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
import torch_geometric.nn
import torch.nn as nn

from utils import *


class joint_model(torch.nn.Module):
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
            add_entropy: bool = False,
            poisson_sum: bool = False,
            k_neighbors_xenium: int = 40,
            k_neighbors_visium: int = 15,
            gcn_hidden_dim: int = 64,
            gat_heads: int = 1,
            gat_dropout: float = 0.1,
        ):
        super().__init__()
        self.xe = xe
        self.vi = vi
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
        self.k_neighbors_xenium = k_neighbors_xenium
        self.k_neighbors_visium = k_neighbors_visium

        # These are observed data
        self.X = torch.tensor(to_dense_mat(self.xe, 0), dtype=torch.float32).to(device)
        self.V = torch.tensor(to_dense_mat(self.vi, 0), dtype=torch.float32).to(device)

        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)

        # Get spatial coordinates
        self.xenium_coords = torch.tensor(self.xe.obsm['spatial'], dtype=torch.float32).to(device)
        self.visium_coords = torch.tensor(self.vi.obsm['spatial'], dtype=torch.float32).to(device)
        
        # Build spatial graph for xenium data
        self.build_spatial_graph()
        self.visium_edge_index = self.prepare_bipartite_graph()

        # These are parameters to be learned
        self.Px = torch.nn.Parameter(
            torch.randn((self.num_xenium_cells, self.n_factors), dtype=torch.float32).to(device) * 0.01
            )
        self.W = torch.nn.Parameter(
            torch.randn((self.n_factors, self.num_genes), dtype=torch.float32).to(device)
            )
        # self.scale_factor_xenium = torch.nn.Parameter(torch.ones((self.num_xenium_cells, 1), dtype=torch.float32).to(device)) # TODO:
        self.scale_factor_visium = torch.nn.Parameter(torch.ones((self.num_visium_cells, 1), dtype=torch.float32).to(device))
        
        # self.platform_sf_visium = torch.nn.Parameter(torch.randn((1, self.num_genes), dtype=torch.float32).to(device))
        self.gene_sf = torch.nn.Parameter(torch.randn((1, self.num_genes), dtype=torch.float32).to(device))

        
        # Initialize GCN layers for spatial smoothing
        # self.gcn_layers = nn.ModuleList()
        # self.gcn_layers.append(GCNConv(self.n_factors, self.n_factors))
        # self.gcn_layers.append(GCNConv(self.n_factors, self.n_factors))
        # self.gcn_layers = self.gcn_layers.to(self.device)

        self.xenium_gat = torch_geometric.nn.GATConv(
            in_channels=self.n_factors,
            out_channels=self.n_factors,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=False  # Average the attention heads
        ).to(self.device)
        """
        # Initialize GAT layer for Visium feature computation
        self.visium_gat = torch_geometric.nn.GATConv(
            in_channels=self.n_factors,
            out_channels=self.n_factors,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=False  # Average the attention heads
        ).to(self.device)
        """
        # Initialize the learnable parameters
        torch.nn.init.xavier_uniform_(self.Px)
        torch.nn.init.xavier_uniform_(self.W)

    def build_spatial_graph(self):
        # Build k-nearest neighbors graph for xenium data
        xenium_coords_np = self.xenium_coords.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors_xenium+1, algorithm='ball_tree').fit(xenium_coords_np)
        _, indices = nbrs.kneighbors(xenium_coords_np)
        
        # Create edge index for PyTorch Geometric
        edge_list = []
        for i in range(len(indices)):
            for j in indices[i][1:]:  # Skip the first one as it's the point itself
                edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = edge_index
        
        # Use gamma matrix to build visium to xenium mapping instead of nearest neighbors
        # gamma shape should be (num_visium_cells, num_xenium_cells)
        v2x_edge_list = []
        
        # For each Visium spot, find the top k Xenium cells with highest gamma values
        gamma_np = self.gamma.cpu().numpy()
        for i in range(self.num_visium_cells):
            # Get indices of top k Xenium cells for this Visium spot
            top_xenium_indices = np.argsort(gamma_np[i])[-self.k_neighbors_visium:]
            for j in top_xenium_indices:
                v2x_edge_list.append([i, j])
        
        self.v2x_edge_index = torch.tensor(v2x_edge_list, dtype=torch.long).t().contiguous().to(self.device)
    
    def prepare_bipartite_graph(self):
        # Prepare edge indices for the bipartite graph using gamma-based connections
        # Original: [visium_idx, xenium_idx]
        # New: [visium_idx, xenium_idx + num_visium_cells]
        visium_indices = self.v2x_edge_index[0]
        xenium_indices = self.v2x_edge_index[1] + self.num_visium_cells
        
        # Create bidirectional edges for better message passing
        src = torch.cat([visium_indices, xenium_indices])
        dst = torch.cat([xenium_indices, visium_indices])
        edge_index = torch.stack([src, dst])
        
        return edge_index
    
    @property
    def W_soft(self):
        return torch.nn.Softmax(dim=1)(self.W)
    
    @property
    def F_soft(self):
        # Apply softmax to ensure each row sums to 1
        return torch.nn.Softmax(dim=1)(self.Px)
    
    @property
    def xenium_factors(self):
        # Apply GCN to smooth the factor matrix
        x = self.F_soft
        # gat
        x = self.xenium_gat(x, self.edge_index)
        return torch.nn.Softmax(dim=1)(x)
    
    @property
    def visium_factors(self):
        # Instead of using GAT, use gamma to compute visium factors as weighted sum of xenium factors
        x = self.xenium_factors
        
        # Compute visium factors as weighted sum of xenium factors
        visium_features = torch.mm(self.gamma.T, x)
        
        # Apply softmax to ensure proper normalization
        visium_features = torch.nn.Softmax(dim=1)(visium_features)
        return visium_features
    
    # Get the estimate of the Xenium counts using the denoised factors
    def get_xenium_est(self, end_at=None):
        if end_at is None:
            end_at = self.num_genes

        # Use the spatially smoothed factors F and the new loading matrix W
        w = self.W_soft[:, :end_at] * torch.exp(self.gene_sf[:, :end_at])
        ret = torch.mm(self.xenium_factors, w)
        # ret = torch.exp(self.scale_factor_xenium) * ret
        return ret

    def get_visium_est(self):

        w = self.W_soft * torch.exp(self.gene_sf)
        ret = torch.mm(self.visium_factors, w)
        ret = torch.exp(self.scale_factor_visium) * ret
        return ret

    def loss(self, verbose=False, loss_type='poisson', num_epoch=1):
        xe_est = self.get_xenium_est(end_at=self.num_xenium_genes)
        vi_est = self.get_visium_est()

        test_tensors = [xe_est, vi_est]
        for idx, t in enumerate(test_tensors):
            assert not has_nan(t), f"Training Failure: Tensor #{idx} contains NaN"
            assert not has_inf(t), f"Training Failure: Tensor #{idx} contains Inf"
            
        # Default to Poisson loss for better modeling of count data
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
            
        # Regularization loss for parameters
        l2_reg_loss = self.l2_w * sum(torch.norm(t, p=2) ** 2 for t in [self.W, self.Px, self.gene_sf, self.scale_factor_visium])
        
        # Entropy loss to encourage sparse cell type assignments
        entropy_loss = 0
        if self.add_entropy:
            if self.adaptive_entropy:
                if num_epoch % 100 == 0:
                    self.entropy_w = np.exp(num_epoch/self.adaptive_entropy_denominator)
                entropy_loss = (self.entropy_w) * torch.mean(torch.sum(self.F_soft * torch.log(self.F_soft + 1e-12), dim=1))
            else:
                entropy_loss = (self.entropy_w) * torch.mean(torch.sum(self.F_soft * torch.log(self.F_soft + 1e-12), dim=1))

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

    def train(self, num_epochs=1000, lr=1e-3, verbose=False, loss_type='poisson', print_freq=100, warm_restart=False):
        print(f"Training the model with {loss_type} loss")
        train_tensors = [self.Px, self.W, self.gene_sf, self.scale_factor_visium]
            
        # Add GCN parameters
        # for layer in self.gcn_layers:
        #     train_tensors.extend(list(layer.parameters()))
        
        # Add GAT parameters for Visium feature computation
        # train_tensors.extend(list(self.visium_gat.parameters()))

        train_tensors.extend(list(self.xenium_gat.parameters()))
        
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
        best_loss = float('inf')
        best_state = None
        early_stop = False
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            """
            # Apply GCN to smooth the factor matrix
            x = self.F_soft
            for i, conv in enumerate(self.gcn_layers):
                x = conv(x, self.edge_index)
                if i < len(self.gcn_layers) - 1:
                    x = F.relu(x)
            with torch.no_grad():
                self.Px.copy_(x)
            """

            if verbose and epoch % 100 == 0:
                run_loss = self.loss(verbose=True, loss_type=loss_type, num_epoch=epoch)
            else:
                run_loss = self.loss(loss_type=loss_type, num_epoch=epoch)

            run_loss[0].backward()
            losses.append(run_loss[0].item())
            optimizer.step()

            if warm_restart:
                scheduler.step(epoch)

            # Track best model
            current_loss = run_loss[0].item()
            if current_loss < best_loss:
                best_loss = current_loss
                # Save a copy of the model state
                best_state = {
                    'Px': self.Px.clone(),
                    'W': self.W.clone(),
                    'gene_sf': self.gene_sf.clone(),
                    'scale_factor_visium': self.scale_factor_visium.clone(),
                    # 'scale_factor_xenium': self.scale_factor_xenium.clone(),
                }
            
            if self.callback(epoch, run_loss, print_freq=print_freq):
                early_stop = True
                break
        
        # Restore best model state if we have one and early stopping was triggered
        if early_stop and best_state is not None:
            with torch.no_grad():
                self.Px.copy_(best_state['Px'])
                self.W.copy_(best_state['W'])
                self.gene_sf.copy_(best_state['gene_sf'])
                self.scale_factor_visium.copy_(best_state['scale_factor_visium'])
                # self.scale_factor_xenium.copy_(best_state['scale_factor_xenium'])

        # Return items of interest
        with torch.no_grad():
            return {
                "F": dcn(self.F_soft),  # Return the spatially smoothed factors
                "W": dcn(self.W_soft),  # Return the new loading matrix
                "gene_sf": dcn(self.gene_sf),
                "scale_factor_visium": dcn(self.scale_factor_visium),
                # "scale_factor_xenium": dcn(self.scale_factor_xenium),
                "training_history": training_history,
                "losses": losses
            }
        
    def callback(self, epoch, run_loss, patience=100, min_delta=0.001, print_freq=10):
        """
        Callback function called after each training epoch.
        
        Parameters:
        -----------
        model : joint_model
            The current model instance
        epoch : int
            Current epoch number
        run_loss : tuple
            Tuple containing different loss components (total_loss, x_loss, v_loss, l2_reg_loss, entropy_loss)
        patience : int, default=50
            Number of epochs to wait for improvement before early stopping
        min_delta : float, default=0.001
            Minimum change in loss to be considered as improvement
        print_freq : int, default=10
            Frequency of epochs to print loss information
            
        Returns:
        --------
        bool
            True if training should stop, False otherwise
        """
        # Initialize tracking variables if they don't exist
        if not hasattr(self, 'best_loss'):
            self.best_loss = float('inf')
            self.patience_counter = 0
            self.loss_history = []
        
        current_loss = run_loss[0].item()
        self.loss_history.append(current_loss)
        
        # Print loss information at specified frequency
        if epoch % print_freq == 0:
            total_loss, x_loss, v_loss, l2_reg_loss, entropy_loss = run_loss
            
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}, "
                    f"Xenium loss: {x_loss.item()/self.lambda_x:.4f}, "
                    f"Visium loss: {v_loss.item()/self.lambda_v:.4f}, "
                    f"Regularization: {l2_reg_loss.item():.4f}, "
                    f"Entropy: {entropy_loss.item() if self.add_entropy else 0:.4f}")
            
            # Print improvement status
            if self.patience_counter > 0:
                print(f"No improvement for {self.patience_counter} epochs. Best loss: {self.best_loss:.4f}")
        
        # Check if the current loss is better than the best loss
        if current_loss < self.best_loss - min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            if epoch % print_freq == 0:
                print(f"New best loss: {self.best_loss:.4f}")
        else:
            self.patience_counter += 1
        
        # Early stopping check
        if self.patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
            return True
        
        return False


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
            obj = joint_model.__new__(joint_model)
            obj.__dict__.update(pickle.load(f))
        return obj
