import torch
import numpy as np
from scipy.sparse import csr_matrix, issparse
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import os

import torch.nn.init as init
import torch.nn as nn

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

def save_results(model, imputed, losses, metrics, output_dir):
    """
    Save model results
    
    Parameters:
    -----------
    model : joint_model
        Trained model
    imputed : AnnData
        Imputed expression data
    losses : list
        Training losses
    metrics : dict
        Evaluation metrics
    output_dir : str
        Directory to save results
    """
    # Save imputed data
    if imputed is not None:
        imputed.write_h5ad(os.path.join(output_dir, 'imputed.h5ad'))
    
    # Save model factors
    with torch.no_grad():
        factors = model.F_soft.cpu().numpy()
        loadings = model.W_soft.cpu().numpy()
    
    np.save(os.path.join(output_dir, 'factors.npy'), factors)
    np.save(os.path.join(output_dir, 'loadings.npy'), loadings)
    
    # Save factors as AnnData for easier visualization
    factors_adata = AnnData(X=factors, obs=model.xe.obs.copy())
    factors_adata.obsm['spatial'] = model.xe.obsm['spatial'].copy()
    factors_adata.write_h5ad(os.path.join(output_dir, 'factors.h5ad'))
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    
    # Save metrics
    if metrics:
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Visualize factors
    plt.figure(figsize=(15, 10))
    sns.heatmap(factors[:100, :], cmap='viridis')
    plt.xlabel('Factors')
    plt.ylabel('Cells')
    plt.title('Denoised Factor Matrix (First 100 Cells)')
    plt.savefig(os.path.join(output_dir, 'factor_heatmap.png'))
    
    # Visualize loadings
    plt.figure(figsize=(15, 10))
    sns.heatmap(loadings[:, :50], cmap='viridis')
    plt.xlabel('Genes (first 50)')
    plt.ylabel('Factors')
    plt.title('Gene Loadings')
    plt.savefig(os.path.join(output_dir, 'loading_heatmap.png'))
    
    # Visualize spatial distribution of dominant factor
    dominant_factor = np.argmax(factors, axis=1)
    
    # Create a new AnnData object with the dominant factor
    spatial_adata = AnnData(X=np.zeros((factors.shape[0], 1)))
    spatial_adata.obs['dominant_factor'] = dominant_factor
    spatial_adata.obsm['spatial'] = model.xe.obsm['spatial'].copy()
    
    # Save for later visualization with scanpy
    spatial_adata.write_h5ad(os.path.join(output_dir, 'spatial_factors.h5ad'))
    
    # Basic spatial plot
    plt.figure(figsize=(10, 10))
    plt.scatter(
        spatial_adata.obsm['spatial'][:, 0],
        spatial_adata.obsm['spatial'][:, 1],
        c=dominant_factor,
        cmap='tab20',
        s=5,
        alpha=0.7
    )
    plt.title('Spatial Distribution of Dominant Factors')
    plt.axis('equal')
    plt.colorbar(label='Factor')
    plt.savefig(os.path.join(output_dir, 'spatial_factors.png'))




