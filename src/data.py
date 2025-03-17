import torch
import numpy as np
from anndata import AnnData
import scanpy as sc
from scipy.spatial import KDTree
from scipy.sparse import issparse


def gen_gamma(vi_sp, xe_sp, max_radius = 100, exclusive = True):
    """
    Generate a binary assignment matrix (gamma) between Xenium cells and Visium spots.
    
    Parameters:
    -----------
    vi_sp : numpy.ndarray
        Spatial coordinates of Visium spots, shape (n_spots, 2)
    xe_sp : numpy.ndarray
        Spatial coordinates of Xenium cells, shape (n_cells, 2)
    max_radius : float, default=100
        Maximum distance threshold for assignment
    exclusive : bool, default=True
        If True, each Xenium cell is assigned to at most one Visium spot
        (Currently only exclusive=True is supported)
        
    Returns:
    --------
    numpy.ndarray
        Binary assignment matrix of shape (n_cells, n_spots)
        Where gamma[i,j] = 1 if Xenium cell i is assigned to Visium spot j
    """
    assert exclusive
    ret = np.zeros((xe_sp.shape[0], vi_sp.shape[0]))
    kdt = KDTree(vi_sp)
    for i in range(xe_sp.shape[0]):
        d, vi_idx = kdt.query(xe_sp[i])  # Find nearest Visium spot for each Xenium cell
        if d <= max_radius:  # Only assign if within the maximum radius
            ret[i, vi_idx] = 1
    return ret

def prepare_data(vi_, xe_, holdouts, n_factors=20, max_radius=100):
    """
    Prepare data for the joint model
    
    Parameters:
    -----------
    vi_ : AnnData
        Visium data
    xe_ : AnnData
        Xenium data
    holdouts : list
        List of genes to hold out for evaluation
    n_factors : int, default=20
        Number of factors for NMF
    max_radius : float, default=100
        Maximum distance threshold for assignment
    
    Returns:
    --------
    sorted_vi : AnnData
        Processed Visium data
    sorted_xe : AnnData
        Processed Xenium data
    gamma : numpy.ndarray
        Assignment matrix between xenium and visium
    F_vi : numpy.ndarray
        NMF factors for Visium data
    F_xe : numpy.ndarray
        NMF factors for Xenium data
    """
    # Make copies to avoid modifying the original data
    vi, xe = vi_.copy(), xe_.copy()
    
    # Get gene names
    vcols = set(vi.var_names)
    xcols = set(xe.var_names)
    
    # Find common genes
    common_genes = vcols.intersection(xcols)
    
    # Identify holdout genes among common genes
    hcols = set(holdouts).intersection(common_genes)
    
    # Define training genes (common genes minus holdouts)
    train_genes = common_genes - hcols
    
    # Define test genes (holdouts)
    test_genes = hcols - train_genes
    
    # Sort xenium data to include only training genes
    sorted_xe = xe[:, list(train_genes)].copy()
    
    # Sort visium data to include both training and test genes
    vi_genes = list(train_genes) + list(test_genes)
    sorted_vi = vi[:, vi_genes]
    
    # Generate gamma matrix for cell-spot assignment
    vi_spatial = sorted_vi.obsm['spatial']
    xe_spatial = sorted_xe.obsm['spatial']
    gamma = gen_gamma(vi_spatial, xe_spatial, max_radius=max_radius)
    
    return sorted_vi, sorted_xe, gamma


# Example usage
if __name__ == "__main__":
    # Load your data
    vi = sc.read_h5ad("path/to/visium_data.h5ad")
    xe = sc.read_h5ad("path/to/xenium_data.h5ad")
    
    # Define holdout genes
    holdout_genes = ["GENE1", "GENE2", "GENE3"]
    
    # Prepare data
    sorted_vi, sorted_xe, gamma = prepare_data(vi, xe, holdout_genes)
    
    print(f"Visium data shape: {sorted_vi.shape}")
    print(f"Xenium data shape: {sorted_xe.shape}")
    print(f"Gamma matrix shape: {gamma.shape}")
    
    # Check which genes are in the training set
    train_genes = sorted_xe.var_names
    print(f"Number of training genes: {len(train_genes)}")
    
    # Check which genes are in the holdout set
    holdout_genes_found = [gene for gene in holdout_genes if gene in sorted_vi.var_names and gene not in sorted_xe.var_names]
    print(f"Holdout genes found: {holdout_genes_found}")