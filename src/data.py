import torch
import numpy as np
from anndata import AnnData
import scanpy as sc

def prepare_data(vi_, xe_, holdouts):
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
    
    Returns:
    --------
    sorted_vi : AnnData
        Processed Visium data
    sorted_xe : AnnData
        Processed Xenium data
    gamma : numpy.ndarray
        Assignment matrix between xenium and visium
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
    
    return sorted_vi, sorted_xe


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