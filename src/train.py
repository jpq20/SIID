import torch
import numpy as np
import scanpy as sc
import os
import time
import argparse
from anndata import AnnData


# Import our data preparation function
from data import prepare_data
from model import joint_model
from eval import evaluate_imputation
from utils import *


def train_and_impute(vi_path, xe_path, output_dir, 
                     hdim=20, ent_div=400, seed=10042, lr=1e-3, 
                     epochs=5000, device="cuda:0", k_neighbors=10, 
                     gcn_hidden_dim=64, gcn_layers=2, verbose=True):
    """
    Train the joint model and impute holdout genes
    
    Parameters:
    -----------
    vi_path : str
        Path to Visium AnnData file
    xe_path : str
        Path to Xenium AnnData file
    holdout_genes : list
        List of genes to hold out for evaluation
    output_dir : str
        Directory to save results
    hdim : int
        Number of latent factors
    ent_div : float
        Entropy regularization denominator
    seed : int
        Random seed
    lr : float
        Learning rate
    epochs : int
        Number of training epochs
    device : str
        Device to run the model on
    k_neighbors : int
        Number of neighbors for spatial graph
    gcn_hidden_dim : int
        Hidden dimension for GCN layers
    gcn_layers : int
        Number of GCN layers
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    imputed : AnnData
        Imputed expression for holdout genes
    model : joint_model
        Trained model
    metrics : dict
        Evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if verbose:
        print(f"Loading data from {vi_path} and {xe_path}")
    vi = sc.read_h5ad(vi_path)
    xe = sc.read_h5ad(xe_path)
    
    # Find intersection of genes and randomly select 1/10 as holdout genes
    all_genes = list(set(vi.var_names).intersection(set(xe.var_names)))
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Randomly select 1/10 of intersection genes as holdout genes
    n_holdout = max(1, int(len(all_genes) / 10))
    holdout_genes = np.random.choice(all_genes, size=n_holdout, replace=False).tolist()
    
    if verbose:
        print(f"Randomly selected {len(holdout_genes)} holdout genes out of {len(all_genes)} intersection genes")
    
    # Prepare data
    if verbose:
        print(f"Preparing data with {len(holdout_genes)} holdout genes")
    sorted_vi, sorted_xe = prepare_data(vi, xe, holdout_genes)
    
    # Save a copy of the original data for the holdout genes
    holdout_xe = xe[:, holdout_genes].copy()
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Start timer
    start_time = time.time()
    
    # Define callback function to track progress
    losses = []
    def callback(model, epoch, loss):
        losses.append(loss[0].item())
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}: Loss = {loss[0].item():.4f}")
    
    # Initialize the joint model
    if verbose:
        print(f"Initializing model with {hdim} factors, {k_neighbors} neighbors, {gcn_layers} GCN layers")
    model = joint_model(
        xe=sorted_xe,
        vi=sorted_vi,
        n_factors=hdim,
        device=device,
        adaptive_entropy=True,
        adaptive_entropy_denominator=ent_div,
        poisson_sum=True,
        add_entropy=True,
        platform_scaling=True,
        k_neighbors=k_neighbors,
        gcn_hidden_dim=gcn_hidden_dim,
        gcn_layers=gcn_layers
    )
    
    # Train the model
    if verbose:
        print(f"Training model for {epochs} epochs")
    results = model.train(
        num_epochs=epochs,
        lr=lr,
        print_freq=100 if verbose else 0,
        loss_type='poisson',
        callback=callback
    )
    
    # Get predictions for all genes
    if verbose:
        print("Generating predictions")
    with torch.no_grad():
        xenium_est = model.get_xenium_est().cpu().numpy()
    
    # Create AnnData object with predictions
    full_pred = AnnData(X=xenium_est, obs=sorted_xe.obs, var=sorted_vi.var)
    
    # Extract predictions for holdout genes
    imputed = full_pred[:, holdout_genes].copy() if len(holdout_genes) > 0 else None
    
    # Calculate training time
    training_time = time.time() - start_time
    if verbose:
        print(f"Training finished in {training_time:.2f} seconds")
    
    # Evaluate predictions if we have holdout data
    metrics = {}
    if holdout_xe is not None and imputed is not None:
        metrics = evaluate_imputation(holdout_xe, imputed, output_dir)
    
    # Save results
    save_results(model, imputed, losses, metrics, output_dir)
    
    return imputed, model, metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train joint model and impute holdout genes')
    parser.add_argument('--vi_path', default='/home/pjiangag/main/mm/siid/BRCA/cast_output/visium_aligned.h5ad', type=str, help='Path to Visium AnnData file')
    parser.add_argument('--xe_path', default='/home/pjiangag/main/mm/siid/BRCA/cast_output/xenium_aligned.h5ad', type=str, help='Path to Xenium AnnData file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--hdim', type=int, default=20, help='Number of latent factors')
    parser.add_argument('--ent_div', type=float, default=400, help='Entropy regularization denominator')
    parser.add_argument('--seed', type=int, default=10042, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--k_neighbors', type=int, default=10, help='Number of neighbors for spatial graph')
    parser.add_argument('--gcn_hidden_dim', type=int, default=64, help='Hidden dimension for GCN layers')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    
    args = parser.parse_args()
    
    # Train and impute
    imputed, model, metrics = train_and_impute(
        vi_path=args.vi_path,
        xe_path=args.xe_path,
        output_dir=args.output_dir,
        hdim=args.hdim,
        ent_div=args.ent_div,
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        k_neighbors=args.k_neighbors,
        gcn_hidden_dim=args.gcn_hidden_dim,
        gcn_layers=args.gcn_layers,
        verbose=args.verbose
    )
    
    print(f"Results saved to {args.output_dir}")