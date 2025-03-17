from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
import numpy as np

def score_corr(x, y, metric="r2"):
    """
    All-in-one scoring function. All metrics are structured such that larger = better correlation.
    
    Parameters:
    -----------
    x : array-like
        First data array
    y : array-like
        Second data array
    metric : str
        Metric to compute: "r2", "pcc", "ssim", "rmse", "js", or "nnz"
        
    Returns:
    --------
    float
        Correlation score
    """
    # Flatten ndarray if needed
    if isinstance(x, np.ndarray):
        xf = x.flatten()
    else:
        xf = np.array(x)
    if isinstance(y, np.ndarray):
        yf = y.flatten()
    else:
        yf = np.array(y)
        
    assert len(xf) == len(yf)
    
    if (np.any(xf == 0)) or (np.any(yf == 0)):
        if metric == "js":
            xf = xf + 1e-10
            yf = yf + 1e-10
            
    if (np.std(xf) < 1e-5) or (np.std(yf) < 1e-5):
        if metric in ['r2', 'pcc', 'ssim']:
            return 0
        elif metric == 'rmse':
            return -1
            
    n = len(xf)
    
    if metric == "r2":
        corr_matrix = np.corrcoef(xf, yf)
        return corr_matrix[0, 1]**2
    elif metric == "pcc":
        mx, my = np.mean(xf), np.mean(yf)
        a = np.mean((xf - mx) * (yf - my))
        b = np.std(xf) * np.std(yf)
        return a / b
    elif metric == "ssim":
        x_, y_ = xf / max(xf), yf / max(yf)
        mx, my = np.mean(x_), np.mean(y_)
        sx, sy = np.std(x_), np.std(y_)
        cov = np.cov(x_, y_)[0][1]
        c1 = 0.01
        c2 = 0.03
        return (2 * mx * my + c1 ** 2) * (2 * cov + c2 ** 2) / (mx ** 2 + my ** 2 + c1 ** 2) / (sx ** 2 + sy ** 2 + c2 ** 2)
    elif metric == "rmse":
        mx, my = np.mean(xf), np.mean(yf)
        sx, sy = np.std(xf), np.std(yf)
        zx = (xf - mx) / sx
        zy = (yf - my) / sy
        avg = np.mean((zx - zy) ** 2)
        return -(avg ** 0.5)
    elif metric == "js":
        xf = xf / sum(xf)
        yf = yf / sum(yf)
        return -(np.sum(xf * np.log(xf / yf)) + np.sum(yf * np.log(yf / xf))) / 2
    elif metric == 'mae':
        return np.mean(np.abs(xf - yf))
    elif metric == "nnz":
        nnz = xf > 0.1
        count = sum(nnz)
        if count == 0:
            return 0  # x is all-zero, meaningless
        sorted_est = np.argsort(yf)
        overlap = 0
        for i in sorted_est[-count:]:
            overlap += nnz[i]
        return overlap / count

def avg_corr_by_row(gt, est, met="r2", eps=0):
    """
    Calculate average correlation by row
    
    Parameters:
    -----------
    gt : array-like
        Ground truth data
    est : array-like
        Estimated data
    met : str
        Metric to use
    eps : float
        Small value to add to avoid division by zero
        
    Returns:
    --------
    float
        Average correlation
    """
    assert gt.shape == est.shape
    data = []
    for i in range(gt.shape[0]):
        data.append(score_corr(gt[i] + eps, est[i] + eps, metric=met))
    return sum(data) / len(data)

def corr_by_col(gt, est, met="r2", eps=0):
    """
    Calculate correlation for each column
    
    Parameters:
    -----------
    gt : array-like
        Ground truth data
    est : array-like
        Estimated data
    met : str
        Metric to use
    eps : float
        Small value to add to avoid division by zero
        
    Returns:
    --------
    list
        List of correlations for each column
    """
    assert gt.shape == est.shape
    data = []
    for i in range(gt.shape[1]):
        data.append(score_corr(gt[:, i] + eps, est[:, i] + eps, metric=met))
    return data

def avg_corr_by_col(gt, est, met="r2", eps=0):
    """
    Calculate average correlation by column
    
    Parameters:
    -----------
    gt : array-like
        Ground truth data
    est : array-like
        Estimated data
    met : str
        Metric to use
    eps : float
        Small value to add to avoid division by zero
        
    Returns:
    --------
    float
        Average correlation
    """
    return np.mean(corr_by_col(gt, est, met, eps))

def row_norm(mat):
    """
    Normalize rows of a matrix to sum to 1
    
    Parameters:
    -----------
    mat : array-like
        Matrix to normalize
        
    Returns:
    --------
    array-like
        Normalized matrix
    """
    if mat is None:
        return None
    row_sums = mat.sum(axis=1)
    mask = row_sums != 0
    ret = np.copy(mat)
    ret[mask] = ret[mask] / row_sums[mask].reshape((-1, 1))
    return ret

def evaluate_imputation(true_data, imputed_data, output_dir, normalize=False):
    """
    Evaluate imputation performance
    
    Parameters:
    -----------
    true_data : AnnData
        True expression data
    imputed_data : AnnData
        Imputed expression data
    output_dir : str
        Directory to save evaluation plots
    normalize : bool
        Whether to normalize data before evaluation
    
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    metrics = {}
    
    # Make sure the data is aligned
    common_cells = list(set(true_data.obs_names).intersection(set(imputed_data.obs_names)))
    common_genes = list(set(true_data.var_names).intersection(set(imputed_data.var_names)))
    
    true_subset = true_data[common_cells, common_genes].X
    imputed_subset = imputed_data[common_cells, common_genes].X
    
    # Convert to dense if sparse
    if isinstance(true_subset, np.ndarray) == False:
        true_subset = true_subset.toarray()
    if isinstance(imputed_subset, np.ndarray) == False:
        imputed_subset = imputed_subset.toarray()
    
    # Normalize if requested
    if normalize:
        true_subset = row_norm(true_subset)
        imputed_subset = row_norm(imputed_subset)
    
    # Calculate metrics
    r2 = score_corr(true_subset, imputed_subset, metric="r2")
    pcc = score_corr(true_subset, imputed_subset, metric="pcc")
    mae = score_corr(true_subset, imputed_subset, metric="mae")
    ssim = score_corr(true_subset, imputed_subset, metric="ssim")
    rmse = score_corr(true_subset, imputed_subset, metric="rmse")
    js = score_corr(true_subset, imputed_subset, metric="js")
    nnz = score_corr(true_subset, imputed_subset, metric="nnz")
    
    # Calculate cell-wise and gene-wise correlations
    cell_r2 = avg_corr_by_row(true_subset, imputed_subset, met="r2")
    gene_r2 = avg_corr_by_col(true_subset, imputed_subset, met="r2")
    
    metrics['r2'] = r2
    metrics['pcc'] = pcc
    metrics['mae'] = mae
    metrics['ssim'] = ssim
    metrics['rmse'] = rmse
    metrics['js'] = js
    metrics['nnz'] = nnz
    metrics['cell_r2'] = cell_r2
    metrics['gene_r2'] = gene_r2
    
    print(f"Evaluation metrics:")
    print(f"  R²: {r2:.4f}")
    print(f"  PCC: {pcc:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  SSIM: {ssim:.4f}")
    print(f"  RMSE: {-rmse:.4f}")  # Convert back to positive for display
    print(f"  JS: {-js:.4f}")      # Convert back to positive for display
    print(f"  NNZ: {nnz:.4f}")
    print(f"  Cell-wise R²: {cell_r2:.4f}")
    print(f"  Gene-wise R²: {gene_r2:.4f}")
    
    # Plot true vs imputed values
    plt.figure(figsize=(10, 10))
    plt.scatter(true_subset.flatten(), imputed_subset.flatten(), alpha=0.1)
    plt.xlabel('True Expression')
    plt.ylabel('Imputed Expression')
    plt.title(f'True vs Imputed Expression (R²={r2:.4f}, PCC={pcc:.4f})')
    plt.savefig(os.path.join(output_dir, 'true_vs_imputed.png'))
    
    # Plot gene-wise correlations
    gene_correlations = corr_by_col(true_subset, imputed_subset, met="r2")
    gene_correlation_pairs = list(zip(common_genes, gene_correlations))
    gene_correlation_pairs.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 6))
    genes, corrs = zip(*gene_correlation_pairs)
    plt.bar(range(len(genes)), corrs)
    plt.xlabel('Gene')
    plt.ylabel('R² Correlation')
    plt.title('Gene-wise Correlation between True and Imputed Expression')
    plt.xticks(range(len(genes)), genes, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_correlations.png'))
    
    return metrics