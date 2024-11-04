# Evaluation: Contains metrics for imputation (for now).
import numpy as np
import pickle
import os.path

version = 2

score_metrics = ["r2", "pcc", "ssim", "rmse", "js", "nnz"]
score_metrics_vs_src = ["r2", "pcc", "ssim", "rmse", "js"]

def score_corr(x, y, metric = "r2"):
    # All-in-one scoring function. All of them are structured such that larger = better correlation.
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
    # The following 5 metrics comes from the Nat Meth paper
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
        return - (avg ** 0.5)
    elif metric == "js":
        xf = xf / sum(xf)
        yf = yf / sum(yf)
        return - (np.sum(xf * np.log(xf / yf)) + np.sum(yf * np.log(yf / xf))) / 2
    elif metric == "nnz":
        # A custom metric. It works as follows: Identify non-zero elements in x, and estimate the overlap between the indices
        # and the top-N indices in y (N = # nnz).
        # Different from everything else: requires x and y to be ordered - x be the observation and y be the estimate.
        nnz = xf > 0.1
        count = sum(nnz)
        if count == 0:
            return 0  # x is all-zero, meaningless
        sorted_est = np.argsort(yf)
        overlap = 0
        for i in sorted_est[-count:]:
            overlap += nnz[i]
        return overlap / count

def avg_corr_by_row(gt, est, met = "r2", eps = 0):
    assert gt.shape == est.shape
    data = []
    for i in range(gt.shape[0]):
        data.append(score_corr(gt[i] + eps, est[i] + eps, metric = met))
    return sum(data) / len(data)

def corr_by_col(gt, est, met = "r2", eps = 0):
    assert gt.shape == est.shape
    data = []
    for i in range(gt.shape[1]):
        data.append(score_corr(gt[:, i] + eps, est[:, i] + eps, metric = met))
    return data

def avg_corr_by_col(gt, est, met = "r2", eps = 0):
    return np.mean(corr_by_col(gt, est, met, eps))

def row_norm(mat):
    if mat is None:
        return None
    row_sums = mat.sum(axis=1)
    mask = row_sums != 0
    ret = np.copy(mat)
    ret[mask] = ret[mask] / row_sums[mask].reshape((-1, 1))
    return ret

def print_triplet_scores(prompt, pred, obs, src, met = "r2", normalize = False):
    '''
    Print triplet scores.
    @param prompt: Prompt text before the output line.
    @param pred: Numpy array of predicted expression.
    @param obs: numpy array of observed counts.
    @param src: Numpy array of Poisson rates for generating obs.
    '''
    if normalize:
        pred, obs, src = row_norm(pred), row_norm(obs), row_norm(src)
    assert pred.shape == obs.shape
    obs_sc = avg_corr_by_col(pred.copy(), obs.copy(), met)
    if src is not None:
        assert obs.shape == src.shape
        ub = avg_corr_by_col(obs.copy(), src.copy(), met)
        src = avg_corr_by_col(pred.copy(), src.copy(), met)
        print(f"[{prompt}] Obs: {obs_sc:.4f} (UB {ub:.4f}) Src: {src:.4f}")
        return obs_sc, ub, src
    else:
        print(f"[{prompt}] Obs: {obs_sc:.4f}")
        return obs_sc, None, None