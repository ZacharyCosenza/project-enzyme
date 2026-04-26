import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, ndcg_score


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho, pval = spearmanr(y_pred, y_true)
    return {'spearman_rho': float(rho), 'spearman_pval': float(pval), 'n': len(y_true)}


def kendall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tau, pval = kendalltau(y_pred, y_true)
    return {'kendall_tau': float(tau), 'kendall_pval': float(pval)}


NDCG_K = 50


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = NDCG_K) -> dict:
    # ndcg_score expects 2D arrays and non-negative relevance scores
    relevance = y_true - y_true.min()
    score = ndcg_score(relevance[np.newaxis], y_pred[np.newaxis], k=k)
    return {f'ndcg_at_{k}': float(score)}


def auc_stabilizing(y_true: np.ndarray, y_pred: np.ndarray, wt_tm: float) -> dict:
    # binary: 1 if mutation is stabilizing (Tm > wildtype), 0 otherwise
    labels = (y_true > wt_tm).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {'auc_stabilizing': float('nan')}
    return {'auc_stabilizing': float(roc_auc_score(labels, y_pred))}


def report(metrics: dict) -> None:
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
