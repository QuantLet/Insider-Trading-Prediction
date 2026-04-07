# evaluator.py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
    brier_score_loss, confusion_matrix
)

# ---------- Top-K ----------
def _topk_index(y_prob, k_frac):
    n = len(y_prob)
    topn = max(1, int(np.ceil(n * k_frac)))
    idx = np.argpartition(y_prob, -topn)[-topn:]
    return idx

def recall_at_k(y_true, y_prob, k_frac):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    idx = _topk_index(y_prob, k_frac)
    pos_total = int(y_true.sum())
    return 0.0 if pos_total == 0 else float(y_true[idx].sum() / pos_total)

def precision_at_k(y_true, y_prob, k_frac):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    idx = _topk_index(y_prob, k_frac)
    return float(y_true[idx].mean())

def topn_at_k(n, k_frac):
    """Top-K 实际样本数（与 evaluator 内部一致：ceil）"""
    return max(1, int(np.ceil(n * k_frac)))

def tp_at_k(y_true, y_prob, k_frac):
    """Top-K 命中的正类数（TP@K）"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    idx = _topk_index(y_prob, k_frac)
    return int(y_true[idx].sum())

# ---------- KS ----------
def ks_statistic(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    pos = np.sort(y_prob[y_true == 1])
    neg = np.sort(y_prob[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    data_all = np.sort(np.unique(y_prob))
    cdf_pos = np.searchsorted(pos, data_all, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, data_all, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))

# ---------- F-beta (safe) ----------
def fbeta_from_pr(p, r, beta):
    if p <= 0 or r <= 0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * p + r
    return 0.0 if denom == 0 else float((1 + b2) * p * r / denom)

def fbeta_score(y_true, y_pred, beta):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return fbeta_from_pr(p, r, beta)

# ---------- Best threshold ----------
def find_best_threshold(y_true, y_prob, metric="f1", grid=2000):
    """
    metric: "f1" / "f0.5" / "f2" / "youden"
    grid: 用分位点构造阈值网格（快，适合百万样本）
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    qs = np.linspace(0.0, 1.0, grid)
    thresholds = np.unique(np.quantile(y_prob, qs))
    thresholds = np.clip(thresholds, 0, 1)

    best_thr, best_val = 0.5, -np.inf

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)

        if metric == "f1":
            val = fbeta_from_pr(p, r, 1.0)
        elif metric in ["f0.5", "f05"]:
            val = fbeta_from_pr(p, r, 0.5)
        elif metric == "f2":
            val = fbeta_from_pr(p, r, 2.0)
        elif metric == "youden":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
            val = r - fpr
        else:
            raise ValueError("Unknown metric for threshold search")

        if val > best_val:
            best_val = val
            best_thr = float(thr)

    return best_thr

def plot_threshold_metrics(y_true, y_prob, best_thr, save_path=None, grid=2000):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    qs = np.linspace(0.0, 1.0, grid)
    thresholds = np.unique(np.quantile(y_prob, qs))
    thresholds = np.clip(thresholds, 0, 1)

    precs, recs, f1s = [], [], []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = fbeta_from_pr(p, r, 1.0)
        precs.append(p); recs.append(r); f1s.append(f1)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")
    plt.axvline(best_thr, linestyle="--", label=f"Best thr={best_thr:.3f}")
    plt.title("Threshold vs Precision / Recall / F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()

# ---------- Main API ----------
def evaluate_all(
    y_true,
    y_prob,
    base_thr=0.5,
    best_metric="f1",
    topk_fracs=(0.0001, 0.001, 0.01, 0.05),
    fig_path=None
):
    """
    返回:
      metrics: dict
      best_thr: float
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    out = {}

    # threshold-free
    if len(np.unique(y_true)) == 2:
        out["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        out["AP"] = average_precision_score(y_true, y_prob)
    else:
        out["ROC-AUC"] = np.nan
        out["AP"] = np.nan

    out["Brier"] = brier_score_loss(y_true, y_prob)
    out["KS"] = ks_statistic(y_true, y_prob)

    # Top-K
    for k in topk_fracs:
        pct = k * 100
        out[f"Recall@{pct:.02f}%"] = recall_at_k(y_true, y_prob, k)
        out[f"Precision@{pct:.02f}%"] = precision_at_k(y_true, y_prob, k)

    # Top-K
    for k in topk_fracs:
        pct = k * 100
        out[f"Recall@{pct:.02f}%"] = recall_at_k(y_true, y_prob, k)
        out[f"Precision@{pct:.02f}%"] = precision_at_k(y_true, y_prob, k)


    # Top-K
    n = len(y_true)
    for k in topk_fracs:
        pct = k * 100
    
        topn = topn_at_k(n, k)
        tp   = tp_at_k(y_true, y_prob, k)
        fp   = topn - tp
    
        out[f"TopN@{pct:.02f}%"] = topn
        out[f"TP@{pct:.02f}%"]   = tp
        out[f"FP@{pct:.02f}%"]   = fp  # ✅ 新增：Top-K 内误报数量
    
        out[f"Recall@{pct:.02f}%"] = recall_at_k(y_true, y_prob, k)
        out[f"Precision@{pct:.02f}%"] = precision_at_k(y_true, y_prob, k)


    # base threshold metrics
    y_pred_base = (y_prob >= base_thr).astype(int)
    out["Precision"] = precision_score(y_true, y_pred_base, zero_division=0)
    out["Recall"] = recall_score(y_true, y_pred_base, zero_division=0)
    out["F1"] = fbeta_score(y_true, y_pred_base, 1.0)
    out["F0.5"] = fbeta_score(y_true, y_pred_base, 0.5)
    out["F2"] = fbeta_score(y_true, y_pred_base, 2.0)
    out["Accuracy"] = accuracy_score(y_true, y_pred_base)
    out["CM@0.5"] = confusion_matrix(y_true, y_pred_base, labels=[0, 1])

    # best threshold
    best_thr = find_best_threshold(y_true, y_prob, metric=best_metric, grid=2000)

    if fig_path is not None:
        plot_threshold_metrics(y_true, y_prob, best_thr, save_path=fig_path, grid=2000)

    y_pred_h = (y_prob >= best_thr).astype(int)
    out["ROC-AUC-H"] = out["ROC-AUC"]
    out["AP-H"] = out["AP"]
    out["Precision-H"] = precision_score(y_true, y_pred_h, zero_division=0)
    out["Recall-H"] = recall_score(y_true, y_pred_h, zero_division=0)
    out["F1-H"] = fbeta_score(y_true, y_pred_h, 1.0)
    out["F0.5-H"] = fbeta_score(y_true, y_pred_h, 0.5)
    out["F2-H"] = fbeta_score(y_true, y_pred_h, 2.0)
    out["Accuracy-H"] = accuracy_score(y_true, y_pred_h)
    out["CM@H"] = confusion_matrix(y_true, y_pred_h, labels=[0, 1])

    out["best_threshold"] = best_thr
    return out, best_thr
