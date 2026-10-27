from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score


def expected_cost(y_true, y_pred, c_fp: float = 1.0, c_fn: float = 100.0) -> float:
    """Average asymmetric classification loss per sample."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((c_fp * fp + c_fn * fn) / max(1, len(y_true)))


def binary_metrics(y_true, y_pred, score=None, c_fp: float = 1.0, c_fn: float = 100.0) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    out = {
        "expected_cost": expected_cost(y_true, y_pred, c_fp=c_fp, c_fn=c_fn),
        "attack_recall": float(recall),
        "fpr": float(fpr),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, np.asarray(score)))
    else:
        out["pr_auc"] = float("nan")
    return out


def best_threshold_by_cost(y_true, score, c_fp: float = 1.0, c_fn: float = 100.0) -> tuple[float, float]:
    """Choose threshold using validation data only.

    The grid combines score quantiles with the Bayes cost threshold and 0.5.
    Returns (threshold, validation_expected_cost).
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        raise ValueError("No finite validation scores available for threshold tuning.")
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 401))
    bayes_threshold = c_fp / (c_fp + c_fn)
    candidates = np.unique(np.clip(np.r_[quantiles, bayes_threshold, 0.5], 0.0, 1.0))
    costs = np.asarray([
        expected_cost(y_true, score >= t, c_fp=c_fp, c_fn=c_fn) for t in candidates
    ])
    idx = int(np.argmin(costs))
    return float(candidates[idx]), float(costs[idx])
