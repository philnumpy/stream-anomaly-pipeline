"""Anomaly detection metrics, including the point-adjusted variant."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def prf(pred: np.ndarray, label: np.ndarray) -> dict:
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """If any point inside a true anomaly segment is flagged, count the whole
    segment as detected. Common in the literature (OmniAnomaly and later), but
    it inflates F1 a lot, so it is reported separately from the plain F1."""
    adjusted = pred.copy()
    edges = np.diff(np.concatenate([[0], label.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for start, end in zip(starts, ends):
        if pred[start:end].any():
            adjusted[start:end] = True
    return adjusted


def best_f1(scores: np.ndarray, label: np.ndarray, adjust: bool = False, steps: int = 200) -> dict:
    """Highest F1 over a sweep of thresholds. This uses the test labels to
    pick the threshold, so it is an upper bound, not a deployable result."""
    best = {"f1": 0.0}
    for threshold in np.quantile(scores, np.linspace(0.5, 0.9999, steps)):
        pred = scores > threshold
        if adjust:
            pred = point_adjust(pred, label)
        result = prf(pred, label)
        if result["f1"] > best["f1"]:
            best = {**result, "threshold": float(threshold)}
    return best


def evaluate(scores: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    label = label.astype(bool)
    pred = scores > threshold
    return {
        "threshold": float(threshold),
        "plain": prf(pred, label),
        "point_adjusted": prf(point_adjust(pred, label), label),
        "best_plain": best_f1(scores, label),
        "best_point_adjusted": best_f1(scores, label, adjust=True),
        "auroc": float(roc_auc_score(label, scores)),
        "auprc": float(average_precision_score(label, scores)),
        "anomaly_rate": float(label.mean()),
    }
