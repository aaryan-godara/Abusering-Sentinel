"""Threshold selection using validation data ONLY.

Implements multiple threshold strategies and selects the primary one by
maximizing F1 on the validation set. All thresholds are frozen after
validation selection; test set is never used for threshold tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


ThresholdStrategy = Literal[
    "max_f1",           # maximize F1 (primary)
    "high_recall",      # recall >= target, maximize precision
    "balanced",         # precision ~= recall
    "low_fp_rate",      # minimize FPR subject to recall floor
]


@dataclass
class ThresholdResult:
    """Result of threshold evaluation on a given dataset."""

    threshold: float
    strategy: str
    precision: float
    recall: float
    f1: float
    specificity: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    n_total: int


def evaluate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    """Compute confusion matrix metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return ThresholdResult(
        threshold=threshold,
        strategy="",
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        true_negatives=tn,
        n_total=len(y_true),
    )


def select_threshold_max_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, ThresholdResult]:
    """Select threshold maximizing F1 on the given data.

    Returns the best threshold and its metrics (computed on the SAME data).
    """
    # Use PR curve to get candidate thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds array has len N-1; add 1.0 for completeness
    thresholds = np.append(thresholds, 1.0)

    best_f1 = -1.0
    best_thresh = 0.5
    best_result = None

    for thresh in thresholds:
        res = evaluate_threshold(y_true, y_prob, thresh)
        if res.f1 > best_f1:
            best_f1 = res.f1
            best_thresh = thresh
            best_result = res

    assert best_result is not None
    return best_thresh, best_result


def select_threshold_high_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.80,
) -> tuple[float, ThresholdResult]:
    """Select the highest-precision threshold achieving at least min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    best_prec = -1.0
    best_thresh = 0.5
    best_result = None

    for thresh in thresholds:
        res = evaluate_threshold(y_true, y_prob, thresh)
        if res.recall >= min_recall and res.precision > best_prec:
            best_prec = res.precision
            best_thresh = thresh
            best_result = res

    if best_result is None:
        # Fallback: max F1
        return select_threshold_max_f1(y_true, y_prob)

    return best_thresh, best_result


def select_threshold_balanced(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, ThresholdResult]:
    """Select threshold where precision ≈ recall (minimize |P - R|)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    best_diff = float("inf")
    best_thresh = 0.5
    best_result = None

    for thresh in thresholds:
        res = evaluate_threshold(y_true, y_prob, thresh)
        diff = abs(res.precision - res.recall)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
            best_result = res

    assert best_result is not None
    return best_thresh, best_result


def select_threshold_low_fp_rate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.50,
) -> tuple[float, ThresholdResult]:
    """Select threshold minimizing FPR subject to min_recall floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    best_fpr = float("inf")
    best_thresh = 0.5
    best_result = None

    for thresh in thresholds:
        res = evaluate_threshold(y_true, y_prob, thresh)
        if res.recall >= min_recall:
            fpr = 1 - res.specificity
            if fpr < best_fpr:
                best_fpr = fpr
                best_thresh = thresh
                best_result = res

    if best_result is None:
        return select_threshold_max_f1(y_true, y_prob)

    return best_thresh, best_result


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: ThresholdStrategy = "max_f1",
) -> tuple[float, ThresholdResult]:
    """Dispatch to the appropriate threshold selection strategy."""
    if strategy == "max_f1":
        thresh, res = select_threshold_max_f1(y_true, y_prob)
        res.strategy = "max_f1"
    elif strategy == "high_recall":
        thresh, res = select_threshold_high_recall(y_true, y_prob)
        res.strategy = "high_recall"
    elif strategy == "balanced":
        thresh, res = select_threshold_balanced(y_true, y_prob)
        res.strategy = "balanced"
    elif strategy == "low_fp_rate":
        thresh, res = select_threshold_low_fp_rate(y_true, y_prob)
        res.strategy = "low_fp_rate"
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return thresh, res


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions at a fixed threshold."""
    return (y_prob >= threshold).astype(int)


__all__ = [
    "ThresholdResult",
    "ThresholdStrategy",
    "evaluate_threshold",
    "select_threshold",
    "select_threshold_max_f1",
    "select_threshold_high_recall",
    "select_threshold_balanced",
    "select_threshold_low_fp_rate",
    "apply_threshold",
]