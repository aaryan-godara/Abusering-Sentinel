"""Evaluation metrics, cost analysis, ring-level evaluation, and bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


@dataclass
class MetricBundle:
    """Aggregated evaluation metrics for one dataset."""
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    specificity: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    n_total: int
    n_pos: int
    n_neg: int


@dataclass
class CostConfig:
    """Configurable cost assumptions (synthetic, not real Razorpay data)."""
    # Cost per false positive (blocked legitimate transaction value)
    fp_cost_per_incident: float = 5000.0  # INR
    # Cost per false negative (exposure from missed abuse)
    fn_cost_per_incident: float = 25000.0  # INR
    # Optional: average transaction value for FP
    avg_txn_value: float = 5000.0  # INR

    def __post_init__(self):
        # Reminder that these are synthetic
        assert self.fp_cost_per_incident > 0
        assert self.fn_cost_per_incident > 0


@dataclass
class CostAnalysis:
    """Cost analysis results."""
    false_positive_count: int
    false_negative_count: int
    fp_cost: float
    fn_cost: float
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "fp_cost": round(self.fp_cost, 2),
            "fn_cost": round(self.fn_cost, 2),
            "total_cost": round(self.total_cost, 2),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> MetricBundle:
    """Compute all core metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Probability-based metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return MetricBundle(
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        specificity=specificity,
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        true_negatives=int(tn),
        n_total=len(y_true),
        n_pos=n_pos,
        n_neg=n_neg,
    )


def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: CostConfig,
) -> CostAnalysis:
    """Compute cost analysis from predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fp_cost = fp * config.fp_cost_per_incident
    fn_cost = fn * config.fn_cost_per_incident

    return CostAnalysis(
        false_positive_count=int(fp),
        false_negative_count=int(fn),
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        total_cost=fp_cost + fn_cost,
    )


def threshold_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CostConfig,
    n_thresholds: int = 100,
) -> list[dict]:
    """Evaluate total cost across a range of thresholds."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cost = compute_cost(y_true, y_pred, config)
        metrics = compute_metrics(y_true, y_prob, thresh)
        results.append({
            "threshold": round(float(thresh), 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "fp": metrics.false_positives,
            "fn": metrics.false_negatives,
            "fp_cost": cost.fp_cost,
            "fn_cost": cost.fn_cost,
            "total_cost": cost.total_cost,
        })
    return results


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CostConfig,
) -> tuple[float, dict]:
    """Find the threshold minimizing total estimated cost."""
    curve = threshold_cost_curve(y_true, y_prob, config)
    best = min(curve, key=lambda x: x["total_cost"])
    return best["threshold"], best


# --- Bootstrap confidence intervals ---

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric.

    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boots = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        sample_prob = y_prob[idx]
        try:
            val = metric_fn(sample_y, sample_prob, threshold)
            boots.append(val)
        except Exception:
            pass

    if not boots:
        return (float("nan"), float("nan"))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boots, alpha * 100))
    upper = float(np.percentile(boots, (1 - alpha) * 100))
    return (lower, upper)


def bootstrap_metric_bundle_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for all core metrics."""
    metrics = [
        ("precision", lambda y, p, t: precision_score(y, (p >= t).astype(int), zero_division=0)),
        ("recall", lambda y, p, t: recall_score(y, (p >= t).astype(int), zero_division=0)),
        ("f1", lambda y, p, t: f1_score(y, (p >= t).astype(int), zero_division=0)),
        ("pr_auc", lambda y, p, t: average_precision_score(y, p)),
        ("roc_auc", lambda y, p, t: roc_auc_score(y, p)),
    ]
    cis = {}
    for name, fn in metrics:
        lower, upper = bootstrap_ci(y_true, y_prob, threshold, fn, n_bootstrap, confidence)
        cis[name] = (round(lower, 4), round(upper, 4))
    return cis


# --- Ring-level evaluation ---

@dataclass
class RingEvalResult:
    ring_id: str
    ring_type: str
    ring_size: int
    detected: bool
    detection_rate: float
    max_member_prob: float
    mean_member_prob: float


def evaluate_rings(
    users_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    min_detection_fraction: float = 0.5,
) -> list[RingEvalResult]:
    """Evaluate ring-level detection.

    A ring is considered detected if >= min_detection_fraction of its abuse
    accounts have probability >= threshold.
    """
    # Build user -> prob map
    user_prob = dict(zip(users_df["user_id"], y_prob))
    user_label = dict(zip(users_df["user_id"], users_df["is_abuse_account"]))

    results = []
    abuse_users = users_df[users_df["is_abuse_account"]].copy()

    for ring_id, group in abuse_users.groupby("ring_id"):
        if ring_id == "":
            continue
        ring_type = group["ring_type"].iloc[0] if "ring_type" in group.columns else "unknown"
        members = group["user_id"].tolist()
        size = len(members)

        probs = [user_prob.get(u, 0.0) for u in members]
        detected_count = sum(1 for p in probs if p >= threshold)
        detection_rate = detected_count / size if size > 0 else 0.0
        detected = detection_rate >= min_detection_fraction

        results.append(RingEvalResult(
            ring_id=ring_id,
            ring_type=ring_type,
            ring_size=size,
            detected=detected,
            detection_rate=round(detection_rate, 4),
            max_member_prob=round(max(probs) if probs else 0.0, 4),
            mean_member_prob=round(float(np.mean(probs)) if probs else 0.0, 4),
        ))

    return results


def ring_detection_summary(results: list[RingEvalResult]) -> dict:
    """Summarize ring-level detection by type and overall."""
    total = len(results)
    if total == 0:
        return {"total_rings": 0}

    detected = sum(1 for r in results if r.detected)
    overall_rate = detected / total

    by_type: dict[str, dict] = {}
    for r in results:
        d = by_type.setdefault(r.ring_type, {"total": 0, "detected": 0})
        d["total"] += 1
        if r.detected:
            d["detected"] += 1

    for rt, d in by_type.items():
        d["detection_rate"] = round(d["detected"] / d["total"], 4) if d["total"] > 0 else 0.0

    return {
        "total_rings": total,
        "rings_detected": detected,
        "overall_detection_rate": round(overall_rate, 4),
        "by_type": by_type,
        "min_detection_fraction": 0.5,
    }


# --- Hidden scenario evaluation ---

def evaluate_hidden_scenarios(
    users_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate performance on hidden test scenarios."""
    results = {}

    # Abuse scenarios: rings with scenario tag
    for scenario_name in ["low_overlap_ring", "high_noise_ring", "indirect_multihop_ring"]:
        abuse_users = users_df[
            (users_df["is_abuse_account"]) &
            (users_df["ring_id"].astype(str).str.contains(scenario_name))
        ]
        if len(abuse_users) == 0:
            results[scenario_name] = {"n_users": 0}
            continue

        scenario_probs = [y_prob[users_df.index.get_loc(uid)] for uid in abuse_users["user_id"]]
        tp = sum(1 for p in scenario_probs if p >= threshold)
        total = len(scenario_probs)
        recall = tp / total if total > 0 else 0.0

        results[scenario_name] = {
            "n_users": total,
            "detected": tp,
            "recall": round(recall, 4),
            "mean_prob": round(float(np.mean(scenario_probs)), 4),
        }

    # Legitimate high-connectivity groups (should be low FPR)
    # These are GRP_HIDDEN groups created as hidden scenarios
    legit_groups = users_df[
        (users_df["group_id"].astype(str).str.startswith("GRP_HIDDEN")) &
        (users_df["is_abuse_account"] == False) &
        (users_df["split"] == "test")
    ]
    # Focus on large groups
    large_groups = legit_groups[legit_groups.groupby("group_id")["user_id"].transform("size") >= 20]
    if len(large_groups) > 0:
        fps = sum(1 for uid in large_groups["user_id"] if y_prob[users_df.index.get_loc(uid)] >= threshold)
        fpr = fps / len(large_groups)
        results["legit_high_connectivity_fpr"] = {
            "n_users": len(large_groups),
            "false_positives": int(fps),
            "fpr": round(fpr, 4),
        }

    return results


__all__ = [
    "MetricBundle",
    "CostConfig",
    "CostAnalysis",
    "compute_metrics",
    "compute_cost",
    "threshold_cost_curve",
    "find_cost_optimal_threshold",
    "bootstrap_ci",
    "bootstrap_metric_bundle_ci",
    "RingEvalResult",
    "evaluate_rings",
    "ring_detection_summary",
    "evaluate_hidden_scenarios",
]