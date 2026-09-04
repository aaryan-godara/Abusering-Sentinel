"""Error analysis: false positive and false negative deep dives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluation import RingEvalResult, evaluate_hidden_scenarios


@dataclass
class FPAnalysis:
    """Detailed analysis of a false positive."""
    user_id: str
    probability: float
    prediction: int
    label: int
    group_type: str | None
    group_id: str | None
    ring_id: str | None
    ring_type: str | None
    shared_infra: dict[str, Any]
    top_features: dict[str, float]


@dataclass
class FNAnalysis:
    """Detailed analysis of a false negative."""
    user_id: str
    probability: float
    prediction: int
    label: int
    ring_id: str | None
    ring_type: str | None
    shared_infra: dict[str, Any]
    top_features: dict[str, float]


def collect_fp_analysis(
    users_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    feature_importance: dict[str, float] | None = None,
    top_k_features: int = 10,
) -> list[FPAnalysis]:
    """Collect detailed analysis for false positives.

    Returns a list of FPAnalysis objects sorted by probability descending.
    """
    # Handle both 'is_abuse_account' and 'label' column names
    label_col = "is_abuse_account" if "is_abuse_account" in users_df.columns else "label"
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]

    fps = []
    for idx in fp_indices:
        uid = users_df.iloc[idx]["user_id"]
        prob = float(y_prob[idx])

        # Shared infrastructure context
        infra_cols = ["device_id", "ip_id", "address_id", "payment_instrument_id"]
        shared = {}
        # We can't easily get this from user row alone; would need projection.
        # For now, record available user-level context.

        # Top contributing features (if importance available)
        top_feats = {}
        if feature_importance:
            sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_feats = dict(sorted_imp[:top_k_features])

        group_type = users_df.iloc[idx].get("group_type")
        group_id = users_df.iloc[idx].get("group_id")
        ring_id = users_df.iloc[idx].get("ring_id")
        ring_type = users_df.iloc[idx].get("ring_type")

        fps.append(FPAnalysis(
            user_id=uid,
            probability=prob,
            prediction=1,
            label=0,
            group_type=group_type,
            group_id=group_id,
            ring_id=ring_id,
            ring_type=ring_type,
            shared_infra=shared,
            top_features=top_feats,
        ))

    # Sort by probability descending (most confident FPs first)
    fps.sort(key=lambda x: x.probability, reverse=True)
    return fps


def collect_fn_analysis(
    users_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    feature_importance: dict[str, float] | None = None,
    top_k_features: int = 10,
) -> list[FNAnalysis]:
    """Collect detailed analysis for false negatives (missed abuse)."""
    label_col = "is_abuse_account" if "is_abuse_account" in users_df.columns else "label"
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    fns = []
    for idx in fn_indices:
        uid = users_df.iloc[idx]["user_id"]
        prob = float(y_prob[idx])

        top_feats = {}
        if feature_importance:
            sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_feats = dict(sorted_imp[:top_k_features])

        ring_id = users_df.iloc[idx].get("ring_id")
        ring_type = users_df.iloc[idx].get("ring_type")

        fns.append(FNAnalysis(
            user_id=uid,
            probability=prob,
            prediction=0,
            label=1,
            ring_id=ring_id,
            ring_type=ring_type,
            shared_infra={},
            top_features=top_feats,
        ))

    fns.sort(key=lambda x: x.probability)  # lowest prob first (hardest to detect)
    return fns


def summarize_fps_by_group(
    fps: list[Any],
    users_df: pd.DataFrame,
) -> dict[str, int]:
    """Summarize false positives by legitimate group type."""
    group_counts = {}
    for fp in fps:
        gt = fp.group_type or "individual"
        group_counts[gt] = group_counts.get(gt, 0) + 1
    return group_counts


def summarize_fns_by_ring_type(
    fns: list[Any],
    users_df: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Summarize false negatives by abuse ring type."""
    ring_summary: dict[str, dict[str, int]] = {}
    for fn in fns:
        rt = fn.ring_type or "unknown"
        d = ring_summary.setdefault(rt, {"total": 0, "fns": 0})
        d["total"] += 1
        d["fns"] += 1
    return ring_summary


def fp_rate_by_legit_group(
    users_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate false positive rate per legitimate group type."""
    label_col = "is_abuse_account" if "is_abuse_account" in users_df.columns else "label"
    legit = users_df[~users_df[label_col].astype(bool)].copy()
    legit["pred"] = y_pred[~y_true.astype(bool)]

    rates = {}
    for gt, group in legit.groupby("group_type"):
        total = len(group)
        fps = int(group["pred"].sum())
        rates[gt] = fps / total if total > 0 else 0.0
    # Also for individuals
    ind = legit[legit["group_type"] == "individual"]
    if len(ind) > 0:
        rates["individual"] = int(ind["pred"].sum()) / len(ind)

    return rates


def fn_recall_by_ring_type(
    users_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate recall per abuse ring type."""
    label_col = "is_abuse_account" if "is_abuse_account" in users_df.columns else "label"
    abuse = users_df[users_df[label_col].astype(bool)].copy()
    abuse["pred"] = y_pred[users_df[label_col].astype(bool)]

    recalls = {}
    for rt, group in abuse.groupby("ring_type"):
        tp = int(group["pred"].sum())
        total = len(group)
        recalls[rt] = tp / total if total > 0 else 0.0

    return recalls


__all__ = [
    "FPAnalysis",
    "FNAnalysis",
    "collect_fp_analysis",
    "collect_fn_analysis",
    "summarize_fps_by_group",
    "summarize_fns_by_ring_type",
    "fp_rate_by_legit_group",
    "fn_recall_by_ring_type",
]