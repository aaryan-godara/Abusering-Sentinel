"""Explainability: feature importance and SHAP analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


FeatureImportanceMethod = Literal["gain", "weight", "cover"]


@dataclass
class FeatureImportanceReport:
    """Aggregated feature importance report."""
    importances: dict[str, float]
    method: str
    top_k: int

    def to_dataframe(self) -> "pd.DataFrame":
        return pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in self.importances.items()]
        ).sort_values("importance", ascending=False)

    def grouped_by_type(self) -> dict[str, dict[str, float]]:
        """Group by feature type (baseline vs graph subgroups)."""
        from src.models.ablation import FEATURE_GROUPS

        feat_to_group = {}
        for group_name, feats in FEATURE_GROUPS.items():
            for f in feats:
                feat_to_group[f] = group_name

        grouped: dict[str, dict[str, float]] = {}
        for feat, imp in self.importances.items():
            group = feat_to_group.get(feat, "unknown")
            if group not in grouped:
                grouped[group] = {}
            grouped[group][feat] = self.importances[feat]

        return grouped


def get_xgb_importance(
    model,
    method: Literal["gain", "weight", "cover"] = "gain",
    top_k: int = 50,
    feature_names: list[str] | None = None,
) -> FeatureImportanceReport:
    """Extract feature importance from fitted XGBoost model."""
    booster = model.get_booster()
    
    # Set feature names if provided
    if feature_names is not None:
        booster.feature_names = feature_names
    
    raw = booster.get_score(importance_type=method)
    # Sort and take top-k
    sorted_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return FeatureImportanceReport(
        importances=dict(sorted_items),
        method=method,
        top_k=top_k,
    )


def compute_shap_values(
    model,
    X: np.ndarray,
    max_samples: int = 1000,
    random_state: int = 42,
) -> np.ndarray | None:
    """Compute SHAP values for a fitted XGBoost model."""
    try:
        import shap
    except ImportError:
        return None

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values


def shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> dict[str, Any]:
    """Generate summary statistics from SHAP values."""
    if shap_values is None:
        return {}

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]

    return {
        "mean_abs_shap": {feat: float(mean_abs[i]) for i, feat in enumerate(feature_names)},
        "top_features": [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in top_idx
        ],
    }


def shap_dependence_data(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    feature_idx: int,
    sample: int = 500,
) -> dict[str, list]:
    """Extract data for SHAP dependence plot of one feature."""
    if shap_values is None:
        return {}

    n = len(X)
    if n > sample:
        idx = np.random.default_rng(42).choice(n, sample, replace=False)
        shap_vals = shap_values[idx, feature_idx]
        x_vals = X[idx, feature_idx]
    else:
        shap_vals = shap_values[:, feature_idx]
        x_vals = X[:, feature_idx]

    return {
        "feature": feature_names[feature_idx],
        "x_values": x_vals.tolist(),
        "shap_values": shap_vals.tolist(),
    }


__all__ = [
    "FeatureImportanceReport",
    "get_xgb_importance",
    "compute_shap_values",
    "shap_summary",
    "shap_dependence_data",
]