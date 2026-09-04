"""Model B/C: XGBoost classifier with sensible defaults for imbalanced fraud data."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import xgboost as xgb


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # set explicitly in build_model
    "n_jobs": -1,
    "verbosity": 0,
}


def build_xgboost_model(
    *,
    scale_pos_weight: float = 1.0,
    **params: Any,
) -> xgb.XGBClassifier:
    """Build an XGBoost classifier with sensible defaults for imbalanced data.

    Parameters
    ----------
    scale_pos_weight: weight for positive (abuse) class; typically n_neg / n_pos.
    params: override any default XGBoost parameter.

    Returns
    -------
    xgboost.XGBClassifier instance (not fitted).
    """
    cfg = {**DEFAULT_XGB_PARAMS, **params}
    cfg["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**cfg)


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: Optional[xgb.XGBClassifier] = None,
    eval_metric: str = "aucpr",
    verbose: bool = False,
    feature_names: list[str] | None = None,
) -> xgb.XGBClassifier:
    """Fit XGBoost with validation-set early stopping.

    Parameters
    ----------
    X_train, y_train: training data.
    X_val, y_val: validation data for early stopping.
    model: optional pre-built XGBClassifier; if None, builds default.
    eval_metric: metric for early stopping (e.g., "aucpr", "logloss").
    verbose: whether to print training progress.
    feature_names: optional list of feature names for the model.

    Returns
    -------
    Fitted XGBClassifier.
    """
    if model is None:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        model = build_xgboost_model(scale_pos_weight=n_neg / max(n_pos, 1))

    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": eval_metric,
        "early_stopping_rounds": 50,
        "verbose": verbose,
    }
    if feature_names is not None:
        fit_kwargs["feature_names"] = feature_names

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_proba_xgb(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) for the positive class (abuse)."""
    return model.predict_proba(X)[:, 1]


def get_feature_importance(model: xgb.XGBClassifier) -> dict[str, float]:
    """Return feature importances by 'gain'."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    return {k: float(v) for k, v in gain.items()}


def get_feature_importance_weight(model: xgb.XGBClassifier) -> dict[str, float]:
    """Return feature importances by 'weight' (usage count)."""
    booster = model.get_booster()
    weight = booster.get_score(importance_type="weight")
    return {k: float(v) for k, v in weight.items()}


__all__ = [
    "build_xgboost_model",
    "fit_xgboost",
    "predict_proba_xgb",
    "get_feature_importance",
    "get_feature_importance_weight",
]