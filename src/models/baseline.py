"""Model A: Logistic Regression baseline.

Uses only the 23 non-graph baseline features with class_weight='balanced'
to handle the 7% abuse prevalence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_model(
    *,
    C: float = 1.0,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str | dict | None = "balanced",
    **kwargs: Any,
) -> Pipeline:
    """Build a scaled Logistic Regression pipeline.

    Parameters
    ----------
    C: inverse regularization strength (smaller = stronger regularization).
    random_state: seed for reproducibility.
    max_iter: maximum iterations for solver convergence.
    class_weight: "balanced" recommended for 7% abuse class; None to disable.
    kwargs: passed to LogisticRegression.

    Returns
    -------
    sklearn.pipeline.Pipeline with StandardScaler + LogisticRegression.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    random_state=random_state,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    solver="lbfgs",
                    **kwargs,
                ),
            ),
        ]
    )


def fit_logistic(X: np.ndarray, y: np.ndarray, model: Pipeline | None = None) -> Pipeline:
    """Fit the logistic regression model.

    Parameters
    ----------
    X: feature matrix (n_samples, n_features).
    y: target vector (0/1).
    model: optional pre-built pipeline; if None, builds a default.

    Returns
    -------
    Fitted pipeline.
    """
    if model is None:
        model = build_logistic_model()
    model.fit(X, y)
    return model


def predict_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) for the positive class (abuse)."""
    return model.predict_proba(X)[:, 1]


__all__ = ["build_logistic_model", "fit_logistic", "predict_proba"]