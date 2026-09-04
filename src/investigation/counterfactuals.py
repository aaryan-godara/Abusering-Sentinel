"""Lightweight counterfactual analysis using the frozen Model C.

For selected features, we modify values within valid bounds, re-score with the
frozen model, and compare probabilities. This describes **model sensitivity**,
NOT causal effects.

The model is never retrained. Ground-truth fields are never modified.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.investigation.schema import CounterfactualResult


# Features suitable for counterfactual analysis with their valid perturbation
# strategies.  Only observable, non-label features are included.
#
# Strategy types:
#   "zero"       → set to 0  (e.g. remove shared-device connections)
#   "median"     → set to the training median
#   "percentile" → set to a specific percentile of training distribution
#   "halve"      → halve the current value
#   "double"     → double the current value
#   "negate"     → replace current value with 0 if > median, else median

_DEFAULT_COUNTERFACTUAL_FEATURES: tuple[str, ...] = (
    "number_of_accounts_sharing_devices",
    "number_of_accounts_sharing_ips",
    "number_of_accounts_sharing_addresses",
    "number_of_accounts_sharing_payment_instruments",
    "promo_usage_rate",
    "clustering_coefficient",
    "transaction_frequency",
    "two_hop_user_count",
    "betweenness_centrality",
    "night_activity_rate",
)


class CounterfactualAnalyzer:
    """Generates counterfactual explanations using the frozen Model C.

    Initialised once with the frozen model, feature names, and training-set
    statistics (medians / percentiles). Per-user counterfactuals are cheap
    numpy operations + one model.predict_proba call per feature.
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        training_medians: np.ndarray,
        training_p25: np.ndarray,
        training_p75: np.ndarray,
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        self._name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(feature_names)
        }
        self._medians = training_medians
        self._p25 = training_p25
        self._p75 = training_p75

    def _predict_proba(self, X_row: np.ndarray) -> float:
        """Score a single sample with the frozen model."""
        row = X_row.reshape(1, -1)
        proba = self._model.predict_proba(row)
        # XGBoost returns (n_samples, 2) → take class-1 probability
        if proba.ndim == 2:
            return float(proba[0, 1])
        return float(proba[0])

    def analyze(
        self,
        X_row: np.ndarray,
        features: Sequence[str] | None = None,
    ) -> list[CounterfactualResult]:
        """Run counterfactual analysis for one user's feature vector.

        Parameters
        ----------
        X_row : 1-D feature vector for the user
        features : features to perturb (defaults to a curated set)

        Returns
        -------
        List of CounterfactualResult sorted by |probability_delta| descending.
        """
        features = features or list(_DEFAULT_COUNTERFACTUAL_FEATURES)
        original_prob = self._predict_proba(X_row)
        results: list[CounterfactualResult] = []

        for feat in features:
            idx = self._name_to_idx.get(feat)
            if idx is None:
                continue

            original_val = float(X_row[idx])
            median_val = float(self._medians[idx])

            # Determine a sensible counterfactual value
            if original_val > median_val:
                cf_val = median_val
            else:
                cf_val = float(self._p75[idx])

            # Skip if the counterfactual is essentially the same
            if abs(cf_val - original_val) < 1e-9:
                continue

            # Modify the feature and re-score
            modified = X_row.copy()
            modified[idx] = cf_val
            cf_prob = self._predict_proba(modified)

            results.append(
                CounterfactualResult(
                    feature=feat,
                    original_value=round(original_val, 6),
                    counterfactual_value=round(cf_val, 6),
                    original_probability=round(original_prob, 6),
                    counterfactual_probability=round(cf_prob, 6),
                    probability_delta=round(cf_prob - original_prob, 6),
                )
            )

        # Sort by absolute delta descending (most impactful first)
        results.sort(key=lambda r: abs(r.probability_delta), reverse=True)
        return results

    @staticmethod
    def from_training_data(
        model,
        feature_names: list[str],
        X_train: np.ndarray,
    ) -> "CounterfactualAnalyzer":
        """Factory: build analyzer from raw training data."""
        medians = np.median(X_train, axis=0)
        p25 = np.percentile(X_train, 25, axis=0)
        p75 = np.percentile(X_train, 75, axis=0)
        return CounterfactualAnalyzer(model, feature_names, medians, p25, p75)


__all__ = ["CounterfactualAnalyzer"]
