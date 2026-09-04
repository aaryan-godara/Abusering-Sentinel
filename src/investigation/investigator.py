"""Investigator: the main Phase 4 facade.

``Investigator`` loads the frozen Model C, prebuilt graphs, and SHAP explainer
once, then provides ``investigate_user(user_id)`` which returns a complete
``InvestigationResult``.

Reuses all existing Phase 2/3 code:
- ``src.models.explainability`` for SHAP
- ``src.graph`` for graph construction / projection / communities
- ``src.models.preprocessing`` for feature loading
- ``src.features.baseline`` for feature definitions
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.baseline import BASELINE_FEATURES
from src.features.pipeline import GRAPH_FEATURE_COLUMNS
from src.graph.builder import build_graph
from src.graph.communities import detect_communities
from src.graph.projection import build_user_projection
from src.models.explainability import compute_shap_values
from src.models.preprocessing import (
    get_feature_lists,
    load_feature_split,
    get_feature_matrix,
)
from src.investigation.counterfactuals import CounterfactualAnalyzer
from src.investigation.graph_evidence import GraphEvidenceCollector
from src.investigation.schema import (
    BehavioralSummary,
    FEATURE_DESCRIPTIONS,
    FeatureAttribution,
    GROUND_TRUTH_FIELDS,
    InvestigationResult,
    RiskLevel,
)
from src.validation.loader import load_tables

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _risk_level(score: float) -> RiskLevel:
    """Map a probability to a discrete risk level."""
    if score >= 0.8:
        return RiskLevel.CRITICAL
    if score >= 0.5:
        return RiskLevel.HIGH
    if score >= 0.2:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class Investigator:
    """Phase 4 investigation facade.

    Loads graphs, models, and feature data once. Subsequent
    ``investigate_user()`` calls are fast lookups + one SHAP + one
    counterfactual pass.

    Parameters
    ----------
    data_dir : directory containing processed CSVs and parquet feature files
    model_dir : directory containing saved joblib models
    split : which split to investigate users from (default "test")
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        model_dir: Path | None = None,
        split: str = "test",
    ) -> None:
        self._data_dir = data_dir or (PROJECT_ROOT / "data" / "processed")
        self._model_dir = model_dir or (PROJECT_ROOT / "models")
        self._split = split

        # -- Load frozen Model C --
        model_path = self._model_dir / "xgboost_graph.joblib"
        self._model = joblib.load(model_path)

        # -- Load feature data --
        self._feat_names = get_feature_lists("graph")
        df = load_feature_split(split, "graph", self._data_dir)
        self._feature_df = df
        self._X, self._y = get_feature_matrix(df, self._feat_names)
        self._user_ids: list[str] = list(df["user_id"])
        self._user_to_idx: dict[str, int] = {
            uid: i for i, uid in enumerate(self._user_ids)
        }

        # -- Scores --
        proba = self._model.predict_proba(self._X)
        self._probas = proba[:, 1] if proba.ndim == 2 else proba

        # -- Build graphs (reuse for all users) --
        tables = load_tables(self._data_dir)
        users_df = tables["users"]
        split_users = users_df[users_df["split"] == split]
        split_user_ids = sorted(split_users["user_id"])
        split_txns = tables["transactions"][
            tables["transactions"]["user_id"].isin(set(split_user_ids))
        ]
        self._hetero = build_graph(tables, transactions=split_txns)
        proj_result = build_user_projection(split_txns, user_ids=split_user_ids)
        self._projection = proj_result.graph
        self._community_df = detect_communities(self._projection)

        # -- Graph evidence collector (shared, never rebuilt) --
        self._graph_evidence = GraphEvidenceCollector(
            self._hetero, self._projection, self._community_df
        )

        # -- SHAP explainer (created once) --
        self._shap_explainer = None
        if SHAP_AVAILABLE:
            self._shap_explainer = shap.TreeExplainer(self._model)

        # -- Counterfactual analyzer (training stats needed) --
        train_df = load_feature_split("train", "graph", self._data_dir)
        X_train, _, _ = get_feature_matrix(train_df, self._feat_names), None, None
        X_train = train_df[self._feat_names].to_numpy(dtype=np.float64)
        self._cf_analyzer = CounterfactualAnalyzer.from_training_data(
            self._model, self._feat_names, X_train
        )

        # -- Baseline feature data for behavioural summaries --
        self._baseline_feature_names = list(BASELINE_FEATURES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def investigate_user(
        self,
        user_id: str,
        top_k_shap: int = 10,
        max_counterfactual_features: int | None = None,
    ) -> InvestigationResult:
        """Generate a complete investigation report for one user.

        Parameters
        ----------
        user_id : the user to investigate
        top_k_shap : number of top SHAP features to include
        max_counterfactual_features : limit counterfactual features (None = all defaults)
        """
        idx = self._user_to_idx.get(user_id)
        if idx is None:
            raise ValueError(
                f"User '{user_id}' not found in {self._split} split. "
                f"Available users: {len(self._user_ids)}"
            )

        X_row = self._X[idx]
        risk_score = float(self._probas[idx])
        risk_level = _risk_level(risk_score)

        # -- SHAP attributions --
        top_factors = self._compute_shap_attributions(X_row, top_k_shap)

        # -- Graph evidence --
        shared_devs = self._graph_evidence.shared_devices(user_id)
        shared_ips_ = self._graph_evidence.shared_ips(user_id)
        shared_addrs = self._graph_evidence.shared_addresses(user_id)
        shared_pis = self._graph_evidence.shared_payment_instruments(user_id)
        related = self._graph_evidence.related_accounts(user_id)
        community = self._graph_evidence.community_context(user_id)
        paths = self._graph_evidence.multi_hop_paths(user_id)

        # -- Behavioural summary --
        behavioral = self._behavioral_summary(X_row)

        # -- Counterfactuals --
        counterfactuals = self._cf_analyzer.analyze(X_row)

        return InvestigationResult(
            user_id=user_id,
            risk_score=round(risk_score, 6),
            risk_level=risk_level,
            top_model_factors=top_factors,
            shared_devices=shared_devs,
            shared_ips=shared_ips_,
            shared_addresses=shared_addrs,
            shared_payment_instruments=shared_pis,
            related_accounts=related,
            community_context=community,
            multi_hop_paths=paths,
            behavioral_summary=behavioral,
            counterfactuals=counterfactuals,
        )

    @property
    def user_ids(self) -> list[str]:
        """All user IDs in the loaded split."""
        return self._user_ids

    @property
    def probabilities(self) -> np.ndarray:
        """Model C probabilities for all users in the split."""
        return self._probas

    @property
    def labels(self) -> np.ndarray:
        """Ground-truth labels (for evaluation selection only, never in output)."""
        return self._y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_shap_attributions(
        self, X_row: np.ndarray, top_k: int
    ) -> list[FeatureAttribution]:
        """Compute per-feature SHAP values for a single user."""
        if self._shap_explainer is None:
            return []

        shap_vals = self._shap_explainer.shap_values(X_row.reshape(1, -1))
        if shap_vals.ndim == 2:
            shap_vals = shap_vals[0]

        # Sort by absolute value, take top-k
        abs_vals = np.abs(shap_vals)
        top_indices = np.argsort(abs_vals)[::-1][:top_k]

        attributions: list[FeatureAttribution] = []
        for i in top_indices:
            fname = self._feat_names[i]
            fval = float(X_row[i])
            sval = float(shap_vals[i])
            direction = "increases_risk" if sval > 0 else "decreases_risk"
            desc = FEATURE_DESCRIPTIONS.get(fname, fname)

            attributions.append(
                FeatureAttribution(
                    feature_name=fname,
                    feature_value=round(fval, 6),
                    shap_value=round(sval, 6),
                    direction=direction,
                    human_readable_description=desc,
                )
            )
        return attributions

    def _behavioral_summary(self, X_row: np.ndarray) -> BehavioralSummary:
        """Generate deterministic behavioural statements from baseline features."""
        feature_vals: dict[str, float] = {}
        statements: list[str] = []

        for fname in self._baseline_feature_names:
            idx = self._feat_names.index(fname) if fname in self._feat_names else None
            if idx is None:
                continue
            val = float(X_row[idx])
            feature_vals[fname] = round(val, 4)

        # Generate concise deterministic statements based on actual values
        tf = feature_vals.get("transaction_frequency", 0.0)
        if tf > 2.0:
            statements.append(f"High transaction frequency ({tf:.1f} per day).")
        elif tf > 0.5:
            statements.append(f"Moderate transaction frequency ({tf:.1f} per day).")

        pur = feature_vals.get("promo_usage_rate", 0.0)
        if pur > 0.5:
            statements.append(f"High promotion usage rate ({pur:.0%} of transactions).")
        elif pur > 0.1:
            statements.append(f"Moderate promotion usage rate ({pur:.0%} of transactions).")

        avg_amt = feature_vals.get("average_transaction_amount", 0.0)
        if avg_amt > 0:
            statements.append(f"Average transaction amount: ₹{avg_amt:,.0f}.")

        tc = feature_vals.get("transaction_count", 0.0)
        if tc > 0:
            statements.append(f"Total transactions: {int(tc)}.")

        umc = feature_vals.get("unique_merchant_count", 0.0)
        if umc <= 2 and tc > 5:
            statements.append("Low merchant diversity relative to transaction count.")
        elif umc > 10:
            statements.append(f"High merchant diversity ({int(umc)} unique merchants).")

        ftr = feature_vals.get("failed_transaction_rate", 0.0)
        if ftr > 0.1:
            statements.append(f"Elevated failed transaction rate ({ftr:.0%}).")

        nar = feature_vals.get("night_activity_rate", 0.0)
        if nar > 0.3:
            statements.append(f"Significant night-time activity ({nar:.0%} of transactions).")

        war = feature_vals.get("weekend_activity_rate", 0.0)
        if war > 0.5:
            statements.append(f"Predominantly weekend activity ({war:.0%} of transactions).")

        r7d = feature_vals.get("recent_transaction_count_7d", 0.0)
        if r7d > 10:
            statements.append(f"High recent activity ({int(r7d)} transactions in last 7 days).")

        age = feature_vals.get("account_age_days", 0.0)
        if age < 7:
            statements.append(f"Very new account ({age:.0f} days old).")

        if not statements:
            statements.append("No notable behavioural patterns detected.")

        return BehavioralSummary(
            statements=statements,
            feature_values=feature_vals,
        )


__all__ = ["Investigator"]
