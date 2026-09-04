"""Tests for Phase 4 — Investigation & Explainability.

Tests cover:
- Investigation schema validation
- Graph evidence correctness
- Multi-hop path validity
- SHAP feature mapping
- No ground-truth fields in investigator output
- Counterfactual bounds and probability generation
- Deterministic output
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.graph.builder import build_graph
from src.graph.communities import detect_communities
from src.graph.projection import build_user_projection
from src.graph.schema import EdgeType, NodeType
from src.investigation.schema import (
    BehavioralSummary,
    CommunityContext,
    CounterfactualResult,
    FEATURE_DESCRIPTIONS,
    FeatureAttribution,
    GROUND_TRUTH_FIELDS,
    InvestigationResult,
    MultiHopPath,
    RelatedAccount,
    RiskLevel,
)
from src.investigation.graph_evidence import GraphEvidenceCollector
from src.investigation.counterfactuals import CounterfactualAnalyzer


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_graph_data(small_tables):
    """Build graph + projection + communities from the small dataset."""
    tables = small_tables
    users = tables["users"]
    test_users = users[users["split"] == "test"]
    test_user_ids = sorted(test_users["user_id"])
    test_txns = tables["transactions"][
        tables["transactions"]["user_id"].isin(set(test_user_ids))
    ]
    hetero = build_graph(tables, transactions=test_txns)
    proj = build_user_projection(test_txns, user_ids=test_user_ids)
    community = detect_communities(proj.graph)
    return {
        "hetero": hetero,
        "projection": proj.graph,
        "community_df": community,
        "user_ids": test_user_ids,
        "tables": tables,
    }


@pytest.fixture(scope="session")
def graph_evidence(small_graph_data):
    """GraphEvidenceCollector from small dataset."""
    return GraphEvidenceCollector(
        small_graph_data["hetero"],
        small_graph_data["projection"],
        small_graph_data["community_df"],
    )


# --------------------------------------------------------------------------
# Schema tests
# --------------------------------------------------------------------------


class TestInvestigationSchema:
    """Test that schema objects validate correctly."""

    def test_feature_attribution_valid(self):
        fa = FeatureAttribution(
            feature_name="promo_usage_rate",
            feature_value=0.85,
            shap_value=1.23,
            direction="increases_risk",
            human_readable_description="Fraction of transactions that used a promotion",
        )
        assert fa.feature_name == "promo_usage_rate"
        assert fa.direction == "increases_risk"

    def test_risk_levels(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_related_account(self):
        ra = RelatedAccount(
            user_id="USR_001",
            relationship_types=["shared_device", "shared_ip"],
            shared_entity_count=3,
        )
        assert len(ra.relationship_types) == 2

    def test_multi_hop_path(self):
        p = MultiHopPath(
            nodes=["USR_A", "DEV_1", "USR_B"],
            node_types=["user", "device", "user"],
            edge_types=["uses_device", "uses_device"],
            hop_count=1,
            path_strength=5.0,
        )
        assert p.hop_count == 1
        assert len(p.nodes) == 3

    def test_community_context(self):
        cc = CommunityContext(
            community_id=0,
            community_size=12,
            community_density=0.35,
            user_degree=5,
            neighbor_count=4,
        )
        assert cc.community_size == 12

    def test_counterfactual_result(self):
        cf = CounterfactualResult(
            feature="promo_usage_rate",
            original_value=0.85,
            counterfactual_value=0.1,
            original_probability=0.92,
            counterfactual_probability=0.45,
            probability_delta=-0.47,
        )
        assert abs(cf.probability_delta - (cf.counterfactual_probability - cf.original_probability)) < 0.01

    def test_investigation_result_no_labels(self):
        """InvestigationResult schema must not contain ground-truth fields."""
        fields = set(InvestigationResult.model_fields.keys())
        for gt_field in GROUND_TRUTH_FIELDS:
            assert gt_field not in fields, f"Schema contains ground-truth field: {gt_field}"

    def test_feature_descriptions_complete(self):
        """All graph + baseline features should have descriptions."""
        from src.features.baseline import BASELINE_FEATURES
        from src.features.pipeline import GRAPH_FEATURE_COLUMNS

        all_features = set(BASELINE_FEATURES) | set(GRAPH_FEATURE_COLUMNS)
        missing = all_features - set(FEATURE_DESCRIPTIONS.keys())
        assert not missing, f"Missing descriptions for: {missing}"


# --------------------------------------------------------------------------
# Graph evidence tests
# --------------------------------------------------------------------------


class TestGraphEvidence:
    """Test graph evidence extraction uses actual graph data."""

    def test_shared_devices_exist_in_graph(self, graph_evidence, small_graph_data):
        """Every returned shared device must be an actual edge in the graph."""
        hetero = small_graph_data["hetero"]
        for uid in small_graph_data["user_ids"][:10]:
            devices = graph_evidence.shared_devices(uid)
            for dev_id in devices:
                assert hetero.has_edge(uid, dev_id), (
                    f"Shared device {dev_id} not connected to {uid} in graph"
                )

    def test_shared_ips_exist_in_graph(self, graph_evidence, small_graph_data):
        """Every returned shared IP must be an actual edge."""
        hetero = small_graph_data["hetero"]
        for uid in small_graph_data["user_ids"][:10]:
            ips = graph_evidence.shared_ips(uid)
            for ip_id in ips:
                assert hetero.has_edge(uid, ip_id)

    def test_related_accounts_from_projection(self, graph_evidence, small_graph_data):
        """Related accounts must exist in the user projection."""
        proj = small_graph_data["projection"]
        for uid in small_graph_data["user_ids"][:10]:
            related = graph_evidence.related_accounts(uid)
            for ra in related:
                assert proj.has_edge(uid, ra.user_id), (
                    f"Related account {ra.user_id} not in projection for {uid}"
                )
                assert ra.shared_entity_count > 0

    def test_related_accounts_sorted_by_strength(self, graph_evidence, small_graph_data):
        """Related accounts should be sorted by shared_entity_count descending."""
        for uid in small_graph_data["user_ids"][:10]:
            related = graph_evidence.related_accounts(uid)
            counts = [ra.shared_entity_count for ra in related]
            assert counts == sorted(counts, reverse=True)

    def test_community_context_valid(self, graph_evidence, small_graph_data):
        """Community context should be valid for known users."""
        for uid in small_graph_data["user_ids"][:5]:
            ctx = graph_evidence.community_context(uid)
            if ctx is not None:
                assert ctx.community_size >= 1
                assert 0.0 <= ctx.community_density <= 1.0
                assert ctx.user_degree >= 0
                assert ctx.neighbor_count >= 0

    def test_unknown_user_returns_empty(self, graph_evidence):
        """Querying a non-existent user returns empty lists / None."""
        assert graph_evidence.shared_devices("NONEXISTENT_USER") == []
        assert graph_evidence.related_accounts("NONEXISTENT_USER") == []
        assert graph_evidence.community_context("NONEXISTENT_USER") is None


# --------------------------------------------------------------------------
# Multi-hop path tests
# --------------------------------------------------------------------------


class TestMultiHopPaths:
    """Test multi-hop path validity."""

    def test_paths_exist_in_graph(self, graph_evidence, small_graph_data):
        """Every edge in every returned path must exist in the actual graph."""
        hetero = small_graph_data["hetero"]
        for uid in small_graph_data["user_ids"][:10]:
            paths = graph_evidence.multi_hop_paths(uid)
            for path in paths:
                for i in range(len(path.nodes) - 1):
                    assert hetero.has_edge(path.nodes[i], path.nodes[i + 1]), (
                        f"Path edge {path.nodes[i]} -> {path.nodes[i+1]} "
                        f"does not exist in graph"
                    )

    def test_max_10_paths(self, graph_evidence, small_graph_data):
        """Never return more than 10 paths per user."""
        for uid in small_graph_data["user_ids"][:10]:
            paths = graph_evidence.multi_hop_paths(uid, max_paths=10)
            assert len(paths) <= 10

    def test_max_3_hops(self, graph_evidence, small_graph_data):
        """No path should exceed 3 hops."""
        for uid in small_graph_data["user_ids"][:10]:
            paths = graph_evidence.multi_hop_paths(uid, max_hops=3)
            for path in paths:
                assert path.hop_count <= 3

    def test_path_starts_with_user(self, graph_evidence, small_graph_data):
        """Every path must start with the investigated user."""
        for uid in small_graph_data["user_ids"][:5]:
            paths = graph_evidence.multi_hop_paths(uid)
            for path in paths:
                assert path.nodes[0] == uid

    def test_path_node_types_consistent(self, graph_evidence, small_graph_data):
        """Path node_types length must match nodes length."""
        for uid in small_graph_data["user_ids"][:5]:
            paths = graph_evidence.multi_hop_paths(uid)
            for path in paths:
                assert len(path.node_types) == len(path.nodes)
                assert len(path.edge_types) == len(path.nodes) - 1


# --------------------------------------------------------------------------
# No ground-truth fields
# --------------------------------------------------------------------------


class TestNoGroundTruth:
    """Ensure investigator output never contains ground-truth labels."""

    def test_schema_fields_clean(self):
        """InvestigationResult must not have ground-truth field names."""
        all_field_names: set[str] = set()
        for model_cls in [
            InvestigationResult,
            FeatureAttribution,
            RelatedAccount,
            MultiHopPath,
            CommunityContext,
            BehavioralSummary,
            CounterfactualResult,
        ]:
            all_field_names.update(model_cls.model_fields.keys())

        for gt in GROUND_TRUTH_FIELDS:
            assert gt not in all_field_names, (
                f"Ground-truth field '{gt}' found in schema"
            )

    def test_feature_descriptions_no_labels(self):
        """Feature description keys must not be ground-truth fields."""
        for gt in GROUND_TRUTH_FIELDS:
            assert gt not in FEATURE_DESCRIPTIONS


# --------------------------------------------------------------------------
# Counterfactual tests
# --------------------------------------------------------------------------


class TestCounterfactuals:
    """Test counterfactual analyzer with a simple mock model."""

    @pytest.fixture
    def simple_cf_analyzer(self):
        """Create a CF analyzer with a mock model."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=(n, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        return CounterfactualAnalyzer.from_training_data(model, feature_names, X)

    def test_counterfactual_probability_bounds(self, simple_cf_analyzer):
        """Counterfactual probabilities must be in [0, 1]."""
        X_row = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
        results = simple_cf_analyzer.analyze(X_row, features=["f0", "f1", "f2"])
        for cf in results:
            assert 0.0 <= cf.original_probability <= 1.0
            assert 0.0 <= cf.counterfactual_probability <= 1.0

    def test_counterfactual_delta_consistent(self, simple_cf_analyzer):
        """Delta must equal counterfactual - original probability."""
        X_row = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
        results = simple_cf_analyzer.analyze(X_row, features=["f0", "f1"])
        for cf in results:
            expected = cf.counterfactual_probability - cf.original_probability
            assert abs(cf.probability_delta - expected) < 0.001

    def test_counterfactual_uses_frozen_model(self, simple_cf_analyzer):
        """Verify the same model is used for all counterfactuals."""
        X_row = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
        results = simple_cf_analyzer.analyze(X_row, features=["f0", "f1", "f2"])
        # All should have the same original probability
        if len(results) > 1:
            orig_probs = [cf.original_probability for cf in results]
            assert all(abs(p - orig_probs[0]) < 0.001 for p in orig_probs)

    def test_counterfactual_sorted_by_impact(self, simple_cf_analyzer):
        """Results should be sorted by |probability_delta| descending."""
        X_row = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
        results = simple_cf_analyzer.analyze(X_row, features=["f0", "f1", "f2", "f3"])
        deltas = [abs(cf.probability_delta) for cf in results]
        assert deltas == sorted(deltas, reverse=True)


# --------------------------------------------------------------------------
# SHAP mapping tests
# --------------------------------------------------------------------------


class TestSHAPMapping:
    """Test SHAP feature mapping."""

    def test_all_model_features_have_descriptions(self):
        """Every feature in the graph variant must have a human-readable description."""
        from src.models.preprocessing import get_feature_lists

        graph_features = get_feature_lists("graph")
        missing = [f for f in graph_features if f not in FEATURE_DESCRIPTIONS]
        assert not missing, f"Missing SHAP descriptions: {missing}"

    def test_descriptions_are_nonempty(self):
        """All descriptions must be non-empty strings."""
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            assert isinstance(desc, str) and len(desc) > 0, (
                f"Empty description for {feat}"
            )


# --------------------------------------------------------------------------
# Deterministic output test
# --------------------------------------------------------------------------


class TestDeterministicOutput:
    """Test that investigation output is deterministic."""

    def test_graph_evidence_deterministic(self, graph_evidence, small_graph_data):
        """Two calls to the same user should return identical results."""
        uid = small_graph_data["user_ids"][0]
        r1_devices = graph_evidence.shared_devices(uid)
        r2_devices = graph_evidence.shared_devices(uid)
        assert r1_devices == r2_devices

        r1_related = graph_evidence.related_accounts(uid)
        r2_related = graph_evidence.related_accounts(uid)
        assert len(r1_related) == len(r2_related)
        for a, b in zip(r1_related, r2_related):
            assert a.user_id == b.user_id
            assert a.shared_entity_count == b.shared_entity_count

    def test_counterfactual_deterministic(self):
        """Counterfactual results should be deterministic."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3))
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        analyzer = CounterfactualAnalyzer.from_training_data(
            model, ["a", "b", "c"], X
        )
        row = np.array([1.5, 0.0, 0.0])
        r1 = analyzer.analyze(row, features=["a", "b"])
        r2 = analyzer.analyze(row, features=["a", "b"])
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.original_probability == b.original_probability
            assert a.counterfactual_probability == b.counterfactual_probability
