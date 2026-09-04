"""Tests for Phase 2 features, communities, and leakage/temporal safety."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from src.features.baseline import BASELINE_FEATURES, build_baseline_features
from src.features.pipeline import (
    GRAPH_FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_split_features,
)
from src.graph.builder import build_graph
from src.graph.communities import detect_communities
from src.graph.features import compute_user_graph_features
from src.graph.projection import build_user_projection


@pytest.fixture(scope="module")
def train_features(small_tables: dict[str, pd.DataFrame]):
    return build_split_features(small_tables, "train")


def test_baseline_has_expected_features(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    users = small_tables["users"]
    txns = small_tables["transactions"]
    feats = build_baseline_features(users, txns, small_tables["devices"])
    cols = set(feats.columns) - {"user_id"}
    assert cols == set(BASELINE_FEATURES)


def test_baseline_is_non_graph_only(train_features) -> None:
    """Baseline dataset must not contain any graph feature."""
    baseline, _ = train_features
    baseline_cols = set(baseline.columns)
    assert not (baseline_cols & set(GRAPH_FEATURE_COLUMNS))


def test_graph_dataset_contains_graph_features(train_features) -> None:
    _, graph_df = train_features
    assert set(GRAPH_FEATURE_COLUMNS).issubset(graph_df.columns)


def test_no_label_columns_in_features(train_features) -> None:
    """Neither dataset may expose a ground-truth field as a feature."""
    baseline, graph_df = train_features
    forbidden = {
        "is_abuse_account",
        "is_abuse_transaction",
        "ring_id",
        "ring_type",
        "estimated_abuse_value",
        "split",
    }
    for df in (baseline, graph_df):
        feature_cols = set(df.columns) - {"user_id", LABEL_COLUMN}
        assert not (feature_cols & forbidden)


def test_graph_features_reproducible(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    txns = small_tables["transactions"]
    hetero = build_graph(small_tables, transactions=txns)
    proj = build_user_projection(txns)
    a = compute_user_graph_features(hetero, proj.graph)
    b = compute_user_graph_features(hetero, proj.graph)
    pd.testing.assert_frame_equal(a, b)


def test_communities_reproducible(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    proj = build_user_projection(small_tables["transactions"])
    a = detect_communities(proj.graph)
    b = detect_communities(proj.graph)
    pd.testing.assert_frame_equal(a, b)


def test_communities_are_label_free(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    """Community output columns contain no label information."""
    proj = build_user_projection(small_tables["transactions"])
    comm = detect_communities(proj.graph)
    assert set(comm.columns) == {
        "user_id",
        "community_id",
        "community_size",
        "community_density",
        "community_edge_count",
    }


def test_recency_features_do_not_use_future(train_features, small_tables) -> None:
    """Recency counts never exceed the user's total transactions."""
    baseline, _ = train_features
    txns = small_tables["transactions"]
    totals = txns.groupby("user_id").size()
    merged = baseline.set_index("user_id")
    for col in (
        "recent_transaction_count_1d",
        "recent_transaction_count_7d",
        "recent_transaction_count_30d",
    ):
        exceeds = merged[col] > merged.index.map(totals)
        assert not exceeds.any(), f"{col} exceeds total (future leak)"
        assert (merged[col] >= 0).all()


def test_split_isolation_no_cross_edges(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    """A split's projection contains only that split's users."""
    users = small_tables["users"]
    split_of = dict(zip(users["user_id"], users["split"]))
    for split in ("train", "val", "test"):
        uids = sorted(users[users["split"] == split]["user_id"])
        uset = set(uids)
        txns = small_tables["transactions"][
            small_tables["transactions"]["user_id"].isin(uset)
        ]
        proj = build_user_projection(txns, user_ids=uids)
        for u, v in proj.graph.edges():
            assert split_of[u] == split and split_of[v] == split


def test_feature_row_counts_match_split_users(
    small_tables: dict[str, pd.DataFrame], train_features
) -> None:
    baseline, graph_df = train_features
    n_train = (small_tables["users"]["split"] == "train").sum()
    assert len(baseline) == n_train
    assert len(graph_df) == n_train
