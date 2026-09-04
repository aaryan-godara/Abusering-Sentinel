"""Phase 2 feature pipeline + CLI.

Produces model-ready feature datasets for Phase 3, in two variants per split:

* ``features_baseline_<split>.parquet``  - non-graph behavioural features only.
* ``features_graph_<split>.parquet``     - baseline features + graph features
                                           + community features.

Split-isolation methodology
----------------------------
Graphs connect entities, so a single full graph would let test relationships
influence training features. We prevent this by building a **split-specific
graph view** for each split: the graph and user projection for a split are built
using ONLY the transactions of users assigned to that split.

Because Phase 1 splitting is group/ring-aware (every abuse ring and every
legitimate shared group lives entirely within one split), restricting to a
split's users keeps those groups intact — a ring is never cut in half — while
guaranteeing that no relationship from another split leaks in. This is the
strictest, simplest isolation and is what we use here.

Row identity: the ``user_id`` and the ground-truth ``label`` (``is_abuse_account``)
are carried in the feature files as separate, clearly-named columns so Phase 3
can join targets. The label is NOT a feature — the leakage audit enforces this by
checking the feature columns only.

Run with::

    python -m src.features.pipeline
    python -m src.features.pipeline --data data/processed --out data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.features.baseline import (
    BASELINE_FEATURES,
    build_baseline_features,
)
from src.graph.builder import build_graph
from src.graph.communities import detect_communities
from src.graph.features import compute_user_graph_features
from src.graph.projection import build_user_projection
from src.validation.loader import load_tables

SPLITS: tuple[str, ...] = ("train", "val", "test")

# Non-feature, carried-through columns.
ID_COLUMN = "user_id"
LABEL_COLUMN = "label"  # = is_abuse_account, carried for Phase 3 targets only

GRAPH_FEATURE_COLUMNS: tuple[str, ...] = (
    # direct
    "degree",
    "weighted_degree",
    "unique_devices",
    "unique_ips",
    "unique_addresses",
    "unique_payment_instruments",
    "unique_merchants",
    "unique_promotions",
    # shared-entity
    "number_of_accounts_sharing_devices",
    "number_of_accounts_sharing_ips",
    "number_of_accounts_sharing_addresses",
    "number_of_accounts_sharing_payment_instruments",
    "number_of_accounts_sharing_promotions",
    # neighbor
    "number_of_user_neighbors",
    "number_of_high_degree_neighbors",
    "average_neighbor_degree",
    "maximum_neighbor_degree",
    # two-hop
    "two_hop_user_count",
    "two_hop_unique_entity_count",
    "two_hop_shared_device_count",
    "two_hop_shared_ip_count",
    # centrality
    "degree_centrality",
    "clustering_coefficient",
    "betweenness_centrality",
    # community
    "community_size",
    "community_density",
    "community_edge_count",
)


def _split_tables(
    tables: dict[str, pd.DataFrame], split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (split_users, split_transactions) for one split."""
    users = tables["users"]
    split_users = users[users["split"] == split]
    user_ids = set(split_users["user_id"])
    split_txns = tables["transactions"][
        tables["transactions"]["user_id"].isin(user_ids)
    ]
    return split_users, split_txns


def build_split_features(
    tables: dict[str, pd.DataFrame], split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (baseline_df, graph_df) for one split, isolated to that split.

    Both frames carry ``user_id`` and ``label``; ``graph_df`` additionally holds
    all graph + community features on top of the baseline features.
    """
    split_users, split_txns = _split_tables(tables, split)
    user_id_list = sorted(split_users["user_id"])

    # --- Baseline (non-graph) ---
    baseline = build_baseline_features(split_users, split_txns, tables["devices"])

    # --- Split-isolated graph views ---
    hetero = build_graph(tables, transactions=split_txns)
    projection = build_user_projection(split_txns, user_ids=user_id_list)

    graph_feats = compute_user_graph_features(hetero, projection.graph)
    # betweenness_approx is provenance (was betweenness sampled?), not a model
    # feature; drop it so it never enters the feature matrix.
    graph_feats = graph_feats.drop(columns=["betweenness_approx"], errors="ignore")
    community = detect_communities(projection.graph)

    graph_df = (
        baseline.merge(graph_feats, on="user_id", how="left")
        .merge(community, on="user_id", how="left")
        .drop(columns=["community_id"], errors="ignore")
    )
    graph_df = graph_df.fillna(0)

    # Attach labels (carried, not a feature).
    label_map = split_users.set_index("user_id")["is_abuse_account"].astype(int)
    baseline[LABEL_COLUMN] = baseline["user_id"].map(label_map).astype(int)
    graph_df[LABEL_COLUMN] = graph_df["user_id"].map(label_map).astype(int)

    return baseline, graph_df


def run(
    data_dir: Path | None = None, out_dir: Path | None = None
) -> dict[str, dict[str, int]]:
    """Build and write all feature datasets. Returns per-file shapes."""
    data_dir = data_dir or (PROJECT_ROOT / "data" / "processed")
    out_dir = out_dir or (PROJECT_ROOT / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(data_dir)

    shapes: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        baseline, graph_df = build_split_features(tables, split)

        b_path = out_dir / f"features_baseline_{split}.parquet"
        g_path = out_dir / f"features_graph_{split}.parquet"
        baseline.to_parquet(b_path, index=False)
        graph_df.to_parquet(g_path, index=False)

        shapes[split] = {
            "baseline_rows": len(baseline),
            "baseline_cols": baseline.shape[1],
            "graph_rows": len(graph_df),
            "graph_cols": graph_df.shape[1],
        }
    return shapes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 feature datasets.")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data) if args.data else None
    out_dir = Path(args.out) if args.out else None

    print("AbuseRing Sentinel - Phase 2 feature build")
    print("=" * 60)
    shapes = run(data_dir, out_dir)
    n_baseline = len(BASELINE_FEATURES)
    n_graph = len(GRAPH_FEATURE_COLUMNS)
    for split, s in shapes.items():
        print(
            f"  {split:<5}: baseline {s['baseline_rows']}x{s['baseline_cols']} | "
            f"graph {s['graph_rows']}x{s['graph_cols']}"
        )
    print(f"\nBaseline features: {n_baseline} | Graph-only features: {n_graph}")
    print("Feature datasets written to data/processed/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run",
    "build_split_features",
    "main",
    "SPLITS",
    "GRAPH_FEATURE_COLUMNS",
    "ID_COLUMN",
    "LABEL_COLUMN",
]
