"""Phase 2 graph & feature validation + reporting, with CLI.

Runs after the graph and feature datasets are built. Responsibilities:

1. **Graph integrity** — every graph node exists in its source table; every
   edge corresponds to a real source relationship; no unjustified self-loops;
   node-type metadata is correct.
2. **No-label-in-graph** — confirm construction used no ground-truth field
   (checked structurally: the builder only reads whitelisted columns, and we
   re-verify no forbidden column influenced edges by rebuilding from a
   label-stripped table and comparing).
3. **Feature leakage audit** — no ground-truth label (or label-derived column)
   appears in any feature dataset's feature columns.
4. **Split isolation** — for each split's graph view, all users belong to that
   split, and no projected edge crosses splits.
5. **Temporal leakage** — recency features never exceed the user's total
   transaction count and are non-negative (no future transactions counted).

Reports written:

* ``reports/graph_summary.json``
* ``reports/community_summary.json``
* ``reports/feature_manifest.json``
* ``reports/feature_quality_report.json``
* ``reports/phase2_validation_report.json``

Run with::

    python -m src.graph.validate
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.baseline import (
    STATIC_BASELINE_FEATURES,
    TIME_AWARE_BASELINE_FEATURES,
)
from src.features.pipeline import (
    GRAPH_FEATURE_COLUMNS,
    LABEL_COLUMN,
    SPLITS,
    build_split_features,
)
from src.graph.builder import build_graph, nodes_of_type
from src.graph.communities import detect_communities
from src.graph.projection import build_user_projection
from src.graph.schema import FORBIDDEN_IN_GRAPH, NODE_SOURCE, EdgeType, NodeType
from src.validation.loader import load_tables
from src.validation.result import ValidationResult

# Ground-truth / label-derived column names that must never be features.
_FORBIDDEN_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "is_abuse_account",
        "is_abuse_transaction",
        "ring_id",
        "ring_type",
        "estimated_abuse_value",
        "split",
    }
)
# Whole-word tokens that would indicate a label-derived feature. Matched on
# underscore-delimited tokens so legitimate names like "sharing" (contains
# "ring") or "clustering" are not falsely flagged.
_SUSPICIOUS_TOKENS: frozenset[str] = frozenset(
    {"abuse", "fraud", "ring", "label", "bad", "malicious", "scam"}
)


# ---------------------------------------------------------------------------
# 1. Graph integrity
# ---------------------------------------------------------------------------


def check_graph_integrity(
    graph: nx.Graph, tables: dict[str, pd.DataFrame]
) -> ValidationResult:
    """Validate node existence, edge provenance, self-loops, type metadata."""
    result = ValidationResult(module="graph_integrity")

    # Source id sets per node type.
    id_sets: dict[NodeType, set] = {}
    for ntype, (table, key) in NODE_SOURCE.items():
        id_sets[ntype] = set(tables[table][key].astype(str))

    # Every node exists in its source table + has valid type metadata.
    bad_type = 0
    missing = 0
    for node, data in graph.nodes(data=True):
        ntype_val = data.get("node_type")
        if ntype_val is None:
            bad_type += 1
            continue
        try:
            ntype = NodeType(ntype_val)
        except ValueError:
            bad_type += 1
            continue
        if str(node) not in id_sets[ntype]:
            missing += 1
    if bad_type:
        result.add("node_type_metadata", "error", f"{bad_type} nodes have invalid node_type")
    if missing:
        result.add("node_existence", "error", f"{missing} graph nodes absent from source tables")

    # Self-loops (none justified in this schema).
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        result.add("self_loops", "error", f"{len(self_loops)} self-loops present")

    # Edge provenance: validate USER->entity edges against transaction rows.
    txns = tables["transactions"]
    observed = {
        EdgeType.USES_DEVICE: set(zip(txns["user_id"], txns["device_id"])),
        EdgeType.CONNECTS_FROM: set(zip(txns["user_id"], txns["ip_id"])),
        EdgeType.ASSOCIATED_WITH: set(zip(txns["user_id"], txns["address_id"])),
        EdgeType.USES_PAYMENT_INSTRUMENT: set(
            zip(txns["user_id"], txns["payment_instrument_id"])
        ),
    }
    bad_edges = 0
    checked = 0
    for u, v, d in graph.edges(data=True):
        etype_val = d.get("edge_type")
        try:
            etype = EdgeType(etype_val)
        except ValueError:
            result.add("edge_type_metadata", "error", f"invalid edge_type {etype_val}")
            continue
        if etype in observed:
            checked += 1
            # Orient (user, entity).
            uu, vv = (u, v)
            if graph.nodes[u].get("node_type") != NodeType.USER.value:
                uu, vv = v, u
            if (uu, vv) not in observed[etype]:
                bad_edges += 1
    if bad_edges:
        result.add(
            "edge_provenance", "error",
            f"{bad_edges} user-entity edges not backed by a transaction row",
        )
    result.add(
        "edge_provenance", "info",
        f"verified {checked} user-entity edges against source rows",
    )

    return result


# ---------------------------------------------------------------------------
# 2. No labels influenced graph construction
# ---------------------------------------------------------------------------


def check_no_label_in_graph(tables: dict[str, pd.DataFrame]) -> ValidationResult:
    """Rebuild the graph from a label-stripped copy and confirm it is identical.

    If any forbidden column had influenced construction, removing those columns
    would change the graph. Identical node/edge sets prove independence.
    """
    result = ValidationResult(module="graph_label_independence")

    full = build_graph(tables)

    stripped = {name: df.copy() for name, df in tables.items()}
    for name, df in stripped.items():
        drop = [c for c in FORBIDDEN_IN_GRAPH if c in df.columns]
        if drop:
            stripped[name] = df.drop(columns=drop)
    stripped_graph = build_graph(stripped)

    same_nodes = set(full.nodes()) == set(stripped_graph.nodes())
    same_edges = set(map(frozenset, full.edges())) == set(
        map(frozenset, stripped_graph.edges())
    )
    if not same_nodes:
        result.add("label_independence", "error", "node set changed when labels removed")
    if not same_edges:
        result.add("label_independence", "error", "edge set changed when labels removed")
    if same_nodes and same_edges:
        result.add(
            "label_independence", "info",
            "graph identical with/without label columns (construction is label-free)",
        )
    return result


# ---------------------------------------------------------------------------
# 3. Feature leakage audit
# ---------------------------------------------------------------------------


def audit_feature_leakage(feature_dir: Path) -> ValidationResult:
    """Confirm no label / label-derived column is present as a feature."""
    result = ValidationResult(module="feature_leakage")

    for variant in ("baseline", "graph"):
        for split in SPLITS:
            path = feature_dir / f"features_{variant}_{split}.parquet"
            if not path.exists():
                result.add("feature_file", "error", f"missing feature file {path.name}")
                continue
            cols = pd.read_parquet(path).columns
            feature_cols = [c for c in cols if c not in ("user_id", LABEL_COLUMN)]

            forbidden = set(feature_cols) & _FORBIDDEN_FEATURE_NAMES
            if forbidden:
                result.add(
                    "explicit_label", "error",
                    f"{path.name} contains label columns {sorted(forbidden)}",
                )
            for col in feature_cols:
                tokens = set(re.split(r"[_\W]+", col.lower()))
                hit = tokens & _SUSPICIOUS_TOKENS
                if hit:
                    result.add(
                        "suspicious_name", "error",
                        f"{path.name} feature '{col}' name token(s) {sorted(hit)} "
                        "suggest label derivation",
                    )
    result.add("feature_leakage", "info", "feature columns scanned for label leakage")
    return result


# ---------------------------------------------------------------------------
# 4. Split isolation
# ---------------------------------------------------------------------------


def check_split_isolation(tables: dict[str, pd.DataFrame]) -> ValidationResult:
    """Each split's projection contains only that split's users, no cross edges."""
    result = ValidationResult(module="split_isolation")
    users = tables["users"]

    split_of = dict(zip(users["user_id"], users["split"]))
    for split in SPLITS:
        uids = sorted(users[users["split"] == split]["user_id"])
        uset = set(uids)
        txns = tables["transactions"][tables["transactions"]["user_id"].isin(uset)]
        proj = build_user_projection(txns, user_ids=uids)

        # All nodes belong to the split.
        foreign_nodes = [n for n in proj.graph.nodes() if split_of.get(n) != split]
        if foreign_nodes:
            result.add(
                "node_membership", "error",
                f"{split} projection has {len(foreign_nodes)} foreign-split users",
            )
        # No edge crosses splits (guaranteed by construction, verified here).
        cross = [
            (u, v)
            for u, v in proj.graph.edges()
            if split_of.get(u) != split_of.get(v)
        ]
        if cross:
            result.add(
                "cross_split_edge", "error",
                f"{split} projection has {len(cross)} cross-split edges",
            )
    result.add("split_isolation", "info", "verified per-split projection isolation")
    return result


# ---------------------------------------------------------------------------
# 5. Temporal leakage
# ---------------------------------------------------------------------------


def check_temporal_leakage(
    tables: dict[str, pd.DataFrame], feature_dir: Path
) -> ValidationResult:
    """Recency features must not exceed total activity or be negative."""
    result = ValidationResult(module="temporal_leakage")
    txns = tables["transactions"]
    total_by_user = txns.groupby("user_id").size()

    for split in SPLITS:
        path = feature_dir / f"features_baseline_{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.set_index("user_id")
        for col in ("recent_transaction_count_1d", "recent_transaction_count_7d", "recent_transaction_count_30d"):
            if (df[col] < 0).any():
                result.add("negative_recency", "error", f"{col} has negative values in {split}")
            totals = df.index.map(total_by_user).astype(float)
            if (df[col].to_numpy() > totals.to_numpy()).any():
                result.add(
                    "recency_exceeds_total", "error",
                    f"{col} exceeds total transaction count in {split} (future leak)",
                )
    result.add("temporal_leakage", "info", "recency features are within total activity bounds")
    return result


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_graph_summary(
    graph: nx.Graph, tables: dict[str, pd.DataFrame]
) -> dict:
    """Compute reports/graph_summary.json content."""
    type_counts = {nt.value: len(nodes_of_type(graph, nt)) for nt in NodeType}

    users = nodes_of_type(graph, NodeType.USER)
    user_degrees = np.array([graph.degree(u) for u in users])

    # Edge counts by relationship type.
    edge_type_counts: dict[str, int] = {}
    for _, _, d in graph.edges(data=True):
        et = d.get("edge_type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

    components = list(nx.connected_components(graph))
    largest = max(components, key=len) if components else set()

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_counts": type_counts,
        "edge_counts_by_type": edge_type_counts,
        "average_user_degree": round(float(user_degrees.mean()), 4) if len(user_degrees) else 0,
        "median_user_degree": float(np.median(user_degrees)) if len(user_degrees) else 0,
        "maximum_user_degree": int(user_degrees.max()) if len(user_degrees) else 0,
        "connected_components": len(components),
        "largest_component_size": len(largest),
        "density": round(nx.density(graph), 8),
        "synthetic_data": True,
    }


def build_community_summary(
    projection: nx.Graph,
    community_df: pd.DataFrame,
    users_table: pd.DataFrame,
) -> dict:
    """Compute reports/community_summary.json content.

    ``abuse_account_fraction_by_community`` is an EVALUATION statistic only —
    computed from ground truth AFTER label-free community detection. It is not a
    feature and never influences community formation.
    """
    sizes = community_df.groupby("community_id")["community_size"].first()
    densities = community_df.groupby("community_id")["community_density"].first()

    # Evaluation-only: abuse fraction per community.
    label = users_table.set_index("user_id")["is_abuse_account"].astype(int)
    cdf = community_df.copy()
    cdf["is_abuse"] = cdf["user_id"].map(label).fillna(0).astype(int)
    abuse_frac = cdf.groupby("community_id")["is_abuse"].mean()

    largest = sizes.sort_values(ascending=False).head(10)

    return {
        "number_of_communities": int(community_df["community_id"].nunique()),
        "community_size_distribution": {
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": round(float(sizes.mean()), 3),
            "median": float(sizes.median()),
            "singletons": int((sizes == 1).sum()),
        },
        "largest_communities": [
            {
                "community_id": int(cid),
                "size": int(sizes[cid]),
                "density": round(float(densities[cid]), 6),
                "abuse_account_fraction_eval_only": round(float(abuse_frac[cid]), 4),
            }
            for cid in largest.index
        ],
        "average_community_density": round(float(densities.mean()), 6),
        "evaluation_note": (
            "abuse_account_fraction_* is computed from ground truth for "
            "evaluation only, AFTER label-free community detection. It is not a "
            "model feature and did not influence community formation."
        ),
    }


def build_feature_manifest() -> dict:
    """Compute reports/feature_manifest.json content."""
    entries: list[dict] = []

    def add(name, group, desc, uses_graph, uses_future, allowed, source, notes=""):
        entries.append(
            {
                "feature_name": name,
                "feature_group": group,
                "description": desc,
                "data_source": source,
                "uses_graph_information": uses_graph,
                "uses_future_information": uses_future,
                "allowed_in_model": allowed,
                "notes": notes,
            }
        )

    # Baseline static.
    for f in STATIC_BASELINE_FEATURES:
        add(f, "baseline_static", f"Per-user behavioural aggregate: {f}", False, False,
            True, "transactions,users", "non-graph")
    # Baseline time-aware.
    for f in TIME_AWARE_BASELINE_FEATURES:
        add(f, "baseline_time_aware",
            f"Recency/temporal feature: {f}", False, False, True,
            "transactions,devices",
            "computed as-of user's last transaction; only past rows used")
    # Graph features.
    for f in GRAPH_FEATURE_COLUMNS:
        group = "graph"
        if f.startswith("community"):
            group = "graph_community"
        add(f, group, f"Graph-derived feature: {f}", True, False, True,
            "user_projection/heterogeneous_graph",
            "computed on split-isolated graph view")
    # Labels (excluded).
    for f in ("is_abuse_account", "is_abuse_transaction", "ring_id", "ring_type",
              "estimated_abuse_value"):
        add(f, "label", f"Ground-truth field: {f}", False, False, False,
            "users/transactions", "excluded from all model features")
    add("split", "metadata", "Data split assignment", False, False, False,
        "users", "data-management field, not a feature")
    add(LABEL_COLUMN, "label", "Carried target (= is_abuse_account) for Phase 3",
        False, False, False, "users",
        "present in feature files as target column, never used as input feature")

    return {
        "note": "Feature manifest for Phase 2. allowed_in_model=false marks fields that must never be inputs.",
        "feature_count": {
            "baseline": len(STATIC_BASELINE_FEATURES) + len(TIME_AWARE_BASELINE_FEATURES),
            "graph_only": len(GRAPH_FEATURE_COLUMNS),
        },
        "features": entries,
    }


def build_feature_quality_report(feature_dir: Path) -> dict:
    """Compute reports/feature_quality_report.json content."""
    report: dict = {"note": "Per-feature stats + train/val/test comparison. Nothing deleted automatically.", "variants": {}}

    for variant in ("baseline", "graph"):
        dfs = {
            split: pd.read_parquet(feature_dir / f"features_{variant}_{split}.parquet")
            for split in SPLITS
        }
        feature_cols = [c for c in dfs["train"].columns if c not in ("user_id", LABEL_COLUMN)]
        variant_report: dict = {"features": {}, "flags": []}

        for col in feature_cols:
            per_split = {}
            for split in SPLITS:
                s = dfs[split][col].astype(float)
                per_split[split] = {
                    "missing_rate": round(float(s.isna().mean()), 6),
                    "unique_count": int(s.nunique()),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "std": round(float(s.std(ddof=0)), 4),
                }
            variant_report["features"][col] = per_split

            train_s = dfs["train"][col].astype(float)
            # Constant / near-constant.
            if train_s.nunique() <= 1:
                variant_report["flags"].append({"feature": col, "flag": "constant", "detail": "single value in train"})
            elif train_s.nunique() <= 2 and (train_s.value_counts(normalize=True).max() > 0.99):
                variant_report["flags"].append({"feature": col, "flag": "near_constant", "detail": ">99% one value"})
            # Distribution shift: mean shift in std units train vs test.
            test_s = dfs["test"][col].astype(float)
            denom = train_s.std(ddof=0)
            if denom > 1e-9:
                shift = abs(train_s.mean() - test_s.mean()) / denom
                if shift > 1.0:
                    variant_report["flags"].append(
                        {"feature": col, "flag": "distribution_shift",
                         "detail": f"|train-test mean| = {shift:.2f} std"}
                    )

        report["variants"][variant] = variant_report
    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_phase2_validation(
    data_dir: Path | None = None,
) -> tuple[bool, dict]:
    """Run all Phase 2 validators + build reports. Returns (ok, aggregate)."""
    data_dir = data_dir or (PROJECT_ROOT / "data" / "processed")
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(data_dir)

    # Full graph (for summary + integrity + label independence).
    full_graph = build_graph(tables)

    results = [
        check_graph_integrity(full_graph, tables),
        check_no_label_in_graph(tables),
        audit_feature_leakage(data_dir),
        check_split_isolation(tables),
        check_temporal_leakage(tables, data_dir),
    ]

    # Reports.
    (reports_dir / "graph_summary.json").write_text(
        json.dumps(build_graph_summary(full_graph, tables), indent=2), encoding="utf-8"
    )

    # Community summary on the FULL projection (label-free), eval stats appended.
    full_proj = build_user_projection(tables["transactions"])
    community_df = detect_communities(full_proj.graph)
    (reports_dir / "community_summary.json").write_text(
        json.dumps(
            build_community_summary(full_proj.graph, community_df, tables["users"]),
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports_dir / "feature_manifest.json").write_text(
        json.dumps(build_feature_manifest(), indent=2), encoding="utf-8"
    )
    (reports_dir / "feature_quality_report.json").write_text(
        json.dumps(build_feature_quality_report(data_dir), indent=2), encoding="utf-8"
    )

    aggregate = {
        "ok": all(r.ok for r in results),
        "total_errors": sum(len(r.errors) for r in results),
        "total_warnings": sum(len(r.warnings) for r in results),
        "modules": [r.to_dict() for r in results],
    }
    (reports_dir / "phase2_validation_report.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate["ok"], aggregate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Phase 2 graph + features.")
    parser.add_argument("--data", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data) if args.data else None

    print("AbuseRing Sentinel - Phase 2 validation")
    print("=" * 60)
    ok, aggregate = run_phase2_validation(data_dir)
    for module in aggregate["modules"]:
        status = "OK" if module["ok"] else "FAIL"
        print(
            f"  [{status:>4}] {module['module']:<26} "
            f"errors={module['error_count']} warnings={module['warning_count']}"
        )
        for issue in module["issues"]:
            if issue["severity"] in ("error", "warning"):
                print(f"         - {issue['severity'].upper()}: {issue['message']}")
    print("=" * 60)
    print(f"Total errors: {aggregate['total_errors']} | warnings: {aggregate['total_warnings']}")
    print("Reports written to reports/.")
    if ok:
        print("PHASE 2 VALIDATION PASSED.")
        return 0
    print("PHASE 2 VALIDATION FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_phase2_validation", "main"]
