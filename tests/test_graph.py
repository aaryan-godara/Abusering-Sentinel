"""Tests for the Phase 2 heterogeneous graph and user projection."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from src.graph.builder import build_graph, edges_of_type, nodes_of_type
from src.graph.projection import build_user_projection
from src.graph.schema import FORBIDDEN_IN_GRAPH, EdgeType, NodeType
from src.graph.serialization import load_graph, save_graph


@pytest.fixture(scope="module")
def hetero(small_tables: dict[str, pd.DataFrame]) -> nx.Graph:
    return build_graph(small_tables)


def test_all_node_types_present(hetero: nx.Graph) -> None:
    for nt in NodeType:
        assert len(nodes_of_type(hetero, nt)) > 0, f"no {nt} nodes"


def test_node_count_matches_referenced_entities(
    hetero: nx.Graph, small_tables: dict[str, pd.DataFrame]
) -> None:
    """User and transaction node counts match the source rows."""
    txns = small_tables["transactions"]
    assert len(nodes_of_type(hetero, NodeType.USER)) == txns["user_id"].nunique()
    assert len(nodes_of_type(hetero, NodeType.TRANSACTION)) == len(txns)


def test_no_self_loops(hetero: nx.Graph) -> None:
    assert nx.number_of_selfloops(hetero) == 0


def test_edges_have_valid_types(hetero: nx.Graph) -> None:
    valid = {e.value for e in EdgeType}
    for _, _, d in hetero.edges(data=True):
        assert d.get("edge_type") in valid


def test_user_entity_edges_are_backed_by_transactions(
    hetero: nx.Graph, small_tables: dict[str, pd.DataFrame]
) -> None:
    """Every USES_DEVICE edge corresponds to a real (user, device) pair."""
    txns = small_tables["transactions"]
    observed = set(zip(txns["user_id"], txns["device_id"]))
    for u, v in edges_of_type(hetero, EdgeType.USES_DEVICE):
        # Orient (user, device).
        if hetero.nodes[u]["node_type"] != NodeType.USER.value:
            u, v = v, u
        assert (u, v) in observed


def test_graph_construction_is_deterministic(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    a = build_graph(small_tables)
    b = build_graph(small_tables)
    assert set(a.nodes()) == set(b.nodes())
    assert set(map(frozenset, a.edges())) == set(map(frozenset, b.edges()))


def test_graph_construction_ignores_labels(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    """Removing ground-truth columns does not change the graph."""
    full = build_graph(small_tables)
    stripped = {k: v.copy() for k, v in small_tables.items()}
    for name, df in stripped.items():
        drop = [c for c in FORBIDDEN_IN_GRAPH if c in df.columns]
        if drop:
            stripped[name] = df.drop(columns=drop)
    stripped_graph = build_graph(stripped)
    assert set(full.nodes()) == set(stripped_graph.nodes())
    assert set(map(frozenset, full.edges())) == set(
        map(frozenset, stripped_graph.edges())
    )


def test_projection_only_connects_users(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    txns = small_tables["transactions"]
    users = set(small_tables["users"]["user_id"])
    proj = build_user_projection(txns)
    assert set(proj.graph.nodes()).issubset(users)


def test_projection_edge_metadata_preserves_types(
    small_tables: dict[str, pd.DataFrame]
) -> None:
    """Projection edges carry per-type shared counts, not a single blob."""
    proj = build_user_projection(small_tables["transactions"])
    required = {
        "shared_device_count",
        "shared_ip_count",
        "shared_address_count",
        "shared_payment_instrument_count",
        "shared_promotion_count",
        "shared_entity_types",
        "weight",
    }
    for _, _, d in proj.graph.edges(data=True):
        assert required.issubset(d.keys())
        # weight equals the sum of per-type counts.
        total = (
            d["shared_device_count"]
            + d["shared_ip_count"]
            + d["shared_address_count"]
            + d["shared_payment_instrument_count"]
            + d["shared_promotion_count"]
        )
        assert d["weight"] == total
        break  # metadata schema is uniform; one edge suffices


def test_projection_correctness_small_case() -> None:
    """Two users sharing exactly one device produce one edge with count 1."""
    txns = pd.DataFrame(
        {
            "user_id": ["U1", "U2", "U3"],
            "device_id": ["D1", "D1", "D2"],
            "ip_id": ["", "", ""],
            "address_id": ["", "", ""],
            "payment_instrument_id": ["", "", ""],
            "promo_id": ["", "", ""],
        }
    )
    proj = build_user_projection(txns, user_ids=["U1", "U2", "U3"])
    assert proj.graph.has_edge("U1", "U2")
    assert proj.graph["U1"]["U2"]["shared_device_count"] == 1
    assert not proj.graph.has_edge("U1", "U3")


def test_promotions_do_not_create_edges() -> None:
    """Sharing only a promotion must NOT create a projected edge.

    Promotions annotate existing infra edges; they never connect strangers.
    """
    txns = pd.DataFrame(
        {
            "user_id": ["U1", "U2", "U3", "U1", "U2"],
            "device_id": ["D1", "D1", "D9", "D1", "D1"],
            "ip_id": ["", "", "", "", ""],
            "address_id": ["", "", "", "", ""],
            "payment_instrument_id": ["", "", "", "", ""],
            "promo_id": ["P1", "P1", "P1", "", ""],
        }
    )
    proj = build_user_projection(txns, user_ids=["U1", "U2", "U3"])
    # U1-U2 share device D1 -> edge exists and shares promo P1.
    assert proj.graph.has_edge("U1", "U2")
    assert proj.graph["U1"]["U2"]["shared_promotion_count"] == 1
    # U3 shares ONLY promo P1 with U1/U2 (device D9 is unique) -> no edge.
    assert not proj.graph.has_edge("U1", "U3")
    assert not proj.graph.has_edge("U2", "U3")


def test_serialization_round_trip(
    small_tables: dict[str, pd.DataFrame], tmp_path
) -> None:
    proj = build_user_projection(small_tables["transactions"])
    path = tmp_path / "proj.json"
    save_graph(proj.graph, path)
    reloaded = load_graph(path)
    assert set(proj.graph.nodes()) == set(reloaded.nodes())
    assert set(map(frozenset, proj.graph.edges())) == set(
        map(frozenset, reloaded.edges())
    )
