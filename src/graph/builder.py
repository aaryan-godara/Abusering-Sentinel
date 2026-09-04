"""Deterministic heterogeneous graph builder.

Builds a ``networkx.Graph`` from the Phase 1 tables. Every edge corresponds to a
real relationship observed in the source data:

* USER -USES_DEVICE-> DEVICE                (user transacted on that device)
* USER -CONNECTS_FROM-> IP                  (user transacted from that IP)
* USER -ASSOCIATED_WITH-> ADDRESS           (user transacted at that address)
* USER -USES_PAYMENT_INSTRUMENT-> PAYMENT   (user paid with that instrument)
* USER -MAKES_TRANSACTION-> TRANSACTION     (user made the transaction)
* TRANSACTION -AT_MERCHANT-> MERCHANT       (transaction's merchant)
* TRANSACTION -USES_PROMOTION-> PROMOTION   (transaction's promo, if any)

USER->entity edges are *observed through transactions* — a user is only linked
to a device/IP/address/payment it actually transacted with. This keeps the graph
grounded in observable behaviour and lets us build time- and split-restricted
views by simply filtering the transaction rows.

Determinism: nodes and edges are added in a fixed, sorted order and construction
uses no randomness, so the same input rows always yield the same graph.

Label safety: only the columns listed below are read. None of
``FORBIDDEN_IN_GRAPH`` is ever accessed here.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from src.graph.schema import EdgeType, NodeType

# USER->entity edge derivation: (transaction column, target node type, edge type)
_USER_ENTITY_EDGES: tuple[tuple[str, NodeType, EdgeType], ...] = (
    ("device_id", NodeType.DEVICE, EdgeType.USES_DEVICE),
    ("ip_id", NodeType.IP, EdgeType.CONNECTS_FROM),
    ("address_id", NodeType.ADDRESS, EdgeType.ASSOCIATED_WITH),
    ("payment_instrument_id", NodeType.PAYMENT_INSTRUMENT, EdgeType.USES_PAYMENT_INSTRUMENT),
)

# Columns the builder is allowed to read (explicitly excludes all labels).
_ALLOWED_TXN_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "merchant_id",
    "payment_instrument_id",
    "device_id",
    "ip_id",
    "address_id",
    "promo_id",
    "transaction_status",
)


def build_graph(
    tables: dict[str, pd.DataFrame],
    *,
    transactions: pd.DataFrame | None = None,
    include_transaction_nodes: bool = True,
) -> nx.Graph:
    """Construct the heterogeneous graph.

    Parameters
    ----------
    tables:
        The Phase 1 tables (from :func:`src.validation.loader.load_tables`).
    transactions:
        Optional pre-filtered transaction frame (for split/time views). Defaults
        to ``tables['transactions']``. Only these rows drive edge creation.
    include_transaction_nodes:
        When False, TRANSACTION nodes and their edges are omitted. The
        user-centric projection and most user features do not need transaction
        nodes, so omitting them speeds up large builds. Merchant/promotion
        linkage is then folded onto USER via lightweight edges is NOT done — set
        True when merchant/promotion structure is required.

    Returns
    -------
    networkx.Graph
        Undirected simple graph; nodes carry ``node_type``, edges carry
        ``edge_type`` and a ``weight`` (co-occurrence count).
    """
    txns = tables["transactions"] if transactions is None else transactions
    # Defensive: never let a label column leak into construction.
    txns = txns[[c for c in _ALLOWED_TXN_COLUMNS if c in txns.columns]]

    graph = nx.Graph()

    # --- Nodes for referenced entities only (keeps the graph grounded) ---
    # Users present in the transaction view.
    user_ids = sorted(txns["user_id"].unique())
    for uid in user_ids:
        graph.add_node(uid, node_type=NodeType.USER.value)

    # Entity nodes are added lazily as edges reference them (below), but we add
    # merchants/promotions/entities from their tables to attach useful metadata.
    def _ensure(node_id: str, node_type: NodeType) -> None:
        if not graph.has_node(node_id):
            graph.add_node(node_id, node_type=node_type.value)

    # --- USER -> entity edges (device/ip/address/payment) ---
    for col, ntype, etype in _USER_ENTITY_EDGES:
        pair_counts = (
            txns.groupby(["user_id", col]).size().reset_index(name="weight")
        )
        for uid, ent, weight in pair_counts.itertuples(index=False):
            if ent == "" or pd.isna(ent):
                continue
            _ensure(ent, ntype)
            graph.add_edge(uid, ent, edge_type=etype.value, weight=int(weight))

    # --- TRANSACTION nodes + txn->merchant / txn->promotion / user->txn ---
    if include_transaction_nodes:
        for row in txns.itertuples(index=False):
            tid = row.transaction_id
            graph.add_node(tid, node_type=NodeType.TRANSACTION.value)
            graph.add_edge(
                row.user_id, tid, edge_type=EdgeType.MAKES_TRANSACTION.value, weight=1
            )
            _ensure(row.merchant_id, NodeType.MERCHANT)
            graph.add_edge(
                tid, row.merchant_id, edge_type=EdgeType.AT_MERCHANT.value, weight=1
            )
            promo = getattr(row, "promo_id", "")
            if promo != "" and not pd.isna(promo):
                _ensure(promo, NodeType.PROMOTION)
                graph.add_edge(
                    tid, promo, edge_type=EdgeType.USES_PROMOTION.value, weight=1
                )

    return graph


def nodes_of_type(graph: nx.Graph, node_type: NodeType) -> list[str]:
    """Return sorted node IDs of the given type."""
    return sorted(
        n for n, d in graph.nodes(data=True) if d.get("node_type") == node_type.value
    )


def edges_of_type(graph: nx.Graph, edge_type: EdgeType) -> list[tuple[str, str]]:
    """Return edges of the given type."""
    return [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("edge_type") == edge_type.value
    ]


__all__ = ["build_graph", "nodes_of_type", "edges_of_type"]
