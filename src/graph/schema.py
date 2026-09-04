"""Graph schema: node and edge type definitions for the heterogeneous graph.

Representation choice
---------------------
We use a single ``networkx.Graph`` (undirected, simple) whose nodes and edges
carry a ``node_type`` / ``edge_type`` attribute. This is the simplest
representation that faithfully preserves node and edge *types* while giving us
NetworkX's full algorithm library (components, clustering, Louvain, etc.).

Node IDs are namespaced by prefix (``USR_...``, ``DEV_...``) so they are already
globally unique across types; we therefore use the source IDs directly as
NetworkX node keys and record the type as an attribute. No synthetic composite
keys are needed.

Only *observable* relationships from the Phase 1 tables become edges. Ground
truth (``is_abuse_account``, ``is_abuse_transaction``, ``ring_id``,
``ring_type``, ``estimated_abuse_value``) is NEVER used to create nodes or edges.
"""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Heterogeneous node types."""

    USER = "user"
    DEVICE = "device"
    IP = "ip"
    ADDRESS = "address"
    PAYMENT_INSTRUMENT = "payment_instrument"
    MERCHANT = "merchant"
    PROMOTION = "promotion"
    TRANSACTION = "transaction"


class EdgeType(str, Enum):
    """Observable relationship types (all derived from source rows)."""

    USES_DEVICE = "uses_device"  # USER  -> DEVICE
    CONNECTS_FROM = "connects_from"  # USER  -> IP
    ASSOCIATED_WITH = "associated_with"  # USER  -> ADDRESS
    USES_PAYMENT_INSTRUMENT = "uses_payment_instrument"  # USER  -> PAYMENT_INSTRUMENT
    MAKES_TRANSACTION = "makes_transaction"  # USER  -> TRANSACTION
    AT_MERCHANT = "at_merchant"  # TRANSACTION -> MERCHANT
    USES_PROMOTION = "uses_promotion"  # TRANSACTION -> PROMOTION


# Node-type -> (source table, primary-key column). Used by the builder and by
# graph integrity validation.
NODE_SOURCE: dict[NodeType, tuple[str, str]] = {
    NodeType.USER: ("users", "user_id"),
    NodeType.DEVICE: ("devices", "device_id"),
    NodeType.IP: ("ips", "ip_id"),
    NodeType.ADDRESS: ("addresses", "address_id"),
    NodeType.PAYMENT_INSTRUMENT: ("payment_instruments", "payment_instrument_id"),
    NodeType.MERCHANT: ("merchants", "merchant_id"),
    NodeType.PROMOTION: ("promotions", "promo_id"),
    NodeType.TRANSACTION: ("transactions", "transaction_id"),
}

# The entity node types a USER can directly share with another USER. Order is
# fixed for deterministic feature columns.
SHARED_ENTITY_TYPES: tuple[NodeType, ...] = (
    NodeType.DEVICE,
    NodeType.IP,
    NodeType.ADDRESS,
    NodeType.PAYMENT_INSTRUMENT,
    NodeType.PROMOTION,
)

# Ground-truth fields that must never influence graph construction.
FORBIDDEN_IN_GRAPH: frozenset[str] = frozenset(
    {
        "is_abuse_account",
        "is_abuse_transaction",
        "ring_id",
        "ring_type",
        "estimated_abuse_value",
    }
)


__all__ = [
    "NodeType",
    "EdgeType",
    "NODE_SOURCE",
    "SHARED_ENTITY_TYPES",
    "FORBIDDEN_IN_GRAPH",
]
