"""User-centric projection.

Projects the heterogeneous graph onto USER nodes: two users are connected when
they share observable **physical infrastructure** (device, IP, address, or
payment instrument). The projected edge preserves *per-type* shared counts so
downstream logic can tell "shared a device" from "shared a payment" — we never
collapse everything into one unexplained edge.

Promotions: policy decision (Phase 3 feature-shift investigation)
-----------------------------------------------------------------
Promotions do **not** create projection edges. A popular promotion is used by
hundreds of unrelated users, so a promotion-based clique connects arbitrary
strangers and (a) carries almost no coordination signal, (b) explodes runtime,
and (c) interacts pathologically with the hub cap: in a large split every
popular promo exceeds the cap and is dropped (feature becomes constant 0), while
in a small split some fall under the cap and survive — an inconsistent feature
definition across splits.

Instead, promotions **annotate edges that already exist** from shared physical
infrastructure: ``shared_promotion_count`` on an edge is the number of
promotions both users used. This keeps promotion coordination as a signal
*between users already linked by infrastructure* (which is what a promotion-abuse
ring looks like) without inventing strangers-as-neighbours edges. The bound is
the number of existing edges, so there is no runtime blow-up and no cap
dependence. See ``reports/feature_shift_investigation.json``.

Edge metadata (all observable, no labels)::

    shared_device_count
    shared_ip_count
    shared_address_count
    shared_payment_instrument_count
    shared_promotion_count   # promos co-used by two infra-linked users
    shared_entity_types      # how many of the 5 types are shared (>0)
    weight                   # total shared entities across all types

Performance
-----------
We build the projection per physical entity: for each entity we connect its
user-set. This is O(sum over entities of k_e^2 / 2), far cheaper than O(users^2).
Entities shared by more than ``hub_cap`` users are skipped (recorded in
:attr:`ProjectionResult.skipped_hubs`) — this documents, not hides, the choice.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import pandas as pd

from src.graph.schema import NodeType

# Physical-infrastructure entity types that CREATE projection edges. Promotions
# are handled separately (annotation only) — see module docstring.
_EDGE_ENTITY_TYPES: tuple[NodeType, ...] = (
    NodeType.DEVICE,
    NodeType.IP,
    NodeType.ADDRESS,
    NodeType.PAYMENT_INSTRUMENT,
)

# Transaction column backing each shared-entity node type.
_ENTITY_COLUMN: dict[NodeType, str] = {
    NodeType.DEVICE: "device_id",
    NodeType.IP: "ip_id",
    NodeType.ADDRESS: "address_id",
    NodeType.PAYMENT_INSTRUMENT: "payment_instrument_id",
    NodeType.PROMOTION: "promo_id",
}

_COUNT_ATTR: dict[NodeType, str] = {
    NodeType.DEVICE: "shared_device_count",
    NodeType.IP: "shared_ip_count",
    NodeType.ADDRESS: "shared_address_count",
    NodeType.PAYMENT_INSTRUMENT: "shared_payment_instrument_count",
    NodeType.PROMOTION: "shared_promotion_count",
}

_ALL_COUNT_ATTRS: tuple[str, ...] = tuple(_COUNT_ATTR[nt] for nt in (
    NodeType.DEVICE,
    NodeType.IP,
    NodeType.ADDRESS,
    NodeType.PAYMENT_INSTRUMENT,
    NodeType.PROMOTION,
))


@dataclass
class ProjectionResult:
    """A user-user projection plus provenance."""

    graph: nx.Graph
    hub_cap: int
    skipped_hubs: dict[str, list[str]] = field(default_factory=dict)


def build_user_projection(
    transactions: pd.DataFrame,
    *,
    user_ids: list[str] | None = None,
    hub_cap: int = 150,
) -> ProjectionResult:
    """Build the user-centric projection from a transaction view.

    Parameters
    ----------
    transactions:
        Transaction rows defining observable user-entity links. Only these rows
        are used, so passing a split/time-restricted frame yields a
        split/time-restricted projection.
    user_ids:
        Optional explicit node set. Users with no shared entities still appear as
        isolated nodes so feature tables cover everyone. Defaults to the users
        present in ``transactions``.
    hub_cap:
        A physical entity shared by more than ``hub_cap`` users is skipped as an
        uninformative mega-hub, recorded in ``skipped_hubs``. Promotions never
        create edges (see module docstring), so the cap no longer produces the
        split-inconsistent promotion feature that the Phase 3 investigation
        found.
    """
    graph = nx.Graph()
    node_set = (
        sorted(user_ids)
        if user_ids is not None
        else sorted(transactions["user_id"].unique())
    )
    graph.add_nodes_from(node_set)

    skipped: dict[str, list[str]] = defaultdict(list)

    # Accumulate per-type shared counts on each user pair (physical infra only).
    pair_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for ntype in _EDGE_ENTITY_TYPES:
        col = _ENTITY_COLUMN[ntype]
        attr = _COUNT_ATTR[ntype]
        sub = transactions[[col, "user_id"]]
        sub = sub[sub[col].notna() & (sub[col] != "")]
        groups = sub.groupby(col)["user_id"].unique()

        for entity, users in groups.items():
            users = sorted(set(users))
            if len(users) < 2:
                continue
            if len(users) > hub_cap:
                skipped[attr].append(str(entity))
                continue
            for a, b in combinations(users, 2):
                key = (a, b) if a < b else (b, a)
                pair_counts[key][attr] += 1

    # Materialise edges from physical sharing.
    for (a, b), counts in pair_counts.items():
        attrs = {attr: 0 for attr in _ALL_COUNT_ATTRS}
        attrs.update(counts)
        graph.add_edge(a, b, **attrs)

    # Annotate existing edges with co-used promotions (no new edges, no cap).
    _annotate_shared_promotions(graph, transactions)

    # Finalise weight + shared_entity_types now that promo counts are set.
    for a, b, d in graph.edges(data=True):
        total = sum(int(d.get(attr, 0)) for attr in _ALL_COUNT_ATTRS)
        d["weight"] = total
        d["shared_entity_types"] = sum(
            1 for attr in _ALL_COUNT_ATTRS if int(d.get(attr, 0)) > 0
        )

    return ProjectionResult(
        graph=graph, hub_cap=hub_cap, skipped_hubs={k: v for k, v in skipped.items()}
    )


def _annotate_shared_promotions(graph: nx.Graph, transactions: pd.DataFrame) -> None:
    """Set ``shared_promotion_count`` on existing edges = co-used promo count.

    Cost is bounded by the number of edges (we look up each edge's endpoints'
    promo sets), so there is no clique blow-up regardless of promo popularity.
    """
    sub = transactions[["user_id", "promo_id"]]
    sub = sub[sub["promo_id"].notna() & (sub["promo_id"] != "")]
    user_promos: dict[str, set] = defaultdict(set)
    for uid, pid in sub.itertuples(index=False):
        user_promos[uid].add(pid)

    for a, b, d in graph.edges(data=True):
        pa = user_promos.get(a)
        pb = user_promos.get(b)
        d["shared_promotion_count"] = len(pa & pb) if pa and pb else 0


__all__ = ["ProjectionResult", "build_user_projection"]

