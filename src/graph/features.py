"""User graph features computed from the heterogeneous graph + user projection.

All features here are derived from *observable* structure only. None reads a
ground-truth label. Feature groups (see the feature manifest for the full list):

* Direct connection: degree in the heterogeneous graph + distinct entity counts.
* Shared-entity: number of other accounts sharing each entity type (from the
  user projection edges).
* Neighbor: statistics over projected user neighbors.
* Two-hop: reach through two projection hops (users and entities).
* Centrality: degree centrality, clustering coefficient, and approximate
  betweenness (sampled — see :func:`compute_centrality`).

Complexity is kept near-linear in edges: we avoid all-pairs shortest paths and
cap betweenness with sampling. Two-hop counts use neighbor-set unions, not
pairwise products over all users.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.graph.builder import nodes_of_type
from src.graph.schema import EdgeType, NodeType

# Heterogeneous edge type -> the distinct-entity feature it feeds.
_ENTITY_DEGREE_FEATURES: tuple[tuple[EdgeType, str], ...] = (
    (EdgeType.USES_DEVICE, "unique_devices"),
    (EdgeType.CONNECTS_FROM, "unique_ips"),
    (EdgeType.ASSOCIATED_WITH, "unique_addresses"),
    (EdgeType.USES_PAYMENT_INSTRUMENT, "unique_payment_instruments"),
)

_SHARED_COUNT_ATTRS: tuple[tuple[str, str], ...] = (
    ("shared_device_count", "number_of_accounts_sharing_devices"),
    ("shared_ip_count", "number_of_accounts_sharing_ips"),
    ("shared_address_count", "number_of_accounts_sharing_addresses"),
    ("shared_payment_instrument_count", "number_of_accounts_sharing_payment_instruments"),
    ("shared_promotion_count", "number_of_accounts_sharing_promotions"),
)


def _hetero_user_features(hetero: nx.Graph) -> pd.DataFrame:
    """Direct-connection features from the heterogeneous graph."""
    users = nodes_of_type(hetero, NodeType.USER)
    rows: dict[str, dict] = {u: {} for u in users}

    # Distinct entity counts per user by edge type.
    for u in users:
        rec = rows[u]
        counts = {name: 0 for _, name in _ENTITY_DEGREE_FEATURES}
        unique_merchants = 0
        unique_promotions = 0
        weighted = 0
        deg = 0
        # Users connect directly to entities and to their transactions.
        for nbr in hetero.neighbors(u):
            edata = hetero[u][nbr]
            etype = edata.get("edge_type")
            weighted += int(edata.get("weight", 1))
            deg += 1
            for et, name in _ENTITY_DEGREE_FEATURES:
                if etype == et.value:
                    counts[name] += 1
        rec["degree"] = deg
        rec["weighted_degree"] = weighted
        rec.update(counts)

    # Merchant / promotion reach requires one hop through transactions.
    # Build once: user -> set(merchant), user -> set(promotion).
    txn_type = NodeType.TRANSACTION.value
    user_merchants: dict[str, set] = {u: set() for u in users}
    user_promos: dict[str, set] = {u: set() for u in users}
    for u in users:
        for t in hetero.neighbors(u):
            if hetero.nodes[t].get("node_type") != txn_type:
                continue
            for m in hetero.neighbors(t):
                mt = hetero.nodes[m].get("node_type")
                if mt == NodeType.MERCHANT.value:
                    user_merchants[u].add(m)
                elif mt == NodeType.PROMOTION.value:
                    user_promos[u].add(m)
    for u in users:
        rows[u]["unique_merchants"] = len(user_merchants[u])
        rows[u]["unique_promotions"] = len(user_promos[u])

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "user_id"
    return df.reset_index()


def _projection_features(projection: nx.Graph, users: list[str]) -> pd.DataFrame:
    """Shared-entity, neighbor, and two-hop features from the user projection.

    Two-hop entity aggregates are computed with precomputed per-node incident
    sums so the cost is O(sum of degrees) rather than O(sum of degree^2) from a
    naive neighbor-of-neighbor triple loop.
    """
    degree = dict(projection.degree())
    high_degree_threshold = _high_degree_threshold(degree)

    # Precompute per-node sums of incident shared-entity counts (for two-hop).
    node_entity_sum: dict[str, int] = {}
    node_dev_sum: dict[str, int] = {}
    node_ip_sum: dict[str, int] = {}
    for v in projection.nodes():
        e_sum = dev_sum = ip_sum = 0
        for w in projection.neighbors(v):
            ed = projection[v][w]
            d = int(ed.get("shared_device_count", 0))
            ip = int(ed.get("shared_ip_count", 0))
            dev_sum += d
            ip_sum += ip
            e_sum += (
                d
                + ip
                + int(ed.get("shared_address_count", 0))
                + int(ed.get("shared_payment_instrument_count", 0))
                + int(ed.get("shared_promotion_count", 0))
            )
        node_entity_sum[v] = e_sum
        node_dev_sum[v] = dev_sum
        node_ip_sum[v] = ip_sum

    rows: list[dict] = []
    for u in users:
        rec: dict = {"user_id": u}

        if not projection.has_node(u):
            for _, name in _SHARED_COUNT_ATTRS:
                rec[name] = 0
            rec.update(
                {
                    "number_of_user_neighbors": 0,
                    "number_of_high_degree_neighbors": 0,
                    "average_neighbor_degree": 0.0,
                    "maximum_neighbor_degree": 0,
                    "two_hop_user_count": 0,
                    "two_hop_unique_entity_count": 0,
                    "two_hop_shared_device_count": 0,
                    "two_hop_shared_ip_count": 0,
                }
            )
            rows.append(rec)
            continue

        neighbors = list(projection.neighbors(u))
        neighbor_set = set(neighbors)

        # Shared-entity counts: sum per-type shared counts over incident edges.
        shared_totals = {name: 0 for _, name in _SHARED_COUNT_ATTRS}
        for v in neighbors:
            edata = projection[u][v]
            for attr, name in _SHARED_COUNT_ATTRS:
                shared_totals[name] += int(edata.get(attr, 0))
        rec.update(shared_totals)

        # Neighbor features.
        nbr_degs = [degree[v] for v in neighbors]
        rec["number_of_user_neighbors"] = len(neighbors)
        rec["number_of_high_degree_neighbors"] = sum(
            1 for d in nbr_degs if d >= high_degree_threshold
        )
        rec["average_neighbor_degree"] = float(np.mean(nbr_degs)) if nbr_degs else 0.0
        rec["maximum_neighbor_degree"] = int(max(nbr_degs)) if nbr_degs else 0

        # Two-hop features. User set via one neighbor-of-neighbor pass; entity
        # aggregates via precomputed per-node sums minus the back-edge to u.
        two_hop_users: set = set()
        two_hop_entity = 0
        two_hop_dev = 0
        two_hop_ip = 0
        for v in neighbors:
            ed_uv = projection[u][v]
            two_hop_entity += node_entity_sum[v] - (
                int(ed_uv.get("shared_device_count", 0))
                + int(ed_uv.get("shared_ip_count", 0))
                + int(ed_uv.get("shared_address_count", 0))
                + int(ed_uv.get("shared_payment_instrument_count", 0))
                + int(ed_uv.get("shared_promotion_count", 0))
            )
            two_hop_dev += node_dev_sum[v] - int(ed_uv.get("shared_device_count", 0))
            two_hop_ip += node_ip_sum[v] - int(ed_uv.get("shared_ip_count", 0))
            for w in projection.neighbors(v):
                if w != u and w not in neighbor_set:
                    two_hop_users.add(w)

        rec["two_hop_user_count"] = len(two_hop_users)
        rec["two_hop_unique_entity_count"] = int(two_hop_entity)
        rec["two_hop_shared_device_count"] = int(two_hop_dev)
        rec["two_hop_shared_ip_count"] = int(two_hop_ip)

        rows.append(rec)

    return pd.DataFrame(rows)


def _high_degree_threshold(degree: dict[str, int]) -> int:
    """A 'high degree' neighbor is one at/above the 90th percentile of degree."""
    if not degree:
        return 1
    values = np.array(list(degree.values()))
    thr = np.percentile(values, 90)
    return max(int(thr), 2)


def compute_centrality(
    projection: nx.Graph,
    *,
    betweenness_sample_cap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Centrality features on the user projection.

    * ``degree_centrality`` — exact.
    * ``clustering_coefficient`` — exact (local clustering).
    * ``betweenness_centrality`` — exact when the graph has
      <= ``betweenness_sample_cap`` nodes, otherwise a deterministic sampled
      approximation using ``k=betweenness_sample_cap`` pivot nodes. Which mode was
      used is recorded in the ``betweenness_approx`` column (bool).

    Betweenness is computed per connected component (Brandes' algorithm cost
    grows with component size); for the sampled case ``k`` is scaled to the
    component but capped, keeping runtime bounded on a laptop.
    """
    users = sorted(projection.nodes())
    deg_cent = nx.degree_centrality(projection) if projection.number_of_nodes() else {}
    clustering = nx.clustering(projection) if projection.number_of_edges() else {
        u: 0.0 for u in users
    }

    n = projection.number_of_nodes()
    approx = n > betweenness_sample_cap
    if projection.number_of_edges() == 0:
        betw = {u: 0.0 for u in users}
        approx = False
    elif approx:
        betw = nx.betweenness_centrality(
            projection, k=min(betweenness_sample_cap, n), seed=seed, normalized=True
        )
    else:
        betw = nx.betweenness_centrality(projection, normalized=True)

    return pd.DataFrame(
        {
            "user_id": users,
            "degree_centrality": [round(deg_cent.get(u, 0.0), 8) for u in users],
            "clustering_coefficient": [round(clustering.get(u, 0.0), 8) for u in users],
            "betweenness_centrality": [round(betw.get(u, 0.0), 8) for u in users],
            "betweenness_approx": [approx for _ in users],
        }
    )


def compute_user_graph_features(
    hetero: nx.Graph,
    projection: nx.Graph,
    *,
    betweenness_sample_cap: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute the full user graph-feature table.

    Returns one row per USER in ``hetero`` with all graph features merged.
    """
    users = nodes_of_type(hetero, NodeType.USER)
    direct = _hetero_user_features(hetero)
    proj = _projection_features(projection, users)
    cent = compute_centrality(
        projection, betweenness_sample_cap=betweenness_sample_cap, seed=seed
    )

    df = direct.merge(proj, on="user_id", how="left").merge(
        cent, on="user_id", how="left"
    )
    # Any user absent from projection centrality (shouldn't happen) -> 0.
    df = df.fillna(0)
    return df.sort_values("user_id").reset_index(drop=True)


__all__ = ["compute_user_graph_features", "compute_centrality"]
