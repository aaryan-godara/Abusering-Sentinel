"""Graph evidence collection for a single user investigation.

Reuses the existing heterogeneous graph and user projection — never rebuilds
them. Extracts:
- shared devices / IPs / addresses / payment instruments
- related accounts with relationship types
- multi-hop paths (up to 3 hops, max 10 per user)
- Louvain community context

No ground-truth label is read or returned.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import networkx as nx

from src.graph.schema import EdgeType, NodeType
from src.investigation.schema import (
    CommunityContext,
    MultiHopPath,
    RelatedAccount,
)


# Edge types that create sharing relationships between users
_SHARING_EDGE_TYPES: dict[str, NodeType] = {
    "shared_device_count": NodeType.DEVICE,
    "shared_ip_count": NodeType.IP,
    "shared_address_count": NodeType.ADDRESS,
    "shared_payment_instrument_count": NodeType.PAYMENT_INSTRUMENT,
}

_ATTR_TO_LABEL: dict[str, str] = {
    "shared_device_count": "shared_device",
    "shared_ip_count": "shared_ip",
    "shared_address_count": "shared_address",
    "shared_payment_instrument_count": "shared_payment_instrument",
}


class GraphEvidenceCollector:
    """Collects graph-based evidence for user investigations.

    Initialised once with the prebuilt heterogeneous graph, user projection,
    and community assignments.  Per-user queries are lightweight dictionary
    lookups and BFS traversals.
    """

    def __init__(
        self,
        hetero_graph: nx.Graph,
        projection: nx.Graph,
        community_df: "import('pandas').DataFrame | None" = None,
    ) -> None:
        self._hetero = hetero_graph
        self._proj = projection
        # Build user → community mapping once
        self._community_map: dict[str, dict] = {}
        if community_df is not None:
            for row in community_df.itertuples(index=False):
                self._community_map[row.user_id] = {
                    "community_id": int(row.community_id),
                    "community_size": int(row.community_size),
                    "community_density": float(row.community_density),
                    "community_edge_count": int(row.community_edge_count),
                }

    # ------------------------------------------------------------------
    # Shared entities
    # ------------------------------------------------------------------

    def _shared_entities_of_type(
        self, user_id: str, edge_type: EdgeType
    ) -> list[str]:
        """Return entity IDs shared via a specific edge type from the hetero graph."""
        if not self._hetero.has_node(user_id):
            return []
        return sorted(
            nbr
            for nbr in self._hetero.neighbors(user_id)
            if self._hetero[user_id][nbr].get("edge_type") == edge_type.value
        )

    def shared_devices(self, user_id: str) -> list[str]:
        return self._shared_entities_of_type(user_id, EdgeType.USES_DEVICE)

    def shared_ips(self, user_id: str) -> list[str]:
        return self._shared_entities_of_type(user_id, EdgeType.CONNECTS_FROM)

    def shared_addresses(self, user_id: str) -> list[str]:
        return self._shared_entities_of_type(user_id, EdgeType.ASSOCIATED_WITH)

    def shared_payment_instruments(self, user_id: str) -> list[str]:
        return self._shared_entities_of_type(
            user_id, EdgeType.USES_PAYMENT_INSTRUMENT
        )

    # ------------------------------------------------------------------
    # Related accounts from the user projection
    # ------------------------------------------------------------------

    def related_accounts(self, user_id: str) -> list[RelatedAccount]:
        """Return accounts related via the user projection with relationship details."""
        if not self._proj.has_node(user_id):
            return []

        results: list[RelatedAccount] = []
        for nbr in self._proj.neighbors(user_id):
            edata = self._proj[user_id][nbr]
            rel_types: list[str] = []
            total = 0
            for attr, label in _ATTR_TO_LABEL.items():
                count = int(edata.get(attr, 0))
                if count > 0:
                    rel_types.append(label)
                    total += count
            if rel_types:
                results.append(
                    RelatedAccount(
                        user_id=nbr,
                        relationship_types=rel_types,
                        shared_entity_count=total,
                    )
                )
        # Sort by shared entity count descending
        results.sort(key=lambda r: r.shared_entity_count, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Multi-hop paths (up to 3 hops, max 10)
    # ------------------------------------------------------------------

    def multi_hop_paths(
        self,
        user_id: str,
        max_hops: int = 3,
        max_paths: int = 10,
    ) -> list[MultiHopPath]:
        """Find multi-hop paths from user_id through the heterogeneous graph.

        Paths alternate user → entity → user (→ entity → user for 3 hops).
        Every returned path exists in the actual graph.
        """
        if not self._hetero.has_node(user_id):
            return []

        paths: list[MultiHopPath] = []
        # BFS-style exploration from user_id through entity nodes to other users
        # Hop 1: user → entity → user2
        # Hop 2: user → entity → user2 → entity2 → user3
        # Hop 3: user → entity → user2 → entity2 → user3 → entity3 → user4

        visited_end_users: set[str] = {user_id}

        # Get direct entity neighbours of user_id
        user_entities: list[tuple[str, str, str]] = []  # (entity_id, node_type, edge_type)
        for nbr in self._hetero.neighbors(user_id):
            ndata = self._hetero.nodes[nbr]
            edata = self._hetero[user_id][nbr]
            ntype = ndata.get("node_type", "")
            etype = edata.get("edge_type", "")
            # Only follow entity edges (not transaction edges)
            if ntype in (
                NodeType.DEVICE.value,
                NodeType.IP.value,
                NodeType.ADDRESS.value,
                NodeType.PAYMENT_INSTRUMENT.value,
            ):
                user_entities.append((nbr, ntype, etype))

        # 1-hop paths: user → entity → user2
        hop1_intermediates: list[tuple[list[str], list[str], list[str], float]] = []
        for ent_id, ent_type, etype in user_entities:
            for u2 in self._hetero.neighbors(ent_id):
                u2data = self._hetero.nodes[u2]
                if u2data.get("node_type") != NodeType.USER.value:
                    continue
                if u2 == user_id:
                    continue
                weight = int(self._hetero[ent_id][u2].get("weight", 1))
                etype2 = self._hetero[ent_id][u2].get("edge_type", "")
                hop1_intermediates.append(
                    (
                        [user_id, ent_id, u2],
                        [NodeType.USER.value, ent_type, NodeType.USER.value],
                        [etype, etype2],
                        float(weight),
                    )
                )

        # Sort by strength descending and deduplicate end users
        hop1_intermediates.sort(key=lambda x: x[3], reverse=True)
        for nodes, ntypes, etypes, strength in hop1_intermediates:
            end_user = nodes[-1]
            if end_user in visited_end_users:
                continue
            visited_end_users.add(end_user)
            paths.append(
                MultiHopPath(
                    nodes=nodes,
                    node_types=ntypes,
                    edge_types=etypes,
                    hop_count=1,
                    path_strength=strength,
                )
            )
            if len(paths) >= max_paths:
                return paths

        if max_hops < 2:
            return paths

        # 2-hop paths: extend each 1-hop path by one more entity→user step
        for p in list(paths):
            if len(paths) >= max_paths:
                break
            last_user = p.nodes[-1]
            for nbr in self._hetero.neighbors(last_user):
                if len(paths) >= max_paths:
                    break
                ndata = self._hetero.nodes[nbr]
                ntype = ndata.get("node_type", "")
                if ntype not in (
                    NodeType.DEVICE.value,
                    NodeType.IP.value,
                    NodeType.ADDRESS.value,
                    NodeType.PAYMENT_INSTRUMENT.value,
                ):
                    continue
                etype1 = self._hetero[last_user][nbr].get("edge_type", "")
                for u3 in self._hetero.neighbors(nbr):
                    if len(paths) >= max_paths:
                        break
                    u3data = self._hetero.nodes[u3]
                    if u3data.get("node_type") != NodeType.USER.value:
                        continue
                    if u3 in visited_end_users:
                        continue
                    etype2 = self._hetero[nbr][u3].get("edge_type", "")
                    weight = int(self._hetero[nbr][u3].get("weight", 1))
                    visited_end_users.add(u3)
                    paths.append(
                        MultiHopPath(
                            nodes=p.nodes + [nbr, u3],
                            node_types=p.node_types + [ntype, NodeType.USER.value],
                            edge_types=p.edge_types + [etype1, etype2],
                            hop_count=2,
                            path_strength=min(p.path_strength, float(weight)),
                        )
                    )

        if max_hops < 3:
            return paths[:max_paths]

        # 3-hop: extend 2-hop paths similarly
        two_hop_paths = [p for p in paths if p.hop_count == 2]
        for p in two_hop_paths:
            if len(paths) >= max_paths:
                break
            last_user = p.nodes[-1]
            for nbr in self._hetero.neighbors(last_user):
                if len(paths) >= max_paths:
                    break
                ndata = self._hetero.nodes[nbr]
                ntype = ndata.get("node_type", "")
                if ntype not in (
                    NodeType.DEVICE.value,
                    NodeType.IP.value,
                    NodeType.ADDRESS.value,
                    NodeType.PAYMENT_INSTRUMENT.value,
                ):
                    continue
                etype1 = self._hetero[last_user][nbr].get("edge_type", "")
                for u4 in self._hetero.neighbors(nbr):
                    if len(paths) >= max_paths:
                        break
                    u4data = self._hetero.nodes[u4]
                    if u4data.get("node_type") != NodeType.USER.value:
                        continue
                    if u4 in visited_end_users:
                        continue
                    etype2 = self._hetero[nbr][u4].get("edge_type", "")
                    weight = int(self._hetero[nbr][u4].get("weight", 1))
                    visited_end_users.add(u4)
                    paths.append(
                        MultiHopPath(
                            nodes=p.nodes + [nbr, u4],
                            node_types=p.node_types + [ntype, NodeType.USER.value],
                            edge_types=p.edge_types + [etype1, etype2],
                            hop_count=3,
                            path_strength=min(p.path_strength, float(weight)),
                        )
                    )

        return paths[:max_paths]

    # ------------------------------------------------------------------
    # Community context
    # ------------------------------------------------------------------

    def community_context(self, user_id: str) -> CommunityContext | None:
        """Return Louvain community context for the user."""
        info = self._community_map.get(user_id)
        if info is None:
            return None

        user_degree = (
            self._proj.degree(user_id) if self._proj.has_node(user_id) else 0
        )
        neighbor_count = len(list(self._proj.neighbors(user_id))) if self._proj.has_node(user_id) else 0

        return CommunityContext(
            community_id=info["community_id"],
            community_size=info["community_size"],
            community_density=info["community_density"],
            user_degree=user_degree,
            neighbor_count=neighbor_count,
        )


__all__ = ["GraphEvidenceCollector"]
