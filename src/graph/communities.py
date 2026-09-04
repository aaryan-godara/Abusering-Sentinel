"""Community detection on the user projection (label-free).

Uses NetworkX's Louvain implementation
(:func:`networkx.algorithms.community.louvain_communities`) on the *user
projection*, weighted by the number of shared entities. Communities are computed
purely from observable structure — no abuse label is ever consulted.

Per-user outputs:

* ``community_id``
* ``community_size``
* ``community_density``     (edge density of the community's induced subgraph)
* ``community_edge_count``

These are graph communities, NOT "fraud rings". Any correspondence between a
community and a known abuse ring is measured later, for evaluation only, and
never feeds back into community formation.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities


def detect_communities(
    projection: nx.Graph, *, seed: int = 42, resolution: float = 1.0
) -> pd.DataFrame:
    """Run Louvain and return per-user community features.

    Determinism: Louvain is seeded. Isolated users (no projection edges) each
    form their own singleton community.
    """
    users = sorted(projection.nodes())

    if projection.number_of_edges() == 0:
        return pd.DataFrame(
            {
                "user_id": users,
                "community_id": list(range(len(users))),
                "community_size": [1] * len(users),
                "community_density": [0.0] * len(users),
                "community_edge_count": [0] * len(users),
            }
        )

    communities = louvain_communities(
        projection, weight="weight", seed=seed, resolution=resolution
    )
    # Deterministic community ids: sort communities by (size desc, min member).
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))

    rows: list[dict] = []
    for cid, members in enumerate(communities):
        sub = projection.subgraph(members)
        size = sub.number_of_nodes()
        edges = sub.number_of_edges()
        density = nx.density(sub) if size > 1 else 0.0
        for u in members:
            rows.append(
                {
                    "user_id": u,
                    "community_id": cid,
                    "community_size": size,
                    "community_density": round(float(density), 8),
                    "community_edge_count": int(edges),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("user_id").reset_index(drop=True)


__all__ = ["detect_communities"]
