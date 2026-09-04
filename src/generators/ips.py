"""IP / network-identity pool generator.

Each IP identity has a network type (home, mobile, office, college, hostel,
public, business) and belongs to a network cluster (e.g. a campus or office
subnet). Office/college/hostel clusters are intended to legitimately connect
many users; that high connectivity is a feature of the data, not evidence of
abuse.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import CITIES, NETWORK_TYPES, GenerationConfig
from src.generators.common import make_id

# Network-type mix. Home/mobile dominate; shared-institution types are rarer
# but each connects many users.
_TYPE_WEIGHTS: dict[str, float] = {
    "home": 3.0,
    "mobile": 3.0,
    "office": 1.2,
    "college": 0.8,
    "hostel": 0.5,
    "public": 1.0,
    "business": 0.9,
}


def generate_ips(config: GenerationConfig, rng: np.random.Generator) -> list[dict]:
    """Return ``config.num_ips`` network-identity records."""
    weights = np.array([_TYPE_WEIGHTS[t] for t in NETWORK_TYPES])
    probs = weights / weights.sum()
    types = rng.choice(list(NETWORK_TYPES), size=config.num_ips, p=probs)
    cities = rng.choice(list(CITIES), size=config.num_ips)

    # Network clusters: shared-infra networks group into subnets.
    num_clusters = max(1, config.num_ips // 8)
    clusters = rng.integers(0, num_clusters, size=config.num_ips)

    ips: list[dict] = []
    for i in range(config.num_ips):
        ips.append(
            {
                "ip_id": make_id("IP", i, width=6),
                "network_type": str(types[i]),
                "city": str(cities[i]),
                "network_cluster": make_id("NET", int(clusters[i]), width=5),
            }
        )
    return ips


__all__ = ["generate_ips"]
