"""Address pool generator.

Addresses carry a type (residential, apartment, hostel, office,
shared_accommodation) and a household group. Multiple users legitimately sharing
a household group is expected (families, flatmates, hostel residents).
"""

from __future__ import annotations

import numpy as np

from src.config.generation import ADDRESS_TYPES, CITIES, GenerationConfig
from src.generators.common import make_id

_TYPE_WEIGHTS: dict[str, float] = {
    "residential": 3.0,
    "apartment": 2.5,
    "hostel": 0.6,
    "office": 1.0,
    "shared_accommodation": 0.9,
}


def generate_addresses(
    config: GenerationConfig, rng: np.random.Generator
) -> list[dict]:
    """Return ``config.num_addresses`` address records."""
    weights = np.array([_TYPE_WEIGHTS[t] for t in ADDRESS_TYPES])
    probs = weights / weights.sum()
    types = rng.choice(list(ADDRESS_TYPES), size=config.num_addresses, p=probs)
    cities = rng.choice(list(CITIES), size=config.num_addresses)

    # Household groups: most addresses are their own household, but some
    # (apartments, hostels, shared_accommodation) collapse into shared groups.
    num_households = max(1, int(config.num_addresses * 0.85))
    households = rng.integers(0, num_households, size=config.num_addresses)

    addresses: list[dict] = []
    for i in range(config.num_addresses):
        addresses.append(
            {
                "address_id": make_id("ADR", i, width=6),
                "address_type": str(types[i]),
                "city": str(cities[i]),
                "household_group": make_id("HH", int(households[i]), width=6),
            }
        )
    return addresses


__all__ = ["generate_addresses"]
