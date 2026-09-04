"""Merchant generator.

Produces a fixed pool of merchants with a non-uniform category mix and a city
assignment. Categories are weighted so that everyday spend (food, groceries,
subscriptions) dominates and big-ticket categories are rarer, mirroring how a
real merchant catalogue skews — without claiming to match any real dataset.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import CITIES, MERCHANT_CATEGORIES, GenerationConfig
from src.generators.common import make_id

# Relative popularity of each merchant category (need not sum to 1).
_CATEGORY_WEIGHTS: dict[str, float] = {
    "food": 3.0,
    "groceries": 2.5,
    "subscriptions": 2.0,
    "fashion": 1.8,
    "services": 1.6,
    "travel": 1.2,
    "electronics": 1.0,
    "home": 1.0,
    "gaming": 0.9,
    "education": 0.8,
}


def generate_merchants(
    config: GenerationConfig, rng: np.random.Generator
) -> list[dict]:
    """Return ``config.num_merchants`` merchant records."""
    weights = np.array([_CATEGORY_WEIGHTS[c] for c in MERCHANT_CATEGORIES])
    probs = weights / weights.sum()

    categories = rng.choice(
        list(MERCHANT_CATEGORIES), size=config.num_merchants, p=probs
    )
    cities = rng.choice(list(CITIES), size=config.num_merchants)

    merchants: list[dict] = []
    for i in range(config.num_merchants):
        merchants.append(
            {
                "merchant_id": make_id("MER", i, width=5),
                "merchant_category": str(categories[i]),
                "city": str(cities[i]),
            }
        )
    return merchants


__all__ = ["generate_merchants"]
