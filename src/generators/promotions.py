"""Promotion generator.

Creates a small pool of promotions with varied types, discount mechanics, and
eligibility rules. Named promo codes (NEW500, FIRSTPAY, ...) are seeded first so
the well-known examples always exist, then filled out to ``num_promotions``.

Deliberately, no promotion is inherently "abusive": abuse rings and legitimate
users both use promotions. Coordination is what later matters, not the promo id.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import GenerationConfig
from src.generators.common import make_id

_PROMO_TYPES: tuple[str, ...] = ("flat_discount", "percentage", "cashback", "referral")
_ELIGIBILITY: tuple[str, ...] = ("new_user", "all_users", "returning_user", "referral")

# Seed codes that should always be present (name, type, flat_value, pct).
_SEED_PROMOS: tuple[tuple[str, str, float, float], ...] = (
    ("NEW500", "flat_discount", 500.0, 0.0),
    ("FIRSTPAY", "cashback", 0.0, 0.0),
    ("REFER200", "referral", 200.0, 0.0),
    ("CASHBACK10", "percentage", 0.0, 10.0),
    ("WELCOME300", "flat_discount", 300.0, 0.0),
)


def generate_promotions(
    config: GenerationConfig, rng: np.random.Generator
) -> list[dict]:
    """Return ``config.num_promotions`` promotion records."""
    promotions: list[dict] = []

    for code, ptype, flat, pct in _SEED_PROMOS[: config.num_promotions]:
        promotions.append(
            {
                "promo_id": code,
                "promo_type": ptype,
                "discount_value": flat,
                "discount_percentage": pct,
                "eligibility_type": (
                    "new_user" if ptype in ("flat_discount", "cashback") else "all_users"
                ),
                "max_usage": int(rng.integers(1, 6)),
            }
        )

    remaining = config.num_promotions - len(promotions)
    for i in range(remaining):
        ptype = str(rng.choice(_PROMO_TYPES))
        if ptype == "percentage":
            flat, pct = 0.0, float(rng.choice([5, 10, 15, 20, 25]))
        elif ptype == "cashback":
            flat, pct = 0.0, float(rng.choice([5, 10, 15]))
        else:
            flat, pct = float(rng.choice([100, 150, 200, 300, 500, 750])), 0.0
        promotions.append(
            {
                "promo_id": make_id("PROMO", i, width=4),
                "promo_type": ptype,
                "discount_value": flat,
                "discount_percentage": pct,
                "eligibility_type": str(rng.choice(_ELIGIBILITY)),
                "max_usage": int(rng.integers(1, 11)),
            }
        )

    return promotions


__all__ = ["generate_promotions"]
