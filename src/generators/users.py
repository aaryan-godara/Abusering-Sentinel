"""Base user-identity generator.

Creates the raw :class:`UserProfile` shells (id, city, age group, account
creation date) for the whole population. Segment, grouping, infrastructure, and
behaviour are filled in later by the group / behaviour / ring stages, because
those depend on how the user is organised.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np

from src.config.generation import AGE_GROUPS, CITIES, GenerationConfig
from src.generators.common import UserProfile, make_id

# Age-group popularity (younger-skewed, typical of digital-payment users).
_AGE_WEIGHTS: dict[str, float] = {
    "18-25": 2.5,
    "26-35": 3.0,
    "36-45": 2.0,
    "46-60": 1.2,
    "60+": 0.5,
}


def generate_users(config: GenerationConfig, rng: np.random.Generator) -> list[UserProfile]:
    """Return ``config.num_users`` bare user profiles.

    City assignment is clustered (a handful of cities dominate) rather than
    uniform, so shared-location signals are meaningful.
    """
    age_weights = np.array([_AGE_WEIGHTS[a] for a in AGE_GROUPS])
    age_probs = age_weights / age_weights.sum()
    ages = rng.choice(list(AGE_GROUPS), size=config.num_users, p=age_probs)

    # City popularity via a Dirichlet draw -> a few big cities, a long tail.
    city_probs = rng.dirichlet(np.linspace(3.0, 0.5, len(CITIES)))
    cities = rng.choice(list(CITIES), size=config.num_users, p=city_probs)

    start = datetime.combine(config.date_start, time.min)
    end = datetime.combine(config.date_end, time.min)
    span_days = max((end - start).days, 1)
    # Account creation biased toward earlier in the window (established base).
    created_offsets = np.clip(
        rng.beta(1.3, 2.5, size=config.num_users) * span_days, 0, span_days
    )

    users: list[UserProfile] = []
    for i in range(config.num_users):
        created_dt = start + timedelta(days=int(created_offsets[i]))
        users.append(
            UserProfile(
                user_id=make_id("USR", i, width=8),
                segment="individual",  # provisional; set during grouping
                city=str(cities[i]),
                age_group=str(ages[i]),
                account_created_at=created_dt,
            )
        )
    return users


__all__ = ["generate_users"]
