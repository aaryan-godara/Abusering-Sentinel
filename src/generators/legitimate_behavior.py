"""Legitimate behaviour model.

Assigns per-user behavioural parameters and personal infrastructure for
legitimate users. Behaviour varies by segment and by random per-user draws, so
no two users are identical and legitimate users can incidentally look
suspicious (few devices, shared promo affinity, bursty activity).

This module never sets any ground-truth label.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import MERCHANT_CATEGORIES
from src.generators.common import ResourcePool, UserProfile

# Segment -> (activity multiplier range, promo affinity range, business-hours bias).
_SEGMENT_BEHAVIOUR: dict[str, dict[str, tuple[float, float]]] = {
    "individual": {"activity": (0.5, 1.6), "promo": (0.05, 0.30), "biz": (0.0, 0.25)},
    "family_member": {"activity": (0.6, 1.5), "promo": (0.10, 0.35), "biz": (0.0, 0.30)},
    "student": {"activity": (0.7, 2.0), "promo": (0.20, 0.55), "biz": (0.0, 0.15)},
    "office_employee": {"activity": (0.6, 1.4), "promo": (0.05, 0.25), "biz": (0.35, 0.7)},
    "small_business_user": {"activity": (1.0, 2.5), "promo": (0.05, 0.20), "biz": (0.3, 0.6)},
}


def assign_behaviour(user: UserProfile, rng: np.random.Generator) -> None:
    """Set merchant preferences, promo affinity, activity, and time bias."""
    params = _SEGMENT_BEHAVIOUR.get(user.segment, _SEGMENT_BEHAVIOUR["individual"])

    lo, hi = params["activity"]
    user.activity_level = float(rng.uniform(lo, hi))
    lo, hi = params["promo"]
    user.promo_affinity = float(rng.uniform(lo, hi))
    lo, hi = params["biz"]
    user.business_hours_bias = float(rng.uniform(lo, hi))

    # Each user prefers a handful of merchant categories (2-5), sampled without
    # replacement, so merchant diversity varies across users.
    n_prefs = int(rng.integers(2, 6))
    user.merchant_category_prefs = list(
        rng.choice(list(MERCHANT_CATEGORIES), size=n_prefs, replace=False)
    )


def assign_personal_infrastructure(
    user: UserProfile,
    pools: dict[str, ResourcePool],
    rng: np.random.Generator,
) -> None:
    """Give an individual user their own devices, IPs, addresses, payments.

    Typical individual: 1 primary device (occasional secondary), 1-3 network
    identities, 1 address, 1-2 payment instruments. Group-shared infrastructure
    is layered on separately by the group builder.
    """
    # Devices: 1 primary, ~30% chance of a secondary.
    user.device_ids = [pools["device"].fresh()]
    if rng.random() < 0.3:
        user.device_ids.append(pools["device"].fresh())

    # Network identities: 1-3 (home/mobile mostly, from the personal pool).
    n_ips = int(rng.integers(1, 4))
    user.ip_ids = pools["ip"].fresh_block(n_ips)

    # Address: usually one personal address.
    user.address_ids = [pools["address"].fresh()]

    # Payment instruments: 1-2.
    n_pay = 1 + int(rng.random() < 0.5)
    user.payment_instrument_ids = pools["payment"].fresh_block(n_pay)


__all__ = ["assign_behaviour", "assign_personal_infrastructure"]
