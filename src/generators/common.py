"""Shared primitives for the synthetic generators.

Holds the deterministic RNG wrapper, ID formatting, sampling helpers, and the
in-memory data structures (:class:`UserProfile`, :class:`RingSpec`,
:class:`GroupSpec`) that flow between generation stages.

Determinism contract: every stochastic draw goes through a single
:class:`numpy.random.Generator` seeded once from the configuration. Given the
same config and seed, the sequence of draws — and therefore the dataset — is
identical across runs and platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# ID helpers. Zero-padded, opaque, prefix-tagged. No real-world PII.
# ---------------------------------------------------------------------------


def make_id(prefix: str, index: int, width: int = 8) -> str:
    """Build an opaque, sortable identifier such as ``USR_00000042``."""
    return f"{prefix}_{index:0{width}d}"


# ---------------------------------------------------------------------------
# Sampling helpers.
# ---------------------------------------------------------------------------


def lognormal_amount(
    rng: np.random.Generator, low: float, high: float, sigma: float = 0.9
) -> float:
    """Draw a positive, right-skewed amount roughly within ``[low, high]``.

    ``low``/``high`` anchor the 10th/90th percentiles of a lognormal. The result
    is not hard-clipped, so realistic outliers above ``high`` occur naturally.
    """
    low = max(low, 1.0)
    high = max(high, low * 2.0)
    mu = (np.log(low) + np.log(high)) / 2.0
    # Spread so that ~80% of mass falls inside the anchors.
    spread = (np.log(high) - np.log(low)) / (2.0 * 1.2816)
    value = float(rng.lognormal(mean=mu, sigma=max(spread, sigma * 0.2)))
    return round(value, 2)


def skewed_counts(
    rng: np.random.Generator, n: int, mean: float, floor: int = 1
) -> np.ndarray:
    """Return ``n`` integer counts from a skewed (gamma) distribution.

    Used for transactions-per-user and similar quantities where a minority of
    users are far more active than the median.
    """
    shape = 1.5  # gamma shape controls skew; smaller => heavier skew
    scale = max(mean - floor, 0.1) / shape
    draws = rng.gamma(shape=shape, scale=scale, size=n)
    return np.maximum(floor, np.round(draws).astype(int) + floor)


def weighted_choice_indices(
    rng: np.random.Generator, weights: np.ndarray, size: int
) -> np.ndarray:
    """Sample ``size`` indices proportional to ``weights`` (with replacement)."""
    probs = weights / weights.sum()
    return rng.choice(len(weights), size=size, p=probs)


def random_timestamp(
    rng: np.random.Generator,
    start: datetime,
    end: datetime,
    *,
    business_hours_bias: float = 0.0,
) -> datetime:
    """Sample a timestamp in ``[start, end)`` with optional daytime bias.

    ``business_hours_bias`` in ``[0, 1]`` raises the probability that the drawn
    hour lands in the 9:00-19:00 window (used for office-like activity).
    """
    total_seconds = int((end - start).total_seconds())
    base = start + timedelta(seconds=int(rng.integers(0, total_seconds)))
    if business_hours_bias > 0 and rng.random() < business_hours_bias:
        # Re-roll the hour into working hours, keep the date.
        hour = int(rng.integers(9, 19))
        base = base.replace(hour=hour, minute=int(rng.integers(0, 60)))
    return base


def coordinated_timestamp(
    rng: np.random.Generator,
    anchor: datetime,
    window_hours: float,
    start: datetime,
    end: datetime,
) -> datetime:
    """Sample a timestamp near ``anchor`` within ``+/- window_hours``.

    Coordination is deliberately loose (uniform jitter), never exact, and the
    result is clamped to the dataset's temporal span.
    """
    jitter = rng.uniform(-window_hours, window_hours)
    ts = anchor + timedelta(hours=float(jitter))
    if ts < start:
        ts = start + timedelta(minutes=int(rng.integers(0, 600)))
    if ts >= end:
        ts = end - timedelta(minutes=int(rng.integers(1, 600)))
    return ts


# ---------------------------------------------------------------------------
# In-memory structures passed between stages.
# ---------------------------------------------------------------------------


class ResourcePool:
    """Hands out IDs from a pre-generated pool.

    ``fresh`` returns previously-unused IDs (a user's own primary device, etc.).
    When the pool is exhausted it falls back to random reuse, which is realistic
    (infrastructure genuinely gets recycled). ``shared`` reserves a small block
    of fresh IDs to be co-used by a group or ring.
    """

    def __init__(self, ids: list[str], rng: np.random.Generator) -> None:
        self._ids = ids
        self._rng = rng
        self._cursor = 0

    def fresh(self) -> str:
        """Return an unused ID, or a random existing one once exhausted."""
        if self._cursor < len(self._ids):
            value = self._ids[self._cursor]
            self._cursor += 1
            return value
        return self._ids[int(self._rng.integers(0, len(self._ids)))]

    def fresh_block(self, n: int) -> list[str]:
        """Return ``n`` fresh IDs (falling back to reuse if exhausted)."""
        return [self.fresh() for _ in range(n)]

    def random_existing(self) -> str:
        """Return a random ID from the whole pool (may repeat)."""
        return self._ids[int(self._rng.integers(0, len(self._ids)))]


@dataclass
class UserProfile:
    """Everything the transaction generator needs about one user.

    Ground-truth fields (``is_abuse_account``, ``ring_id``, ``ring_type``) are
    kept here for generation only; they are excluded from model features via the
    label-exclusion manifest.
    """

    user_id: str
    segment: str
    city: str
    age_group: str
    account_created_at: datetime

    # Infrastructure pools this user draws from.
    device_ids: list[str] = field(default_factory=list)
    ip_ids: list[str] = field(default_factory=list)
    address_ids: list[str] = field(default_factory=list)
    payment_instrument_ids: list[str] = field(default_factory=list)

    # Behaviour.
    merchant_category_prefs: list[str] = field(default_factory=list)
    promo_affinity: float = 0.15
    activity_level: float = 1.0
    business_hours_bias: float = 0.0

    # Grouping / ground truth.
    group_id: str | None = None
    group_type: str | None = None  # family | office | college | hostel | individual
    is_abuse_account: bool = False
    ring_id: str | None = None
    ring_type: str | None = None

    # Assigned split (filled by the splitter).
    split: str | None = None


@dataclass
class GroupSpec:
    """A legitimate group of users sharing some infrastructure."""

    group_id: str
    group_type: str  # family | office | college | hostel
    city: str
    user_ids: list[str] = field(default_factory=list)
    shared_ip_ids: list[str] = field(default_factory=list)
    shared_address_ids: list[str] = field(default_factory=list)
    shared_device_ids: list[str] = field(default_factory=list)


@dataclass
class RingSpec:
    """An abuse ring and its coordination parameters."""

    ring_id: str
    ring_type: str
    city: str
    user_ids: list[str] = field(default_factory=list)

    shared_device_ids: list[str] = field(default_factory=list)
    shared_ip_ids: list[str] = field(default_factory=list)
    shared_address_ids: list[str] = field(default_factory=list)
    shared_payment_ids: list[str] = field(default_factory=list)
    shared_promo_ids: list[str] = field(default_factory=list)

    coordination_window_hours: float = 48.0
    amount_coordination_sigma: float = 0.35
    abuse_txn_fraction: float = 0.5
    target_merchant_categories: list[str] = field(default_factory=list)

    # Hidden-scenario tag, if this ring was built for a specific scenario.
    scenario: str | None = None

    ground_truth_loss: float = 0.0  # filled after transactions are generated


__all__ = [
    "make_id",
    "lognormal_amount",
    "skewed_counts",
    "weighted_choice_indices",
    "random_timestamp",
    "coordinated_timestamp",
    "ResourcePool",
    "UserProfile",
    "GroupSpec",
    "RingSpec",
]
