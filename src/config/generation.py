"""Configuration for synthetic dataset generation (Phase 1).

All tunable quantities live here so no magic numbers leak into the generators.
The configuration is a plain, validated Pydantic model (not environment-backed
settings): a dataset is defined entirely by a :class:`GenerationConfig` instance
plus the random seed, which together guarantee reproducibility.

Nothing in this module produces data or labels; it only describes *how much* and
*with what parameters* to generate.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Categorical vocabularies. Kept here so generators and validators agree.
# ---------------------------------------------------------------------------

USER_SEGMENTS: tuple[str, ...] = (
    "individual",
    "family_member",
    "student",
    "office_employee",
    "small_business_user",
)

AGE_GROUPS: tuple[str, ...] = ("18-25", "26-35", "36-45", "46-60", "60+")

DEVICE_TYPES: tuple[str, ...] = ("android", "ios", "web", "tablet")

NETWORK_TYPES: tuple[str, ...] = (
    "home",
    "mobile",
    "office",
    "college",
    "hostel",
    "public",
    "business",
)

ADDRESS_TYPES: tuple[str, ...] = (
    "residential",
    "apartment",
    "hostel",
    "office",
    "shared_accommodation",
)

INSTRUMENT_TYPES: tuple[str, ...] = ("card", "upi", "wallet")

MERCHANT_CATEGORIES: tuple[str, ...] = (
    "food",
    "electronics",
    "fashion",
    "travel",
    "gaming",
    "subscriptions",
    "groceries",
    "home",
    "education",
    "services",
)

TRANSACTION_STATUSES: tuple[str, ...] = ("success", "failed", "refunded")

# Synthetic Indian cities. Purely for realistic-looking grouping; no real data.
CITIES: tuple[str, ...] = (
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Surat",
    "Lucknow",
    "Kochi",
)

RING_TYPES: tuple[str, ...] = (
    "direct_infrastructure",  # Type A
    "promotion_coordination",  # Type B
    "indirect_multihop",  # Type C
    "distributed_behavioral",  # Type D
)

# Per-category synthetic amount distributions (INR). Modelled as lognormal in
# rupee space via (low, high) anchors -> these are NOT claims about real data.
CATEGORY_AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "food": (100, 2_000),
    "electronics": (1_000, 100_000),
    "fashion": (300, 15_000),
    "travel": (500, 50_000),
    "gaming": (100, 8_000),
    "subscriptions": (100, 10_000),
    "groceries": (150, 6_000),
    "home": (500, 40_000),
    "education": (500, 60_000),
    "services": (200, 20_000),
}


class RingTypeConfig(BaseModel):
    """Parameters controlling one abuse-ring archetype.

    Overlap fractions are expressed as the share of ring members that share a
    given piece of infrastructure. Values below 1.0 are intentional: rings must
    be noisy, not perfectly connected.
    """

    weight: float = Field(gt=0, description="Relative frequency of this ring type.")
    min_size: int = Field(ge=2, description="Minimum accounts in the ring.")
    max_size: int = Field(ge=2, description="Maximum accounts in the ring.")

    device_overlap: float = Field(ge=0, le=1)
    ip_overlap: float = Field(ge=0, le=1)
    address_overlap: float = Field(ge=0, le=1)
    payment_overlap: float = Field(ge=0, le=1)
    promo_overlap: float = Field(ge=0, le=1)

    # Fraction of each abuse account's transactions that are coordinated abuse
    # (the rest is legitimate-looking noise activity).
    abuse_txn_fraction: float = Field(ge=0, le=1, default=0.5)

    # Temporal coordination window width, in hours, for coordinated bursts.
    coordination_window_hours: float = Field(gt=0, default=48.0)

    # How tightly coordinated amounts cluster (lognormal sigma). Larger = looser.
    amount_coordination_sigma: float = Field(gt=0, default=0.35)

    @model_validator(mode="after")
    def _check_sizes(self) -> "RingTypeConfig":
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        return self


class HiddenScenarioConfig(BaseModel):
    """A scenario forced into the test split to probe generalization.

    ``kind`` selects the construction. ``count`` is how many such rings/groups
    to build. These are additional to the base ring population.
    """

    name: str
    kind: str  # one of: low_overlap, high_noise, indirect_multihop, legit_high_connectivity
    count: int = Field(ge=1)
    size: int = Field(ge=3)


class GenerationConfig(BaseModel):
    """Top-level, fully-configurable synthetic dataset specification."""

    # --- Reproducibility -------------------------------------------------
    random_seed: int = Field(default=42, ge=0)
    dataset_version: str = Field(default="1.0.0")

    # --- Scale (defaults match the Phase 1 brief) ------------------------
    num_users: int = Field(default=10_000, ge=1)
    num_transactions: int = Field(default=100_000, ge=1)
    num_devices: int = Field(default=12_000, ge=1)
    num_ips: int = Field(default=5_000, ge=1)
    num_addresses: int = Field(default=8_000, ge=1)
    num_payment_instruments: int = Field(default=9_000, ge=1)
    num_merchants: int = Field(default=100, ge=1)
    num_promotions: int = Field(default=20, ge=1)
    num_abuse_rings: int = Field(default=80, ge=1)

    # --- Temporal span ---------------------------------------------------
    date_start: date = Field(default=date(2025, 1, 1))
    date_end: date = Field(default=date(2025, 6, 30))

    # --- Abuse prevalence target ----------------------------------------
    # Approximate share of accounts that belong to abuse rings.
    abuse_account_fraction: float = Field(default=0.07, ge=0.01, le=0.30)

    # --- Legitimate group mix (as fractions of the user population) ------
    family_user_fraction: float = Field(default=0.22, ge=0, le=1)
    office_user_fraction: float = Field(default=0.16, ge=0, le=1)
    college_user_fraction: float = Field(default=0.12, ge=0, le=1)
    hostel_user_fraction: float = Field(default=0.06, ge=0, le=1)
    # Remainder become independent individuals.

    # --- Split fractions -------------------------------------------------
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    val_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)

    # --- Per-ring-type parameters ---------------------------------------
    ring_types: dict[str, RingTypeConfig] = Field(
        default_factory=lambda: {
            # Type A: strong but partial infrastructure overlap.
            "direct_infrastructure": RingTypeConfig(
                weight=0.30,
                min_size=5,
                max_size=12,
                device_overlap=0.6,
                ip_overlap=0.7,
                address_overlap=0.45,
                payment_overlap=0.35,
                promo_overlap=0.8,
                abuse_txn_fraction=0.55,
                coordination_window_hours=48.0,
                amount_coordination_sigma=0.35,
            ),
            # Type B: promotion-centric, little infra overlap.
            "promotion_coordination": RingTypeConfig(
                weight=0.30,
                min_size=6,
                max_size=15,
                device_overlap=0.15,
                ip_overlap=0.2,
                address_overlap=0.1,
                payment_overlap=0.1,
                promo_overlap=0.95,
                abuse_txn_fraction=0.6,
                coordination_window_hours=72.0,
                amount_coordination_sigma=0.2,
            ),
            # Type C: indirect, chain-like connections (no hub entity).
            "indirect_multihop": RingTypeConfig(
                weight=0.20,
                min_size=6,
                max_size=14,
                device_overlap=0.25,
                ip_overlap=0.25,
                address_overlap=0.25,
                payment_overlap=0.15,
                promo_overlap=0.5,
                abuse_txn_fraction=0.5,
                coordination_window_hours=96.0,
                amount_coordination_sigma=0.4,
            ),
            # Type D: distributed behavioral, minimal infra sharing.
            "distributed_behavioral": RingTypeConfig(
                weight=0.20,
                min_size=5,
                max_size=12,
                device_overlap=0.1,
                ip_overlap=0.1,
                address_overlap=0.05,
                payment_overlap=0.05,
                promo_overlap=0.7,
                abuse_txn_fraction=0.65,
                coordination_window_hours=36.0,
                amount_coordination_sigma=0.18,
            ),
        }
    )

    # --- Hidden test-only scenarios -------------------------------------
    hidden_scenarios: list[HiddenScenarioConfig] = Field(
        default_factory=lambda: [
            HiddenScenarioConfig(
                name="low_overlap_ring", kind="low_overlap", count=2, size=8
            ),
            HiddenScenarioConfig(
                name="high_noise_ring", kind="high_noise", count=2, size=9
            ),
            HiddenScenarioConfig(
                name="indirect_multihop_ring",
                kind="indirect_multihop",
                count=2,
                size=10,
            ),
            HiddenScenarioConfig(
                name="legit_high_connectivity",
                kind="legit_high_connectivity",
                count=2,
                size=40,
            ),
        ]
    )

    # --- Behavioral knobs ------------------------------------------------
    # Mean transactions per user; actual counts follow a skewed distribution
    # and are rescaled to hit num_transactions.
    mean_txns_per_user: float = Field(default=10.0, gt=0)
    failed_txn_rate: float = Field(default=0.06, ge=0, le=1)
    refunded_txn_rate: float = Field(default=0.03, ge=0, le=1)

    @model_validator(mode="after")
    def _validate(self) -> "GenerationConfig":
        if self.date_end <= self.date_start:
            raise ValueError("date_end must be after date_start")

        split_total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(
                f"train/val/test fractions must sum to 1.0, got {split_total}"
            )

        group_total = (
            self.family_user_fraction
            + self.office_user_fraction
            + self.college_user_fraction
            + self.hostel_user_fraction
        )
        if group_total >= 1.0:
            raise ValueError(
                "legitimate group fractions must leave room for individuals "
                f"(sum was {group_total})"
            )

        for name in self.ring_types:
            if name not in RING_TYPES:
                raise ValueError(f"unknown ring type: {name}")
        for canonical in RING_TYPES:
            if canonical not in self.ring_types:
                raise ValueError(f"missing ring type config: {canonical}")

        if self.failed_txn_rate + self.refunded_txn_rate >= 1.0:
            raise ValueError("failed + refunded rates must be < 1.0")
        return self

    def small(self) -> "GenerationConfig":
        """Return a fast, small configuration for tests.

        Keeps every structural feature (all ring types, hidden scenarios,
        legitimate groups) but at a fraction of the scale.
        """
        return self.model_copy(
            update={
                "num_users": 800,
                "num_transactions": 6_000,
                "num_devices": 950,
                "num_ips": 400,
                "num_addresses": 650,
                "num_payment_instruments": 720,
                "num_merchants": 40,
                "num_promotions": 12,
                "num_abuse_rings": 8,
                "abuse_account_fraction": 0.12,
                "mean_txns_per_user": 7.0,
            }
        )


DEFAULT_CONFIG = GenerationConfig()
"""Ready-to-use default configuration matching the Phase 1 brief."""


__all__ = [
    "GenerationConfig",
    "RingTypeConfig",
    "HiddenScenarioConfig",
    "DEFAULT_CONFIG",
    "USER_SEGMENTS",
    "AGE_GROUPS",
    "DEVICE_TYPES",
    "NETWORK_TYPES",
    "ADDRESS_TYPES",
    "INSTRUMENT_TYPES",
    "MERCHANT_CATEGORIES",
    "TRANSACTION_STATUSES",
    "CITIES",
    "RING_TYPES",
    "CATEGORY_AMOUNT_RANGES",
]
