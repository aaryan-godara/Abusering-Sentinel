"""Pydantic schemas for investigation results.

All types here represent investigator-facing output. None contains ground-truth
labels (is_abuse_account, ring_id, ring_type, is_abuse_transaction,
estimated_abuse_value).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ground-truth fields that MUST NOT appear in investigator output
# ---------------------------------------------------------------------------
GROUND_TRUTH_FIELDS: frozenset[str] = frozenset({
    "is_abuse_account",
    "ring_id",
    "ring_type",
    "is_abuse_transaction",
    "estimated_abuse_value",
})


class RiskLevel(str, Enum):
    """Discretised risk level for investigator display."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeatureAttribution(BaseModel):
    """Single SHAP-based feature attribution for an individual user."""
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str = Field(description="'increases_risk' or 'decreases_risk'")
    human_readable_description: str


class RelatedAccount(BaseModel):
    """An account connected to the investigated user through shared entities."""
    user_id: str
    relationship_types: list[str]
    shared_entity_count: int


class MultiHopPath(BaseModel):
    """A path through the heterogeneous graph connecting users via shared entities."""
    nodes: list[str] = Field(description="Alternating user/entity node IDs")
    node_types: list[str] = Field(description="Node type for each node in the path")
    edge_types: list[str] = Field(description="Edge type for each hop")
    hop_count: int
    path_strength: float = Field(
        description="Aggregate relationship strength along the path"
    )


class CommunityContext(BaseModel):
    """Louvain community context for the investigated user."""
    community_id: int
    community_size: int
    community_density: float
    user_degree: int
    neighbor_count: int


class BehavioralSummary(BaseModel):
    """Deterministic behavioural summary derived from baseline features."""
    statements: list[str] = Field(
        description="Human-readable statements about observed behaviour"
    )
    feature_values: dict[str, float] = Field(
        description="Raw baseline feature values backing the statements"
    )


class CounterfactualResult(BaseModel):
    """Result of modifying a single feature and re-scoring with the frozen model."""
    feature: str
    original_value: float
    counterfactual_value: float
    original_probability: float
    counterfactual_probability: float
    probability_delta: float


class InvestigationResult(BaseModel):
    """Complete investigation report for a single user.

    This is the top-level output of ``Investigator.investigate_user()``.
    No field contains ground-truth labels.
    """
    user_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_model_factors: list[FeatureAttribution]
    shared_devices: list[str]
    shared_ips: list[str]
    shared_addresses: list[str]
    shared_payment_instruments: list[str]
    related_accounts: list[RelatedAccount]
    community_context: CommunityContext | None = None
    multi_hop_paths: list[MultiHopPath] = Field(default_factory=list)
    behavioral_summary: BehavioralSummary | None = None
    counterfactuals: list[CounterfactualResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Human-readable feature descriptions
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Graph: direct connection
    "degree": "Number of entities (devices, IPs, etc.) directly connected to this account",
    "weighted_degree": "Total transaction-weighted connections to entities",
    "unique_devices": "Number of distinct devices used for transactions",
    "unique_ips": "Number of distinct IP addresses used for transactions",
    "unique_addresses": "Number of distinct addresses associated with transactions",
    "unique_payment_instruments": "Number of distinct payment instruments used",
    "unique_merchants": "Number of distinct merchants transacted with",
    "unique_promotions": "Number of distinct promotions used",
    # Graph: shared-entity
    "number_of_accounts_sharing_devices": "Number of other accounts connected through shared devices",
    "number_of_accounts_sharing_ips": "Number of other accounts connected through shared IP addresses",
    "number_of_accounts_sharing_addresses": "Number of other accounts connected through shared addresses",
    "number_of_accounts_sharing_payment_instruments": "Number of other accounts sharing payment instruments",
    "number_of_accounts_sharing_promotions": "Number of other accounts sharing promotions",
    # Graph: neighbor
    "number_of_user_neighbors": "Number of directly connected user accounts in the graph",
    "number_of_high_degree_neighbors": "Number of highly-connected neighbouring accounts",
    "average_neighbor_degree": "Average connectivity of neighbouring accounts",
    "maximum_neighbor_degree": "Maximum connectivity among neighbouring accounts",
    # Graph: two-hop
    "two_hop_user_count": "Number of accounts reachable within two hops through shared entities",
    "two_hop_unique_entity_count": "Number of distinct shared entities across two-hop neighbours",
    "two_hop_shared_device_count": "Number of shared devices across two-hop neighbours",
    "two_hop_shared_ip_count": "Number of shared IP addresses across two-hop neighbours",
    # Graph: centrality
    "degree_centrality": "How central this account is in the user network (degree-based)",
    "clustering_coefficient": "How tightly clustered this account's neighbours are",
    "betweenness_centrality": "How often this account lies on shortest paths between others",
    # Graph: community
    "community_size": "Size of the graph community this account belongs to",
    "community_density": "Edge density within the account's graph community",
    "community_edge_count": "Number of edges within the account's graph community",
    # Baseline: static
    "transaction_count": "Total number of transactions made",
    "average_transaction_amount": "Average transaction amount",
    "median_transaction_amount": "Median transaction amount",
    "transaction_amount_std": "Standard deviation of transaction amounts",
    "promo_usage_rate": "Fraction of transactions that used a promotion",
    "unique_merchant_count": "Number of distinct merchants transacted with (baseline)",
    "unique_promo_count": "Number of distinct promotions used (baseline)",
    "unique_payment_instrument_count": "Number of distinct payment instruments (baseline)",
    "unique_device_count": "Number of distinct devices used (baseline)",
    "unique_ip_count": "Number of distinct IP addresses used (baseline)",
    "unique_address_count": "Number of distinct addresses used (baseline)",
    "account_age_days": "Account age in days from creation to last activity",
    "transaction_frequency": "Transactions per day over the active period",
    "failed_transaction_rate": "Fraction of transactions that failed",
    "refund_rate": "Fraction of transactions that were refunded",
    "weekend_activity_rate": "Fraction of transactions on weekends",
    "night_activity_rate": "Fraction of transactions at night (22:00-06:00)",
    # Baseline: time-aware
    "recent_transaction_count_1d": "Number of transactions in the last 1 day",
    "recent_transaction_count_7d": "Number of transactions in the last 7 days",
    "recent_transaction_count_30d": "Number of transactions in the last 30 days",
    "recent_amount_sum_30d": "Total transaction amount in the last 30 days",
    "days_since_first_device_seen": "Days since the earliest device was first seen",
    "recent_unique_entity_count_30d": "Distinct entities used in the last 30 days",
}


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "GROUND_TRUTH_FIELDS",
    "BehavioralSummary",
    "CommunityContext",
    "CounterfactualResult",
    "FeatureAttribution",
    "InvestigationResult",
    "MultiHopPath",
    "RelatedAccount",
    "RiskLevel",
]
