"""Graph ablation experiments.

Systematically evaluates the contribution of graph feature groups by
training XGBoost models with incremental feature additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Feature group definitions (exact column names from Phase 2 feature pipeline)
FEATURE_GROUPS: dict[str, list[str]] = {
    # Baseline only (23 features)
    "baseline": [
        "transaction_count", "average_transaction_amount", "median_transaction_amount",
        "transaction_amount_std", "promo_usage_rate", "unique_merchant_count",
        "unique_promo_count", "unique_payment_instrument_count", "unique_device_count",
        "unique_ip_count", "unique_address_count", "account_age_days",
        "transaction_frequency", "failed_transaction_rate", "refund_rate",
        "weekend_activity_rate", "night_activity_rate",
        "recent_transaction_count_1d", "recent_transaction_count_7d",
        "recent_transaction_count_30d", "recent_amount_sum_30d",
        "recent_unique_entity_count_30d", "days_since_first_device_seen",
    ],
    # Graph feature groups (27 total, from Phase 2)
    "direct": [
        "degree", "weighted_degree", "unique_devices", "unique_ips",
        "unique_addresses", "unique_payment_instruments", "unique_merchants",
        "unique_promotions",
    ],
    "shared_entity": [
        "number_of_accounts_sharing_devices", "number_of_accounts_sharing_ips",
        "number_of_accounts_sharing_addresses",
        "number_of_accounts_sharing_payment_instruments",
        "number_of_accounts_sharing_promotions",
    ],
    "neighbor": [
        "number_of_user_neighbors", "number_of_high_degree_neighbors",
        "average_neighbor_degree", "maximum_neighbor_degree",
    ],
    "two_hop": [
        "two_hop_user_count", "two_hop_unique_entity_count",
        "two_hop_shared_device_count", "two_hop_shared_ip_count",
    ],
    "centrality": [
        "degree_centrality", "clustering_coefficient", "betweenness_centrality",
    ],
    "community": [
        "community_size", "community_density", "community_edge_count",
    ],
}


@dataclass
class AblationResult:
    """Results for one ablation experiment."""
    experiment: str
    feature_groups: list[str]
    feature_count: int
    validation_metrics: dict
    test_metrics: dict
    training_time_seconds: float
    selected_threshold: float


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    feature_groups: list[str]  # e.g., ["direct"], ["direct", "shared_entity"], etc.


# Pre-defined ablation experiments
ABLATION_EXPERIMENTS: list[AblationConfig] = [
    AblationConfig(name="baseline_only", feature_groups=[]),
    AblationConfig(name="baseline_plus_direct", feature_groups=["direct"]),
    AblationConfig(name="baseline_plus_direct_shared", feature_groups=["direct", "shared_entity"]),
    AblationConfig(name="baseline_plus_neighbor_twohop", feature_groups=["neighbor", "two_hop"]),
    AblationConfig(name="baseline_plus_centrality_community", feature_groups=["centrality", "community"]),
    AblationConfig(name="baseline_plus_all_graph", feature_groups=[
        "direct", "shared_entity", "neighbor", "two_hop", "centrality", "community"
    ]),
]


def get_experiment_feature_columns(groups: list[str]) -> list[str]:
    """Get the feature column names for an experiment (baseline + specified groups)."""
    cols = list(FEATURE_GROUPS["baseline"])
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {g}")
        cols.extend(FEATURE_GROUPS[g])
    return list(dict.fromkeys(cols))


def get_feature_indices(
    full_feature_names: list[str],
    groups: list[str],
) -> list[int]:
    """Return indices into the full feature array for the given groups."""
    exp_cols = get_experiment_feature_columns(groups)
    name_to_idx = {name: i for i, name in enumerate(full_feature_names)}
    return [name_to_idx[c] for c in exp_cols if c in name_to_idx]


__all__ = [
    "FEATURE_GROUPS",
    "AblationResult",
    "AblationConfig",
    "ABLATION_EXPERIMENTS",
    "get_experiment_feature_columns",
    "get_feature_indices",
]