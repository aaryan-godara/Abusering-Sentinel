"""Baseline (non-graph) behavioural feature builder.

Produces one row per USER using ONLY that user's own transaction/account
behaviour. No graph structure, no cross-user information, and no ground-truth
label is read. This is the honest non-graph baseline that Phase 3 will compare
against the graph-augmented feature set.

Time-aware features are computed "as of decision time" = the user's most recent
transaction timestamp, using only transactions at or before that time. Because
decision time is the user's own latest activity, this is inherently
future-safe at the account level; the recency windows are what carry temporal
signal. See the feature manifest for the static-vs-time-aware split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Windows (days) for recency features.
RECENT_WINDOWS_DAYS: tuple[int, ...] = (1, 7, 30)

# Static behavioural features (order fixed for stable columns).
STATIC_BASELINE_FEATURES: tuple[str, ...] = (
    "transaction_count",
    "average_transaction_amount",
    "median_transaction_amount",
    "transaction_amount_std",
    "promo_usage_rate",
    "unique_merchant_count",
    "unique_promo_count",
    "unique_payment_instrument_count",
    "unique_device_count",
    "unique_ip_count",
    "unique_address_count",
    "account_age_days",
    "transaction_frequency",
    "failed_transaction_rate",
    "refund_rate",
    "weekend_activity_rate",
    "night_activity_rate",
)

TIME_AWARE_BASELINE_FEATURES: tuple[str, ...] = (
    "recent_transaction_count_1d",
    "recent_transaction_count_7d",
    "recent_transaction_count_30d",
    "recent_amount_sum_30d",
    "days_since_first_device_seen",
    "recent_unique_entity_count_30d",
)

BASELINE_FEATURES: tuple[str, ...] = (
    STATIC_BASELINE_FEATURES + TIME_AWARE_BASELINE_FEATURES
)


def build_baseline_features(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-user baseline feature table.

    Parameters
    ----------
    users:
        Users table (only ``user_id`` and ``account_created_at`` are read).
    transactions:
        Transaction rows for the users of interest (no label columns are read).
    devices:
        Devices table for ``first_seen_at`` (used by a time-aware feature).
    """
    txns = transactions.copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], errors="coerce")
    txns["promo_id"] = txns["promo_id"].fillna("").astype(str)

    created = pd.to_datetime(users.set_index("user_id")["account_created_at"])
    device_first_seen = pd.to_datetime(
        devices.set_index("device_id")["first_seen_at"]
    )

    # Precompute vectorised helper columns to avoid per-group Python lambdas.
    txns["has_promo"] = (txns["promo_id"] != "").astype(int)
    txns["is_failed"] = (txns["transaction_status"] == "failed").astype(int)
    txns["is_refunded"] = (txns["transaction_status"] == "refunded").astype(int)
    txns["is_weekend"] = (txns["timestamp"].dt.weekday >= 5).astype(int)
    hour = txns["timestamp"].dt.hour
    txns["is_night"] = ((hour < 6) | (hour >= 22)).astype(int)
    txns["promo_or_nan"] = txns["promo_id"].where(txns["promo_id"] != "")

    grp = txns.groupby("user_id")
    feats = pd.DataFrame(index=pd.Index(sorted(users["user_id"]), name="user_id"))

    # --- Static behavioural aggregates (vectorised) ---
    feats["transaction_count"] = grp.size()
    feats["average_transaction_amount"] = grp["amount"].mean()
    feats["median_transaction_amount"] = grp["amount"].median()
    feats["transaction_amount_std"] = grp["amount"].std()
    feats["promo_usage_rate"] = grp["has_promo"].mean()
    feats["unique_merchant_count"] = grp["merchant_id"].nunique()
    feats["unique_promo_count"] = grp["promo_or_nan"].nunique()
    feats["unique_payment_instrument_count"] = grp["payment_instrument_id"].nunique()
    feats["unique_device_count"] = grp["device_id"].nunique()
    feats["unique_ip_count"] = grp["ip_id"].nunique()
    feats["unique_address_count"] = grp["address_id"].nunique()

    last_txn = grp["timestamp"].max()
    first_txn = grp["timestamp"].min()
    feats["account_age_days"] = (
        last_txn - feats.index.map(created)
    ).dt.total_seconds() / 86400.0
    active_span_days = (last_txn - first_txn).dt.total_seconds() / 86400.0
    feats["transaction_frequency"] = feats["transaction_count"] / active_span_days.clip(
        lower=1.0
    )

    feats["failed_transaction_rate"] = grp["is_failed"].mean()
    feats["refund_rate"] = grp["is_refunded"].mean()
    feats["weekend_activity_rate"] = grp["is_weekend"].mean()
    feats["night_activity_rate"] = grp["is_night"].mean()

    # --- Time-aware features (as of each user's decision time = last txn) ---
    recent = _recent_features(txns, last_txn, device_first_seen)
    feats = feats.join(recent)

    feats = feats.fillna(0.0)
    return feats.reset_index()


def _recent_features(
    txns: pd.DataFrame,
    decision_time: pd.Series,
    device_first_seen: pd.Series,
) -> pd.DataFrame:
    """Recency features referenced to each user's decision time.

    Only transactions with ``timestamp <= decision_time[user]`` are used, so no
    future information leaks. Since decision time is the user's own latest
    transaction, this equals all their transactions; the windows carry the
    temporal signal. Implemented with vectorised group operations (no per-user
    Python loop) so it scales to the full dataset quickly.
    """
    entity_cols = ["device_id", "ip_id", "address_id", "payment_instrument_id"]

    df = txns[["user_id", "timestamp", "amount", *entity_cols]].copy()
    df["decision_time"] = df["user_id"].map(decision_time)
    # Guard against any row after decision time (none expected).
    df = df[df["timestamp"] <= df["decision_time"]]
    df["age_days"] = (df["decision_time"] - df["timestamp"]).dt.total_seconds() / 86400.0

    out = pd.DataFrame(index=pd.Index(sorted(decision_time.index), name="user_id"))

    for w in RECENT_WINDOWS_DAYS:
        in_window = df[df["age_days"] <= w]
        out[f"recent_transaction_count_{w}d"] = (
            in_window.groupby("user_id").size()
        )
        if w == 30:
            out["recent_amount_sum_30d"] = (
                in_window.groupby("user_id")["amount"].sum().round(2)
            )
            # Unique entities across the four entity columns in the 30d window.
            melted = in_window.melt(
                id_vars="user_id", value_vars=entity_cols, value_name="entity"
            )
            melted = melted[(melted["entity"].notna()) & (melted["entity"] != "")]
            out["recent_unique_entity_count_30d"] = (
                melted.groupby("user_id")["entity"].nunique()
            )

    # days_since_first_device_seen: earliest device first_seen vs decision time.
    dev = df[["user_id", "device_id", "decision_time"]].copy()
    dev = dev[(dev["device_id"].notna()) & (dev["device_id"] != "")]
    dev["first_seen"] = dev["device_id"].map(device_first_seen)
    dev = dev.dropna(subset=["first_seen"])
    earliest = dev.groupby("user_id").agg(
        first_seen=("first_seen", "min"), decision_time=("decision_time", "first")
    )
    out["days_since_first_device_seen"] = (
        (earliest["decision_time"] - earliest["first_seen"]).dt.total_seconds()
        / 86400.0
    ).clip(lower=0.0)

    return out.fillna(0.0)


__all__ = [
    "build_baseline_features",
    "BASELINE_FEATURES",
    "STATIC_BASELINE_FEATURES",
    "TIME_AWARE_BASELINE_FEATURES",
    "RECENT_WINDOWS_DAYS",
]
