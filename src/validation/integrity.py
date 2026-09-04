"""Structural integrity checks.

Verifies the mechanical correctness of the dataset: unique primary keys, no
unexpected nulls, valid foreign keys, valid enum values, sane timestamps, and
positive transaction amounts. These are hard requirements; any failure is an
error-severity issue.
"""

from __future__ import annotations

import pandas as pd

from src.config.generation import (
    ADDRESS_TYPES,
    DEVICE_TYPES,
    INSTRUMENT_TYPES,
    MERCHANT_CATEGORIES,
    NETWORK_TYPES,
    TRANSACTION_STATUSES,
    USER_SEGMENTS,
    GenerationConfig,
)
from src.validation.result import ValidationResult

# (table, primary-key column)
_PRIMARY_KEYS: tuple[tuple[str, str], ...] = (
    ("users", "user_id"),
    ("devices", "device_id"),
    ("ips", "ip_id"),
    ("addresses", "address_id"),
    ("payment_instruments", "payment_instrument_id"),
    ("promotions", "promo_id"),
    ("merchants", "merchant_id"),
    ("transactions", "transaction_id"),
)

# (child table, fk column, parent table, parent key, allow_blank)
_FOREIGN_KEYS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("transactions", "user_id", "users", "user_id", False),
    ("transactions", "merchant_id", "merchants", "merchant_id", False),
    ("transactions", "payment_instrument_id", "payment_instruments", "payment_instrument_id", False),
    ("transactions", "device_id", "devices", "device_id", False),
    ("transactions", "ip_id", "ips", "ip_id", False),
    ("transactions", "address_id", "addresses", "address_id", False),
    ("transactions", "promo_id", "promotions", "promo_id", True),
)

# (table, column, allowed values)
_ENUMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("users", "user_segment", USER_SEGMENTS),
    ("devices", "device_type", DEVICE_TYPES),
    ("ips", "network_type", NETWORK_TYPES),
    ("addresses", "address_type", ADDRESS_TYPES),
    ("payment_instruments", "instrument_type", INSTRUMENT_TYPES),
    ("merchants", "merchant_category", MERCHANT_CATEGORIES),
    ("transactions", "transaction_status", TRANSACTION_STATUSES),
)


def check_integrity(
    tables: dict[str, pd.DataFrame], config: GenerationConfig
) -> ValidationResult:
    """Run all structural integrity checks."""
    result = ValidationResult(module="integrity")

    # --- Unique primary keys, no nulls in keys ---
    for table, key in _PRIMARY_KEYS:
        df = tables[table]
        if key not in df.columns:
            result.add("primary_key", "error", f"{table} missing key column {key}")
            continue
        n_dupes = int(df[key].duplicated().sum())
        if n_dupes:
            result.add(
                "primary_key", "error",
                f"{table}.{key} has {n_dupes} duplicate values", table=table,
            )
        n_null = int(df[key].isna().sum())
        if n_null:
            result.add(
                "primary_key", "error",
                f"{table}.{key} has {n_null} null values", table=table,
            )

    # --- Foreign keys ---
    for child, fk, parent, pk, allow_blank in _FOREIGN_KEYS:
        cdf, pdf = tables[child], tables[parent]
        valid = set(pdf[pk])
        col = cdf[fk].fillna("")
        mask = ~col.isin(valid)
        if allow_blank:
            mask &= col != ""
        n_bad = int(mask.sum())
        if n_bad:
            result.add(
                "foreign_key", "error",
                f"{child}.{fk} has {n_bad} values not in {parent}.{pk}",
                child=child, fk=fk, examples=col[mask].unique()[:5].tolist(),
            )

    # --- Nulls in required columns ---
    required_non_null = {
        "users": ["user_segment", "city", "age_group", "account_created_at"],
        "transactions": [
            "user_id", "timestamp", "amount", "merchant_id",
            "payment_instrument_id", "device_id", "ip_id", "address_id",
            "transaction_status",
        ],
    }
    for table, cols in required_non_null.items():
        df = tables[table]
        for col in cols:
            n_null = int(df[col].isna().sum())
            if n_null:
                result.add(
                    "null_check", "error",
                    f"{table}.{col} has {n_null} nulls", table=table, column=col,
                )

    # --- Enum validity ---
    for table, col, allowed in _ENUMS:
        df = tables[table]
        bad = set(df[col].dropna().unique()) - set(allowed)
        if bad:
            result.add(
                "enum_check", "error",
                f"{table}.{col} has invalid values: {sorted(bad)}",
                table=table, column=col,
            )

    # --- Transaction amounts positive ---
    txns = tables["transactions"]
    n_nonpos = int((txns["amount"] <= 0).sum())
    if n_nonpos:
        result.add(
            "amount_check", "error",
            f"transactions has {n_nonpos} non-positive amounts",
        )

    # --- Timestamps within span ---
    ts = txns["timestamp"]
    if ts.isna().any():
        result.add(
            "timestamp_check", "error",
            f"{int(ts.isna().sum())} transactions have unparseable timestamps",
        )
    start = pd.Timestamp(config.date_start)
    end = pd.Timestamp(config.date_end)
    out_of_range = int(((ts < start) | (ts >= end)).sum())
    if out_of_range:
        result.add(
            "timestamp_check", "warning",
            f"{out_of_range} transactions fall outside [{start.date()}, {end.date()})",
        )

    return result


__all__ = ["check_integrity"]
