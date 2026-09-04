"""Label consistency checks.

Validates the ground-truth labels for internal consistency (not for how
"separable" they are — that's :mod:`distribution`). Checks:

* Every abuse account has a ring_id + ring_type; every non-abuse account has
  neither.
* ring_type values are drawn from the canonical set.
* Every abuse transaction belongs to an abuse account and carries a ring_id
  matching that account's ring.
* No legitimate account carries an abuse transaction.
* All four ring types are present at the account level.
* estimated_abuse_value is zero for legit accounts and non-negative overall.
"""

from __future__ import annotations

import pandas as pd

from src.config.generation import RING_TYPES, GenerationConfig
from src.validation.result import ValidationResult


def check_labels(
    tables: dict[str, pd.DataFrame], config: GenerationConfig
) -> ValidationResult:
    """Run label-consistency checks."""
    result = ValidationResult(module="labels")
    users = tables["users"]
    txns = tables["transactions"]

    abuse = users[users["is_abuse_account"]]
    legit = users[~users["is_abuse_account"]]

    # --- Abuse accounts must have ring identity ---
    missing_ring = abuse[abuse["ring_id"] == ""]
    if len(missing_ring):
        result.add(
            "abuse_ring_identity", "error",
            f"{len(missing_ring)} abuse accounts have no ring_id",
        )
    missing_type = abuse[abuse["ring_type"] == ""]
    if len(missing_type):
        result.add(
            "abuse_ring_identity", "error",
            f"{len(missing_type)} abuse accounts have no ring_type",
        )

    # --- Legit accounts must NOT have ring identity ---
    legit_with_ring = legit[legit["ring_id"] != ""]
    if len(legit_with_ring):
        result.add(
            "legit_ring_identity", "error",
            f"{len(legit_with_ring)} legit accounts carry a ring_id",
        )

    # --- ring_type validity ---
    bad_types = set(abuse["ring_type"].unique()) - set(RING_TYPES)
    if bad_types:
        result.add(
            "ring_type_valid", "error",
            f"invalid ring_type values: {sorted(bad_types)}",
        )

    # --- All four ring types present ---
    present = set(abuse["ring_type"].unique())
    missing = set(RING_TYPES) - present
    if missing:
        result.add(
            "ring_types_present", "error",
            f"missing ring types at account level: {sorted(missing)}",
        )

    # --- Transaction-level consistency ---
    abuse_user_ids = set(abuse["user_id"])
    abuse_txns = txns[txns["is_abuse_transaction"]]

    orphan = abuse_txns[~abuse_txns["user_id"].isin(abuse_user_ids)]
    if len(orphan):
        result.add(
            "abuse_txn_owner", "error",
            f"{len(orphan)} abuse transactions belong to non-abuse accounts",
        )

    no_ring = abuse_txns[abuse_txns["ring_id"] == ""]
    if len(no_ring):
        result.add(
            "abuse_txn_ring", "error",
            f"{len(no_ring)} abuse transactions have no ring_id",
        )

    # Abuse txn ring must match the owner's ring.
    user_ring = dict(zip(users["user_id"], users["ring_id"]))
    mismatch = 0
    for _, row in abuse_txns.iterrows():
        if user_ring.get(row["user_id"], "") != row["ring_id"]:
            mismatch += 1
    if mismatch:
        result.add(
            "abuse_txn_ring_match", "error",
            f"{mismatch} abuse transactions have a ring_id != owner's ring_id",
        )

    # --- estimated_abuse_value sanity ---
    if "estimated_abuse_value" in users.columns:
        neg = int((users["estimated_abuse_value"] < 0).sum())
        if neg:
            result.add(
                "estimated_value", "error",
                f"{neg} accounts have negative estimated_abuse_value",
            )
        legit_value = legit[legit["estimated_abuse_value"] > 0]
        if len(legit_value):
            result.add(
                "estimated_value", "warning",
                f"{len(legit_value)} legit accounts have positive estimated_abuse_value",
            )

    # --- Every abuse account should have at least some abuse transactions? ---
    # Not strictly required (high-noise members may have few), so info only.
    abuse_txn_users = set(abuse_txns["user_id"])
    silent = abuse_user_ids - abuse_txn_users
    if silent:
        result.add(
            "abuse_txn_coverage", "info",
            f"{len(silent)} abuse accounts have zero coordinated abuse transactions "
            "(acceptable: high-noise members)",
        )

    return result


__all__ = ["check_labels"]
