"""Leakage audit.

Two responsibilities:

1. **Feature-whitelist guard** — confirm that no ground-truth field appears in
   the set of columns a future model would be allowed to use. This is the
   mechanical guarantee behind ``reports/label_exclusion_manifest.json``.

2. **Artifact detection** — actively look for generation artifacts that would
   let a model cheat: an abuse-only city, device type, transaction status, or a
   near-perfectly separable amount/timestamp distribution. Findings are written
   to ``reports/leakage_audit.json`` with feature, reason, severity, and action.

The audit is intentionally conservative: it flags anything suspicious for human
review rather than asserting the data is leak-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.generation import GenerationConfig
from src.validation.distribution import _overlap_coefficient, build_account_features
from src.validation.result import ValidationResult

# Ground-truth fields that must never be model features.
GROUND_TRUTH_FIELDS: frozenset[str] = frozenset(
    {
        "is_abuse_account",
        "ring_id",
        "ring_type",
        "is_abuse_transaction",
        "estimated_abuse_value",
        "split",
    }
)

# A categorical value present for abuse but essentially absent for legit (or vice
# versa) beyond this dominance implies an artifact.
_CATEGORICAL_DOMINANCE = 0.98
# Amount/behaviour overlap below this at the transaction level is suspicious.
_TXN_OVERLAP_FLOOR = 0.30


def _categorical_leakage(
    df: pd.DataFrame,
    label: pd.Series,
    column: str,
    findings: list[dict],
) -> None:
    """Flag categories that are almost exclusive to one class."""
    if column not in df.columns:
        return
    ct = pd.crosstab(df[column], label)
    if ct.shape[1] < 2:
        return
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    for value, row in ct_norm.iterrows():
        # row indexed by label (False/True). Check dominance toward abuse=True.
        abuse_share = row.get(True, 0.0)
        total_for_value = ct.loc[value].sum()
        if total_for_value < 20:
            continue  # too rare to judge
        if abuse_share >= _CATEGORICAL_DOMINANCE:
            findings.append(
                {
                    "feature": column,
                    "value": str(value),
                    "possible_leakage_reason": (
                        f"value '{value}' is {abuse_share:.0%} abuse — nearly "
                        "class-exclusive"
                    ),
                    "severity": "high",
                    "action": "investigate generator; ensure value is shared with legit users",
                }
            )


def audit_leakage(
    tables: dict[str, pd.DataFrame],
    config: GenerationConfig,
    feature_whitelist: set[str] | None = None,
) -> tuple[ValidationResult, dict]:
    """Run the leakage audit.

    ``feature_whitelist`` is the set of columns a future model would be allowed
    to use. If provided, we assert it contains no ground-truth field. If not
    provided, we synthesise a plausible whitelist (all non-ground-truth columns)
    and audit that instead.

    Returns ``(result, report)``.
    """
    result = ValidationResult(module="leakage")
    findings: list[dict] = []

    users = tables["users"]
    txns = tables["transactions"]

    # --- 1. Whitelist guard ---
    if feature_whitelist is None:
        all_cols = set()
        for name, df in tables.items():
            all_cols.update(f"{name}.{c}" for c in df.columns)
        # Strip table prefix for ground-truth comparison.
        feature_whitelist = {
            c.split(".", 1)[1] for c in all_cols
        } - GROUND_TRUTH_FIELDS

    leaked_fields = feature_whitelist & GROUND_TRUTH_FIELDS
    if leaked_fields:
        result.add(
            "whitelist_guard", "error",
            f"ground-truth fields present in feature whitelist: {sorted(leaked_fields)}",
        )
        for f in sorted(leaked_fields):
            findings.append(
                {
                    "feature": f,
                    "possible_leakage_reason": "ground-truth label in feature set",
                    "severity": "critical",
                    "action": "remove from feature whitelist immediately",
                }
            )

    # --- 2. Categorical artifact detection ---
    # Merge transaction categorical context with account label.
    label_by_user = users.set_index("user_id")["is_abuse_account"]

    # 2a. Account-level city / segment.
    _categorical_leakage(users, users["is_abuse_account"], "city", findings)
    _categorical_leakage(users, users["is_abuse_account"], "user_segment", findings)
    _categorical_leakage(users, users["is_abuse_account"], "age_group", findings)

    # 2b. Transaction-level device_type / network via joins.
    txn_label = txns["user_id"].map(label_by_user).fillna(False)
    dev_type = txns.merge(
        tables["devices"][["device_id", "device_type"]], on="device_id", how="left"
    )["device_type"]
    _categorical_leakage(
        pd.DataFrame({"device_type": dev_type}), txn_label, "device_type", findings
    )
    net_type = txns.merge(
        tables["ips"][["ip_id", "network_type"]], on="ip_id", how="left"
    )["network_type"]
    _categorical_leakage(
        pd.DataFrame({"network_type": net_type}), txn_label, "network_type", findings
    )

    # 2c. Transaction status must not be class-exclusive.
    _categorical_leakage(
        pd.DataFrame({"transaction_status": txns["transaction_status"]}),
        txn_label,
        "transaction_status",
        findings,
    )

    # --- 3. Amount / timestamp separability at transaction level ---
    legit_amt = txns.loc[~txn_label.to_numpy(), "amount"].to_numpy(dtype=float)
    abuse_amt = txns.loc[txn_label.to_numpy(), "amount"].to_numpy(dtype=float)
    amt_overlap = _overlap_coefficient(legit_amt, abuse_amt)
    if amt_overlap < _TXN_OVERLAP_FLOOR:
        findings.append(
            {
                "feature": "transactions.amount",
                "possible_leakage_reason": (
                    f"legit/abuse amount overlap only {amt_overlap:.2f}"
                ),
                "severity": "medium",
                "action": "widen abuse amount distribution to overlap legit",
            }
        )
        result.add(
            "amount_separability", "warning",
            f"transaction amount overlap {amt_overlap:.2f} below floor",
        )

    # Timestamp hour-of-day separability.
    legit_hr = txns.loc[~txn_label.to_numpy(), "timestamp"].dt.hour.to_numpy(dtype=float)
    abuse_hr = txns.loc[txn_label.to_numpy(), "timestamp"].dt.hour.to_numpy(dtype=float)
    hr_overlap = _overlap_coefficient(legit_hr, abuse_hr, bins=24)
    if hr_overlap < _TXN_OVERLAP_FLOOR:
        findings.append(
            {
                "feature": "transactions.timestamp(hour)",
                "possible_leakage_reason": (
                    f"legit/abuse hour-of-day overlap only {hr_overlap:.2f}"
                ),
                "severity": "medium",
                "action": "loosen abuse temporal coordination",
            }
        )
        result.add(
            "timestamp_separability", "warning",
            f"hour-of-day overlap {hr_overlap:.2f} below floor",
        )

    # --- 4. Account-level single-feature separability (behavioural) ---
    feats = build_account_features(tables)
    for col in ("n_devices", "n_ips", "n_payments", "avg_amount", "txn_count"):
        legit_vals = feats.loc[~feats["is_abuse_account"], col].to_numpy(dtype=float)
        abuse_vals = feats.loc[feats["is_abuse_account"], col].to_numpy(dtype=float)
        ov = _overlap_coefficient(legit_vals, abuse_vals)
        if ov < _TXN_OVERLAP_FLOOR:
            findings.append(
                {
                    "feature": f"account.{col}",
                    "possible_leakage_reason": f"legit/abuse overlap only {ov:.2f}",
                    "severity": "medium",
                    "action": "adjust generator so this feature overlaps across classes",
                }
            )
            result.add(
                "account_feature_separability", "warning",
                f"account feature '{col}' overlap {ov:.2f} below floor",
            )

    report = {
        "note": (
            "Automated leakage audit. 'findings' lists suspicious features with "
            "reason, severity, and recommended action. An empty findings list "
            "does not prove the data is leak-free."
        ),
        "ground_truth_fields": sorted(GROUND_TRUTH_FIELDS),
        "amount_overlap_coefficient": round(amt_overlap, 4),
        "hour_overlap_coefficient": round(hr_overlap, 4),
        "findings": findings,
        "finding_count": len(findings),
    }

    if not findings:
        result.add(
            "leakage_audit", "info",
            "no leakage artifacts detected by automated checks",
        )

    return result, report


__all__ = ["audit_leakage", "GROUND_TRUTH_FIELDS"]
