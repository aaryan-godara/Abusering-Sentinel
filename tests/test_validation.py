"""Tests for the validation subsystem and leakage guards."""

from __future__ import annotations

import pandas as pd

from src.config.generation import GenerationConfig
from src.generators.pipeline import LABEL_FIELDS
from src.validation.distribution import analyze_distributions, build_account_features
from src.validation.integrity import check_integrity
from src.validation.labels import check_labels
from src.validation.leakage import GROUND_TRUTH_FIELDS, audit_leakage
from src.validation.splits import check_splits


def test_integrity_passes_on_generated_data(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """A freshly generated dataset has no integrity errors."""
    result = check_integrity(small_tables, small_config)
    assert result.ok, [i.message for i in result.errors]


def test_labels_consistent(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """Ground-truth labels are internally consistent."""
    result = check_labels(small_tables, small_config)
    assert result.ok, [i.message for i in result.errors]


def test_ring_isolation_across_splits(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """No abuse ring spans more than one split."""
    result = check_splits(small_tables, small_config)
    isolation_errors = [
        i for i in result.errors if i.check == "ring_isolation"
    ]
    assert not isolation_errors, [i.message for i in isolation_errors]
    assert result.ok, [i.message for i in result.errors]


def test_ring_isolation_explicit(small_tables: dict[str, pd.DataFrame]) -> None:
    """Directly assert each ring lives in exactly one split."""
    users = small_tables["users"]
    abuse = users[users["is_abuse_account"]]
    spans = abuse.groupby("ring_id")["split"].nunique()
    assert (spans == 1).all(), spans[spans > 1].to_dict()


def test_integrity_detects_broken_foreign_key(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """Corrupting a foreign key is caught as an integrity error."""
    tables = {k: v.copy() for k, v in small_tables.items()}
    tables["transactions"].loc[
        tables["transactions"].index[0], "user_id"
    ] = "USR_DOES_NOT_EXIST"
    result = check_integrity(tables, small_config)
    assert not result.ok
    assert any(i.check == "foreign_key" for i in result.errors)


def test_labels_detect_legit_with_ring(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """A legit account carrying a ring_id is flagged."""
    tables = {k: v.copy() for k, v in small_tables.items()}
    users = tables["users"]
    legit_idx = users[~users["is_abuse_account"]].index[0]
    users.loc[legit_idx, "ring_id"] = "RING_9999"
    result = check_labels(tables, small_config)
    assert not result.ok
    assert any(i.check == "legit_ring_identity" for i in result.errors)


def test_no_ground_truth_field_in_whitelist(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """A clean feature whitelist passes the leakage guard."""
    whitelist = {
        "amount", "merchant_category", "device_type", "network_type",
        "instrument_type", "city", "age_group", "user_segment",
    }
    result, report = audit_leakage(small_tables, small_config, whitelist)
    guard_errors = [i for i in result.errors if i.check == "whitelist_guard"]
    assert not guard_errors


def test_leakage_guard_flags_ground_truth_field(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """A whitelist containing a ground-truth field is rejected."""
    bad_whitelist = {"amount", "is_abuse_account"}
    result, report = audit_leakage(small_tables, small_config, bad_whitelist)
    assert not result.ok
    assert any(i.check == "whitelist_guard" for i in result.errors)


def test_label_fields_match_ground_truth_constant() -> None:
    """The pipeline's LABEL_FIELDS align with the leakage module's set."""
    assert set(LABEL_FIELDS) == set(GROUND_TRUTH_FIELDS)


def test_amounts_not_trivially_separable(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """Transaction amounts overlap substantially between classes."""
    _, report = audit_leakage(small_tables, small_config)
    assert report["amount_overlap_coefficient"] >= 0.3


def test_distribution_report_has_all_features(
    small_tables: dict[str, pd.DataFrame], small_config: GenerationConfig
) -> None:
    """The distribution report covers the expected behavioural features."""
    _, report = analyze_distributions(small_tables, small_config)
    for col in (
        "txn_count", "avg_amount", "n_devices", "n_ips", "n_payments",
        "n_merchant_categories", "promo_rate", "active_days_span",
    ):
        assert col in report["features"]


def test_account_features_are_not_ground_truth(
    small_tables: dict[str, pd.DataFrame],
) -> None:
    """Behavioural feature builder emits no ground-truth field except the label col."""
    feats = build_account_features(small_tables)
    forbidden = GROUND_TRUTH_FIELDS - {"is_abuse_account"}
    assert not (set(feats.columns) & forbidden)
