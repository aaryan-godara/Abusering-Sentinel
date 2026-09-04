"""Tests for deterministic generation and dataset structure."""

from __future__ import annotations

import pandas as pd

from src.config.generation import RING_TYPES, GenerationConfig
from src.generators.pipeline import GeneratedDataset, generate_dataset


def test_generation_is_deterministic() -> None:
    """Same config + seed produces byte-identical tables."""
    cfg = GenerationConfig().small()
    a = generate_dataset(cfg)
    b = generate_dataset(cfg)
    for name in a.table_map():
        pd.testing.assert_frame_equal(
            a.table_map()[name], b.table_map()[name], check_like=False
        )


def test_different_seed_changes_data() -> None:
    """A different seed yields a different transaction table."""
    a = generate_dataset(GenerationConfig().small())
    b = generate_dataset(GenerationConfig().small().model_copy(update={"random_seed": 999}))
    assert not a.transactions.equals(b.transactions)


def test_expected_tables_and_columns(small_dataset: GeneratedDataset) -> None:
    """All required tables exist with their required columns."""
    tables = small_dataset.table_map()
    expected = {
        "users",
        "devices",
        "ips",
        "addresses",
        "payment_instruments",
        "promotions",
        "merchants",
        "transactions",
    }
    assert set(tables) == expected

    assert {
        "user_id", "account_created_at", "user_segment", "city", "age_group",
        "is_abuse_account", "ring_id", "ring_type", "estimated_abuse_value", "split",
    }.issubset(small_dataset.users.columns)

    assert {
        "transaction_id", "user_id", "timestamp", "amount", "merchant_id",
        "payment_instrument_id", "device_id", "ip_id", "address_id", "promo_id",
        "transaction_status", "is_abuse_transaction", "ring_id",
    }.issubset(small_dataset.transactions.columns)


def test_unique_primary_keys(small_dataset: GeneratedDataset) -> None:
    """Primary keys are unique in every table."""
    keys = {
        "users": "user_id",
        "devices": "device_id",
        "ips": "ip_id",
        "addresses": "address_id",
        "payment_instruments": "payment_instrument_id",
        "promotions": "promo_id",
        "merchants": "merchant_id",
        "transactions": "transaction_id",
    }
    for table, key in keys.items():
        df = small_dataset.table_map()[table]
        assert df[key].is_unique, f"{table}.{key} not unique"


def test_all_four_ring_types_present(small_dataset: GeneratedDataset) -> None:
    """The four abuse-ring archetypes all appear."""
    types = {r.ring_type for r in small_dataset.rings}
    assert set(RING_TYPES).issubset(types)


def test_abuse_rate_is_realistic(small_dataset: GeneratedDataset) -> None:
    """Abuse prevalence stays a minority class (roughly the configured target)."""
    rate = small_dataset.users["is_abuse_account"].mean()
    assert 0.02 <= rate <= 0.25


def test_hidden_scenarios_exist_and_are_test_only(
    small_dataset: GeneratedDataset,
) -> None:
    """Hidden abuse rings exist and land exclusively in the test split."""
    hidden = [r for r in small_dataset.rings if r.scenario is not None]
    assert len(hidden) >= 4  # low_overlap, high_noise, indirect (x2 each)

    users = small_dataset.users
    hidden_ring_ids = {r.ring_id for r in hidden}
    hidden_members = users[users["ring_id"].isin(hidden_ring_ids)]
    assert len(hidden_members) > 0
    assert set(hidden_members["split"].unique()) == {"test"}


def test_legitimate_shared_infrastructure_exists(
    small_tables: dict[str, pd.DataFrame],
) -> None:
    """Legitimate users genuinely share IPs, addresses, and sometimes devices."""
    users = small_tables["users"]
    txns = small_tables["transactions"]
    legit_ids = set(users[~users["is_abuse_account"]]["user_id"])
    tl = txns[txns["user_id"].isin(legit_ids)]

    ip_share = tl.groupby("ip_id")["user_id"].nunique()
    adr_share = tl.groupby("address_id")["user_id"].nunique()
    dev_share = tl.groupby("device_id")["user_id"].nunique()

    assert (ip_share > 1).any(), "no legit shared IPs"
    assert (adr_share > 1).any(), "no legit shared addresses"
    assert (dev_share > 1).any(), "no legit shared devices"


def test_abuse_accounts_have_noise(small_tables: dict[str, pd.DataFrame]) -> None:
    """Abuse accounts also have non-abuse (legitimate-looking) transactions."""
    users = small_tables["users"]
    txns = small_tables["transactions"]
    abuse_ids = set(users[users["is_abuse_account"]]["user_id"])
    abuse_txns = txns[txns["user_id"].isin(abuse_ids)]
    # Some of an abuse account's transactions must be non-abuse noise.
    assert (~abuse_txns["is_abuse_transaction"]).sum() > 0
