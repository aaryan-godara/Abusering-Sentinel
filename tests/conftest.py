"""Shared pytest fixtures for Phase 1 tests.

A single small dataset is generated once per test session (it is deterministic
and fast at the small scale) and shared across tests via the ``small_dataset``
and ``small_tables`` fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.generation import GenerationConfig
from src.generators.pipeline import GeneratedDataset, generate_dataset


@pytest.fixture(scope="session")
def small_config() -> GenerationConfig:
    """A small, fast configuration retaining all structural features."""
    return GenerationConfig().small()


@pytest.fixture(scope="session")
def small_dataset(small_config: GenerationConfig) -> GeneratedDataset:
    """Generate the small dataset once for the whole session."""
    return generate_dataset(small_config)


@pytest.fixture(scope="session")
def small_tables(small_dataset: GeneratedDataset) -> dict[str, pd.DataFrame]:
    """The generated tables as a name -> DataFrame map, typed like the loader.

    Mirrors :func:`src.validation.loader.load_tables` dtype normalisation so the
    validators behave identically to running on loaded CSVs.
    """
    tables = small_dataset.table_map()
    tables = {k: v.copy() for k, v in tables.items()}
    users = tables["users"]
    users["is_abuse_account"] = users["is_abuse_account"].astype(bool)
    users["ring_id"] = users["ring_id"].fillna("").astype(str)
    users["ring_type"] = users["ring_type"].fillna("").astype(str)
    txns = tables["transactions"]
    txns["is_abuse_transaction"] = txns["is_abuse_transaction"].astype(bool)
    txns["ring_id"] = txns["ring_id"].fillna("").astype(str)
    txns["promo_id"] = txns["promo_id"].fillna("").astype(str)
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], errors="coerce")
    return tables
