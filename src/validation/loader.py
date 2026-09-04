"""Loading generated CSVs back into DataFrames for validation.

Keeps dtype handling in one place so every validator sees consistent types
(notably the boolean label columns and the parsed transaction timestamp).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

TABLE_NAMES: tuple[str, ...] = (
    "users",
    "devices",
    "ips",
    "addresses",
    "payment_instruments",
    "promotions",
    "merchants",
    "transactions",
)


def default_data_dir() -> Path:
    """Return the default processed-data directory."""
    return PROJECT_ROOT / "data" / "processed"


def load_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all dataset tables from ``data_dir`` into DataFrames."""
    data_dir = data_dir or default_data_dir()
    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        tables[name] = pd.read_csv(path)

    # Normalise types.
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


__all__ = ["TABLE_NAMES", "default_data_dir", "load_tables"]
