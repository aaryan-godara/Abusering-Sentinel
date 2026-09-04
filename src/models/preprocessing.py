"""Preprocessing, feature selection, and label audit for Phase 3 models.

Loads the Phase 2 feature datasets, performs label-leakage audits, and
prepares clean (X, y) matrices for Model A / B / C.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.pipeline import BASELINE_FEATURES, GRAPH_FEATURE_COLUMNS, LABEL_COLUMN
from src.validation.loader import load_tables

# Ground-truth / label-derived column names that must never be features.
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "is_abuse_account",
        "is_abuse_transaction",
        "ring_id",
        "ring_type",
        "estimated_abuse_value",
        "split",
    }
)

# Substrings that would suggest a feature is label-derived (token-based).
SUSPICIOUS_TOKENS: frozenset[str] = frozenset(
    {"abuse", "fraud", "ring", "label", "bad", "malicious", "scam"}
)


def load_feature_split(
    split: str, variant: str = "graph", data_dir: Path | None = None
) -> pd.DataFrame:
    """Load a feature dataset for one split and variant.

    Parameters
    ----------
    split: one of "train", "val", "test"
    variant: "baseline" or "graph"
    """
    data_dir = data_dir or (PROJECT_ROOT / "data" / "processed")
    path = data_dir / f"features_{variant}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return pd.read_parquet(path)


def audit_features(df: pd.DataFrame, variant: str, split: str) -> list[str]:
    """Run the Phase 3 label-leakage audit on a feature DataFrame.

    Returns a list of error messages (empty if clean).
    """
    errors: list[str] = []

    # Carried columns (allowed, not features).
    allowed_carry = {"user_id", "label"}

    feature_cols = [c for c in df.columns if c not in allowed_carry]

    # 1. Exact forbidden fields present?
    forbidden_hits = set(feature_cols) & FORBIDDEN_FEATURES
    if forbidden_hits:
        errors.append(
            f"[{variant}/{split}] explicit label columns in features: {sorted(forbidden_hits)}"
        )

    # 2. Suspicious name tokens?
    import re
    for col in feature_cols:
        tokens = set(re.split(r"[_\W]+", col.lower()))
        hit = tokens & SUSPICIOUS_TOKENS
        if hit:
            errors.append(
                f"[{variant}/{split}] feature '{col}' name token(s) {sorted(hit)} suggest label derivation"
            )

    # 3. Sanity: target must be binary 0/1.
    if "label" in df.columns:
        uniq = df["label"].dropna().unique()
        if not set(uniq).issubset({0, 1}):
            errors.append(f"[{variant}/{split}] label not binary: {uniq}")

    return errors


def get_feature_matrix(
    df: pd.DataFrame, feature_names: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract X and y from a feature DataFrame.

    Parameters
    ----------
    df: feature DataFrame (with user_id and label columns).
    feature_names: optional explicit feature column list. If None, uses all
        columns except user_id and label.

    Returns
    -------
    X: shape (n_samples, n_features), float64
    y: shape (n_samples,), int (0/1)
    """
    if feature_names is None:
        feature_names = [c for c in df.columns if c not in ("user_id", "label")]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


def get_feature_lists(variant: str) -> list[str]:
    """Return the feature column names for a given variant (excludes user_id, label)."""
    if variant == "baseline":
        return list(BASELINE_FEATURES)
    if variant == "graph":
        return list(GRAPH_FEATURE_COLUMNS) + list(BASELINE_FEATURES)
    raise ValueError(f"unknown variant: {variant}")


def verify_split_identities(
    data_dir: Path | None = None
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Verify that train/val/test user identities match the Phase 1 splits.

    Returns (split_user_ids, split_labels) maps.
    """
    tables = load_tables(data_dir)
    users = tables["users"]
    splits = {}
    labels = {}
    for s in ("train", "val", "test"):
        subset = users[users["split"] == s]
        splits[s] = set(subset["user_id"])
        labels[s] = dict(zip(subset["user_id"], subset["is_abuse_account"].astype(int)))
    return splits, labels


def load_split_data(
    split: str, variant: str = "graph", data_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convenience: load (X, y, feature_names) for a split/variant."""
    df = load_feature_split(split, variant, data_dir)
    errs = audit_features(df, variant, split)
    if errs:
        raise ValueError(f"Label-leakage audit failed for {variant}/{split}:\n" + "\n".join(errs))
    feat_names = get_feature_lists(variant)
    X, y = get_feature_matrix(df, feat_names)
    return X, y, feat_names


__all__ = [
    "FORBIDDEN_FEATURES",
    "SUSPICIOUS_TOKENS",
    "load_feature_split",
    "audit_features",
    "get_feature_matrix",
    "get_feature_lists",
    "verify_split_identities",
    "load_split_data",
]