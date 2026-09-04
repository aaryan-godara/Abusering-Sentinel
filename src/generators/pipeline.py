"""End-to-end synthetic data generation pipeline + CLI.

Orchestrates every stage in a fixed, deterministic order:

1. Seed one RNG from the configuration.
2. Generate entity pools (merchants, promotions, devices, IPs, addresses,
   payment instruments).
3. Generate bare user profiles.
4. Partition users into abuse accounts vs legitimate; carve out hidden-scenario
   allocations.
5. Build legitimate groups + individuals (personal + shared infrastructure).
6. Build the four abuse-ring archetypes + hidden scenarios.
7. Generate transactions (baseline + coordinated abuse).
8. Assign leakage-safe, group-aware splits.
9. Write CSVs, the dataset manifest, and the split/label-exclusion reports.

Run with::

    python -m src.generators.pipeline                # default (full-scale) config
    python -m src.generators.pipeline --small         # fast small dataset
    python -m src.generators.pipeline --seed 7        # override seed
    python -m src.generators.pipeline --out data/processed
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.config.generation import RING_TYPES, GenerationConfig
from src.generators.abuse_rings import build_abuse_rings, build_hidden_scenarios
from src.generators.addresses import generate_addresses
from src.generators.common import RingSpec, ResourcePool, UserProfile
from src.generators.devices import generate_devices
from src.generators.ips import generate_ips
from src.generators.legitimate_groups import build_legitimate_groups
from src.generators.merchants import generate_merchants
from src.generators.payment_instruments import generate_payment_instruments
from src.generators.promotions import generate_promotions
from src.generators.transactions import generate_transactions
from src.generators.users import generate_users
from src.splitting.group_split import assign_splits

# Fields that are ground truth and must never become model features.
LABEL_FIELDS: tuple[str, ...] = (
    "is_abuse_account",
    "ring_id",
    "ring_type",
    "is_abuse_transaction",
    "estimated_abuse_value",
    "split",
)


@dataclass
class GeneratedDataset:
    """Container for all generated tables (as DataFrames) + specs."""

    users: pd.DataFrame
    devices: pd.DataFrame
    ips: pd.DataFrame
    addresses: pd.DataFrame
    payment_instruments: pd.DataFrame
    promotions: pd.DataFrame
    merchants: pd.DataFrame
    transactions: pd.DataFrame
    rings: list[RingSpec]
    config: GenerationConfig

    def table_map(self) -> dict[str, pd.DataFrame]:
        return {
            "users": self.users,
            "devices": self.devices,
            "ips": self.ips,
            "addresses": self.addresses,
            "payment_instruments": self.payment_instruments,
            "promotions": self.promotions,
            "merchants": self.merchants,
            "transactions": self.transactions,
        }


def _split_population(
    config: GenerationConfig, users: list[UserProfile], rng: np.random.Generator
) -> tuple[list[UserProfile], list[UserProfile], dict[str, list[UserProfile]]]:
    """Partition users into (legit, abuse-base, hidden-scenario allocations).

    Hidden scenarios are carved from the tail of the population first so they are
    disjoint from the base rings and legitimate groups.
    """
    n = len(users)
    n_abuse_target = int(round(n * config.abuse_account_fraction))

    # Reserve users for hidden scenarios.
    scenario_needs: dict[str, int] = {}
    for sc in config.hidden_scenarios:
        scenario_needs[sc.name] = sc.count * sc.size
    total_scenario = sum(scenario_needs.values())

    scenario_users: dict[str, list[UserProfile]] = {}
    cursor = n
    for sc in config.hidden_scenarios:
        need = scenario_needs[sc.name]
        scenario_users[sc.name] = users[cursor - need : cursor]
        cursor -= need

    remaining = users[:cursor]
    # Abuse-base users (excluding legit-connectivity scenarios which are legit).
    abuse_scenario_users = sum(
        sc.count * sc.size
        for sc in config.hidden_scenarios
        if sc.kind != "legit_high_connectivity"
    )
    n_base_abuse = max(0, n_abuse_target - abuse_scenario_users)
    # Guarantee enough base abuse users to build one ring of every archetype
    # (the first N rings are forced to distinct types). Without this floor,
    # small configs can starve some ring types.
    min_forced = sum(rt.min_size for rt in config.ring_types.values())
    n_base_abuse = max(n_base_abuse, min_forced)
    n_base_abuse = min(n_base_abuse, len(remaining) - 1)

    abuse_base = remaining[:n_base_abuse]
    legit = remaining[n_base_abuse:]
    return legit, abuse_base, scenario_users


def generate_dataset(config: GenerationConfig) -> GeneratedDataset:
    """Run the full generation pipeline in-memory and return all tables."""
    rng = np.random.default_rng(config.random_seed)

    # --- Entity pools ---
    merchants = generate_merchants(config, rng)
    promotions = generate_promotions(config, rng)
    devices = generate_devices(config, rng)
    ips = generate_ips(config, rng)
    addresses = generate_addresses(config, rng)
    payments = generate_payment_instruments(config, rng)

    pools = {
        "device": ResourcePool([d["device_id"] for d in devices], rng),
        "ip": ResourcePool([i["ip_id"] for i in ips], rng),
        "address": ResourcePool([a["address_id"] for a in addresses], rng),
        "payment": ResourcePool([p["payment_instrument_id"] for p in payments], rng),
    }
    promo_ids = [p["promo_id"] for p in promotions]

    # --- Users ---
    users = generate_users(config, rng)
    legit, abuse_base, scenario_users = _split_population(config, users, rng)

    # --- Legitimate groups + individuals ---
    build_legitimate_groups(config, legit, pools, rng)

    # --- Abuse rings ---
    rings = build_abuse_rings(config, abuse_base, pools, promo_ids, rng)

    # --- Hidden scenarios ---
    hidden_rings, forced_test_groups = build_hidden_scenarios(
        config, scenario_users, pools, promo_ids, rng
    )
    rings.extend(hidden_rings)

    # Hidden abuse rings are pinned to test as well.
    forced_test = set(forced_test_groups) | {r.ring_id for r in hidden_rings}

    # --- Transactions ---
    txns, ring_loss = generate_transactions(
        config, users, merchants, promotions, rings, rng
    )
    for r in rings:
        r.ground_truth_loss = round(ring_loss.get(r.ring_id, 0.0), 2)

    # --- Splits ---
    assign_splits(config, users, forced_test, rng)

    # --- Assemble DataFrames ---
    est_loss_by_user: dict[str, float] = {}
    ring_size = {r.ring_id: len(r.user_ids) for r in rings}
    for r in rings:
        per_member = r.ground_truth_loss / max(len(r.user_ids), 1)
        for uid in r.user_ids:
            est_loss_by_user[uid] = round(per_member, 2)

    users_df = pd.DataFrame(
        [
            {
                "user_id": u.user_id,
                "account_created_at": u.account_created_at.date().isoformat(),
                "user_segment": u.segment,
                "city": u.city,
                "age_group": u.age_group,
                "is_abuse_account": u.is_abuse_account,
                "ring_id": u.ring_id or "",
                "ring_type": u.ring_type or "",
                "group_id": u.group_id or "",
                "group_type": u.group_type or "",
                "estimated_abuse_value": est_loss_by_user.get(u.user_id, 0.0),
                "split": u.split or "train",
            }
            for u in users
        ]
    )

    dataset = GeneratedDataset(
        users=users_df,
        devices=pd.DataFrame(devices),
        ips=pd.DataFrame(ips),
        addresses=pd.DataFrame(addresses),
        payment_instruments=pd.DataFrame(payments),
        promotions=pd.DataFrame(promotions),
        merchants=pd.DataFrame(merchants),
        transactions=pd.DataFrame(txns),
        rings=rings,
        config=config,
    )
    return dataset


def _write_csvs(dataset: GeneratedDataset, out_dir: Path) -> dict[str, int]:
    """Write all tables to ``out_dir`` and return row counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, df in dataset.table_map().items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        counts[name] = len(df)
    return counts


def _write_manifest(
    dataset: GeneratedDataset, out_dir: Path, counts: dict[str, int]
) -> Path:
    """Write ``data/manifests/dataset_manifest.json``."""
    manifests_dir = PROJECT_ROOT / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": dataset.config.random_seed,
        "dataset_version": dataset.config.dataset_version,
        "configuration": json.loads(dataset.config.model_dump_json()),
        "row_counts": counts,
        "columns": {
            name: list(df.columns) for name, df in dataset.table_map().items()
        },
        "output_directory": str(out_dir),
        "synthetic_data": True,
    }
    path = manifests_dir / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def _write_reports(dataset: GeneratedDataset) -> list[Path]:
    """Write the split manifest and label-exclusion manifest reports."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    users = dataset.users
    txns = dataset.transactions

    # --- split_manifest.json ---
    ring_split: dict[str, set[str]] = {}
    split_manifest = {}
    for split in ("train", "val", "test"):
        split_users = users[users["split"] == split]
        user_ids = set(split_users["user_id"])
        split_txns = txns[txns["user_id"].isin(user_ids)]
        ring_ids = sorted(
            r for r in split_users["ring_id"].unique() if r
        )
        ring_types = sorted(
            t for t in split_users[split_users["is_abuse_account"]]["ring_type"].unique() if t
        )
        for r in ring_ids:
            ring_split.setdefault(r, set()).add(split)
        split_manifest[split] = {
            "users": int(len(split_users)),
            "transactions": int(len(split_txns)),
            "abuse_accounts": int(split_users["is_abuse_account"].sum()),
            "rings": len(ring_ids),
            "ring_types": ring_types,
        }
    # Ring isolation check embedded in the manifest.
    leaked = {r: sorted(s) for r, s in ring_split.items() if len(s) > 1}
    split_manifest["ring_isolation_ok"] = len(leaked) == 0
    split_manifest["rings_spanning_multiple_splits"] = leaked

    path = reports_dir / "split_manifest.json"
    path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    written.append(path)

    # --- label_exclusion_manifest.json ---
    exclusion = {
        "description": (
            "Ground-truth fields that must NEVER be used as model features. "
            "These reveal or trivially encode the abuse label."
        ),
        "excluded_fields": list(LABEL_FIELDS),
        "excluded_by_table": {
            "users": [
                "is_abuse_account",
                "ring_id",
                "ring_type",
                "estimated_abuse_value",
                "split",
            ],
            "transactions": ["is_abuse_transaction", "ring_id"],
        },
        "rationale": {
            "is_abuse_account": "Account-level ground-truth label.",
            "ring_id": "Directly identifies the abuse ring membership.",
            "ring_type": "Reveals abuse archetype.",
            "is_abuse_transaction": "Transaction-level ground-truth label.",
            "estimated_abuse_value": "Derived from ground-truth abuse activity.",
            "split": "Data-management field, not a behavioural signal.",
        },
    }
    path = reports_dir / "label_exclusion_manifest.json"
    path.write_text(json.dumps(exclusion, indent=2), encoding="utf-8")
    written.append(path)

    return written


def run(config: GenerationConfig, out_dir: Path | None = None) -> GeneratedDataset:
    """Generate, write CSVs, manifest, and reports. Return the dataset."""
    out_dir = out_dir or (PROJECT_ROOT / "data" / "processed")
    dataset = generate_dataset(config)
    counts = _write_csvs(dataset, out_dir)
    _write_manifest(dataset, out_dir, counts)
    _write_reports(dataset)
    return dataset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the AbuseRing Sentinel synthetic dataset.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--small", action="store_true", help="Use the small test-scale config.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for CSVs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = GenerationConfig()
    if args.small:
        config = config.small()
    if args.seed is not None:
        config = config.model_copy(update={"random_seed": args.seed})

    out_dir = Path(args.out) if args.out else None
    print("AbuseRing Sentinel - synthetic data generation")
    print("=" * 60)
    print(f"Seed: {config.random_seed} | version: {config.dataset_version}")
    print(f"Target users: {config.num_users} | transactions: {config.num_transactions}")

    dataset = run(config, out_dir)

    n_abuse = int(dataset.users["is_abuse_account"].sum())
    n_users = len(dataset.users)
    print("\nGenerated:")
    for name, df in dataset.table_map().items():
        print(f"  {name:<20}: {len(df):>8,} rows")
    print(f"\nAbuse accounts: {n_abuse:,} / {n_users:,} "
          f"({100 * n_abuse / n_users:.2f}%)")
    print(f"Abuse rings: {len(dataset.rings)}")
    ring_type_counts = {rt: 0 for rt in RING_TYPES}
    for r in dataset.rings:
        ring_type_counts[r.ring_type] = ring_type_counts.get(r.ring_type, 0) + 1
    for rt, c in ring_type_counts.items():
        print(f"  {rt:<24}: {c}")
    print("\nDone. CSVs in data/processed/, manifest in data/manifests/, "
          "reports in reports/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GeneratedDataset", "generate_dataset", "run", "main", "LABEL_FIELDS"]
