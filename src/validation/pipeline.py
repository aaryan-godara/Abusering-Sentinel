"""Validation + reporting pipeline and CLI.

Loads the generated dataset, runs every validation module, computes the data
quality summary and distribution report, runs the leakage audit, and writes:

* ``reports/dataset_summary.json``
* ``reports/distribution_report.json``
* ``reports/leakage_audit.json``
* ``reports/validation_report.json`` (aggregate of all module results)

Run with::

    python -m src.validation.pipeline               # validate data/processed
    python -m src.validation.pipeline --data DIR     # validate a custom dir

Exit code is non-zero if any error-severity issue is found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.config.generation import RING_TYPES, GenerationConfig
from src.validation.distribution import analyze_distributions
from src.validation.integrity import check_integrity
from src.validation.labels import check_labels
from src.validation.leakage import audit_leakage
from src.validation.loader import default_data_dir, load_tables
from src.validation.result import ValidationResult
from src.validation.splits import check_splits


def _load_config_from_manifest() -> GenerationConfig:
    """Rebuild the generation config from the dataset manifest, if present."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            return GenerationConfig(**manifest["configuration"])
        except Exception:
            pass
    return GenerationConfig()


def build_dataset_summary(
    tables: dict[str, pd.DataFrame], config: GenerationConfig
) -> dict:
    """Compute the data quality summary (reports/dataset_summary.json)."""
    users = tables["users"]
    txns = tables["transactions"]
    abuse = users[users["is_abuse_account"]]

    ring_sizes = abuse.groupby("ring_id").size()
    ring_type_by_ring = abuse.groupby("ring_id")["ring_type"].first()
    ring_count_by_type = {rt: 0 for rt in RING_TYPES}
    for rt in ring_type_by_ring:
        ring_count_by_type[rt] = ring_count_by_type.get(rt, 0) + 1

    # Shared-infrastructure statistics (users per device/ip/address).
    users_per_device = txns.groupby("device_id")["user_id"].nunique()
    users_per_ip = txns.groupby("ip_id")["user_id"].nunique()
    users_per_address = txns.groupby("address_id")["user_id"].nunique()

    promo_txns = txns[txns["promo_id"] != ""]

    summary = {
        "synthetic_data": True,
        "counts": {
            "users": int(len(users)),
            "transactions": int(len(txns)),
            "devices": int(len(tables["devices"])),
            "ips": int(len(tables["ips"])),
            "addresses": int(len(tables["addresses"])),
            "payment_instruments": int(len(tables["payment_instruments"])),
            "merchants": int(len(tables["merchants"])),
            "promotions": int(len(tables["promotions"])),
        },
        "labels": {
            "legitimate_accounts": int((~users["is_abuse_account"]).sum()),
            "abuse_accounts": int(users["is_abuse_account"].sum()),
            "abuse_rate": round(float(users["is_abuse_account"].mean()), 4),
            "abuse_transactions": int(txns["is_abuse_transaction"].sum()),
            "abuse_transaction_rate": round(
                float(txns["is_abuse_transaction"].mean()), 4
            ),
        },
        "rings": {
            "total": int(abuse["ring_id"].nunique()),
            "count_by_type": ring_count_by_type,
            "size_stats": {
                "min": int(ring_sizes.min()) if len(ring_sizes) else 0,
                "max": int(ring_sizes.max()) if len(ring_sizes) else 0,
                "mean": round(float(ring_sizes.mean()), 2) if len(ring_sizes) else 0,
                "median": float(ring_sizes.median()) if len(ring_sizes) else 0,
            },
            "estimated_abuse_value_total": round(
                float(users["estimated_abuse_value"].sum()), 2
            ),
        },
        "transaction_amounts": {
            "mean": round(float(txns["amount"].mean()), 2),
            "median": round(float(txns["amount"].median()), 2),
            "p90": round(float(txns["amount"].quantile(0.90)), 2),
            "p99": round(float(txns["amount"].quantile(0.99)), 2),
            "max": round(float(txns["amount"].max()), 2),
        },
        "promotions_usage": {
            "transactions_with_promo": int(len(promo_txns)),
            "promo_usage_rate": round(float((txns["promo_id"] != "").mean()), 4),
            "distinct_promos_used": int(promo_txns["promo_id"].nunique()),
        },
        "shared_infrastructure": {
            "max_users_per_device": int(users_per_device.max()),
            "devices_shared_by_multiple_users": int((users_per_device > 1).sum()),
            "max_users_per_ip": int(users_per_ip.max()),
            "ips_shared_by_multiple_users": int((users_per_ip > 1).sum()),
            "max_users_per_address": int(users_per_address.max()),
            "addresses_shared_by_multiple_users": int((users_per_address > 1).sum()),
        },
        "status_breakdown": {
            k: int(v) for k, v in txns["transaction_status"].value_counts().items()
        },
        "splits": {
            k: int(v) for k, v in users["split"].value_counts().items()
        },
    }
    return summary


def run_validation(data_dir: Path | None = None) -> tuple[bool, dict]:
    """Run all validators + reports. Returns ``(ok, aggregate_report)``.

    Also writes the four report files under ``reports/``.
    """
    data_dir = data_dir or default_data_dir()
    config = _load_config_from_manifest()
    tables = load_tables(data_dir)

    results: list[ValidationResult] = [
        check_integrity(tables, config),
        check_labels(tables, config),
        check_splits(tables, config),
    ]
    dist_result, dist_report = analyze_distributions(tables, config)
    results.append(dist_result)
    leak_result, leak_report = audit_leakage(tables, config)
    results.append(leak_result)

    summary = build_dataset_summary(tables, config)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (reports_dir / "distribution_report.json").write_text(
        json.dumps(dist_report, indent=2), encoding="utf-8"
    )
    (reports_dir / "leakage_audit.json").write_text(
        json.dumps(leak_report, indent=2), encoding="utf-8"
    )

    aggregate = {
        "ok": all(r.ok for r in results),
        "total_errors": sum(len(r.errors) for r in results),
        "total_warnings": sum(len(r.warnings) for r in results),
        "modules": [r.to_dict() for r in results],
    }
    (reports_dir / "validation_report.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    return aggregate["ok"], aggregate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the AbuseRing Sentinel synthetic dataset."
    )
    parser.add_argument("--data", type=str, default=None, help="Data directory to validate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data) if args.data else None

    print("AbuseRing Sentinel - dataset validation")
    print("=" * 60)
    ok, aggregate = run_validation(data_dir)

    for module in aggregate["modules"]:
        status = "OK" if module["ok"] else "FAIL"
        print(
            f"  [{status:>4}] {module['module']:<14} "
            f"errors={module['error_count']} warnings={module['warning_count']}"
        )
        for issue in module["issues"]:
            if issue["severity"] in ("error", "warning"):
                print(f"         - {issue['severity'].upper()}: {issue['message']}")

    print("=" * 60)
    print(
        f"Total errors: {aggregate['total_errors']} | "
        f"warnings: {aggregate['total_warnings']}"
    )
    print("Reports written to reports/.")
    if ok:
        print("VALIDATION PASSED (no error-severity issues).")
        return 0
    print("VALIDATION FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_validation", "build_dataset_summary", "main"]
