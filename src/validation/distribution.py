"""Distribution analysis: legitimate vs abuse populations.

Computes per-account behavioural aggregates and compares the legitimate and
abuse populations across the dimensions the brief asks for. Writes
``reports/distribution_report.json``.

Philosophy (brief sections 13, 25): the data must NOT be trivially separable. We
quantify separability per feature with a simple, model-free statistic — the
overlap coefficient of the two histograms — and flag any single feature whose
distributions barely overlap, because that would indicate a generation artifact
handing a future model an unrealistic freebie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.generation import GenerationConfig
from src.validation.result import ValidationResult

# If a single behavioural feature has histogram overlap below this, we warn:
# it means legit and abuse populations are almost disjoint on that feature.
_MIN_ACCEPTABLE_OVERLAP = 0.30


def build_account_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate transactions to per-account behavioural features.

    None of these are ground-truth fields; they are ordinary behavioural
    aggregates a future feature pipeline might legitimately compute.
    """
    users = tables["users"]
    txns = tables["transactions"].copy()

    grp = txns.groupby("user_id")
    feats = pd.DataFrame(index=users["user_id"])
    feats["txn_count"] = grp.size()
    feats["avg_amount"] = grp["amount"].mean()
    feats["total_amount"] = grp["amount"].sum()
    feats["n_devices"] = grp["device_id"].nunique()
    feats["n_ips"] = grp["ip_id"].nunique()
    feats["n_payments"] = grp["payment_instrument_id"].nunique()
    feats["n_merchant_categories"] = grp["merchant_id"].nunique()
    feats["promo_rate"] = grp["promo_id"].apply(lambda s: (s != "").mean())

    # Temporal spread in days between first and last transaction.
    span = grp["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400.0)
    feats["active_days_span"] = span

    feats = feats.fillna(0.0)
    feats["is_abuse_account"] = users.set_index("user_id")["is_abuse_account"]
    return feats.reset_index()


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Histogram overlap coefficient in [0, 1]; 1 = identical distributions."""
    if len(a) == 0 or len(b) == 0:
        return 1.0
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(ha, hb) * width))


def _summ(values: pd.Series) -> dict:
    """Summary statistics for a numeric series."""
    return {
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "std": round(float(values.std(ddof=0)), 4),
        "p10": round(float(values.quantile(0.10)), 4),
        "p90": round(float(values.quantile(0.90)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
    }


_FEATURE_COLUMNS = (
    "txn_count",
    "avg_amount",
    "total_amount",
    "n_devices",
    "n_ips",
    "n_payments",
    "n_merchant_categories",
    "promo_rate",
    "active_days_span",
)


def analyze_distributions(
    tables: dict[str, pd.DataFrame], config: GenerationConfig
) -> tuple[ValidationResult, dict]:
    """Compare legit vs abuse distributions.

    Returns ``(result, report)`` where ``report`` is the JSON-serialisable
    distribution report.
    """
    result = ValidationResult(module="distribution")
    feats = build_account_features(tables)

    legit = feats[~feats["is_abuse_account"]]
    abuse = feats[feats["is_abuse_account"]]

    report: dict = {
        "note": (
            "Model-free comparison of legitimate vs abuse accounts. Overlap "
            "coefficient near 1 means the feature alone does NOT separate the "
            "classes (desired). Low overlap on a single feature is flagged as a "
            "possible generation artifact."
        ),
        "legit_account_count": int(len(legit)),
        "abuse_account_count": int(len(abuse)),
        "features": {},
    }

    for col in _FEATURE_COLUMNS:
        overlap = _overlap_coefficient(
            legit[col].to_numpy(dtype=float), abuse[col].to_numpy(dtype=float)
        )
        report["features"][col] = {
            "legit": _summ(legit[col]),
            "abuse": _summ(abuse[col]),
            "overlap_coefficient": round(overlap, 4),
        }
        if overlap < _MIN_ACCEPTABLE_OVERLAP:
            result.add(
                "single_feature_separability", "warning",
                f"feature '{col}' has low legit/abuse overlap ({overlap:.2f}); "
                "a single feature should not separate classes this well",
                feature=col, overlap=round(overlap, 4),
            )

    result.add(
        "distribution_summary", "info",
        f"compared {len(_FEATURE_COLUMNS)} behavioural features across "
        f"{len(legit)} legit / {len(abuse)} abuse accounts",
    )

    return result, report


__all__ = ["analyze_distributions", "build_account_features"]
