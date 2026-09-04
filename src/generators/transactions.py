"""Transaction generator.

Produces the transaction table from finalized user profiles. Two transaction
flavours are emitted:

* **Baseline activity** for every user (legit and abuse alike): drawn from the
  user's own behavioural parameters, personal/shared infrastructure, preferred
  merchant categories, and skewed amounts/timestamps.
* **Coordinated abuse activity** for ring members: a fraction of a ring's
  transactions cluster in time (loose window, never exact), lean on the ring's
  shared promotion, target shared merchant categories, and cluster amounts —
  but always mixed with the member's own noise.

Only coordinated abuse transactions get ``is_abuse_transaction = True`` and the
``ring_id``. Baseline transactions by abuse accounts are labelled legitimate at
the transaction level (an abuse account still does normal things).

The per-transaction ``amount`` and ``timestamp`` distributions are deliberately
*not* separable by label on their own — see the leakage audit.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np

from src.config.generation import (
    CATEGORY_AMOUNT_RANGES,
    GenerationConfig,
    TRANSACTION_STATUSES,
)
from src.generators.common import (
    RingSpec,
    UserProfile,
    coordinated_timestamp,
    lognormal_amount,
    make_id,
    skewed_counts,
)


def _pick(rng: np.random.Generator, options: list[str], fallback: str) -> str:
    """Choose from a list, or return ``fallback`` if empty."""
    if not options:
        return fallback
    return str(rng.choice(options))


def _status(rng: np.random.Generator, config: GenerationConfig) -> str:
    """Draw a transaction status with configured failure/refund rates.

    Status is independent of abuse to avoid leakage (see leakage audit).
    """
    r = rng.random()
    if r < config.failed_txn_rate:
        return "failed"
    if r < config.failed_txn_rate + config.refunded_txn_rate:
        return "refunded"
    return "success"


def generate_transactions(
    config: GenerationConfig,
    users: list[UserProfile],
    merchants: list[dict],
    promotions: list[dict],
    rings: list[RingSpec],
    rng: np.random.Generator,
) -> tuple[list[dict], dict[str, float]]:
    """Generate transactions for all users.

    Returns ``(transactions, ring_loss)`` where ``ring_loss`` maps ring_id to the
    summed amount of that ring's coordinated abuse transactions (ground-truth
    estimated loss).
    """
    start = datetime.combine(config.date_start, time.min)
    end = datetime.combine(config.date_end, time.min)

    # Merchant lookup by category for targeted selection.
    merchants_by_cat: dict[str, list[str]] = {}
    for m in merchants:
        merchants_by_cat.setdefault(m["merchant_category"], []).append(m["merchant_id"])
    all_merchant_ids = [m["merchant_id"] for m in merchants]
    merchant_category = {m["merchant_id"]: m["merchant_category"] for m in merchants}

    # Popularity-weighted promo pool for legitimate usage. A few promos are
    # genuinely popular among legit users (Zipf-like weights).
    all_promo_ids = [p["promo_id"] for p in promotions]
    promo_weights = 1.0 / (1.0 + np.arange(len(all_promo_ids)))
    rng.shuffle(promo_weights)
    promo_weights = promo_weights / promo_weights.sum()

    ring_by_id = {r.ring_id: r for r in rings}
    ring_loss: dict[str, float] = {r.ring_id: 0.0 for r in rings}

    # Per-user transaction counts: skewed, scaled to hit the global target.
    activity = np.array([u.activity_level for u in users])
    raw_counts = skewed_counts(rng, len(users), config.mean_txns_per_user, floor=1)
    weighted = raw_counts * activity
    scale = config.num_transactions / max(weighted.sum(), 1.0)
    counts = np.maximum(1, np.round(weighted * scale).astype(int))

    transactions: list[dict] = []
    txn_index = 0

    # Precompute one coordination anchor per ring (loose shared timing).
    ring_anchor: dict[str, datetime] = {}
    span_seconds = int((end - start).total_seconds())
    for r in rings:
        ring_anchor[r.ring_id] = start + timedelta(
            seconds=int(rng.integers(0, span_seconds))
        )

    for user, n_txns in zip(users, counts):
        n_txns = int(n_txns)
        # How many of this user's txns are coordinated abuse?
        n_abuse = 0
        ring: RingSpec | None = None
        if user.is_abuse_account and user.ring_id in ring_by_id:
            ring = ring_by_id[user.ring_id]
            n_abuse = int(round(n_txns * ring.abuse_txn_fraction))

        # Available promos for this user: shared ring promo (if any) biases the draw.
        for j in range(n_txns):
            is_abuse_txn = j < n_abuse and ring is not None

            # --- Merchant / category ---
            if is_abuse_txn and ring.target_merchant_categories:
                cat = _pick(rng, ring.target_merchant_categories, "food")
            elif user.merchant_category_prefs and rng.random() < 0.75:
                cat = _pick(rng, user.merchant_category_prefs, "food")
            else:
                cat = merchant_category[
                    all_merchant_ids[int(rng.integers(0, len(all_merchant_ids)))]
                ]
            cand = merchants_by_cat.get(cat, all_merchant_ids)
            merchant_id = _pick(rng, cand, all_merchant_ids[0])

            # --- Amount ---
            low, high = CATEGORY_AMOUNT_RANGES.get(cat, (100, 5_000))
            if is_abuse_txn:
                # Coordinated amounts cluster: draw around a ring-level center
                # with tight sigma, but still within the category's realm.
                center = (low * high) ** 0.5
                amount = round(
                    float(
                        rng.lognormal(
                            mean=np.log(center),
                            sigma=ring.amount_coordination_sigma,
                        )
                    ),
                    2,
                )
            else:
                amount = lognormal_amount(rng, low, high)
            # Rare outliers for everyone.
            if rng.random() < 0.01:
                amount = round(amount * float(rng.uniform(3, 10)), 2)

            # --- Timestamp ---
            if is_abuse_txn:
                ts = coordinated_timestamp(
                    rng,
                    ring_anchor[ring.ring_id],
                    ring.coordination_window_hours,
                    start,
                    end,
                )
            else:
                # Personal timing with optional business-hours bias.
                sec = int(rng.integers(0, span_seconds))
                ts = start + timedelta(seconds=sec)
                if (
                    user.business_hours_bias > 0
                    and rng.random() < user.business_hours_bias
                ):
                    ts = ts.replace(hour=int(rng.integers(9, 19)))

            # --- Infrastructure ---
            device_id = _pick(rng, user.device_ids, "DEV_000000")
            ip_id = _pick(rng, user.ip_ids, "IP_000000")
            address_id = _pick(rng, user.address_ids, "ADR_000000")
            payment_id = _pick(rng, user.payment_instrument_ids, "PAY_000000")

            # --- Promo usage ---
            promo_id = ""
            if is_abuse_txn and ring.shared_promo_ids and rng.random() < 0.8:
                promo_id = _pick(rng, ring.shared_promo_ids, "")
            elif rng.random() < user.promo_affinity:
                # Legit promo usage: popularity-weighted, so some promos are
                # genuinely popular across the legitimate population.
                promo_id = str(rng.choice(all_promo_ids, p=promo_weights))

            status = _status(rng, config)

            txn = {
                "transaction_id": make_id("TXN", txn_index, width=9),
                "user_id": user.user_id,
                "timestamp": ts.replace(microsecond=0).isoformat(sep=" "),
                "amount": amount,
                "merchant_id": merchant_id,
                "payment_instrument_id": payment_id,
                "device_id": device_id,
                "ip_id": ip_id,
                "address_id": address_id,
                "promo_id": promo_id,
                "transaction_status": status,
                "is_abuse_transaction": bool(is_abuse_txn),
                "ring_id": ring.ring_id if is_abuse_txn else "",
            }
            transactions.append(txn)
            txn_index += 1

            if is_abuse_txn and status != "failed":
                ring_loss[ring.ring_id] += amount

    return transactions, ring_loss


__all__ = ["generate_transactions"]
