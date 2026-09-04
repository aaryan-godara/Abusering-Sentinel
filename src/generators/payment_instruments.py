"""Payment-instrument pool generator.

Instruments are opaque IDs with a type (card, upi, wallet) and an issuer group
(a synthetic bank/issuer bucket). No real card numbers, UPI handles, or bank
details are ever generated — only opaque identifiers.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import INSTRUMENT_TYPES, GenerationConfig
from src.generators.common import make_id

_TYPE_WEIGHTS: dict[str, float] = {
    "upi": 3.0,
    "card": 2.0,
    "wallet": 1.2,
}


def generate_payment_instruments(
    config: GenerationConfig, rng: np.random.Generator
) -> list[dict]:
    """Return ``config.num_payment_instruments`` payment-instrument records."""
    weights = np.array([_TYPE_WEIGHTS[t] for t in INSTRUMENT_TYPES])
    probs = weights / weights.sum()
    types = rng.choice(
        list(INSTRUMENT_TYPES), size=config.num_payment_instruments, p=probs
    )

    # Issuer groups: a modest number of synthetic issuers.
    num_issuers = max(1, config.num_payment_instruments // 40)
    issuers = rng.integers(0, num_issuers, size=config.num_payment_instruments)

    instruments: list[dict] = []
    for i in range(config.num_payment_instruments):
        instruments.append(
            {
                "payment_instrument_id": make_id("PAY", i, width=6),
                "instrument_type": str(types[i]),
                "issuer_group": make_id("ISS", int(issuers[i]), width=4),
            }
        )
    return instruments


__all__ = ["generate_payment_instruments"]
