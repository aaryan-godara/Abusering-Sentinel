"""Device pool generator.

Creates a pool of devices with a realistic type mix and a fingerprint group.
The fingerprint group is a coarse hardware/browser cluster: several devices can
share one, which is what makes naive "same fingerprint => same actor" rules
noisy. Reuse patterns (one primary user, occasional sharing) are decided later
by the behaviour/group/ring stages, not here.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import DEVICE_TYPES, GenerationConfig
from src.generators.common import make_id

# Device-type popularity (android-heavy, tablet-light).
_TYPE_WEIGHTS: dict[str, float] = {
    "android": 3.0,
    "ios": 1.6,
    "web": 1.2,
    "tablet": 0.5,
}


def generate_devices(config: GenerationConfig, rng: np.random.Generator) -> list[dict]:
    """Return ``config.num_devices`` device records."""
    weights = np.array([_TYPE_WEIGHTS[t] for t in DEVICE_TYPES])
    probs = weights / weights.sum()
    types = rng.choice(list(DEVICE_TYPES), size=config.num_devices, p=probs)

    # Fingerprint groups: roughly one group per ~6 devices, so overlaps are
    # common but far from universal.
    num_groups = max(1, config.num_devices // 6)
    fp_groups = rng.integers(0, num_groups, size=config.num_devices)

    start = np.datetime64(config.date_start)
    span_days = (config.date_end - config.date_start).days
    # Devices are "first seen" across the window, biased earlier.
    offsets = np.clip(rng.beta(1.5, 3.0, size=config.num_devices) * span_days, 0, span_days)

    devices: list[dict] = []
    for i in range(config.num_devices):
        first_seen = start + np.timedelta64(int(offsets[i]), "D")
        devices.append(
            {
                "device_id": make_id("DEV", i, width=6),
                "device_type": str(types[i]),
                "device_fingerprint_group": make_id("FPG", int(fp_groups[i]), width=5),
                "first_seen_at": np.datetime_as_string(first_seen, unit="D"),
            }
        )
    return devices


__all__ = ["generate_devices"]
