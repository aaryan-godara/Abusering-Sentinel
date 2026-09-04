"""Leakage-safe, group-aware dataset splitting.

The unit of splitting is the **group**, never the individual transaction or user.
A group is either an abuse ring or a legitimate shared group (family, office,
college, hostel). Independent individuals are singleton groups.

Hard rules:

* No abuse ring may appear in more than one split (prevents a model from
  memorising a ring in train and being tested on the same ring).
* Legitimate shared groups are also kept whole, so shared-infrastructure signals
  don't straddle splits.
* Hidden-scenario rings and the legit high-connectivity groups are pinned to the
  **test** split.

Assignment is deterministic given the seed. We greedily bin groups to hit the
target user-count fractions while honouring the pins.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config.generation import GenerationConfig
from src.generators.common import UserProfile


def assign_splits(
    config: GenerationConfig,
    users: list[UserProfile],
    forced_test_group_ids: set[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Assign each user a split label; return a ``group_id -> split`` map.

    Every user with the same ``group_id`` lands in the same split. Users without
    a group id are treated as their own singleton group.
    """
    # Collect groups and their member counts.
    group_members: dict[str, list[UserProfile]] = defaultdict(list)
    for u in users:
        gid = u.group_id if u.group_id is not None else f"__solo__{u.user_id}"
        group_members[gid].append(u)

    group_ids = list(group_members.keys())
    group_sizes = {gid: len(members) for gid, members in group_members.items()}
    total_users = len(users)

    # Pinned-to-test groups.
    pinned_test = {gid for gid in group_ids if gid in forced_test_group_ids}

    # Shuffle the remaining groups deterministically.
    free_groups = [gid for gid in group_ids if gid not in pinned_test]
    order = rng.permutation(len(free_groups))
    free_groups = [free_groups[i] for i in order]

    targets = {
        "train": config.train_fraction * total_users,
        "val": config.val_fraction * total_users,
        "test": config.test_fraction * total_users,
    }
    assigned = {"train": 0, "val": 0, "test": 0}

    group_split: dict[str, str] = {}

    # Seed the test split with pinned groups first.
    for gid in pinned_test:
        group_split[gid] = "test"
        assigned["test"] += group_sizes[gid]

    # Greedily assign free groups to whichever split is furthest below target.
    for gid in free_groups:
        size = group_sizes[gid]
        deficits = {s: (targets[s] - assigned[s]) / max(targets[s], 1) for s in targets}
        split = max(deficits, key=deficits.get)
        group_split[gid] = split
        assigned[split] += size

    # Write back onto users.
    for gid, members in group_members.items():
        split = group_split[gid]
        for u in members:
            u.split = split

    return group_split


__all__ = ["assign_splits"]
