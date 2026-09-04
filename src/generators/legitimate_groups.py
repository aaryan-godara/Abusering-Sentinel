"""Legitimate group construction.

Partitions part of the user population into families, offices, colleges, and
hostels that share infrastructure the way real groups do. These groups are the
source of *hard legitimate cases*: they produce genuinely high graph
connectivity (shared IPs, addresses, occasional shared devices) without being
abuse.

Sharing rules by group type:

* family  - shared home IP + shared address; sometimes a shared device;
            separate payment instruments and independent timing.
* office  - shared office IP (+ sometimes office address); separate devices and
            payments; daytime-biased activity.
* college - large shared campus network; separate devices/payments/behaviour.
* hostel  - shared hostel network + shared accommodation address; separate
            devices/payments.

No ground-truth abuse label is ever set here.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import GenerationConfig
from src.generators.common import GroupSpec, ResourcePool, UserProfile, make_id
from src.generators.legitimate_behavior import (
    assign_behaviour,
    assign_personal_infrastructure,
)

# Typical size ranges per group type.
_GROUP_SIZE: dict[str, tuple[int, int]] = {
    "family": (2, 5),
    "office": (5, 40),
    "college": (20, 80),
    "hostel": (10, 40),
}

_GROUP_SEGMENT: dict[str, str] = {
    "family": "family_member",
    "office": "office_employee",
    "college": "student",
    "hostel": "student",
}


def _build_group(
    group_type: str,
    members: list[UserProfile],
    pools: dict[str, ResourcePool],
    rng: np.random.Generator,
    group_index: int,
) -> GroupSpec:
    """Wire one legitimate group: set segment, behaviour, shared + personal infra."""
    group_id = make_id(f"GRP_{group_type.upper()}", group_index, width=5)
    city = members[0].city
    spec = GroupSpec(group_id=group_id, group_type=group_type, city=city)

    # Shared infrastructure for the group.
    shared_ip = pools["ip"].fresh()
    spec.shared_ip_ids = [shared_ip]
    shared_address: str | None = None
    shared_device: str | None = None

    if group_type in ("family", "hostel", "shared_accommodation"):
        shared_address = pools["address"].fresh()
        spec.shared_address_ids = [shared_address]
    if group_type == "office":
        # ~50% of offices also share a registered office address.
        if rng.random() < 0.5:
            shared_address = pools["address"].fresh()
            spec.shared_address_ids = [shared_address]
    if group_type == "college":
        # Large campuses often have multiple subnets.
        extra = pools["ip"].fresh_block(int(rng.integers(0, 3)))
        spec.shared_ip_ids.extend(extra)

    # Families occasionally share one device across the household.
    if group_type == "family" and rng.random() < 0.4:
        shared_device = pools["device"].fresh()
        spec.shared_device_ids = [shared_device]

    for user in members:
        user.segment = _GROUP_SEGMENT[group_type]
        user.city = city  # group co-locates
        user.group_id = group_id
        user.group_type = group_type
        assign_behaviour(user, rng)
        assign_personal_infrastructure(user, pools, rng)

        # Layer shared infra onto personal infra.
        user.ip_ids.extend(spec.shared_ip_ids)
        if shared_address is not None:
            # Family/hostel members mostly live at the shared address; offices
            # add it alongside personal ones.
            if group_type in ("family", "hostel"):
                user.address_ids = [shared_address]
            else:
                user.address_ids.append(shared_address)
        if shared_device is not None and rng.random() < 0.6:
            user.device_ids.append(shared_device)

        spec.user_ids.append(user.user_id)

    return spec


def build_legitimate_groups(
    config: GenerationConfig,
    legit_users: list[UserProfile],
    pools: dict[str, ResourcePool],
    rng: np.random.Generator,
) -> list[GroupSpec]:
    """Assign legitimate users to groups or leave them as individuals.

    ``legit_users`` are consumed in order. Users not placed in a group become
    independent individuals with purely personal infrastructure.
    """
    n = len(legit_users)
    targets = {
        "family": int(config.family_user_fraction * n),
        "office": int(config.office_user_fraction * n),
        "college": int(config.college_user_fraction * n),
        "hostel": int(config.hostel_user_fraction * n),
    }

    groups: list[GroupSpec] = []
    cursor = 0
    group_counters: dict[str, int] = {k: 0 for k in _GROUP_SIZE}

    for group_type, target in targets.items():
        assigned = 0
        lo, hi = _GROUP_SIZE[group_type]
        while assigned < target and cursor < n:
            size = int(rng.integers(lo, hi + 1))
            size = min(size, target - assigned, n - cursor)
            if size < 2:
                break
            members = legit_users[cursor : cursor + size]
            cursor += size
            assigned += size
            spec = _build_group(
                group_type, members, pools, rng, group_counters[group_type]
            )
            group_counters[group_type] += 1
            groups.append(spec)

    # Remaining users are independent individuals.
    for user in legit_users[cursor:]:
        user.segment = "individual" if rng.random() < 0.8 else "small_business_user"
        user.group_type = "individual"
        assign_behaviour(user, rng)
        assign_personal_infrastructure(user, pools, rng)

    return groups


__all__ = ["build_legitimate_groups"]
