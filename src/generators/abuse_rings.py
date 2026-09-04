"""Abuse-ring construction (four archetypes) and hidden test scenarios.

Design principles enforced here (see the Phase 1 brief, sections 12-13):

* Rings are **noisy**: infrastructure is shared by a *fraction* of members, never
  all of them. Overlap fractions come from :class:`RingTypeConfig`.
* Every abuse account keeps personal infrastructure and legitimate-looking
  activity; only a fraction of its transactions are coordinated abuse.
* Coordination is temporal/behavioural, not a deterministic tell. No abuse
  account gets a unique city, device type, status, or amount rule.
* Type C (indirect multi-hop) is built as a **chain**: no single entity connects
  every member, so the ring only appears through multi-hop reasoning.

Ground-truth fields (``is_abuse_account``, ``ring_id``, ``ring_type``) are set
here because this is where abuse is *defined*. They are excluded from features
downstream via the label-exclusion manifest.
"""

from __future__ import annotations

import numpy as np

from src.config.generation import (
    MERCHANT_CATEGORIES,
    GenerationConfig,
    RingTypeConfig,
)
from src.generators.common import RingSpec, ResourcePool, UserProfile, make_id
from src.generators.legitimate_behavior import (
    assign_behaviour,
    assign_personal_infrastructure,
)


def _assign_shared(
    members: list[UserProfile],
    shared_ids: list[str],
    fraction: float,
    attr: str,
    rng: np.random.Generator,
) -> list[str]:
    """Attach ``shared_ids`` to a random ``fraction`` of members.

    Returns the subset of member indices that actually received sharing (unused
    by callers but handy for reasoning). The randomness is what injects noise:
    e.g. only 6 of 8 members share the IP cluster.
    """
    if not shared_ids or fraction <= 0:
        return []
    k = max(1, int(round(len(members) * fraction)))
    chosen = rng.choice(len(members), size=min(k, len(members)), replace=False)
    for idx in chosen:
        getattr(members[idx], attr).extend(shared_ids)
    return [members[i].user_id for i in chosen]


def _base_setup(
    members: list[UserProfile],
    pools: dict[str, ResourcePool],
    rng: np.random.Generator,
    promotions: list[str],
) -> None:
    """Give ring members legitimate-looking behaviour + personal infrastructure."""
    for user in members:
        # Abuse accounts still look like ordinary segments.
        user.segment = str(
            rng.choice(["individual", "student", "small_business_user"])
        )
        assign_behaviour(user, rng)
        assign_personal_infrastructure(user, pools, rng)


def _build_ring(
    ring_id: str,
    ring_type: str,
    members: list[UserProfile],
    cfg: RingTypeConfig,
    pools: dict[str, ResourcePool],
    promotions: list[str],
    rng: np.random.Generator,
    scenario: str | None = None,
    overlap_scale: float = 1.0,
) -> RingSpec:
    """Construct one ring of the given archetype.

    ``overlap_scale`` multiplies every overlap fraction (used by the low-overlap
    hidden scenario). ``scenario`` tags rings created for hidden test scenarios.
    """
    city = members[0].city
    spec = RingSpec(
        ring_id=ring_id,
        ring_type=ring_type,
        city=city,
        coordination_window_hours=cfg.coordination_window_hours,
        amount_coordination_sigma=cfg.amount_coordination_sigma,
        abuse_txn_fraction=cfg.abuse_txn_fraction,
        scenario=scenario,
    )

    _base_setup(members, pools, rng, promotions)

    # Shared target merchant categories give behavioural coordination.
    n_cats = int(rng.integers(1, 3))
    spec.target_merchant_categories = list(
        rng.choice(list(MERCHANT_CATEGORIES), size=n_cats, replace=False)
    )

    def scaled(x: float) -> float:
        return float(np.clip(x * overlap_scale, 0.0, 1.0))

    if ring_type == "indirect_multihop":
        # Chain construction: consecutive members share ONE entity, alternating
        # entity kind. No hub connects everyone.
        entity_kinds = ["device", "ip", "address"]
        for i in range(len(members) - 1):
            kind = entity_kinds[i % len(entity_kinds)]
            shared = pools[kind].fresh()
            attr = {
                "device": "device_ids",
                "ip": "ip_ids",
                "address": "address_ids",
            }[kind]
            getattr(members[i], attr).append(shared)
            getattr(members[i + 1], attr).append(shared)
            {
                "device": spec.shared_device_ids,
                "ip": spec.shared_ip_ids,
                "address": spec.shared_address_ids,
            }[kind].append(shared)
        # A loose promo affinity across the chain.
        if promotions and rng.random() < scaled(cfg.promo_overlap):
            promo = str(rng.choice(promotions))
            spec.shared_promo_ids = [promo]
    else:
        # Pooled overlap: reserve small blocks of shared infra, then attach each
        # to a random fraction of members.
        dev_block = pools["device"].fresh_block(max(1, len(members) // 3))
        ip_block = pools["ip"].fresh_block(max(1, len(members) // 3))
        adr_block = pools["address"].fresh_block(max(1, len(members) // 4))
        pay_block = pools["payment"].fresh_block(max(1, len(members) // 4))

        _assign_shared(members, dev_block, scaled(cfg.device_overlap), "device_ids", rng)
        _assign_shared(members, ip_block, scaled(cfg.ip_overlap), "ip_ids", rng)
        _assign_shared(members, adr_block, scaled(cfg.address_overlap), "address_ids", rng)
        _assign_shared(
            members, pay_block, scaled(cfg.payment_overlap), "payment_instrument_ids", rng
        )

        spec.shared_device_ids = dev_block
        spec.shared_ip_ids = ip_block
        spec.shared_address_ids = adr_block
        spec.shared_payment_ids = pay_block

        # Coordinated promotion usage (a fraction of members).
        if promotions:
            promo = str(rng.choice(promotions))
            spec.shared_promo_ids = [promo]
            k = max(1, int(round(len(members) * scaled(cfg.promo_overlap))))
            chosen = rng.choice(len(members), size=min(k, len(members)), replace=False)
            for idx in chosen:
                # Raise promo affinity so these members lean on the promo.
                members[idx].promo_affinity = float(
                    np.clip(members[idx].promo_affinity + 0.4, 0, 0.95)
                )

    # Mark ground truth on members.
    for user in members:
        user.is_abuse_account = True
        user.ring_id = ring_id
        user.ring_type = ring_type
        user.group_id = ring_id
        user.group_type = "abuse_ring"
        spec.user_ids.append(user.user_id)

    return spec


def _draw_ring_type(config: GenerationConfig, rng: np.random.Generator) -> str:
    """Pick a ring type according to configured weights."""
    names = list(config.ring_types.keys())
    weights = np.array([config.ring_types[n].weight for n in names])
    probs = weights / weights.sum()
    return str(rng.choice(names, p=probs))


def build_abuse_rings(
    config: GenerationConfig,
    abuse_users: list[UserProfile],
    pools: dict[str, ResourcePool],
    promotions: list[str],
    rng: np.random.Generator,
) -> list[RingSpec]:
    """Partition ``abuse_users`` into ``config.num_abuse_rings`` base rings.

    Ring sizes are drawn from each type's ``[min_size, max_size]`` and clipped so
    the whole abuse population is consumed. All four archetypes are guaranteed
    present (the first four rings are forced to distinct types).
    """
    rings: list[RingSpec] = []
    cursor = 0
    n_users = len(abuse_users)
    forced_types = list(config.ring_types.keys())

    # Decide the type sequence up front so we can reserve min-sizes correctly.
    type_sequence: list[str] = []
    for r in range(config.num_abuse_rings):
        if r < len(forced_types):
            type_sequence.append(forced_types[r])
        else:
            type_sequence.append(_draw_ring_type(config, rng))

    for r in range(config.num_abuse_rings):
        if cursor >= n_users:
            break
        ring_type = type_sequence[r]
        cfg = config.ring_types[ring_type]

        remaining_users = n_users - cursor
        # Reserve at least the min_size of every subsequent ring.
        reserved = sum(
            config.ring_types[type_sequence[k]].min_size
            for k in range(r + 1, config.num_abuse_rings)
        )
        max_allowed = remaining_users - reserved
        # Draw a size within the type's band, capped by what's available.
        hi = min(cfg.max_size, max(cfg.min_size, max_allowed))
        size = int(rng.integers(cfg.min_size, hi + 1))
        size = min(size, remaining_users)
        size = max(size, 2)

        members = abuse_users[cursor : cursor + size]
        cursor += size
        if len(members) < 2:
            break

        ring = _build_ring(
            ring_id=make_id("RING", r, width=4),
            ring_type=ring_type,
            members=members,
            cfg=cfg,
            pools=pools,
            promotions=promotions,
            rng=rng,
        )
        rings.append(ring)

    # Any leftover abuse users join the last ring (keeps labels consistent).
    if cursor < n_users and rings:
        leftovers = abuse_users[cursor:]
        last = rings[-1]
        cfg = config.ring_types[last.ring_type]
        _base_setup(leftovers, pools, rng, promotions)
        for user in leftovers:
            user.is_abuse_account = True
            user.ring_id = last.ring_id
            user.ring_type = last.ring_type
            user.group_id = last.ring_id
            user.group_type = "abuse_ring"
            # Give leftovers a slice of the ring's shared infra.
            if last.shared_ip_ids and rng.random() < cfg.ip_overlap:
                user.ip_ids.append(str(rng.choice(last.shared_ip_ids)))
            if last.shared_device_ids and rng.random() < cfg.device_overlap:
                user.device_ids.append(str(rng.choice(last.shared_device_ids)))
            last.user_ids.append(user.user_id)

    return rings


def build_hidden_scenarios(
    config: GenerationConfig,
    scenario_users: dict[str, list[UserProfile]],
    pools: dict[str, ResourcePool],
    promotions: list[str],
    rng: np.random.Generator,
) -> tuple[list[RingSpec], list[str]]:
    """Build the forced test-only scenarios.

    ``scenario_users`` maps scenario name -> the users allocated to it. Returns
    the abuse ring specs created (legit high-connectivity groups are handled as
    groups, not rings) plus the list of legit "group" ids created so the splitter
    can pin them to the test split.

    Returns ``(ring_specs, legit_group_ids_for_test)``.
    """
    ring_specs: list[RingSpec] = []
    legit_test_group_ids: list[str] = []
    counter = 0

    for scenario in config.hidden_scenarios:
        for _ in range(scenario.count):
            users = scenario_users.get(scenario.name, [])
            take = users[:scenario.size]
            del users[:scenario.size]
            if len(take) < 2:
                continue

            if scenario.kind == "legit_high_connectivity":
                # A large legitimate group with heavy shared infra but NO abuse
                # label. Tests false-positive resistance.
                group_id = make_id("GRP_HIDDEN", counter, width=4)
                city = take[0].city
                shared_ips = pools["ip"].fresh_block(2)
                shared_addr = pools["address"].fresh()
                for user in take:
                    user.segment = "student"
                    user.city = city
                    user.group_id = group_id
                    user.group_type = "college"
                    assign_behaviour(user, rng)
                    assign_personal_infrastructure(user, pools, rng)
                    user.ip_ids.extend(shared_ips)
                    if rng.random() < 0.5:
                        user.address_ids.append(shared_addr)
                legit_test_group_ids.append(group_id)
                counter += 1
                continue

            # Abuse scenarios.
            if scenario.kind == "low_overlap":
                ring_type = "direct_infrastructure"
                overlap_scale = 0.35  # 20-40% overlap
            elif scenario.kind == "high_noise":
                ring_type = "distributed_behavioral"
                overlap_scale = 1.0
            elif scenario.kind == "indirect_multihop":
                ring_type = "indirect_multihop"
                overlap_scale = 1.0
            else:
                ring_type = "direct_infrastructure"
                overlap_scale = 1.0

            cfg = config.ring_types[ring_type]
            if scenario.kind == "high_noise":
                # Halve the coordinated-transaction fraction => more legit noise.
                cfg = cfg.model_copy(update={"abuse_txn_fraction": cfg.abuse_txn_fraction * 0.4})

            ring = _build_ring(
                ring_id=make_id("RING_HIDDEN", counter, width=4),
                ring_type=ring_type,
                members=take,
                cfg=cfg,
                pools=pools,
                promotions=promotions,
                rng=rng,
                scenario=scenario.name,
                overlap_scale=overlap_scale,
            )
            ring_specs.append(ring)
            counter += 1

    return ring_specs, legit_test_group_ids


__all__ = ["build_abuse_rings", "build_hidden_scenarios"]
