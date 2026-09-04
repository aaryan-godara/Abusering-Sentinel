import type { GraphEvidenceResponse, InvestigationResponse } from '../api/client';

/* ==========================================================================
   Evidence interpretation helpers.
   Everything here is derived from the existing investigation payload — no
   probabilities, no inference, no fabricated values.
   ========================================================================== */

/** Qualitative strength of a link type. NOT a fraud probability. */
export type EvidenceStrength = 'HIGH' | 'MEDIUM' | 'INDIRECT';

/** Strength is a property of the relationship type itself. */
export const RELATIONSHIP_STRENGTH: Record<string, EvidenceStrength> = {
  shared_device: 'HIGH',
  shared_payment_instrument: 'HIGH',
  shared_address: 'MEDIUM',
  shared_ip: 'INDIRECT',
};

/** Same mapping keyed by graph node kind. */
export const ENTITY_STRENGTH: Record<string, EvidenceStrength> = {
  DEVICE: 'HIGH',
  PAYMENT: 'HIGH',
  ADDRESS: 'MEDIUM',
  IP: 'INDIRECT',
};

interface EvidenceGroup {
  relType: string;
  key: keyof GraphEvidenceResponse;
  title: string;
  noun: string;
}

const GROUPS: EvidenceGroup[] = [
  { relType: 'shared_device', key: 'shared_devices', title: 'Shared Device', noun: 'device' },
  {
    relType: 'shared_payment_instrument',
    key: 'shared_payment_instruments',
    title: 'Shared Payment',
    noun: 'payment instrument',
  },
  { relType: 'shared_address', key: 'shared_addresses', title: 'Shared Address', noun: 'address' },
  { relType: 'shared_ip', key: 'shared_ips', title: 'Shared IP', noun: 'IP' },
];

const plural = (n: number, noun: string) => `${n} ${noun}${n === 1 ? '' : 's'}`;

export interface WhyItem {
  title: string;
  detail: string;
  strength: EvidenceStrength;
}

/**
 * Human-readable "why this user?" summary. Every line restates a value that is
 * already present in the investigation response.
 */
export function buildWhyThisUser(inv: InvestigationResponse): WhyItem[] {
  const items: WhyItem[] = [];

  const accountsWith = (relType: string) =>
    inv.related_accounts.filter((a) => a.relationship_types.includes(relType)).length;

  for (const group of GROUPS) {
    const entities = inv.graph_evidence[group.key].length;
    if (entities === 0) continue;
    const accounts = accountsWith(group.relType);
    items.push({
      title: group.title,
      strength: RELATIONSHIP_STRENGTH[group.relType] ?? 'INDIRECT',
      detail:
        accounts > 0
          ? `${plural(accounts, 'account')} share ${plural(entities, group.noun)} with this subject.`
          : `${plural(entities, group.noun)} linked to this subject; no related account shares them in the loaded evidence.`,
    });
  }

  for (const statement of inv.behavioral_evidence?.statements.slice(0, 2) ?? []) {
    items.push({ title: 'Behavioral Signal', strength: 'MEDIUM', detail: statement });
  }

  for (const factor of inv.top_model_factors.filter((f) => f.shap_value > 0).slice(0, 2)) {
    items.push({
      title: 'Model Factor',
      strength: 'INDIRECT',
      detail: `${factor.human_readable_description} (SHAP +${factor.shap_value.toFixed(3)}).`,
    });
  }

  if (inv.community_context && inv.community_context.community_size > 1) {
    items.push({
      title: 'Community Context',
      strength: 'INDIRECT',
      detail: `Sits in a community of ${plural(
        inv.community_context.community_size,
        'account',
      )} at density ${inv.community_context.community_density.toFixed(3)}.`,
    });
  }

  return items;
}

export interface TimelineItem {
  stage: string;
  detail: string;
  red?: boolean;
}

/**
 * Ordered observation sequence. The evidence payload carries no timestamps, so
 * this is a processing order, never a wall-clock timeline.
 */
export function buildTimeline(
  inv: InvestigationResponse,
  decision?: string,
  priority?: string | null,
): TimelineItem[] {
  const items: TimelineItem[] = [];
  const high = inv.risk_level === 'high' || inv.risk_level === 'critical';

  items.push({
    stage: 'Subject scored',
    detail: `Risk score ${(inv.risk_score * 100).toFixed(1)}% · level ${inv.risk_level.toUpperCase()}.`,
    red: high,
  });

  items.push({
    stage: 'Graph expanded',
    detail: `${plural(
      inv.related_accounts.length,
      'related account',
    )} resolved from the shared-entity graph.`,
  });

  for (const group of GROUPS) {
    const entities = inv.graph_evidence[group.key].length;
    if (entities === 0) continue;
    const accounts = inv.related_accounts.filter((a) =>
      a.relationship_types.includes(group.relType),
    ).length;
    items.push({
      stage: `${group.title} detected`,
      detail: `${plural(entities, group.noun)} shared across ${plural(accounts, 'account')}.`,
      red: RELATIONSHIP_STRENGTH[group.relType] === 'HIGH' && accounts > 0,
    });
  }

  for (const statement of inv.behavioral_evidence?.statements.slice(0, 3) ?? []) {
    items.push({ stage: 'Behavioral observation', detail: statement });
  }

  if (decision) {
    items.push({
      stage: 'Decision issued',
      detail: `${decision.toUpperCase()}${priority ? ` · priority ${priority.toUpperCase()}` : ''}.`,
      red: true,
    });
  }

  return items;
}
