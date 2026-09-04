import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  graph_loaded: boolean;
  investigator_loaded: boolean;
  risk_engine_loaded: boolean;
}

/** Matches src.investigation.schema.RiskLevel (serialised lowercase). */
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

/** Matches src.investigation.schema.FeatureAttribution. */
export interface FeatureAttribution {
  feature_name: string;
  feature_value: number;
  shap_value: number;
  direction: string;
  human_readable_description: string;
}

/** Matches src.investigation.schema.RelatedAccount. */
export interface RelatedAccount {
  user_id: string;
  relationship_types: string[];
  shared_entity_count: number;
}

/** Matches src.investigation.schema.CommunityContext. */
export interface CommunityContext {
  community_id: number;
  community_size: number;
  community_density: number;
  user_degree: number;
  neighbor_count: number;
}

/** Matches src.investigation.schema.BehavioralSummary. */
export interface BehavioralSummary {
  statements: string[];
  feature_values: Record<string, number>;
}

/** Matches src.investigation.schema.CounterfactualResult. */
export interface CounterfactualResult {
  feature: string;
  original_value: number;
  counterfactual_value: number;
  original_probability: number;
  counterfactual_probability: number;
  probability_delta: number;
}

/** Matches src.api.schemas.GraphEvidenceResponse. */
export interface GraphEvidenceResponse {
  shared_devices: string[];
  shared_ips: string[];
  shared_addresses: string[];
  shared_payment_instruments: string[];
}

/** Matches src.api.schemas.InvestigationResponse. */
export interface InvestigationResponse {
  user_id: string;
  risk_score: number;
  risk_level: RiskLevel;
  top_model_factors: FeatureAttribution[];
  behavioral_evidence: BehavioralSummary | null;
  graph_evidence: GraphEvidenceResponse;
  related_accounts: RelatedAccount[];
  community_context: CommunityContext | null;
  counterfactuals: CounterfactualResult[];
}

/** Matches src.risk.schema.Reason. */
export interface Reason {
  code: string;
  description: string;
}

/** Matches src.api.schemas.RiskAssessmentResponse. */
export interface RiskAssessmentResponse {
  user_id: string;
  risk_score: number;
  risk_level: string;
  decision: string;
  review_priority: string | null;
  reasons: Reason[];
  evidence: InvestigationResponse;
}

export const api = {
  getHealth: async () => {
    const res = await apiClient.get<HealthResponse>('/health');
    return res.data;
  },
  getRisk: async (userId: string) => {
    const res = await apiClient.get<RiskAssessmentResponse>(`/users/${userId}/risk`);
    return res.data;
  },
  getInvestigation: async (userId: string) => {
    const res = await apiClient.get<InvestigationResponse>(`/users/${userId}/investigation`);
    return res.data;
  },
  getReviewQueueTop: async (n: number = 20) => {
    const res = await apiClient.get<RiskAssessmentResponse[]>(`/review-queue/top/${n}`);
    return res.data;
  },
  /** Batch investigation. Unknown users are omitted by the backend. */
  batchInvestigate: async (userIds: string[]) => {
    const res = await apiClient.post<InvestigationResponse[]>('/investigate', {
      user_ids: userIds,
    });
    return res.data;
  }
};
