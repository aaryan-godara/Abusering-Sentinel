"""Pydantic schemas for the Phase 6 API.

Ensures that ground-truth labels are never exposed via the API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from src.investigation.schema import (
    BehavioralSummary,
    CommunityContext,
    CounterfactualResult,
    FeatureAttribution,
    MultiHopPath,
    RelatedAccount,
    RiskLevel,
)
from src.risk.schema import Decision, Reason, ReviewPriority


# ---------------------------------------------------------------------------
# Investigation API Responses
# ---------------------------------------------------------------------------

class GraphEvidenceResponse(BaseModel):
    """Sub-document grouping direct graph evidence."""
    shared_devices: List[str]
    shared_ips: List[str]
    shared_addresses: List[str]
    shared_payment_instruments: List[str]


class InvestigationResponse(BaseModel):
    """Complete investigation details for a user.
    
    Contains no ground-truth fields.
    """
    user_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_model_factors: List[FeatureAttribution]
    behavioral_evidence: Optional[BehavioralSummary]
    graph_evidence: GraphEvidenceResponse
    related_accounts: List[RelatedAccount]
    community_context: Optional[CommunityContext]
    counterfactuals: List[CounterfactualResult]


# ---------------------------------------------------------------------------
# Risk API Responses
# ---------------------------------------------------------------------------

class RiskAssessmentResponse(BaseModel):
    """Risk assessment response for a single user."""
    user_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    decision: Decision
    review_priority: Optional[ReviewPriority]
    reasons: List[Reason]
    evidence: InvestigationResponse


# ---------------------------------------------------------------------------
# API Requests
# ---------------------------------------------------------------------------

class BatchUserRequest(BaseModel):
    """Request schema for batch operations."""
    user_ids: List[str] = Field(min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Health Response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    graph_loaded: bool
    investigator_loaded: bool
    risk_engine_loaded: bool


# ---------------------------------------------------------------------------
# Error Responses
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    detail: str

__all__ = [
    "BatchUserRequest",
    "ErrorDetail",
    "GraphEvidenceResponse",
    "HealthResponse",
    "InvestigationResponse",
    "RiskAssessmentResponse",
]
