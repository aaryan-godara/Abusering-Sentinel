"""Pydantic schemas for the Phase 5 Risk Engine.

All types represent operational layer decisions. No ground-truth labels.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.investigation.schema import InvestigationResult


class Decision(str, Enum):
    """Operational decision taken by the risk engine."""
    ALLOW = "allow"
    MONITOR = "monitor"
    REVIEW = "review"
    ESCALATE = "escalate"


class ReviewPriority(str, Enum):
    """Priority for manual review."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ReasonCode(str, Enum):
    """Deterministic reason codes for evidence."""
    # Graph evidence
    GRAPH_SHARED_DEVICE = "GRAPH_SHARED_DEVICE"
    GRAPH_SHARED_IP = "GRAPH_SHARED_IP"
    GRAPH_SHARED_ADDRESS = "GRAPH_SHARED_ADDRESS"
    GRAPH_SHARED_PAYMENT = "GRAPH_SHARED_PAYMENT"
    GRAPH_CONNECTED_ACCOUNTS = "GRAPH_CONNECTED_ACCOUNTS"
    GRAPH_MULTI_HOP = "GRAPH_MULTI_HOP"
    GRAPH_DENSE_COMMUNITY = "GRAPH_DENSE_COMMUNITY"
    
    # Model/SHAP
    MODEL_HIGH_RISK_FACTOR = "MODEL_HIGH_RISK_FACTOR"
    
    # Behavioural
    BEHAVIORAL_HIGH_ACTIVITY = "BEHAVIORAL_HIGH_ACTIVITY"
    BEHAVIORAL_HIGH_PROMO = "BEHAVIORAL_HIGH_PROMO"
    BEHAVIORAL_NEW_ACCOUNT = "BEHAVIORAL_NEW_ACCOUNT"
    BEHAVIORAL_NIGHT_ACTIVITY = "BEHAVIORAL_NIGHT_ACTIVITY"
    BEHAVIORAL_WEEKEND_ACTIVITY = "BEHAVIORAL_WEEKEND_ACTIVITY"
    BEHAVIORAL_HIGH_FAIL_RATE = "BEHAVIORAL_HIGH_FAIL_RATE"


class Reason(BaseModel):
    """A generated reason for a decision, backed by evidence."""
    code: ReasonCode
    description: str


class RiskAssessment(BaseModel):
    """Complete risk assessment and operational decision for a user."""
    user_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str  # Inherits from InvestigationResult's RiskLevel
    decision: Decision
    review_priority: ReviewPriority | None
    model_version: str
    policy_version: str
    reasons: list[Reason]
    evidence: InvestigationResult


__all__ = [
    "Decision",
    "Reason",
    "ReasonCode",
    "ReviewPriority",
    "RiskAssessment",
]
