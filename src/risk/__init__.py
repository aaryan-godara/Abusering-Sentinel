"""Phase 5 — Risk & Decision Engine.

Provides an operational layer on top of Model C and Investigator to assign
decisions and manage manual review queues based on deterministic policies
and reason extraction.
"""

from __future__ import annotations

from src.risk.engine import RiskEngine
from src.risk.policies import DefaultRiskPolicy
from src.risk.queue import ReviewQueue
from src.risk.reasons import ReasonExtractor
from src.risk.schema import (
    Decision,
    Reason,
    ReasonCode,
    ReviewPriority,
    RiskAssessment,
)

__all__ = [
    "Decision",
    "DefaultRiskPolicy",
    "Reason",
    "ReasonCode",
    "ReasonExtractor",
    "ReviewPriority",
    "ReviewQueue",
    "RiskAssessment",
    "RiskEngine",
]
