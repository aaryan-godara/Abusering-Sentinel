"""Policies for the Phase 5 Risk Engine."""

from __future__ import annotations

from src.investigation.schema import RiskLevel
from src.risk.schema import Decision, ReviewPriority


class DefaultRiskPolicy:
    """Configurable risk policy mapping risk level and score to decisions."""
    
    VERSION = "1.0.0"

    def __init__(
        self,
        medium_threshold: float = 0.2,
        high_threshold: float = 0.5,
        critical_threshold: float = 0.8,
    ) -> None:
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold

    def evaluate(self, risk_score: float) -> tuple[Decision, ReviewPriority | None]:
        """Evaluate a risk score and return the corresponding decision and priority."""
        if risk_score >= self.critical_threshold:
            return Decision.ESCALATE, ReviewPriority.URGENT
        elif risk_score >= self.high_threshold:
            return Decision.REVIEW, ReviewPriority.HIGH
        elif risk_score >= self.medium_threshold:
            return Decision.MONITOR, ReviewPriority.NORMAL
        else:
            return Decision.ALLOW, None

__all__ = ["DefaultRiskPolicy"]
