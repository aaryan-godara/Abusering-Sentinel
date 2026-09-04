"""Core Risk Engine tying Investigator to Risk Policy."""

from __future__ import annotations

import time

from src.investigation.investigator import Investigator
from src.risk.policies import DefaultRiskPolicy
from src.risk.reasons import ReasonExtractor
from src.risk.schema import RiskAssessment


class RiskEngine:
    """Operational layer that evaluates users for risk and assigns decisions."""
    
    # We use a hardcoded model version for the scope of this project.
    MODEL_VERSION = "C-1.0.0"

    def __init__(
        self,
        investigator: Investigator | None = None,
        policy: DefaultRiskPolicy | None = None,
    ) -> None:
        # If no investigator is passed, instantiate it here.
        # This keeps the graphs/model loaded once in memory.
        self._investigator = investigator or Investigator()
        self._policy = policy or DefaultRiskPolicy()

    def evaluate(self, user_id: str) -> RiskAssessment:
        """Evaluate a single user and generate a complete RiskAssessment."""
        # Get evidence and risk score from Investigator
        evidence = self._investigator.investigate_user(user_id)
        
        # Policy evaluation
        decision, review_priority = self._policy.evaluate(evidence.risk_score)
        
        # Reason extraction
        reasons = ReasonExtractor.extract_reasons(evidence)
        
        return RiskAssessment(
            user_id=user_id,
            risk_score=evidence.risk_score,
            risk_level=evidence.risk_level.value,
            decision=decision,
            review_priority=review_priority,
            model_version=self.MODEL_VERSION,
            policy_version=self._policy.VERSION,
            reasons=reasons,
            evidence=evidence,
        )

    def evaluate_batch(self, user_ids: list[str]) -> list[RiskAssessment]:
        """Evaluate multiple users sequentially."""
        return [self.evaluate(uid) for uid in user_ids]

__all__ = ["RiskEngine"]
