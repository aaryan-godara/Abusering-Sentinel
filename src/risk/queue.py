"""In-memory Review Queue for the Risk Engine."""

from __future__ import annotations

from typing import Optional

from src.risk.engine import RiskEngine
from src.risk.schema import Decision, ReviewPriority, RiskAssessment


class ReviewQueue:
    """In-memory queue for accounts requiring manual review."""

    def __init__(self, risk_engine: RiskEngine) -> None:
        self._engine = risk_engine
        self._queue: list[RiskAssessment] = []
        
        self._priority_map = {
            ReviewPriority.URGENT: 4,
            ReviewPriority.HIGH: 3,
            ReviewPriority.NORMAL: 2,
            ReviewPriority.LOW: 1,
        }

    def evaluate(self, user_id: str) -> RiskAssessment:
        """Evaluate a user and optionally add to queue."""
        assessment = self._engine.evaluate(user_id)
        if assessment.decision in (Decision.REVIEW, Decision.ESCALATE):
            self._queue.append(assessment)
            self._sort_queue()
        return assessment

    def score_users(self, user_ids: list[str]) -> list[RiskAssessment]:
        """Score a batch of users, adding relevant ones to queue."""
        assessments = []
        for uid in user_ids:
            assessments.append(self.evaluate(uid))
        return assessments

    def _sort_queue(self) -> None:
        """Sort the queue by review priority, then risk score descending."""
        self._queue.sort(
            key=lambda x: (
                self._priority_map.get(x.review_priority, 0),
                x.risk_score
            ),
            reverse=True
        )

    def top_risk_users(self, n: int = 10) -> list[RiskAssessment]:
        """Get the top N highest priority users in the queue."""
        return self._queue[:n]

    @property
    def queue_size(self) -> int:
        return len(self._queue)

__all__ = ["ReviewQueue"]
