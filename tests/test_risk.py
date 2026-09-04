"""Tests for Phase 5 — Risk Engine.

Tests cover:
- RiskAssessment schema
- Model C probability preservation
- Policy mapping
- Reason generation
- Queue ordering and batch scoring
- No ground-truth fields in output
- Deterministic output
"""

from __future__ import annotations

import pytest
import numpy as np

from src.investigation.schema import GROUND_TRUTH_FIELDS, InvestigationResult, RiskLevel
from src.risk.engine import RiskEngine
from src.risk.policies import DefaultRiskPolicy
from src.risk.queue import ReviewQueue
from src.risk.reasons import ReasonExtractor
from src.risk.schema import Decision, ReasonCode, ReviewPriority, RiskAssessment


class MockInvestigator:
    """Mock Investigator for testing the Risk Engine."""
    
    def __init__(self):
        self.user_ids = ["U1", "U2", "U3"]
        self.probabilities = np.array([0.1, 0.6, 0.9])
        self._investigations = {
            "U1": InvestigationResult(
                user_id="U1",
                risk_score=0.1,
                risk_level=RiskLevel.LOW,
                top_model_factors=[],
                shared_devices=[],
                shared_ips=[],
                shared_addresses=[],
                shared_payment_instruments=[],
                related_accounts=[],
                multi_hop_paths=[],
                counterfactuals=[],
            ),
            "U2": InvestigationResult(
                user_id="U2",
                risk_score=0.6,
                risk_level=RiskLevel.HIGH,
                top_model_factors=[],
                shared_devices=["D1", "D2"],
                shared_ips=[],
                shared_addresses=[],
                shared_payment_instruments=[],
                related_accounts=[],
                multi_hop_paths=[],
                counterfactuals=[],
            ),
            "U3": InvestigationResult(
                user_id="U3",
                risk_score=0.9,
                risk_level=RiskLevel.CRITICAL,
                top_model_factors=[],
                shared_devices=[],
                shared_ips=["IP1"],
                shared_addresses=[],
                shared_payment_instruments=[],
                related_accounts=[],
                multi_hop_paths=[],
                counterfactuals=[],
            )
        }

    def investigate_user(self, user_id: str, **kwargs) -> InvestigationResult:
        return self._investigations[user_id]


@pytest.fixture
def mock_engine():
    investigator = MockInvestigator()
    return RiskEngine(investigator=investigator)


@pytest.fixture
def mock_queue(mock_engine):
    return ReviewQueue(risk_engine=mock_engine)


class TestRiskAssessmentSchema:
    """Test RiskAssessment schema and constraints."""

    def test_schema_valid(self, mock_engine):
        assessment = mock_engine.evaluate("U2")
        assert assessment.user_id == "U2"
        assert assessment.decision == Decision.REVIEW
        assert assessment.review_priority == ReviewPriority.HIGH

    def test_no_ground_truth_fields(self):
        """RiskAssessment must not contain ground-truth fields."""
        fields = set(RiskAssessment.model_fields.keys())
        for gt_field in GROUND_TRUTH_FIELDS:
            assert gt_field not in fields, f"Schema contains ground-truth field: {gt_field}"


class TestModelCProbabilityPreservation:
    def test_probability_preserved(self, mock_engine):
        """Risk score MUST come directly from frozen Model C probability."""
        a1 = mock_engine.evaluate("U1")
        a2 = mock_engine.evaluate("U2")
        a3 = mock_engine.evaluate("U3")
        
        assert a1.risk_score == 0.1
        assert a2.risk_score == 0.6
        assert a3.risk_score == 0.9


class TestPolicyMapping:
    def test_default_policy(self):
        policy = DefaultRiskPolicy()
        
        # LOW
        d, p = policy.evaluate(0.1)
        assert d == Decision.ALLOW and p is None
        
        # MEDIUM
        d, p = policy.evaluate(0.3)
        assert d == Decision.MONITOR and p == ReviewPriority.NORMAL
        
        # HIGH
        d, p = policy.evaluate(0.6)
        assert d == Decision.REVIEW and p == ReviewPriority.HIGH
        
        # CRITICAL
        d, p = policy.evaluate(0.9)
        assert d == Decision.ESCALATE and p == ReviewPriority.URGENT


class TestReasonGeneration:
    def test_reason_generation(self, mock_engine):
        a2 = mock_engine.evaluate("U2")
        # U2 has shared devices
        codes = [r.code for r in a2.reasons]
        assert ReasonCode.GRAPH_SHARED_DEVICE in codes
        
        a3 = mock_engine.evaluate("U3")
        # U3 has shared ips
        codes = [r.code for r in a3.reasons]
        assert ReasonCode.GRAPH_SHARED_IP in codes
        assert ReasonCode.GRAPH_SHARED_DEVICE not in codes


class TestQueueOrdering:
    def test_queue_ordering(self, mock_queue):
        # Add U1 (LOW), U2 (HIGH), U3 (CRITICAL)
        mock_queue.evaluate("U1")
        mock_queue.evaluate("U2")
        mock_queue.evaluate("U3")
        
        # Only REVIEW and ESCALATE should be in queue
        assert mock_queue.queue_size == 2
        
        top = mock_queue.top_risk_users(2)
        # U3 is CRITICAL (URGENT priority)
        # U2 is HIGH (HIGH priority)
        assert top[0].user_id == "U3"
        assert top[1].user_id == "U2"


class TestBatchScoring:
    def test_batch_scoring(self, mock_queue):
        assessments = mock_queue.score_users(["U1", "U2", "U3"])
        assert len(assessments) == 3
        assert mock_queue.queue_size == 2


class TestDeterministicOutput:
    def test_deterministic_evaluation(self, mock_engine):
        a1 = mock_engine.evaluate("U2")
        a2 = mock_engine.evaluate("U2")
        assert a1.model_dump() == a2.model_dump()

