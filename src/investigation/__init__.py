"""Phase 4 — Investigation and Explainability.

Turns Model C predictions into investigator-friendly explanations:
SHAP-based feature attribution, graph relationship evidence, multi-hop paths,
community context, behavioural summaries, and counterfactual analysis.

No ground-truth labels (is_abuse_account, ring_id, ring_type,
is_abuse_transaction, estimated_abuse_value) appear in any investigator-facing
output.
"""

from __future__ import annotations

from src.investigation.schema import (
    BehavioralSummary,
    CommunityContext,
    CounterfactualResult,
    FeatureAttribution,
    InvestigationResult,
    MultiHopPath,
    RelatedAccount,
    RiskLevel,
)
from src.investigation.investigator import Investigator
from src.investigation.graph_evidence import GraphEvidenceCollector
from src.investigation.counterfactuals import CounterfactualAnalyzer

__all__ = [
    "BehavioralSummary",
    "CommunityContext",
    "CounterfactualAnalyzer",
    "CounterfactualResult",
    "FeatureAttribution",
    "GraphEvidenceCollector",
    "InvestigationResult",
    "Investigator",
    "MultiHopPath",
    "RelatedAccount",
    "RiskLevel",
]
