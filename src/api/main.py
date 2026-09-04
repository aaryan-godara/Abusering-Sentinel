"""FastAPI application entry point for AbuseRing Sentinel."""

from __future__ import annotations

import logging
from typing import Any, List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from src.api.dependencies import (
    get_investigator,
    get_review_queue,
    get_risk_engine,
    load_resources,
)
from src.api.schemas import (
    BatchUserRequest,
    GraphEvidenceResponse,
    HealthResponse,
    InvestigationResponse,
    RiskAssessmentResponse,
)
from src.investigation.investigator import Investigator
from src.risk.engine import RiskEngine
from src.risk.queue import ReviewQueue
from src.investigation.schema import InvestigationResult

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AbuseRing Sentinel API",
    description="Operational API for Fraud Ring Investigation and Decisioning.",
    version="1.0.0",
)

# CORS configuration for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load resources exactly once at application startup."""
    load_resources()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_investigation_response(inv: InvestigationResult) -> InvestigationResponse:
    """Map the core InvestigationResult to the API InvestigationResponse schema."""
    graph_evidence = GraphEvidenceResponse(
        shared_devices=inv.shared_devices,
        shared_ips=inv.shared_ips,
        shared_addresses=inv.shared_addresses,
        shared_payment_instruments=inv.shared_payment_instruments,
    )
    return InvestigationResponse(
        user_id=inv.user_id,
        risk_score=inv.risk_score,
        risk_level=inv.risk_level,
        top_model_factors=inv.top_model_factors,
        behavioral_evidence=inv.behavioral_summary,
        graph_evidence=graph_evidence,
        related_accounts=inv.related_accounts,
        community_context=inv.community_context,
        counterfactuals=inv.counterfactuals,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and loaded dependencies health."""
    try:
        inv = get_investigator()
        is_inv_loaded = inv is not None
    except Exception:
        is_inv_loaded = False
        
    try:
        re = get_risk_engine()
        is_re_loaded = re is not None
    except Exception:
        is_re_loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=is_inv_loaded,  # model is part of investigator
        graph_loaded=is_inv_loaded,  # graph is part of investigator
        investigator_loaded=is_inv_loaded,
        risk_engine_loaded=is_re_loaded,
    )


@app.get("/users/{user_id}/risk", response_model=RiskAssessmentResponse)
def get_user_risk(
    user_id: str,
    queue: ReviewQueue = Depends(get_review_queue)
):
    """Get the full risk assessment for a single user."""
    try:
        # evaluate pushes to queue if appropriate
        assessment = queue.evaluate(user_id)
        
        return RiskAssessmentResponse(
            user_id=assessment.user_id,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            decision=assessment.decision,
            review_priority=assessment.review_priority,
            reasons=assessment.reasons,
            evidence=_map_investigation_response(assessment.evidence),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}/investigation", response_model=InvestigationResponse)
def get_user_investigation(
    user_id: str,
    investigator: Investigator = Depends(get_investigator)
):
    """Get the detailed explainability investigation for a single user."""
    try:
        inv = investigator.investigate_user(user_id)
        return _map_investigation_response(inv)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}/evidence", response_model=GraphEvidenceResponse)
def get_user_evidence(
    user_id: str,
    investigator: Investigator = Depends(get_investigator)
):
    """Get only the raw graph evidence for a user."""
    try:
        inv = investigator.investigate_user(user_id)
        return GraphEvidenceResponse(
            shared_devices=inv.shared_devices,
            shared_ips=inv.shared_ips,
            shared_addresses=inv.shared_addresses,
            shared_payment_instruments=inv.shared_payment_instruments,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/review-queue", response_model=List[RiskAssessmentResponse])
def get_review_queue_all(
    queue: ReviewQueue = Depends(get_review_queue)
):
    """Get all items currently in the manual review queue."""
    res = []
    for assessment in queue._queue:
        res.append(RiskAssessmentResponse(
            user_id=assessment.user_id,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            decision=assessment.decision,
            review_priority=assessment.review_priority,
            reasons=assessment.reasons,
            evidence=_map_investigation_response(assessment.evidence),
        ))
    return res


@app.get("/review-queue/top/{n}", response_model=List[RiskAssessmentResponse])
def get_review_queue_top(
    n: int,
    queue: ReviewQueue = Depends(get_review_queue)
):
    """Get the top N highest priority users in the review queue."""
    if n <= 0:
        raise HTTPException(status_code=400, detail="n must be positive")
    
    top_users = queue.top_risk_users(n)
    res = []
    for assessment in top_users:
        res.append(RiskAssessmentResponse(
            user_id=assessment.user_id,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            decision=assessment.decision,
            review_priority=assessment.review_priority,
            reasons=assessment.reasons,
            evidence=_map_investigation_response(assessment.evidence),
        ))
    return res


@app.post("/score", response_model=List[RiskAssessmentResponse])
def batch_score(
    request: BatchUserRequest,
    queue: ReviewQueue = Depends(get_review_queue)
):
    """Batch evaluate a list of users for risk."""
    res = []
    for uid in request.user_ids:
        try:
            assessment = queue.evaluate(uid)
            res.append(RiskAssessmentResponse(
                user_id=assessment.user_id,
                risk_score=assessment.risk_score,
                risk_level=assessment.risk_level,
                decision=assessment.decision,
                review_priority=assessment.review_priority,
                reasons=assessment.reasons,
                evidence=_map_investigation_response(assessment.evidence),
            ))
        except ValueError:
            # Skip unknown users in batch, or we could return partial failures
            pass
    return res


@app.post("/investigate", response_model=List[InvestigationResponse])
def batch_investigate(
    request: BatchUserRequest,
    investigator: Investigator = Depends(get_investigator)
):
    """Batch investigate a list of users."""
    res = []
    for uid in request.user_ids:
        try:
            inv = investigator.investigate_user(uid)
            res.append(_map_investigation_response(inv))
        except ValueError:
            pass
    return res

