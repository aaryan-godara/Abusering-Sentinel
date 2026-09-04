"""API Dependencies and App State.

Loads heavy resources (Investigator, RiskEngine, ReviewQueue) exactly once
at application startup so they can be injected into route handlers.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.investigation.investigator import Investigator
from src.risk.engine import RiskEngine
from src.risk.queue import ReviewQueue

logger = logging.getLogger(__name__)

# Global instances loaded at startup
_investigator: Optional[Investigator] = None
_risk_engine: Optional[RiskEngine] = None
_review_queue: Optional[ReviewQueue] = None


def load_resources() -> None:
    """Load Investigator, RiskEngine, and ReviewQueue into memory."""
    global _investigator, _risk_engine, _review_queue
    logger.info("Loading AbuseRing Sentinel resources...")
    try:
        _investigator = Investigator()
        _risk_engine = RiskEngine(investigator=_investigator)
        _review_queue = ReviewQueue(risk_engine=_risk_engine)
        logger.info("Resources loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise


def get_investigator() -> Investigator:
    """Dependency injection for Investigator."""
    if _investigator is None:
        raise RuntimeError("Investigator is not loaded.")
    return _investigator


def get_risk_engine() -> RiskEngine:
    """Dependency injection for Risk Engine."""
    if _risk_engine is None:
        raise RuntimeError("RiskEngine is not loaded.")
    return _risk_engine


def get_review_queue() -> ReviewQueue:
    """Dependency injection for Review Queue."""
    if _review_queue is None:
        raise RuntimeError("ReviewQueue is not loaded.")
    return _review_queue

__all__ = [
    "get_investigator",
    "get_review_queue",
    "get_risk_engine",
    "load_resources",
]
