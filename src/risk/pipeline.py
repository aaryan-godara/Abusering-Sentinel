"""Phase 5 pipeline: generate risk assessment examples and distributions.

Run with:
    python -m src.risk.pipeline

Produces:
    reports/risk_assessment_examples.json
    reports/risk_distribution.json
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT
from src.investigation.investigator import Investigator
from src.risk.engine import RiskEngine
from src.risk.queue import ReviewQueue

REPORTS_DIR = PROJECT_ROOT / "reports"


def run_pipeline() -> dict:
    """Run the complete Phase 5 risk pipeline."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AbuseRing Sentinel - Phase 5 Risk & Decision Engine")
    print("=" * 60)

    # 1. Initialize
    print("\nInitialising Investigator and Risk Engine...")
    t0 = time.time()
    investigator = Investigator()
    engine = RiskEngine(investigator=investigator)
    queue = ReviewQueue(risk_engine=engine)
    init_time = time.time() - t0
    print(f"  Initialised in {init_time:.1f}s")
    
    user_ids = investigator.user_ids
    probas = investigator.probabilities
    
    print(f"  Users in split: {len(user_ids)}")

    # 2. Select targets for examples
    # We want 5 high-risk, 3 medium-risk, 2 low-risk
    print("\nSelecting investigation targets...")
    high_idx = np.where(probas >= 0.5)[0]
    med_idx = np.where((probas >= 0.2) & (probas < 0.5))[0]
    low_idx = np.where(probas < 0.2)[0]

    # Deterministic subset for examples
    high_subset = [user_ids[i] for i in high_idx[:5]]
    med_subset = [user_ids[i] for i in med_idx[:3]]
    low_subset = [user_ids[i] for i in low_idx[:2]]
    
    example_uids = high_subset + med_subset + low_subset
    
    examples = []
    
    print("Running example assessments...")
    for uid in example_uids:
        t_start = time.time()
        assessment = queue.evaluate(uid)
        elapsed = time.time() - t_start
        examples.append({
            "user_id": uid,
            "runtime_seconds": round(elapsed, 3),
            "assessment": assessment.model_dump(mode="json")
        })
        print(f"  {uid}: risk={assessment.risk_score:.4f} decision={assessment.decision.value} reasons={len(assessment.reasons)} [{elapsed:.2f}s]")

    # Also evaluate the next 5 high-risk users just to put them in the queue for the report
    queue_uids = [user_ids[i] for i in high_idx[5:10]]
    for uid in queue_uids:
        queue.evaluate(uid)

    examples_path = REPORTS_DIR / "risk_assessment_examples.json"
    examples_path.write_text(json.dumps(examples, indent=2, default=str))
    print(f"\nSaved {len(examples)} risk assessment examples to {examples_path}")

    # 3. Batch Performance Test
    print("\nRunning 100-user batch test...")
    batch_uids = user_ids[:100]
    t_start_batch = time.time()
    for uid in batch_uids:
        queue.evaluate(uid)
    batch_elapsed = time.time() - t_start_batch
    print(f"  Evaluated 100 users in {batch_elapsed:.2f}s ({batch_elapsed/100:.3f}s per user)")

    # 4. Generate distributions over the 100-user batch
    print("\nGenerating risk distribution report...")
    assessments = [q for q in queue._queue]
    all_scores = probas
    
    # We will simulate the whole set's decisions for distribution
    # since running 1500 could take ~15s and we just need the policy mapping
    decisions = []
    levels = []
    for p in all_scores:
        d, r = engine._policy.evaluate(p)
        decisions.append(d.value)
        
        # Risk level logic from investigator
        if p >= 0.8: l = "critical"
        elif p >= 0.5: l = "high"
        elif p >= 0.2: l = "medium"
        else: l = "low"
        levels.append(l)

    distribution = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_users": len(all_scores),
        "risk_levels": dict(Counter(levels)),
        "decisions": dict(Counter(decisions)),
        "queue_size": queue.queue_size,
        "risk_scores": {
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores)),
            "mean": float(np.mean(all_scores)),
            "median": float(np.median(all_scores)),
            "p95": float(np.percentile(all_scores, 95)),
            "p99": float(np.percentile(all_scores, 99)),
        },
        "performance_100_batch_seconds": round(batch_elapsed, 2)
    }

    dist_path = REPORTS_DIR / "risk_distribution.json"
    dist_path.write_text(json.dumps(distribution, indent=2))
    print(f"  Saved risk distribution to {dist_path}")

    return {
        "examples_count": len(examples),
        "batch_elapsed": batch_elapsed,
        "init_time": init_time,
    }


def main() -> int:
    try:
        result = run_pipeline()
        print("\n" + "=" * 60)
        print("Phase 5 complete!")
        print(f"  Risk examples generated: {result['examples_count']}")
        print(f"  100-user batch time: {result['batch_elapsed']:.2f}s")
        return 0
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
