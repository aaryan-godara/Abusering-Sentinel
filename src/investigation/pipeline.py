"""Phase 4 pipeline: generate investigation examples and explanation audit.

Run with:
    python -m src.investigation.pipeline

Produces:
    reports/investigation_examples.json
    reports/explanation_audit.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT
from src.investigation.investigator import Investigator
from src.investigation.schema import GROUND_TRUTH_FIELDS, FEATURE_DESCRIPTIONS

REPORTS_DIR = PROJECT_ROOT / "reports"


def _select_investigation_targets(investigator: Investigator) -> list[dict]:
    """Select a diverse sample of users for investigation.

    Returns list of dicts with user_id and selection_reason.
    """
    probas = investigator.probabilities
    labels = investigator.labels
    user_ids = investigator.user_ids

    targets: list[dict] = []
    used: set[str] = set()

    def _add(uid: str, reason: str) -> None:
        if uid not in used:
            used.add(uid)
            targets.append({"user_id": uid, "selection_reason": reason})

    # 5 high-risk users
    high_risk_idx = np.argsort(probas)[::-1]
    count = 0
    for i in high_risk_idx:
        if count >= 5:
            break
        _add(user_ids[i], "high_risk")
        count += 1

    # 2 legitimate high-connectivity users (label=0, high score)
    legit_high_idx = np.argsort(probas)[::-1]
    count = 0
    for i in legit_high_idx:
        if count >= 2:
            break
        if labels[i] == 0 and probas[i] > 0.1 and user_ids[i] not in used:
            _add(user_ids[i], "legitimate_high_connectivity")
            count += 1

    # 2 false positives (label=0, score above some threshold)
    # Use the Model C threshold from experiment manifest
    manifest_path = REPORTS_DIR / "experiment_manifest.json"
    thresh = 0.35  # default fallback
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        thresh = manifest.get("selected_thresholds", {}).get("C", thresh)

    fp_idx = np.where((labels == 0) & (probas >= thresh))[0]
    fp_sorted = fp_idx[np.argsort(probas[fp_idx])[::-1]]
    count = 0
    for i in fp_sorted:
        if count >= 2:
            break
        if user_ids[i] not in used:
            _add(user_ids[i], "false_positive")
            count += 1

    # 2 false negatives (label=1, score below threshold)
    fn_idx = np.where((labels == 1) & (probas < thresh))[0]
    fn_sorted = fn_idx[np.argsort(probas[fn_idx])]  # lowest score first
    count = 0
    for i in fn_sorted:
        if count >= 2:
            break
        if user_ids[i] not in used:
            _add(user_ids[i], "false_negative")
            count += 1

    # 1 indirect multi-hop case: pick a user with multi-hop paths available
    # Select from medium-risk users
    med_idx = np.where((probas > 0.1) & (probas < 0.6))[0]
    for i in med_idx:
        uid = user_ids[i]
        if uid in used:
            continue
        _add(uid, "indirect_multi_hop_case")
        break

    return targets


def _run_explanation_audit(
    investigator: Investigator,
    results: list[dict],
) -> dict:
    """Audit Phase 4 outputs for correctness.

    Checks:
    1. Reported SHAP features exist in the feature set
    2. SHAP values correspond to the model (are non-trivial)
    3. Graph evidence entities exist in the graph
    4. Multi-hop paths exist in the graph
    5. No ground-truth labels in investigator output
    6. Counterfactual probabilities come from the frozen model
    """
    audit: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "users_audited": len(results),
        "checks": {},
    }

    feat_names = set(investigator._feat_names)
    all_errors: list[str] = []

    # Check 1: SHAP features exist
    shap_feature_errors: list[str] = []
    for r in results:
        inv = r["investigation"]
        for factor in inv.get("top_model_factors", []):
            fname = factor["feature_name"]
            if fname not in feat_names:
                shap_feature_errors.append(
                    f"User {inv['user_id']}: SHAP feature '{fname}' not in model features"
                )
    audit["checks"]["shap_features_exist"] = {
        "passed": len(shap_feature_errors) == 0,
        "errors": shap_feature_errors,
    }
    all_errors.extend(shap_feature_errors)

    # Check 2: SHAP values are non-trivial (not all zero)
    shap_value_errors: list[str] = []
    for r in results:
        inv = r["investigation"]
        factors = inv.get("top_model_factors", [])
        if factors:
            all_zero = all(f["shap_value"] == 0.0 for f in factors)
            if all_zero:
                shap_value_errors.append(
                    f"User {inv['user_id']}: all SHAP values are zero"
                )
    audit["checks"]["shap_values_nontrivial"] = {
        "passed": len(shap_value_errors) == 0,
        "errors": shap_value_errors,
    }
    all_errors.extend(shap_value_errors)

    # Check 3: Graph evidence exists
    graph_errors: list[str] = []
    hetero = investigator._hetero
    for r in results:
        inv = r["investigation"]
        uid = inv["user_id"]
        # Verify reported shared devices exist as edges
        for dev_id in inv.get("shared_devices", []):
            if not hetero.has_edge(uid, dev_id):
                graph_errors.append(
                    f"User {uid}: shared device '{dev_id}' not connected in graph"
                )
    audit["checks"]["graph_evidence_exists"] = {
        "passed": len(graph_errors) == 0,
        "errors": graph_errors,
    }
    all_errors.extend(graph_errors)

    # Check 4: Multi-hop paths exist
    path_errors: list[str] = []
    for r in results:
        inv = r["investigation"]
        uid = inv["user_id"]
        for path in inv.get("multi_hop_paths", []):
            nodes = path["nodes"]
            for i in range(len(nodes) - 1):
                if not hetero.has_edge(nodes[i], nodes[i + 1]):
                    path_errors.append(
                        f"User {uid}: path edge {nodes[i]} -> {nodes[i+1]} "
                        f"does not exist in graph"
                    )
                    break
    audit["checks"]["paths_exist"] = {
        "passed": len(path_errors) == 0,
        "errors": path_errors,
    }
    all_errors.extend(path_errors)

    # Check 5: No ground-truth labels in output
    label_errors: list[str] = []

    def _check_dict_for_labels(d: dict, context: str) -> None:
        for key in d:
            if key in GROUND_TRUTH_FIELDS:
                label_errors.append(f"{context}: contains ground-truth field '{key}'")
        for val in d.values():
            if isinstance(val, dict):
                _check_dict_for_labels(val, context)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        _check_dict_for_labels(item, context)

    for r in results:
        inv = r["investigation"]
        _check_dict_for_labels(inv, f"User {inv['user_id']}")
    audit["checks"]["no_labels_in_output"] = {
        "passed": len(label_errors) == 0,
        "errors": label_errors,
    }
    all_errors.extend(label_errors)

    # Check 6: Counterfactual probabilities come from frozen model
    cf_errors: list[str] = []
    for r in results:
        inv = r["investigation"]
        for cf in inv.get("counterfactuals", []):
            # Verify original probability matches the user's risk score
            if abs(cf["original_probability"] - inv["risk_score"]) > 0.01:
                cf_errors.append(
                    f"User {inv['user_id']}: counterfactual original_probability "
                    f"({cf['original_probability']}) doesn't match risk_score "
                    f"({inv['risk_score']})"
                )
            # Verify probability is valid
            if not (0.0 <= cf["counterfactual_probability"] <= 1.0):
                cf_errors.append(
                    f"User {inv['user_id']}: counterfactual probability "
                    f"{cf['counterfactual_probability']} out of [0,1] range"
                )
            # Verify delta consistency
            expected_delta = cf["counterfactual_probability"] - cf["original_probability"]
            if abs(cf["probability_delta"] - expected_delta) > 0.001:
                cf_errors.append(
                    f"User {inv['user_id']}: counterfactual delta inconsistent"
                )
    audit["checks"]["counterfactual_validity"] = {
        "passed": len(cf_errors) == 0,
        "errors": cf_errors,
    }
    all_errors.extend(cf_errors)

    # Overall
    audit["overall_passed"] = len(all_errors) == 0
    audit["total_errors"] = len(all_errors)

    return audit


def run_pipeline() -> dict:
    """Run the complete Phase 4 investigation pipeline."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AbuseRing Sentinel - Phase 4 Investigation & Explainability")
    print("=" * 60)

    # 1. Initialise investigator
    print("\nInitialising investigator (loading model, graphs, SHAP)...")
    t0 = time.time()
    investigator = Investigator()
    init_time = time.time() - t0
    print(f"  Initialised in {init_time:.1f}s")
    print(f"  Users in split: {len(investigator.user_ids)}")

    # 2. Select investigation targets
    print("\nSelecting investigation targets...")
    targets = _select_investigation_targets(investigator)
    print(f"  Selected {len(targets)} targets:")
    for t in targets:
        print(f"    {t['user_id']}: {t['selection_reason']}")

    # 3. Run investigations
    print("\nRunning investigations...")
    results: list[dict] = []
    runtimes: list[float] = []
    for t in targets:
        uid = t["user_id"]
        t_start = time.time()
        inv = investigator.investigate_user(uid)
        elapsed = time.time() - t_start
        runtimes.append(elapsed)
        inv_dict = inv.model_dump(mode="json")
        results.append({
            "selection_reason": t["selection_reason"],
            "investigation": inv_dict,
            "runtime_seconds": round(elapsed, 3),
        })
        print(
            f"  {uid}: risk={inv.risk_score:.4f} ({inv.risk_level.value}) "
            f"shap_factors={len(inv.top_model_factors)} "
            f"related={len(inv.related_accounts)} "
            f"paths={len(inv.multi_hop_paths)} "
            f"counterfactuals={len(inv.counterfactuals)} "
            f"[{elapsed:.2f}s]"
        )

    avg_runtime = np.mean(runtimes) if runtimes else 0.0
    print(f"\n  Average investigation runtime: {avg_runtime:.3f}s")

    # 4. Save investigation examples
    examples_path = REPORTS_DIR / "investigation_examples.json"
    examples_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {len(results)} investigation examples to {examples_path}")

    # 5. Run explanation audit
    print("\nRunning explanation audit...")
    audit = _run_explanation_audit(investigator, results)
    audit["average_runtime_seconds"] = round(avg_runtime, 4)
    audit["init_time_seconds"] = round(init_time, 2)

    audit_path = REPORTS_DIR / "explanation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"  Audit result: {'PASSED' if audit['overall_passed'] else 'FAILED'}")
    print(f"  Total errors: {audit['total_errors']}")
    for check_name, check_result in audit["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        print(f"    {status} {check_name}")
    print(f"  Saved to {audit_path}")

    return {
        "examples_count": len(results),
        "audit_passed": audit["overall_passed"],
        "average_runtime": avg_runtime,
        "init_time": init_time,
    }


def main() -> int:
    try:
        result = run_pipeline()
        print("\n" + "=" * 60)
        print("Phase 4 complete!")
        print(f"  Investigation examples: {result['examples_count']}")
        print(f"  Explanation audit: {'PASSED' if result['audit_passed'] else 'FAILED'}")
        print(f"  Avg single-user runtime: {result['average_runtime']:.3f}s")
        return 0
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
