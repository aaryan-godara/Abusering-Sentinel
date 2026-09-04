"""Phase 3 model training and evaluation pipeline.

Orchestrates the complete experiment:
1. Load feature datasets
2. Run label-leakage audit
3. Train Model A (Logistic Regression + baseline)
4. Train Model B (XGBoost + baseline)
5. Train Model C (XGBoost + baseline + graph)
6. Select thresholds on VALIDATION only
7. Evaluate on TEST (once, frozen)
8. Run ablation experiments
9. Generate all reports
10. Save models

Run with:
    python -m src.models.pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.config.generation import GenerationConfig
from src.models.ablation import (
    ABLATION_EXPERIMENTS,
    AblationResult,
    get_experiment_feature_columns,
    get_feature_indices,
    FEATURE_GROUPS,
)
from src.models.baseline import build_logistic_model, fit_logistic, predict_proba
from src.models.evaluation import (
    MetricBundle,
    CostConfig,
    compute_metrics,
    compute_cost,
    find_cost_optimal_threshold,
    bootstrap_metric_bundle_ci,
    evaluate_rings,
    ring_detection_summary,
    evaluate_hidden_scenarios,
    RingEvalResult,
)
from src.models.error_analysis import (
    collect_fp_analysis,
    collect_fn_analysis,
    summarize_fps_by_group,
    fn_recall_by_ring_type,
    fp_rate_by_legit_group,
)
from src.models.explainability import get_xgb_importance, compute_shap_values, shap_summary
from src.models.preprocessing import (
    load_feature_split,
    audit_features,
    get_feature_lists,
    load_split_data,
    verify_split_identities,
    get_feature_matrix,
)
from src.models.thresholds import select_threshold_max_f1, select_threshold
from src.models.xgboost_model import build_xgboost_model, fit_xgboost, predict_proba_xgb, get_feature_importance

SPLITS: tuple[str, ...] = ("train", "val", "test")
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42

# --- Cost configuration (synthetic, not real Razorpay data) ---
DEFAULT_COST_CONFIG = CostConfig(
    fp_cost_per_incident=5000.0,   # INR per false positive
    fn_cost_per_incident=25000.0,  # INR per false negative
    avg_txn_value=5000.0,
)


def _load_split_data(split: str, variant: str = "graph"):
    """Load X, y, feature_names for a split/variant."""
    return load_split_data(split, variant)


def _audit_split(variant: str, split: str, df: pd.DataFrame):
    """Audit one split for label leakage."""
    errs = audit_features(df, variant, split)
    if errs:
        raise ValueError(f"Label audit failed for {variant}/{split}:\n" + "\n".join(errs))


def _prepare_matrices(variant: str) -> dict[str, dict]:
    """Load all splits for a variant and return dict with X, y, feat_names."""
    result = {}
    for split in SPLITS:
        df = load_feature_split(split, variant)
        _audit_split(variant, split, df)

        feat_names = get_feature_lists(variant)
        X, y = get_feature_matrix(df, feat_names)
        result[split] = {"X": X, "y": y, "features": feat_names, "df": df}
    return result


def train_model_a(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train Model A: Logistic Regression on baseline features."""
    from src.models.baseline import build_logistic_model, fit_logistic, predict_proba

    model = build_logistic_model(random_state=RANDOM_SEED, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_model_bc(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scale_pos_weight: float,
    feature_names: list[str] | None = None,
):
    """Train XGBoost (Model B or C) with validation early stopping."""
    from src.models.xgboost_model import build_xgboost_model, predict_proba_xgb

    model = build_xgboost_model(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    model.fit(X_train, y_train, **fit_kwargs)
    # Set feature names on booster for proper feature importance
    if feature_names is not None:
        model.get_booster().feature_names = feature_names
    return model


def run_full_pipeline(
    data_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run the complete Phase 3 experiment."""
    data_dir = data_dir or (PROJECT_ROOT / "data" / "processed")
    out_dir = out_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AbuseRing Sentinel - Phase 3 Model Training")
    print("=" * 60)

    # 1. Load Phase 1 tables for ring evaluation
    from src.validation.loader import load_tables
    tables = load_tables(data_dir)
    users_df = tables["users"]

    # 2. Verify split identities match Phase 1
    split_ids, split_labels = verify_split_identities(data_dir)
    print("Split user counts:", {s: len(ids) for s, ids in split_ids.items()})

    # 3. Load feature datasets
    print("\nLoading feature datasets...")
    baseline_data = _prepare_matrices("baseline")
    graph_data = _prepare_matrices("graph")

    X_train_b = baseline_data["train"]["X"]
    y_train_b = baseline_data["train"]["y"]
    X_val_b = baseline_data["val"]["X"]
    y_val_b = baseline_data["val"]["y"]
    X_test_b = baseline_data["test"]["X"]
    y_test_b = baseline_data["test"]["y"]
    feat_names_b = baseline_data["train"]["features"]

    X_train_g = graph_data["train"]["X"]
    y_train_g = graph_data["train"]["y"]
    X_val_g = graph_data["val"]["X"]
    y_val_g = graph_data["val"]["y"]
    X_test_g = graph_data["test"]["X"]
    y_test_g = graph_data["test"]["y"]
    feat_names_g = graph_data["train"]["features"]
    train_df_g = graph_data["train"]["df"]
    val_df_g = graph_data["val"]["df"]
    test_df_g = graph_data["test"]["df"]

    # Verify targets match
    assert np.array_equal(y_train_b, y_train_g)
    assert np.array_equal(y_val_b, y_val_g)
    assert np.array_equal(y_test_b, y_test_g)

    # 4. Compute scale_pos_weight
    n_pos = int(y_train_b.sum())
    n_neg = len(y_train_b) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"Train: {n_pos} abuse / {n_neg} legit (scale_pos_weight={scale_pos_weight:.2f})")

    # 5. Train all three models
    print("\n--- Training Models ---")

    # Model A: Logistic Regression (baseline only)
    print("Training Model A (Logistic Regression, baseline)...")
    t0 = time.time()
    model_a = train_model_a(X_train_b, y_train_b, X_val_b, y_val_b)
    time_a = time.time() - t0
    print(f"  done in {time_a:.1f}s")

    # Model B: XGBoost baseline
    print("Training Model B (XGBoost, baseline)...")
    t0 = time.time()
    model_b = train_model_bc(X_train_b, y_train_b, X_val_b, y_val_b, scale_pos_weight, feat_names_b)
    time_b = time.time() - t0
    print(f"  done in {time_b:.1f}s")

    # Model C: XGBoost + graph
    print("Training Model C (XGBoost, baseline + graph)...")
    t0 = time.time()
    model_c = train_model_bc(X_train_g, y_train_g, X_val_g, y_val_g, scale_pos_weight, feat_names_g)
    time_c = time.time() - t0
    print(f"  done in {time_c:.1f}s")

    # 6. Get validation probabilities
    from src.models.baseline import predict_proba
    from src.models.xgboost_model import predict_proba_xgb

    val_prob_a = predict_proba(model_a, X_val_b)
    val_prob_b = predict_proba_xgb(model_b, X_val_b)
    val_prob_c = predict_proba_xgb(model_c, X_val_g)

    # 7. Select thresholds on VALIDATION only
    print("\n--- Threshold Selection (validation only) ---")
    thresh_a, res_a = select_threshold_max_f1(y_val_b, val_prob_a)
    thresh_b, res_b = select_threshold_max_f1(y_val_b, val_prob_b)
    thresh_c, res_c = select_threshold_max_f1(y_val_g, val_prob_c)

    print(f"  Model A threshold: {thresh_a:.4f} (F1={res_a.f1:.4f}, P={res_a.precision:.4f}, R={res_a.recall:.4f})")
    print(f"  Model B threshold: {thresh_b:.4f} (F1={res_b.f1:.4f}, P={res_b.precision:.4f}, R={res_b.recall:.4f})")
    print(f"  Model C threshold: {thresh_c:.4f} (F1={res_c.f1:.4f}, P={res_c.precision:.4f}, R={res_c.recall:.4f})")

    # 8. Final TEST evaluation (once, frozen)
    print("\n--- Final Test Evaluation (frozen) ---")
    test_prob_a = predict_proba(model_a, X_test_b)
    test_prob_b = predict_proba_xgb(model_b, X_test_b)
    test_prob_c = predict_proba_xgb(model_c, X_test_g)

    metrics_a_test = compute_metrics(y_test_b, test_prob_a, thresh_a)
    metrics_b_test = compute_metrics(y_test_b, test_prob_b, thresh_b)
    metrics_c_test = compute_metrics(y_test_g, test_prob_c, thresh_c)

    # Bootstrap CIs
    ci_a = bootstrap_metric_bundle_ci(y_test_b, test_prob_a, thresh_a, n_bootstrap=500)
    ci_b = bootstrap_metric_bundle_ci(y_test_b, test_prob_b, thresh_b, n_bootstrap=500)
    ci_c = bootstrap_metric_bundle_ci(y_test_g, test_prob_c, thresh_c, n_bootstrap=500)

    # 9. Cost analysis
    cost_config = DEFAULT_COST_CONFIG
    cost_a = compute_cost(y_test_b, (test_prob_a >= thresh_a).astype(int), cost_config)
    cost_b = compute_cost(y_test_b, (test_prob_b >= thresh_b).astype(int), cost_config)
    cost_c = compute_cost(y_test_g, (test_prob_c >= thresh_c).astype(int), cost_config)

    # 9b. Ring-level evaluation
    test_users = tables["users"][tables["users"]["split"] == "test"].copy()
    test_users.index = test_users["user_id"]

    ring_results_a = evaluate_rings(test_users, test_prob_a, thresh_a)
    ring_results_b = evaluate_rings(test_users, test_prob_b, thresh_b)
    ring_results_c = evaluate_rings(test_users, test_prob_c, thresh_c)

    ring_summary_a = ring_detection_summary(ring_results_a)
    ring_summary_b = ring_detection_summary(ring_results_b)
    ring_summary_c = ring_detection_summary(ring_results_c)

    # 9c. Hidden scenario evaluation
    hidden_a = evaluate_hidden_scenarios(test_users, test_prob_a, thresh_a)
    hidden_b = evaluate_hidden_scenarios(test_users, test_prob_b, thresh_b)
    hidden_c = evaluate_hidden_scenarios(test_users, test_prob_c, thresh_c)

    # 10. Error analysis (Model C - final model)
    test_pred_c = (test_prob_c >= thresh_c).astype(int)
    # Use original users table (test_users) for error analysis since it has ring_type, group_type, etc.
    fps = collect_fp_analysis(test_users, y_test_g, test_prob_c, test_pred_c, feat_names_g)
    fns = collect_fn_analysis(test_users, y_test_g, test_prob_c, test_pred_c, feat_names_g)

    fp_by_group = summarize_fps_by_group(fps, test_users)
    fn_by_ring = fn_recall_by_ring_type(test_users, y_test_g, test_pred_c)
    fp_rate_groups = fp_rate_by_legit_group(test_users, y_test_g, test_pred_c)
    fn_recall_ring = fn_recall_by_ring_type(test_users, y_test_g, test_pred_c)

    # 11. Feature importance (Model C)
    imp_gain = get_xgb_importance(model_c, method="gain", top_k=30, feature_names=feat_names_g)
    imp_weight = get_xgb_importance(model_c, method="weight", top_k=30, feature_names=feat_names_g)

    # 12. SHAP (sampled)
    shap_vals = compute_shap_values(model_c, X_test_g[:1000])
    shap_sum = shap_summary(shap_vals, feat_names_g) if shap_vals is not None else {}

    # 13. Ablation experiments
    print("\n--- Ablation Experiments ---")
    ablation_results = []
    full_feat_names = feat_names_g
    full_X_train = X_train_g
    full_X_val = X_val_g
    full_X_test = X_test_g
    full_y_train = y_train_g
    full_y_val = y_val_g
    full_y_test = y_test_g

    for exp in ABLATION_EXPERIMENTS:
        print(f"  Running ablation: {exp.name}...")
        t0 = time.time()
        cols = get_experiment_feature_columns(exp.feature_groups)
        idx = get_feature_indices(full_feat_names, exp.feature_groups)

        Xtr = full_X_train[:, idx]
        Xva = full_X_val[:, idx]
        Xte = full_X_test[:, idx]

        exp_feat_names = [full_feat_names[i] for i in idx]
        model = build_xgboost_model(scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED)
        model.fit(
            Xtr, full_y_train,
            eval_set=[(Xva, full_y_val)],
            verbose=False,
        )
        # Ensure feature names are set on the booster for feature importance
        model.get_booster().feature_names = exp_feat_names
        val_prob = predict_proba_xgb(model, Xva)
        thresh, _ = select_threshold_max_f1(full_y_val, val_prob)
        test_prob = predict_proba_xgb(model, Xte)
        val_metrics = compute_metrics(full_y_val, val_prob, thresh)
        test_metrics = compute_metrics(full_y_test, test_prob, thresh)

        ablation_results.append(AblationResult(
            experiment=exp.name,
            feature_groups=exp.feature_groups,
            feature_count=len(idx),
            validation_metrics=asdict(compute_metrics(full_y_val, val_prob, thresh)),
            test_metrics=asdict(compute_metrics(full_y_test, test_prob, thresh)),
            training_time_seconds=0.0,
            selected_threshold=thresh,
        ))
        print(f"    {exp.name}: test F1={test_metrics.f1:.4f}, PR-AUC={test_metrics.pr_auc:.4f}")

    # 14. Compile final results
    print("\n--- Compiling Reports ---")
    results = {
        "experiment_manifest": {
            "random_seed": RANDOM_SEED,
            "dataset_version": GenerationConfig().dataset_version,
            "train_users": int(len(y_train_b)),
            "val_users": int(len(y_val_b)),
            "test_users": int(len(y_test_b)),
            "abuse_rate_train": float(y_train_b.mean()),
            "abuse_rate_test": float(y_test_b.mean()),
            "models": {
                "A": "LogisticRegression (baseline)",
                "B": "XGBoost (baseline)",
                "C": "XGBoost (baseline + graph)",
            },
            "selected_thresholds": {
                "A": round(thresh_a, 4),
                "B": round(thresh_b, 4),
                "C": round(thresh_c, 4),
            },
            "selected_strategy": "max_f1 on validation",
            "cost_config": {"fp_cost_per_incident": 5000.0, "fn_cost_per_incident": 25000.0, "avg_txn_value": 5000.0},
        },
        "model_comparison": {
            "A": {
                "validation": asdict(compute_metrics(y_val_b, val_prob_a, thresh_a)),
                "test": asdict(compute_metrics(y_test_b, test_prob_a, thresh_a)),
                "bootstrap_ci": ci_a,
                "threshold": round(thresh_a, 4),
                "training_time_s": 0.0,
            },
            "B": {
                "validation": asdict(compute_metrics(y_val_b, val_prob_b, thresh_b)),
                "test": asdict(compute_metrics(y_test_b, test_prob_b, thresh_b)),
                "bootstrap_ci": ci_b,
                "threshold": round(thresh_b, 4),
                "training_time_s": 0.0,
            },
            "C": {
                "validation": asdict(compute_metrics(y_val_g, val_prob_c, thresh_c)),
                "test": asdict(compute_metrics(y_test_g, test_prob_c, thresh_c)),
                "bootstrap_ci": ci_c,
                "threshold": round(thresh_c, 4),
                "training_time_s": 0.0,
            },
        },
        "graph_ablation": [asdict(r) for r in ablation_results],
        "feature_importance": {
            "gain": imp_gain.importances,
            "weight": imp_weight.importances,
            "grouped_by_type": imp_gain.grouped_by_type(),
        },
        "shap_summary": shap_sum,
        "cost_analysis": {
            "A": cost_a.to_dict(),
            "B": cost_b.to_dict(),
            "C": cost_c.to_dict(),
        },
        "ring_detection": {
            "A": ring_summary_a,
            "B": ring_summary_b,
            "C": ring_summary_c,
        },
        "hidden_scenarios": {
            "A": hidden_a,
            "B": hidden_b,
            "C": hidden_c,
        },
        "error_analysis": {
            "false_positives": [
                {
                    "user_id": fp.user_id,
                    "probability": fp.probability,
                    "group_type": fp.group_type,
                    "group_id": fp.group_id,
                } for fp in fps[:20]
            ],
            "false_negatives": [
                {
                    "user_id": fn.user_id,
                    "probability": fn.probability,
                    "ring_id": fn.ring_id,
                    "ring_type": fn.ring_type,
                } for fn in fns[:20]
            ],
            "fp_by_legit_group": fp_by_group,
            "fn_by_ring_type": fn_by_ring,
            "fp_rate_by_legit_group": fp_rate_groups,
            "fn_recall_by_ring_type": fn_recall_ring,
        },
    }

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2, default=str))
    (REPORTS_DIR / "graph_ablation.json").write_text(json.dumps([asdict(r) for r in ablation_results], indent=2))
    (REPORTS_DIR / "feature_importance.json").write_text(json.dumps({
        "gain": imp_gain.importances,
        "weight": imp_weight.importances,
        "grouped_by_type": imp_gain.grouped_by_type(),
    }, indent=2))
    (REPORTS_DIR / "shap_summary.json").write_text(json.dumps(shap_sum, indent=2))
    (REPORTS_DIR / "error_analysis.json").write_text(json.dumps({
        "false_positives": [
            {
                "user_id": fp.user_id,
                "probability": fp.probability,
                "group_type": fp.group_type,
                "group_id": fp.group_id,
            } for fp in fps[:20]
        ],
        "false_negatives": [
            {
                "user_id": fn.user_id,
                "probability": fn.probability,
                "ring_id": fn.ring_id,
                "ring_type": fn.ring_type,
            } for fn in fns[:20]
        ],
        "fp_by_legit_group": fp_by_group,
        "fn_by_ring_type": fn_by_ring,
        "fp_rate_by_legit_group": fp_rate_groups,
        "fn_recall_by_ring_type": fn_recall_ring,
    }, indent=2))
    (REPORTS_DIR / "experiment_manifest.json").write_text(json.dumps({
        "random_seed": RANDOM_SEED,
        "dataset_version": GenerationConfig().dataset_version,
        "train_users": int(len(y_train_b)),
        "val_users": int(len(y_val_b)),
        "test_users": int(len(y_test_b)),
        "abuse_rate_train": float(y_train_b.mean()),
        "abuse_rate_test": float(y_test_b.mean()),
        "models": {
            "A": "LogisticRegression (baseline)",
            "B": "XGBoost (baseline)",
            "C": "XGBoost (baseline + graph)",
        },
        "selected_thresholds": {
            "A": round(thresh_a, 4),
            "B": round(thresh_b, 4),
            "C": round(thresh_c, 4),
        },
        "selected_strategy": "max_f1 on validation",
        "cost_config": {"fp_cost_per_incident": 5000.0, "fn_cost_per_incident": 25000.0, "avg_txn_value": 5000.0},
    }, indent=2, default=str))

    # Save models
    out_dir = out_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model_a, out_dir / "logistic_baseline.joblib")
    joblib.dump(model_b, out_dir / "xgboost_baseline.joblib")
    joblib.dump(model_c, out_dir / "xgboost_graph.joblib")

    print("\n" + "=" * 60)
    print("Phase 3 complete!")
    print(f"Reports written to {REPORTS_DIR}")
    print(f"Models saved to {out_dir}")
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 model training and evaluation.")
    parser.add_argument("--data", type=str, default=None, help="Data directory")
    parser.add_argument("--out", type=str, default=None, help="Model output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data) if args.data else None
    out_dir = Path(args.out) if args.out else None

    try:
        run_full_pipeline(data_dir, out_dir)
        return 0
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_full_pipeline", "main"]