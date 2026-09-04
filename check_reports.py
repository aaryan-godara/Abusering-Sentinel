import json

# Check graph ablation
with open("reports/graph_ablation.json") as f:
    d = json.load(f)
print("=== Graph Ablation ===")
for r in d:
    print(f'{r["experiment"]}: feat={r["feature_count"]} test_F1={r["test_metrics"]["f1"]:.4f} test_PR-AUC={r["test_metrics"]["pr_auc"]:.4f}')

# Check feature importance
print("\n=== Feature Importance (top 20 by gain) ===")
with open("reports/feature_importance.json") as f:
    d = json.load(f)
for i, (feat, imp) in enumerate(list(d["gain"].items())[:20]):
    print(f'{i+1}. {feat}: {imp:.4f}')

# Check error analysis
print("\n=== Error Analysis ===")
with open("reports/error_analysis.json") as f:
    d = json.load(f)
print(f"FP by group: {d['fp_by_legit_group']}")
print(f"FN by ring type: {d['fn_by_ring_type']}")
print(f"FP rate by legit group: {d['fp_rate_by_legit_group']}")
print(f"FN recall by ring type: {d['fn_recall_by_ring_type']}")

# Check hidden scenarios
print("\n=== Hidden Scenarios (Model C) ===")
with open("reports/model_comparison.json") as f:
    d = json.load(f)
for m in ["A", "B", "C"]:
    print(f"Model {m}: {d['hidden_scenarios'][m]}")

# Check ring detection
print("\n=== Ring Detection (Model C) ===")
with open("reports/model_comparison.json") as f:
    d = json.load(f)
c = d["ring_detection"]["C"]
print(f"Overall: {c['overall_detection_rate']:.2%} ({c['rings_detected']}/{c['total_rings']})")
for rt, v in c["by_type"].items():
    print(f"  {rt}: {v['detected']}/{v['total']} = {v['detection_rate']:.2%}")