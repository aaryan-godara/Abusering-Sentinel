import json

with open("reports/model_comparison.json") as f:
    d = json.load(f)

print("=== Model Comparison ===")
for k, v in d["model_comparison"].items():
    t = v["test"]
    vld = v["validation"]
    print(f'{k}:')
    print(f'  val:  F1={vld["f1"]:.4f} PR-AUC={vld["pr_auc"]:.4f} ROC-AUC={vld["roc_auc"]:.4f} P={vld["precision"]:.4f} R={vld["recall"]:.4f}')
    print(f'  test: F1={t["f1"]:.4f} PR-AUC={t["pr_auc"]:.4f} ROC-AUC={t["roc_auc"]:.4f} P={t["precision"]:.4f} R={t["recall"]:.4f}')
    print(f'  threshold: {v["threshold"]:.4f}')
    print(f'  CI: {v["bootstrap_ci"]}')
    print()

print("=== Cost Analysis ===")
for k, v in d["cost_analysis"].items():
    print(f'  {k}: {v}')

print("\n=== Hidden Scenarios ===")
for m in ["A", "B", "C"]:
    print(f'  {m}: {d["hidden_scenarios"][m]}')

print("\n=== Ring Detection (Model C) ===")
c = d["ring_detection"]["C"]
print(f'Overall: {c["overall_detection_rate"]:.2%} ({c["rings_detected"]}/{c["total_rings"]})')
for rt, v in c["by_type"].items():
    print(f'  {rt}: {v["detected"]}/{v["total"]} = {v["detection_rate"]:.2%}')

print("\n=== Ablation Experiments ===")
with open("reports/graph_ablation.json") as f:
    abl = json.load(f)
for r in abl:
    print(f'{r["experiment"]}: feat={r["feature_count"]} test_F1={r["test_metrics"]["f1"]:.4f} test_PR-AUC={r["test_metrics"]["pr_auc"]:.4f}')