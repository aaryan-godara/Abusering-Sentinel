# Technical Explanation

This document details the technical pipeline driving AbuseRing Sentinel.

## 1. Data Generation and Split Isolation
Because coordinated abuse signals are inherently relational, testing requires massive structured overlap. We use a deterministic synthetic pipeline:
- **Legitimate Structure**: ~90% of the dataset consists of independent users or grouped legitimate users (e.g., families or university campuses sharing IPs and devices).
- **Ring Injection**: Hidden abuse rings are injected under four distinct archetypes (direct infrastructure, promotion coordination, multi-hop chains, distributed behavior).
- **Group-Aware Splitting**: A purely random train/test split would leak ring structures (part of a ring in train, part in test). We perform group-aware splitting to ensure entire rings are pinned to a single split, guaranteeing strict separation.

## 2. Graph Construction
We build an undirected heterogeneous graph using `networkx`. 
- **Nodes**: `user`, `device`, `ip`, `address`, `payment_instrument`, `merchant`, `promotion`, `transaction`.
- **Edges**: Connect a user to their transacted entities.
- **User Projection**: Two users are connected via an edge that preserves the exact count of shared entities (e.g., `shared_device_count = 2`).

## 3. Feature Engineering
Features fall into two primary buckets:
1. **Behavioral Baselines**: Metrics like transaction frequency, average amount, standard deviation of amounts, and specific time-bound counts (last 7/30 days).
2. **Structural Graph Features**:
   - *Direct*: Unique devices, IPs, addresses used.
   - *Neighbor*: Average degree of neighbor users.
   - *Two-Hop*: Count of users reachable via two hops.
   - *Centrality*: Exact (or sampled) Brandes betweenness centrality and clustering coefficients.
   - *Community*: NetworkX Louvain community sizes and density (computed completely blind to the abuse label).

## 4. Leakage Prevention
Rigorous automated leakage audits (`reports/leakage_audit.json`) prevent target leakage. Ground truth fields (e.g., `is_abuse_account`, `ring_id`) are structurally excluded from the feature space. 

## 5. Model Training & Evaluation
We compared three paradigms:
- **Model A**: Logistic Regression (Baseline)
- **Model B**: XGBoost (Baseline behavioral)
- **Model C**: XGBoost (Baseline + Graph)

**Model C** decisively outperforms the others by leveraging the structural overlap that standard behavioral models miss. It achieved an F1 of 0.7901 and ROC-AUC of 0.9724 on the strictly isolated test set.

## 6. Explainability (XAI)
To make the model actionable for risk investigators:
- **SHAP Integration**: We compute SHAP values for the prediction, extracting the top features pulling the score toward abuse.
- **Graph Evidence**: The explainer automatically extracts the specific entities (e.g., "Device 405") causing the structural features to spike.
- **Counterfactuals**: The system answers "What if?" scenarios, estimating the new probability if a specific feature were removed.

## 7. Risk Decisions
The `RiskEngine` abstracts the continuous probability into discrete operational states according to a predefined `DefaultRiskPolicy`. High-risk users are auto-blocked; medium-risk users are routed to a prioritized `ReviewQueue` for human intervention.
