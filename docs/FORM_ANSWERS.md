# Form Answers

**Project Name / Title**
AbuseRing Sentinel: Graph Intelligence for Coordinated Fraud Detection

**Project Objectives — What does it solve?**
Single-account fraud models completely miss organized abuse rings where individual accounts look legitimate in isolation. AbuseRing Sentinel solves this by modeling users, devices, IPs, and transactions as a heterogeneous graph, extracting deep structural features (centrality, community density), and combining them with behavioral ML. This detects the coordinated relationships driving fraud, while avoiding false positives on legitimate shared infrastructure like university campuses.

**GitHub Repository URL**
[PASTE YOUR GITHUB REPOSITORY URL]

**5-min Pitch Video Link**
[PASTE YOUR 5-MINUTE PITCH VIDEO URL]

**Build Challenges & Technical Obstacles**
- **Synthetic Data Generation**: Constructing a massive, deterministic synthetic dataset that accurately modeled both subtle fraud rings and highly-connected legitimate structures (offices/campuses) required complex NumPy weighting.
- **Leakage Prevention**: Preventing ground-truth ring IDs from leaking into graph features required a custom, group-aware Train/Val/Test splitting algorithm.
- **Graph Mega-Hubs**: Naive graph projection collapsed due to "mega-hubs" (massive merchants or common promotions). We had to implement strict degree-capping to prune uninformative nodes.
- **Explainability**: Fusing the output of the XGBoost classifier, SHAP explainers, and real-time graph path extraction into a single, cohesive, sub-10ms API response required significant engineering in the `Investigator` facade.
- **False Positive Avoidance**: Tuning the model to differentiate between a fraud ring sharing an IP and a legitimate university dorm required extensive feature ablation.
- **Integration**: Correcting frontend build issues (TypeScript) and missing backend dependencies (`xgboost`) under time constraints to deliver a flawless zero-error production build.

**Final Submission Confirmation**
Yes, the project is complete, fully tested, and ready for judging.
