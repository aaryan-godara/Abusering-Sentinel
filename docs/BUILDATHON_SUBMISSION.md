# Buildathon Submission Summary

### Project Title
AbuseRing Sentinel: Graph Intelligence for Coordinated Fraud Detection

### Project Objective
To detect organized fraud and abuse rings by analyzing the structural graph relationships (shared devices, IPs, physical addresses) between users, combining these topological features with traditional behavioral machine learning.

### AI Innovation
Standard fraud models evaluate users individually, missing coordinated abuse where each account looks relatively ordinary. AbuseRing Sentinel bridges this gap by constructing a heterogeneous graph, projecting it into user-to-user edges based on shared entities, extracting deep structural features (centralities, Louvain communities, two-hop reach), and integrating them directly into an XGBoost classifier. We further innovate by providing a custom `Investigator` facade that unifies SHAP explainability with graph path extraction and counterfactual analysis.

### Build Challenges
- **Synthetic Data Generation**: Constructing a realistic, massive synthetic dataset that accurately models both malicious fraud rings and massive legitimate shared infrastructure (offices, campuses) was a significant engineering challenge.
- **Leakage Prevention**: Ensuring that ground truth labels did not leak into the graph structural features required designing a rigorous group-aware Train/Val/Test splitting algorithm.
- **Promotion Mega-Hub Distortion**: Early graph builds were completely distorted by "mega-hub" nodes (e.g., a massive merchant or promotion code). We implemented degree-capping during user projection to prune uninformative heavy hitters.
- **Explainability UX**: Surfacing complex ML probabilities, graph data, and SHAP values in a digestible React dashboard required careful API design and state management.

### Verified Results
On the completely isolated, synthetic test set:
- **Model C (Graph + Behavioral XGBoost)**: F1: 0.7901, ROC-AUC: 0.9724.
- **Ring Detection**: Successfully detected 80% (12 of 15) of hidden test rings.
- **Performance**: The frontend bundles to ~164 kB gzipped, and the backend `/investigation` API resolves in ~9.6ms average runtime per user.

### Demo Story
Our demo walks judges through four pre-configured scenarios directly in the React dashboard:
1. A dense, obvious shared-infrastructure fraud ring.
2. A sprawling, complex, multi-hop related account set.
3. A subtle ring attempting to evade detection via multi-entity variation.
4. A legitimate high-connectivity user, proving the model avoids naive false positives on shared campus IPs.

### Limitations & Future Scope
The primary limitation is the use of synthetic data, meaning real-world noise and adversarial adaptation are not fully captured. Future scope includes transitioning the static graph to a streaming Kafka architecture, exploring Graph Neural Networks (GNNs) for automated embedding generation, and implementing active learning feedback loops for the risk review queue.
