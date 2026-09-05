# AbuseRing Sentinel
**Graph Intelligence for Coordinated Fraud Detection**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![React](https://img.shields.io/badge/React-18-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange) ![Status](https://img.shields.io/badge/Status-Complete-success)

AbuseRing Sentinel is an end-to-end machine learning and graph intelligence platform designed to detect coordinated fraud and abuse rings. By combining behavioral ML with heterogeneous graph relationships, it identifies groups of accounts that appear independent in isolation but are linked through shared infrastructure and behavior.

---

### Main Dashboard

<img width="1917" height="962" alt="image" src="https://github.com/user-attachments/assets/1e7bcd6b-dcc5-498d-98b2-bf1d30e80d45" />

### Investigation Workspace

<img width="1917" height="852" alt="image" src="https://github.com/user-attachments/assets/af91d948-7c15-443f-bdee-8f33270ac2c3" />




## 🛑 Problem Statement
Single-account fraud rules miss organized abuse. A ring operator registers many accounts, each individually unremarkable, then farms promotions, launders funds, or resells credit. The signal is not in any one account but in the *relationships* between users, devices, IPs, and transactions.

## ⚠️ Why Traditional Detection Fails
Traditional models evaluate users individually. A user checking out with a clean device and no prior chargebacks looks perfectly legitimate to a standard ML model. By the time enough chargebacks occur to flag the account, the fraudster has already cycled to the next account in the ring. Simple rule-based overlap (e.g., "block all users on this IP") fails because legitimate families and university campuses also share infrastructure, leading to massive false positives.

## 💡 Solution Overview
AbuseRing Sentinel models all entities as a heterogeneous graph, derives relational features (centrality, clustering, multi-hop reach, community structure), and combines them with tabular behavioral machine learning to surface high-risk rings for investigation. 

### Core Workflow
**Detect** → **Explain** → **Trace** → **Investigate** → **Decide**

## 🚀 Main Capabilities
- **Behavioral ML**: Standard transactional baselines.
- **Graph Intelligence**: Deep structural features revealing multi-hop linkages and communities.
- **Explainable AI (XAI)**: SHAP value integration providing exact reasons for model scores.
- **Counterfactual Analysis**: Tells investigators exactly how probability would change if an entity were removed.
- **Risk Decision Engine**: Automated policy engine mapping risk scores to operational decisions.
- **Investigation Dashboard**: Visual exploration of user relationships and overlapping entities.

---

## 🏗️ Product Architecture
The system consists of a Python-based core engine, a FastAPI backend serving ML and graph insights, and a React (Vite) frontend for investigation.

For deep technical details, see the [Architecture Guide](docs/ARCHITECTURE.md).

## 📊 End-to-End Data Flow
1. **Generation**: Synthetic datasets are generated simulating users, transactions, and hidden abuse rings.
2. **Graph Construction**: A heterogeneous `networkx` graph connects all entities.
3. **Feature Engineering**: Features are computed per user (baseline behavioral + structural graph metrics).
4. **Model Training**: An XGBoost classifier is trained to detect abuse accounts.
5. **Operationalization**: FastAPI serves the Investigator module which runs SHAP explainers and extracts multi-hop paths.
6. **Review**: The Risk Engine assigns a priority score, placing users in a review queue accessed by the React frontend.

---

## 🧪 Synthetic Dataset Explanation
**All data in this project is strictly synthetic.** Entities, transactions, and abuse patterns are generated programmatically with Faker and NumPy. No real user data or financial information is used.

### Entity Types
- Users, Devices, IP Addresses, Physical Addresses, Payment Instruments, Promotions, Merchants, Transactions.

### Abuse-Ring Archetypes
1. **Direct infrastructure abuse**: High overlap on devices/IPs/addresses.
2. **Promotion coordination**: Minimal infrastructure overlap, heavy promotion abuse.
3. **Indirect multi-hop**: Chains where no single entity connects everyone.
4. **Distributed behavioral**: Minimal sharing, aligned behavioral transaction windows.

### Group-Aware Splitting & Leakage Prevention
Legitimate structures (families, offices, hostels) deliberately share infrastructure. To prevent data leakage, splitting is group-aware: whole rings and legitimate groups stay within a single split (Train, Val, or Test). Ground truth labels (`is_abuse_account`, `ring_id`) are structurally excluded from features.

---

## 🕸️ Graph Methodology

### Construction & Projection
The raw heterogeneous graph maps users to their transactions, devices, IPs, etc. We perform a **User Projection** where two users are linked when they share observable entities, preserving per-type shared counts (`shared_device_count`, `shared_ip_count`).

### Graph Features & Community Detection
- **Features**: Direct connection degree, neighbor statistics, two-hop reach, exact/sampled Brandes betweenness centrality.
- **Communities**: Detected via NetworkX's Louvain algorithm purely on graph structure, *without* any abuse labels.

---

## 🤖 Machine Learning

### ML Model Comparison
We evaluated three approaches:
- **Model A**: Logistic Regression (Baseline)
- **Model B**: XGBoost (Baseline behavioral features only)
- **Model C**: XGBoost (Baseline + Graph features)

### Final Model C Metrics (Test Set)
Graph features significantly improved detection, particularly on distributed and multi-hop rings.
- **F1 Score**: 0.7901
- **PR-AUC**: 0.8952
- **ROC-AUC**: 0.9724
- **Precision**: 0.9505
- **Recall**: 0.6761
- **Ring Detection Rate**: 80.00% (12 of 15 hidden test rings detected)

---

## 🔍 Investigation & Explainability

### Explainability Workflow
The `Investigator` facade runs the ML model and a SHAP explainer simultaneously. It extracts the top driving factors for the score and fetches the exact shared entities (devices, IPs) and multi-hop paths from the graph.

### Counterfactual Sensitivity Analysis
The system calculates counterfactuals (e.g., "If we ignore this shared device, the risk score drops by X%"), giving investigators confidence in the causal drivers of the score.

### Risk Decision Engine & Review Queue
The `RiskEngine` maps continuous ML probabilities to discrete actions (e.g., `AUTO_BLOCK`, `MANUAL_REVIEW`) via the `DefaultRiskPolicy`. Users requiring review are sent to the `ReviewQueue` sorted by priority.

---

## 🖥️ System Interfaces

### FastAPI Endpoints
- `GET /health`
- `GET /users/{user_id}/risk`
- `GET /users/{user_id}/investigation`
- `GET /users/{user_id}/evidence`
- `GET /review-queue`
- `GET /review-queue/top/{n}`

### Frontend Workflow
The investigator searches a `User ID` in the top bar. The dashboard fetches the investigation context, visualizes the risk score, surfaces SHAP explanations, and renders a fully interactive node-link graph of the user's infrastructure neighborhood.

---

## 🎬 Demo Scenarios
Four specific users from the test set are pre-configured to showcase the system's capabilities:
- **Example A: Dense Shared Infrastructure** (`USR_00000613`)
- **Example B: Large Related-Account Set** (`USR_00000522`)
- **Example C: Multi-Entity Overlap** (`USR_00000085`)
- **Example D: Legitimate High Connectivity** (`USR_00000646`) - A false positive avoidance showcase.

---

## ⚙️ Setup and Installation

### Project Structure
See `docs/ARCHITECTURE.md` for deep structural details. 
- `/src`: Backend API, generators, features, models, risk engine.
- `/frontend`: React + Vite application.
- `/tests`: Pytest suite.

### Backend Setup (Python 3.11+)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Frontend Setup
```powershell
cd frontend
npm install
```

### Environment Variables
Copy `.env.example` to `.env` in the root. (Variables: `RANDOM_SEED`, `ENVIRONMENT`, `DATA_DIRECTORY`).

---

## 🏃‍♂️ Running the System

### Start the API Backend
```powershell
python -m uvicorn src.api.main:app --reload
```

### Start the Frontend
```powershell
cd frontend
npm run dev
```

### Testing and Validation
```powershell
python -m pytest
npm run build
```
Current Status: Backend tests pass (98 passed, 3 warnings), Frontend builds successfully (0 errors, ~507 kB bundle).

---

## 🛡️ Responsible AI & Limitations
- **Synthetic Data**: The models are tuned on synthetic data and do not reflect real-world Razorpay distributions.
- **Human in the Loop**: Graph relationships provide *evidence*, not undeniable *proof*. The system is designed to empower human investigators via the XAI dashboard, not to replace them entirely.
- For full details, see [Responsible AI](docs/RESPONSIBLE_AI.md).

## 🔮 Future Improvements
- **Streaming Graph Updates**: Moving from static dataset snapshots to streaming graph updates via Kafka.
- **Graph Neural Networks (GNNs)**: Exploring GNNs to replace manual graph feature engineering.
- **Temporal Graphs**: Modeling the precise temporal decay of shared entities.

## 🏆 Buildathon Relevance
AbuseRing Sentinel directly addresses the **AI Risk Manager** track by providing a technically deep, highly usable, and explainable AI solution to one of the hardest problems in financial risk: coordinated abuse networks. 

---
*Built for the Razorpay AI Buildathon 2026*
