# Responsible AI & Limitations

AbuseRing Sentinel is designed with Responsible AI principles at its core. Detecting coordinated fraud inherently requires analyzing user behavior and relationships, which demands strict ethical guardrails.

## 1. Evidence vs. Proof
Graph connections (e.g., sharing an IP address) represent *evidence*, not undeniable *proof* of malicious intent. Legitimate families, university students, and office workers routinely share infrastructure. The system is tuned to evaluate the density and behavioral context of these connections, but human investigators must ultimately review the evidence before finalizing high-impact actions like account termination.

## 2. False Positives & Legitimate High Connectivity
We explicitly modeled "Legitimate High Connectivity" scenarios in our test set to ensure the system does not lazily block users purely based on graph degree centrality. The model has learned that high connectivity *without* anomalous behavioral overlap is a strong indicator of a false positive. 

## 3. Human in the Loop
The entire frontend dashboard is an exercise in empowering the "Human in the Loop". The system does not operate as a black box. The `Investigator` facade extracts SHAP values, visualizes the exact overlapping graph entities, and provides counterfactual analysis ("If this shared device was ignored, the risk drops by X%"). This ensures that a human reviewer can override the AI with complete context.

## 4. Synthetic Data Limitation
**All data used to train, evaluate, and demonstrate this system is purely synthetic.** The distributions, overlap probabilities, and ring archetypes were programmatically generated to simulate realistic fraud scenarios, but they do not represent real Razorpay customer data. The metrics reported (e.g., F1 0.79) are strictly on this synthetic benchmark and do not establish guaranteed real-world production performance.

## 5. Privacy & Production Requirements
In a production deployment, all personally identifiable information (PII) would need to be tokenized or hashed prior to graph construction. A production implementation would also require rigorous continuous monitoring for model drift to ensure the graph feature weights remain unbiased over time.
