# 5-Minute Pitch Script

**0:00–0:20: Opening and Problem**
*(Camera on speaker)*
"Hello judges, my name is [Your Name], and this is AbuseRing Sentinel. Today, financial institutions are locked in an arms race with fraudsters. But the nature of fraud has changed. It’s no longer just isolated bad actors. It's organized, coordinated abuse rings."

**0:20–0:55: Why coordinated abuse is difficult**
*(Show architectural diagram)*
"Traditional fraud models evaluate users one by one. To a standard ML model, a user checking out on a clean device looks perfectly legitimate. By the time the chargeback arrives, the fraudster has cycled to the next account. If we try simple rules—like blocking an IP—we accidentally block legitimate families and university campuses. The signal isn’t in any single account. The signal is in the *relationships* between them."

**0:55–1:25: Product Introduction**
*(Switch to screen recording of the main dashboard)*
"Enter AbuseRing Sentinel. We solve this by modeling every user, device, IP, and transaction as a massive heterogeneous graph. We extract deep structural features—like betweenness centrality and community density—and combine them with behavioral ML. This allows us to detect organized abuse that traditional models miss."

**1:25–2:05: Technical approach**
*(Show the graph construction flow)*
"We built a robust, end-to-end pipeline. First, we generated a highly realistic synthetic dataset containing complex legitimate overlap and hidden abuse rings. Then, we ensure zero data leakage through rigorous group-aware splitting. Finally, we train an XGBoost classifier on the combined structural and behavioral features. Our final model achieved an ROC-AUC of 0.97 on a strictly held-out test set."

**2:05–2:45: Live Demo**
*(Switch to the React Dashboard. Enter ID `USR_00000613`)*
"Let's look at the operational dashboard. Here is a detected fraud ring. Notice the interactive graph—this isn't just a pretty picture; this is exactly how the model views the data. You can clearly see multiple users heavily sharing the same devices and physical addresses."

**2:45–3:25: Explainability and Investigator Workflow**
*(Highlight the Risk Evidence and SHAP panels)*
"But providing a risk score isn't enough; investigators need proof. Our custom Investigator facade unifies the ML model with a SHAP explainer. It tells the investigator exactly *why* the score is high. Right here, we see that the primary driver of this 95% risk score is the 'number of accounts sharing devices', and we extract the exact device IDs from the graph as evidence."

**3:25–4:00: Counterfactuals & False Positives**
*(Enter ID `USR_00000646`)*
"Now, let's look at false positive avoidance. This user shares an IP with dozens of other users—think of a university campus. Standard rules would block them. Our model correctly assigns a low risk score because it learned that sharing an IP *without* anomalous behavioral overlap is legitimate."

**4:00–4:30: Responsible AI and Limitations**
*(Camera on speaker)*
"We built this with Responsible AI in mind. Graph connections are evidence, not undeniable proof. That's why we emphasize the human-in-the-loop dashboard and counterfactual sensitivity analysis. It's important to note that our data is entirely synthetic, simulating realistic patterns but requiring tuning for production."

**4:30–5:00: Closing**
"AbuseRing Sentinel proves that by moving from single-account evaluation to graph intelligence, we can dismantle coordinated fraud networks before they scale. Thank you for your time, and I look forward to your questions."
