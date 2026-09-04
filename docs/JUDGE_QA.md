# Judge Q&A

**1. What is the core business problem you are solving?**
We are solving the problem of coordinated fraud rings. Traditional ML looks at single accounts in isolation. Fraudsters know this, so they spread their activity across hundreds of accounts. We use graph intelligence to detect the relationships between these accounts.

**2. Why use a Graph instead of just adding "shared IP count" to a normal ML model?**
A simple count doesn't capture topology. If two users share an IP, are they part of a massive 500-user ring, or are they just two roommates? Graph algorithms like Louvain community detection and betweenness centrality capture the *shape* and *density* of the network, which simple counts miss.

**3. Are the graph features actually effective?**
Yes. In our ablation studies, adding structural graph features to the baseline behavioral XGBoost model improved the F1 score from ~0.74 to 0.79, and ROC-AUC from 0.95 to 0.97.

**4. How did you get the data?**
The data is 100% synthetic. We built a custom deterministic generator using Python and Faker.

**5. How do you prevent data leakage during training?**
We use group-aware splitting. If we randomly split the data, 70% of a fraud ring would be in the training set and 30% in the test set, giving the model an unfair advantage. We ensure entire rings are grouped into a single split. We also run automated leakage audits to guarantee ground-truth labels aren't accidentally used as features.

**6. If the data is synthetic, how do we know this works in the real world?**
While the exact thresholds and distributions will change in production, the *architecture* is sound. Real fraud rings definitively reuse infrastructure (devices, addresses). Graph features mathematically capture this reuse. The synthetic data serves to validate the pipeline, not establish exact real-world precision rates.

**7. How do you handle Legitimate Shared Infrastructure, like a university campus?**
We specifically modeled legitimate high-connectivity groups (offices, hostels, families). The model learns that sharing an IP is not malicious *unless* it correlates with suspicious transaction velocity or overlaps across multiple vectors (e.g., sharing an IP *and* a device). 

**8. Explainability is crucial in risk. How is your model explainable?**
We don't just output a risk score. The Investigator facade runs a SHAP explainer alongside the model, extracting the top contributing features. It also returns exact multi-hop paths from the graph as concrete evidence for human reviewers.

**9. What are Counterfactuals in your XAI implementation?**
Counterfactuals answer "What if?". They tell an investigator: "If this user had not shared this specific device, their risk score would drop by 45%." This isolates the exact causal driver of the score.

**10. How do you manage False Positives?**
High-risk users are auto-blocked, but borderline cases are sent to a prioritized Review Queue. Our dashboard gives human investigators the graph context they need to quickly dismiss false positives.

**11. Is this scalable to millions of users?**
The current implementation builds a static snapshot using NetworkX, which is fine for batch processing. For real-time production at Razorpay scale, the graph representation would be migrated to a streaming architecture like Kafka and a graph database like Neo4j or Amazon Neptune.

**12. How do you handle Mega-Hubs like popular promotions?**
During the graph projection phase, we implement degree-capping. If an entity (like a merchant or a generic promo code) is shared by more than a configurable cap of users, we ignore it, as it contains no discriminatory signal and distorts centrality metrics.

**13. What ML model are you using?**
We are using XGBoost. Tree-based models are highly effective for tabular data with mixed feature types and handle the non-linear interactions between behavioral and graph features very well.

**14. What are the known limitations?**
The system relies on synthetic data. Furthermore, sophisticated fraudsters using unique residential proxies and pristine devices per account will evade structural detection (though they may still trigger behavioral alerts). 

**15. Does the graph provide proof of fraud?**
No, it provides evidence. Sharing an address is a strong signal, but human oversight is required for edge cases, which is why we built the Risk Decision Engine and Review Queue.

**16. How did you handle the API integration?**
We built a FastAPI backend that loads the model and graph exactly once on startup into memory. This allows the `/investigate` endpoint to run SHAP and graph traversal in sub-10ms.

**17. What is the Frontend stack?**
React, Vite, and TailwindCSS, communicating with the FastAPI backend. 

**18. Why not use Graph Neural Networks (GNNs)?**
GNNs are powerful but notoriously difficult to explain to risk operators. We chose manual feature engineering + XGBoost + SHAP because it provides exact, deterministic reasons (e.g., "Score is high because you share 3 devices") which is a strict regulatory requirement in finance.

**19. How do you ensure privacy?**
In production, PII like IP addresses and physical addresses would be securely hashed or tokenized before entering the graph. The topology is what matters, not the raw string value.

**20. What is your model threshold strategy?**
We chose a threshold that maximizes the F1 score on the validation set, balancing false positives and false negatives based on our simulated cost matrix.

**21. How do you handle model drift?**
Fraudsters adapt. The graph features would require continuous monitoring. If fraudsters stop sharing devices, the weight of the "shared_device" feature will drift, requiring retraining.

**22. How fast is the scoring?**
Because features are pre-computed in the batch pipeline, the FastAPI endpoint only runs the XGBoost inference and SHAP calculation, which averages around 9.6ms per user.

**23. Can this detect new types of rings?**
Yes. Because the model learns structural anomalies (high centrality, tight communities) rather than specific hardcoded rules, it can flag structurally suspicious clusters even if they are using a new tactic.

**24. What happens if a user is completely new?**
For "cold start" users with no graph history, the model relies entirely on their behavioral baseline features (transaction velocity, time of day) until they build a relationship history.

**25. What would you do differently if you had more time?**
We would implement a streaming feature store to update graph metrics in real-time as transactions occur, rather than relying on batch extraction.
