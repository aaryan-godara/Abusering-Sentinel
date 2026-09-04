# Demo Guide

To effectively showcase the power of AbuseRing Sentinel, we have pre-configured four specific users from the hidden test set. These users perfectly highlight the spectrum of coordinated abuse and legitimate overlap.

## Recommended Demo Order

1. **Example A: Dense Shared Infrastructure**
   - **ID**: `USR_00000613`
   - **What it shows**: The classic fraud ring. Multiple users heavily sharing the same devices, IPs, and physical addresses. 
   - **Focus on**: The interactive graph visualization showing a tight cluster of overlapping nodes, and the SHAP explanation highlighting "number of accounts sharing devices" as the primary risk driver.

2. **Example C: Multi-Entity Overlap**
   - **ID**: `USR_00000085`
   - **What it shows**: A more sophisticated ring that attempts to evade simple IP blocks by varying infrastructure, but gets caught through overlapping payment instruments and promotion farming.
   - **Focus on**: The risk evidence panel showing exactly which specific entities triggered the structural graph features.

3. **Example B: Large Related-Account Set**
   - **ID**: `USR_00000522`
   - **What it shows**: An extensive, sprawling ring spanning dozens of nodes. 
   - **Focus on**: The Louvain community context metrics and the betweenness centrality score. Explain how standard ML would struggle to flag this user without seeing the massive surrounding network.

4. **Example D: Legitimate High Connectivity**
   - **ID**: `USR_00000646`
   - **What it shows**: A crucial false-positive avoidance scenario. This user shares an IP with dozens of other users (representing a university campus or large corporate office). 
   - **Focus on**: The low risk score despite massive connectivity. The model correctly learns that sharing an IP is not malicious *unless* accompanied by suspicious behavioral features or dense overlap across *multiple* distinct entity types (like sharing an IP *and* a device *and* a payment instrument).
