"""Deterministic reason extraction for the Phase 5 Risk Engine."""

from __future__ import annotations

from src.investigation.schema import InvestigationResult
from src.risk.schema import Reason, ReasonCode


class ReasonExtractor:
    """Extracts deterministic reasons from an InvestigationResult."""

    @staticmethod
    def extract_reasons(evidence: InvestigationResult) -> list[Reason]:
        """Generate reasons based on the actual evidence present."""
        reasons: list[Reason] = []

        # Graph reasons
        if evidence.shared_devices:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_SHARED_DEVICE,
                    description=f"User shares {len(evidence.shared_devices)} device(s)."
                )
            )
        if evidence.shared_ips:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_SHARED_IP,
                    description=f"User shares {len(evidence.shared_ips)} IP address(es)."
                )
            )
        if evidence.shared_addresses:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_SHARED_ADDRESS,
                    description=f"User shares {len(evidence.shared_addresses)} physical address(es)."
                )
            )
        if evidence.shared_payment_instruments:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_SHARED_PAYMENT,
                    description=f"User shares {len(evidence.shared_payment_instruments)} payment instrument(s)."
                )
            )
        
        if evidence.related_accounts:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_CONNECTED_ACCOUNTS,
                    description=f"User is directly connected to {len(evidence.related_accounts)} other account(s) via shared entities."
                )
            )

        if evidence.multi_hop_paths:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_MULTI_HOP,
                    description=f"User is part of {len(evidence.multi_hop_paths)} multi-hop path(s) spanning across the network."
                )
            )
        
        if evidence.community_context and evidence.community_context.community_density > 0.5:
            reasons.append(
                Reason(
                    code=ReasonCode.GRAPH_DENSE_COMMUNITY,
                    description=f"User is in a highly dense community (density: {evidence.community_context.community_density:.2f})."
                )
            )

        # Behavioural reasons
        bs = evidence.behavioral_summary
        if bs:
            vals = bs.feature_values
            if vals.get("transaction_frequency", 0.0) > 2.0:
                reasons.append(
                    Reason(
                        code=ReasonCode.BEHAVIORAL_HIGH_ACTIVITY,
                        description="User exhibits high transaction frequency."
                    )
                )
            if vals.get("promo_usage_rate", 0.0) > 0.5:
                reasons.append(
                    Reason(
                        code=ReasonCode.BEHAVIORAL_HIGH_PROMO,
                        description="User has a high promotion usage rate."
                    )
                )
            if vals.get("account_age_days", 999) < 7:
                reasons.append(
                    Reason(
                        code=ReasonCode.BEHAVIORAL_NEW_ACCOUNT,
                        description="Account is very new (less than 7 days old)."
                    )
                )
            if vals.get("night_activity_rate", 0.0) > 0.3:
                reasons.append(
                    Reason(
                        code=ReasonCode.BEHAVIORAL_NIGHT_ACTIVITY,
                        description="Significant night-time transaction activity."
                    )
                )
            if vals.get("weekend_activity_rate", 0.0) > 0.5:
                reasons.append(
                    Reason(
                        code=ReasonCode.BEHAVIORAL_WEEKEND_ACTIVITY,
                        description="Predominantly weekend transaction activity."
                    )
                )
            if vals.get("failed_transaction_rate", 0.0) > 0.1:
                reasons.append(
                    Reason(
                        code=ReasonCode.BEHAVIORAL_HIGH_FAIL_RATE,
                        description="Elevated failed transaction rate."
                    )
                )

        # Model / SHAP reasons
        if evidence.top_model_factors:
            top_factor = evidence.top_model_factors[0]
            if top_factor.direction == "increases_risk":
                reasons.append(
                    Reason(
                        code=ReasonCode.MODEL_HIGH_RISK_FACTOR,
                        description=f"Primary risk factor: {top_factor.human_readable_description}"
                    )
                )

        return reasons

__all__ = ["ReasonExtractor"]
