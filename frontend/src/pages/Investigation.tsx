import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ChevronDown } from 'lucide-react';
import { api } from '../api/client';
import type { RiskAssessmentResponse } from '../api/client';
import GraphVisualizer from '../components/GraphVisualizer';
import { Bar, Metric, Panel, Row, Tag, WorkspaceNotice } from '../components/ui';

/* ==========================================================================
   Investigation workspace — same visual language as the home workspace.
   The graph fills the viewport; model / behavioural / counterfactual evidence
   sits in compact panels floating over it and in a detail strip below it.
   ========================================================================== */

export default function Investigation() {
  const { userId } = useParams();
  const [assessment, setAssessment] = useState<RiskAssessmentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        if (!userId) return;
        const res = await api.getRisk(userId);
        if (!cancelled) setAssessment(res);
      } catch (err: unknown) {
        if (cancelled) return;
        const status = (err as { response?: { status?: number } })?.response?.status;
        setError(status === 404 ? 'User not found.' : 'Failed to fetch investigation details.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadData();
    return () => {
      cancelled = true;
    };
  }, [userId]);

  if (loading) {
    return (
      <div className="tech-grid absolute inset-0">
        <WorkspaceNotice title="Loading investigation" detail={userId} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="tech-grid absolute inset-0">
        <WorkspaceNotice red title="Investigation unavailable" detail={error} />
      </div>
    );
  }

  if (!assessment) return null;

  const inv = assessment.evidence;
  const factors = inv.top_model_factors;
  const counterfactuals = inv.counterfactuals;
  const behavioral = inv.behavioral_evidence;

  /* ---------------- floating left panel: subject + decision ---------------- */
  const leftPanels = (
    <>
      <Panel title="Subject" meta="Investigation Report" className="shrink-0">
        <div className="space-y-2">
          <Metric label="User" value={assessment.user_id} />
          <div className="grid grid-cols-2 gap-2 border-t border-black/[0.09] pt-2">
            <Metric label="Risk Score" value={(assessment.risk_score * 100).toFixed(1)} unit="%" red />
            <Metric label="Level" value={assessment.risk_level.toUpperCase()} red mono={false} />
          </div>
          <div className="border-t border-black/[0.09] pt-1">
            <Row label="Decision" value={assessment.decision.toUpperCase()} strong />
            <Row
              label="Priority"
              value={assessment.review_priority ? assessment.review_priority.toUpperCase() : '—'}
              red={!!assessment.review_priority}
            />
          </div>
          {assessment.reasons.length > 0 && (
            <div className="flex flex-wrap gap-1 border-t border-black/[0.09] pt-2">
              {assessment.reasons.map((reason) => (
                <Tag key={reason.code} red>
                  {reason.code}
                </Tag>
              ))}
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Graph Connectivity" className="shrink-0">
        <div>
          <Row label="Shared Devices" value={String(inv.graph_evidence.shared_devices.length)} mono />
          <Row label="Shared IPs" value={String(inv.graph_evidence.shared_ips.length)} mono />
          <Row
            label="Shared Addresses"
            value={String(inv.graph_evidence.shared_addresses.length)}
            mono
          />
          <Row
            label="Payment Instruments"
            value={String(inv.graph_evidence.shared_payment_instruments.length)}
            mono
          />
          <Row label="Related Accounts" value={String(inv.related_accounts.length)} mono />
          <Row
            label="Community Size"
            value={String(inv.community_context?.community_size ?? 0)}
            mono
          />
        </div>
      </Panel>
    </>
  );

  /* ---------------- floating right panel: top model factors ---------------- */
  const rightPanels = (
    <Panel title="Model Explanation" meta="SHAP" className="max-h-[38%] shrink-0">
      <div className="space-y-2.5">
        {factors.slice(0, 4).map((f) => (
          <div key={f.feature_name}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="tech-id truncate text-[12.5px] font-medium text-ink-50">
                {f.feature_name}
              </span>
              <span
                className={`tech-num shrink-0 text-[13px] font-semibold ${
                  f.shap_value > 0 ? 'text-signal-bright' : 'text-ink-50'
                }`}
              >
                {f.shap_value > 0 ? '+' : ''}
                {f.shap_value.toFixed(3)}
              </span>
            </div>
            <div className="mt-1.5">
              <Bar ratio={Math.min(Math.abs(f.shap_value) * 2, 1)} red={f.shap_value > 0} />
            </div>
          </div>
        ))}
        {factors.length === 0 && <p className="body-text">No attributions returned.</p>}
      </div>
    </Panel>
  );

  return (
    <div className="absolute inset-0 overflow-y-auto">
      {/* graph workspace — hero surface */}
      <section className="relative h-full min-h-[520px] w-full">
        <GraphVisualizer
          key={inv.user_id}
          investigation={inv}
          leftPanels={leftPanels}
          rightPanels={rightPanels}
        />
        <div className="label-caps pointer-events-none absolute bottom-3 right-3 hidden items-center gap-1 xl:flex">
          <ChevronDown className="h-3 w-3" strokeWidth={1.8} />
          Detailed evidence below
        </div>
      </section>

      {/* detail strip — compact, secondary */}
      <section className="tech-grid border-t border-black/[0.09] p-3">
        <div className="grid grid-cols-1 gap-2 xl:grid-cols-3">
          <Panel title="Model Factors" meta="SHAP attribution" bodyClassName="overflow-visible">
            <div className="space-y-3">
              {factors.map((f) => (
                <div key={f.feature_name}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="tech-id truncate text-[13px] font-medium text-ink-50">
                      {f.feature_name}
                    </span>
                    <span
                      className={`tech-num shrink-0 text-[14px] font-semibold ${
                        f.shap_value > 0 ? 'text-signal-bright' : 'text-ink-50'
                      }`}
                    >
                      {f.shap_value > 0 ? '+' : ''}
                      {f.shap_value.toFixed(3)}
                    </span>
                  </div>
                  <div className="mt-1.5">
                    <Bar ratio={Math.min(Math.abs(f.shap_value) * 2, 1)} red={f.shap_value > 0} />
                  </div>
                  <p className="body-muted mt-1">
                    {f.human_readable_description} · value{' '}
                    <span className="tech-num font-medium text-ink-50">{f.feature_value}</span>
                  </p>
                </div>
              ))}
              {factors.length === 0 && <p className="body-text">No attributions returned.</p>}
            </div>
          </Panel>

          <Panel title="Behavioral Evidence" meta="Feature observations">
            {behavioral && behavioral.statements.length > 0 ? (
              <ul className="space-y-1.5">
                {behavioral.statements.map((statement, i) => (
                  <li key={i} className="body-text flex gap-1.5">
                    <span className="text-ink-350">—</span>
                    <span>{statement}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="body-text">No behavioral statements returned.</p>
            )}
            {behavioral && Object.keys(behavioral.feature_values).length > 0 && (
              <div className="mt-2 border-t border-black/[0.09] pt-1">
                {Object.entries(behavioral.feature_values).map(([name, value]) => (
                  <Row key={name} label={name} value={String(value)} mono />
                ))}
              </div>
            )}
          </Panel>

          <Panel title="Counterfactuals" meta="Model sensitivity — not causal">
            {counterfactuals.length > 0 ? (
              <div className="space-y-2.5">
                {counterfactuals.map((c) => (
                  <div
                    key={c.feature}
                    className="border-b border-black/[0.09] pb-2.5 last:border-b-0"
                  >
                    <div className="tech-id truncate text-[13px] font-medium text-ink-50">
                      {c.feature}
                    </div>
                    <div className="tech-num mt-1 flex items-baseline justify-between gap-2 text-[12.5px] font-medium leading-[1.5] text-ink-350">
                      <span>
                        {c.original_value.toFixed(1)} → {c.counterfactual_value.toFixed(1)}
                      </span>
                      <span className="text-ink-50">
                        {(c.original_probability * 100).toFixed(1)}% →{' '}
                        {(c.counterfactual_probability * 100).toFixed(1)}%
                      </span>
                      <span
                        className={
                          c.probability_delta > 0
                            ? 'text-[13px] font-semibold text-signal-bright'
                            : 'text-[13px] font-semibold text-ink-50'
                        }
                      >
                        {c.probability_delta > 0 ? '+' : ''}
                        {(c.probability_delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="body-text">No counterfactuals returned.</p>
            )}
          </Panel>
        </div>
      </section>
    </div>
  );
}
