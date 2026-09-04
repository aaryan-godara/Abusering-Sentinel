import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { HealthResponse, RiskAssessmentResponse } from '../api/client';
import { demoScenarios } from '../config/demoScenarios';
import GraphVisualizer from '../components/GraphVisualizer';
import { ActionButton, Metric, Panel, Row, Tag, WorkspaceNotice } from '../components/ui';

/* ==========================================================================
   Home workspace — the network graph is the hero surface.
   Subject graph is driven by the review queue payload already returned by
   /review-queue/top/{n} (each item carries its full investigation evidence),
   so switching subject costs no additional requests.
   ========================================================================== */

const QUEUE_SIZE = 20;
const QUEUE_VISIBLE = 8;

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [queue, setQueue] = useState<RiskAssessmentResponse[]>([]);
  const [subjectId, setSubjectId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    const loadDashboard = async () => {
      try {
        const [h, q] = await Promise.all([
          api.getHealth(),
          api.getReviewQueueTop(QUEUE_SIZE),
        ]);
        if (cancelled) return;
        setHealth(h);
        setQueue(q);
        setSubjectId(q.length > 0 ? q[0].user_id : null);
      } catch {
        if (!cancelled) setError('Failed to connect to AbuseRing Sentinel backend.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadDashboard();
    return () => {
      cancelled = true;
    };
  }, []);

  const subject = useMemo(
    () => queue.find((item) => item.user_id === subjectId) ?? null,
    [queue, subjectId],
  );

  if (loading) {
    return (
      <div className="tech-grid absolute inset-0">
        <WorkspaceNotice title="Initialising workspace" detail="Loading review queue." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="tech-grid absolute inset-0">
        <WorkspaceNotice red title="Backend unreachable" detail={error} />
      </div>
    );
  }

  if (!subject) {
    return (
      <div className="tech-grid absolute inset-0">
        <WorkspaceNotice
          title="No accounts in review queue"
          detail="Nothing is currently queued for manual review. Open a subject directly to load its network."
        >
          <div className="flex flex-col gap-1.5">
            {demoScenarios.map((scenario) => (
              <ActionButton key={scenario.userId} onClick={() => navigate(`/users/${scenario.userId}`)}>
                {scenario.userId}
              </ActionButton>
            ))}
          </div>
        </WorkspaceNotice>
      </div>
    );
  }

  const priority = subject.review_priority;

  /* --------------- floating left panel: investigation status --------------- */
  const leftPanels = (
    <>
      <Panel
        title="Investigation Status"
        meta={health ? health.status.toUpperCase() : undefined}
        className="shrink-0"
      >
        <div className="space-y-2">
          <Metric label="Subject" value={subject.user_id} />
          <div className="grid grid-cols-2 gap-2 border-t border-black/[0.09] pt-2">
            <Metric
              label="Risk Score"
              value={(subject.risk_score * 100).toFixed(1)}
              unit="%"
              red
            />
            <Metric label="Level" value={subject.risk_level.toUpperCase()} red mono={false} />
          </div>
          <div className="border-t border-black/[0.09] pt-1">
            <Row label="Decision" value={subject.decision.toUpperCase()} strong />
            <Row label="Priority" value={priority ? priority.toUpperCase() : '—'} red={!!priority} />
            <Row label="In Queue" value={String(queue.length)} mono />
            <Row label="Engine" value="Model C" />
          </div>
          {subject.reasons.length > 0 && (
            <div className="flex flex-wrap gap-1 border-t border-black/[0.09] pt-2">
              {subject.reasons.slice(0, 4).map((reason) => (
                <Tag key={reason.code} red>
                  {reason.code}
                </Tag>
              ))}
            </div>
          )}
          <ActionButton primary onClick={() => navigate(`/users/${subject.user_id}`)}>
            Investigate User
          </ActionButton>
        </div>
      </Panel>

      <Panel title="Entry Points" className="shrink-0">
        <div className="flex flex-col gap-1">
          {demoScenarios.map((scenario) => (
            <button
              key={scenario.userId}
              type="button"
              onClick={() => navigate(`/users/${scenario.userId}`)}
              className="flex items-baseline gap-2 rounded-md border border-black/[0.1] bg-white/45 px-2 py-1.5 text-left transition-colors hover:border-ink-500 hover:bg-white/65"
            >
              <span className="tech-id text-[12.5px] font-medium text-ink-50">
                {scenario.userId}
              </span>
              <span className="label-caps truncate">{scenario.label}</span>
            </button>
          ))}
        </div>
      </Panel>
    </>
  );

  /* --------------- floating right panel: review queue --------------- */
  const rightPanels = (
    <Panel
      title="Review Queue"
      meta={`top ${Math.min(QUEUE_VISIBLE, queue.length)} of ${queue.length}`}
      className="max-h-[42%] shrink-0"
    >
      <div className="flex flex-col gap-0.5">
        {queue.slice(0, QUEUE_VISIBLE).map((item) => {
          const active = item.user_id === subject.user_id;
          return (
            <button
              key={item.user_id}
              type="button"
              onClick={() => setSubjectId(item.user_id)}
              aria-pressed={active}
              className={`flex items-baseline gap-2 rounded-md border px-2 py-1.5 text-left transition-colors ${
                active
                  ? 'border-signal bg-signal-deep/60 text-signal-bright'
                  : 'border-transparent text-ink-100 hover:border-black/[0.1] hover:bg-white/55'
              }`}
            >
              <span className="tech-id w-[100px] shrink-0 truncate text-[12.5px] font-medium">
                {item.user_id}
              </span>
              <span className="tech-num w-[52px] shrink-0 text-[13px] font-semibold">
                {(item.risk_score * 100).toFixed(1)}%
              </span>
              <span
                className={`label-caps ml-auto truncate ${active ? 'text-signal-bright' : ''}`}
              >
                {item.review_priority ?? item.decision}
              </span>
            </button>
          );
        })}
      </div>
    </Panel>
  );

  return (
    <GraphVisualizer
      key={subject.user_id}
      investigation={subject.evidence}
      leftPanels={leftPanels}
      rightPanels={rightPanels}
    />
  );
}
