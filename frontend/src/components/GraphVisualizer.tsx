import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import {
  RotateCcw,
  Maximize2,
  Eraser,
  Tag,
  Highlighter,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { api } from '../api/client';
import type {
  GraphEvidenceResponse,
  InvestigationResponse,
  RelatedAccount,
} from '../api/client';
import {
  ActionButton,
  ControlButton,
  IconButton,
  Panel,
  Row,
} from './ui';

/* ==========================================================================
   AbuseRing Sentinel — Network Workspace
   The graph is the workspace: full-bleed canvas over a technical grid, with
   floating panels, a floating inspector and a compact floating control bar.
   Palette is strictly black / white / gray / red.
   All nodes and relationships come from the API investigation response only.
   ========================================================================== */

const C = {
  /** White workspace: node fills read as light objects with dark outlines. */
  bg: '#ffffff',
  canvasHollow: 'rgba(255,255,255,0.97)',
  nodeUser: '#ffffff',
  nodeUserShade: '#d2d2d8',
  nodeEntity: '#f4f4f6',
  nodeEntityShade: '#c2c2ca',
  highlight: 'rgba(255,255,255,0.95)',
  bevel: 'rgba(255,255,255,0.9)',
  shadow: 'rgba(10,10,12,0.26)',
  ink: '#111111',
  inkSoft: '#2b2b31',
  glyph: '#22222a',
  glyphSoft: '#3a3a42',
  gray: '#5f5f68',
  grayMid: '#7a7a83',
  grayDim: '#b6b6be',
  grayFaint: '#dcdce0',
  labelSoft: '#555555',
  red: '#dc2626',
  redBright: '#bf0d1e',
  redDim: '#f0b8b8',
} as const;

type EntityType = 'DEVICE' | 'IP' | 'ADDRESS' | 'PAYMENT';
type NodeKind = 'USER' | EntityType;

interface GNode {
  id: string;
  kind: NodeKind;
  isSubject: boolean;
  /** Only present for related accounts returned by the backend. */
  related?: RelatedAccount;
  /** Human label for the entity relationship, e.g. "Shared device". */
  relationshipLabel?: string;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

type LinkKind = 'subject_entity' | 'related_entity' | 'user_user';

interface GLink {
  source: string;
  target: string;
  linkKind: LinkKind;
  /** Backend relationship types backing this edge. */
  relationshipTypes: string[];
  label: string;
  /** True when the backend reports >1 shared entity for the related account. */
  emphasis: boolean;
}

const ENTITY_GROUPS: {
  key: keyof InvestigationResponse['graph_evidence'];
  kind: EntityType;
  relationshipType: string;
  label: string;
}[] = [
  { key: 'shared_devices', kind: 'DEVICE', relationshipType: 'shared_device', label: 'Shared device' },
  { key: 'shared_ips', kind: 'IP', relationshipType: 'shared_ip', label: 'Shared IP' },
  { key: 'shared_addresses', kind: 'ADDRESS', relationshipType: 'shared_address', label: 'Shared address' },
  {
    key: 'shared_payment_instruments',
    kind: 'PAYMENT',
    relationshipType: 'shared_payment_instrument',
    label: 'Shared payment instrument',
  },
];

const RELATIONSHIP_LABELS: Record<string, string> = {
  shared_device: 'Shared device',
  shared_ip: 'Shared IP',
  shared_address: 'Shared address',
  shared_payment_instrument: 'Shared payment instrument',
};

/** relationship_type -> the graph_evidence list that backs it. */
const RELATIONSHIP_TO_EVIDENCE_KEY: Record<string, keyof GraphEvidenceResponse> = {
  shared_device: 'shared_devices',
  shared_ip: 'shared_ips',
  shared_address: 'shared_addresses',
  shared_payment_instrument: 'shared_payment_instruments',
};

const RELATIONSHIP_TO_KIND: Record<string, EntityType> = {
  shared_device: 'DEVICE',
  shared_ip: 'IP',
  shared_address: 'ADDRESS',
  shared_payment_instrument: 'PAYMENT',
};

const KIND_LABELS: Record<NodeKind, string> = {
  USER: 'USER',
  DEVICE: 'DEVICE',
  IP: 'IP',
  ADDRESS: 'ADDRESS',
  PAYMENT: 'PAYMENT INSTRUMENT',
};

/** Rendering bounds — the full 10k-user graph is never loaded or drawn. */
const MAX_ENTITIES_PER_TYPE = 40;
const MAX_RELATED_ACCOUNTS = 40;
/** Neighbour evidence lookups are batched and capped. */
const MAX_NEIGHBOR_EVIDENCE_LOOKUPS = 40;
const MAX_BRIDGES_PER_RELATIONSHIP = 3;
const BFS_MAX_DEPTH = 6;
const BFS_MAX_VISITS = 600;

const edgeKey = (a: string, b: string) => (a < b ? `${a}\u0000${b}` : `${b}\u0000${a}`);

const nodeIdOf = (v: unknown): string => {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  if (v && typeof v === 'object' && 'id' in (v as Record<string, unknown>)) {
    return String((v as { id: unknown }).id);
  }
  return '';
};

/** USR_00000085 -> U85 ; DEV_000123 -> D123 ; otherwise truncated. */
function shortLabel(id: string): string {
  const m = /^([A-Za-z]+)[_-]?0*(\d+)$/.exec(id);
  if (m) return `${m[1][0].toUpperCase()}${m[2]}`;
  return id.length > 9 ? `${id.slice(0, 8)}\u2026` : id;
}

function relationshipLabel(relationshipType: string): string {
  return RELATIONSHIP_LABELS[relationshipType] ?? relationshipType.replace(/_/g, ' ');
}

/* --------------------------------------------------------------------------
   Graph construction — strictly from the investigation response
   -------------------------------------------------------------------------- */

/** Evidence for neighbouring accounts, keyed by user id (from /investigate). */
type NeighborEvidence = Map<string, GraphEvidenceResponse>;

interface BuiltGraph {
  nodes: GNode[];
  links: GLink[];
  nodeById: Map<string, GNode>;
  hiddenEntities: number;
  hiddenRelated: number;
  /** Relationship types whose bridging entity could not be uniquely resolved. */
  unresolvedTypes: Set<string>;
}

function buildGraph(inv: InvestigationResponse, neighbors: NeighborEvidence): BuiltGraph {
  const nodes: GNode[] = [];
  const nodeById = new Map<string, GNode>();
  const links: GLink[] = [];
  const seenEdges = new Set<string>();
  const unresolvedTypes = new Set<string>();

  const addNode = (node: GNode) => {
    if (nodeById.has(node.id)) return nodeById.get(node.id)!;
    nodes.push(node);
    nodeById.set(node.id, node);
    return node;
  };

  const addLink = (link: GLink) => {
    const key = edgeKey(link.source, link.target);
    if (seenEdges.has(key)) return;
    seenEdges.add(key);
    links.push(link);
  };

  // Subject user is pinned at the origin so it stays the focal node.
  addNode({ id: inv.user_id, kind: 'USER', isSubject: true, fx: 0, fy: 0, x: 0, y: 0 });

  let hiddenEntities = 0;
  // Entity id that uniquely carries a relationship type (only when unambiguous).
  const uniqueBridge = new Map<string, string>();

  for (const group of ENTITY_GROUPS) {
    const ids = inv.graph_evidence[group.key];
    if (ids.length === 1) uniqueBridge.set(group.relationshipType, ids[0]);
    const shown = ids.slice(0, MAX_ENTITIES_PER_TYPE);
    hiddenEntities += ids.length - shown.length;
    for (const id of shown) {
      addNode({ id, kind: group.kind, isSubject: false, relationshipLabel: group.label });
      addLink({
        source: inv.user_id,
        target: id,
        linkKind: 'subject_entity',
        relationshipTypes: [group.relationshipType],
        label: group.label,
        emphasis: false,
      });
    }
  }

  const related = inv.related_accounts.slice(0, MAX_RELATED_ACCOUNTS);
  const hiddenRelated = inv.related_accounts.length - related.length;

  for (const account of related) {
    addNode({ id: account.user_id, kind: 'USER', isSubject: false, related: account });
    const emphasis = account.shared_entity_count > 1;
    const typeOnly: string[] = [];
    const neighborEvidence = neighbors.get(account.user_id);

    for (const relType of account.relationship_types) {
      const evidenceKey = RELATIONSHIP_TO_EVIDENCE_KEY[relType];
      const kind = RELATIONSHIP_TO_KIND[relType];

      // Preferred: intersect the subject's entity list with the neighbour's own
      // entity list, both returned by the backend. The overlap is the actual
      // shared infrastructure — nothing is inferred.
      let bridges: string[] = [];
      if (neighborEvidence && evidenceKey) {
        const mine = new Set(inv.graph_evidence[evidenceKey]);
        bridges = neighborEvidence[evidenceKey]
          .filter((id) => mine.has(id))
          .slice(0, MAX_BRIDGES_PER_RELATIONSHIP);
      }

      // Fallback: the subject holds exactly one entity of this type, so the
      // shared entity reported by the backend can only be that one.
      if (bridges.length === 0) {
        const unique = uniqueBridge.get(relType);
        if (unique) bridges = [unique];
      }

      if (bridges.length > 0 && kind) {
        for (const bridge of bridges) {
          addNode({
            id: bridge,
            kind,
            isSubject: false,
            relationshipLabel: relationshipLabel(relType),
          });
          // The bridge is in the subject's own evidence list, so the subject edge
          // is real even if the entity fell outside the per-type render cap.
          addLink({
            source: inv.user_id,
            target: bridge,
            linkKind: 'subject_entity',
            relationshipTypes: [relType],
            label: relationshipLabel(relType),
            emphasis: false,
          });
          addLink({
            source: bridge,
            target: account.user_id,
            linkKind: 'related_entity',
            relationshipTypes: [relType],
            label: relationshipLabel(relType),
            emphasis,
          });
        }
      } else {
        typeOnly.push(relType);
        unresolvedTypes.add(relType);
      }
    }

    if (typeOnly.length > 0) {
      addLink({
        source: inv.user_id,
        target: account.user_id,
        linkKind: 'user_user',
        relationshipTypes: typeOnly,
        label: typeOnly.map(relationshipLabel).join(', '),
        emphasis,
      });
    }
  }

  return { nodes, links, nodeById, hiddenEntities, hiddenRelated, unresolvedTypes };
}

/* --------------------------------------------------------------------------
   Bounded path search over the loaded neighborhood only
   -------------------------------------------------------------------------- */

type Adjacency = Map<string, { to: string; key: string }[]>;

function buildAdjacency(links: GLink[]): Adjacency {
  const adjacency: Adjacency = new Map();
  const push = (from: string, to: string, key: string) => {
    const list = adjacency.get(from);
    if (list) list.push({ to, key });
    else adjacency.set(from, [{ to, key }]);
  };
  for (const link of links) {
    const a = nodeIdOf(link.source);
    const b = nodeIdOf(link.target);
    const key = edgeKey(a, b);
    push(a, b, key);
    push(b, a, key);
  }
  return adjacency;
}

function shortestPath(adjacency: Adjacency, from: string, to: string): string[] | null {
  if (from === to) return [from];
  const previous = new Map<string, string>([[from, '']]);
  let frontier = [from];
  let depth = 0;
  let visits = 0;

  while (frontier.length > 0 && depth < BFS_MAX_DEPTH && visits < BFS_MAX_VISITS) {
    const next: string[] = [];
    for (const current of frontier) {
      for (const edge of adjacency.get(current) ?? []) {
        if (previous.has(edge.to)) continue;
        previous.set(edge.to, current);
        visits += 1;
        if (edge.to === to) {
          const path = [to];
          let cursor = current;
          while (cursor) {
            path.push(cursor);
            cursor = previous.get(cursor) ?? '';
          }
          return path.reverse();
        }
        next.push(edge.to);
      }
    }
    frontier = next;
    depth += 1;
  }
  return null;
}

/* --------------------------------------------------------------------------
   Canvas helpers
   -------------------------------------------------------------------------- */

function tracePolygon(ctx: CanvasRenderingContext2D, points: [number, number][]) {
  ctx.beginPath();
  points.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
  ctx.closePath();
}

function traceRoundedSquare(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  const s = r * 0.92;
  const radius = s * 0.35;
  ctx.beginPath();
  ctx.moveTo(x - s + radius, y - s);
  ctx.lineTo(x + s - radius, y - s);
  ctx.quadraticCurveTo(x + s, y - s, x + s, y - s + radius);
  ctx.lineTo(x + s, y + s - radius);
  ctx.quadraticCurveTo(x + s, y + s, x + s - radius, y + s);
  ctx.lineTo(x - s + radius, y + s);
  ctx.quadraticCurveTo(x - s, y + s, x - s, y + s - radius);
  ctx.lineTo(x - s, y - s + radius);
  ctx.quadraticCurveTo(x - s, y - s, x - s + radius, y - s);
  ctx.closePath();
}

function traceHexagon(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  const points: [number, number][] = [];
  for (let i = 0; i < 6; i += 1) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    points.push([x + r * Math.cos(angle), y + r * Math.sin(angle)]);
  }
  tracePolygon(ctx, points);
}

function traceDiamond(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  tracePolygon(ctx, [
    [x, y - r * 1.15],
    [x + r * 1.05, y],
    [x, y + r * 1.15],
    [x - r * 1.05, y],
  ]);
}

/* --------------------------------------------------------------------------
   Node glyphs — simple monochrome pictograms drawn inside each node body.
   Every glyph is authored inside a unit box (-1..1) and scaled by `g`.
   -------------------------------------------------------------------------- */

function traceRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  radius: number,
) {
  const rr = Math.min(radius, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x, y + rr);
  ctx.closePath();
}

/** Person: head + shoulders. */
function glyphUser(ctx: CanvasRenderingContext2D, g: number) {
  ctx.beginPath();
  ctx.arc(0, -0.42 * g, 0.36 * g, 0, 2 * Math.PI);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(-0.66 * g, 0.66 * g);
  ctx.quadraticCurveTo(-0.62 * g, 0.02 * g, 0, 0.02 * g);
  ctx.quadraticCurveTo(0.62 * g, 0.02 * g, 0.66 * g, 0.66 * g);
  ctx.closePath();
  ctx.fill();
}

/** Laptop: screen + base. */
function glyphDevice(ctx: CanvasRenderingContext2D, g: number, lw: number) {
  ctx.lineWidth = lw;
  ctx.lineJoin = 'round';
  traceRoundedRect(ctx, -0.6 * g, -0.62 * g, 1.2 * g, 0.9 * g, 0.16 * g);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-0.86 * g, 0.5 * g);
  ctx.lineTo(0.86 * g, 0.5 * g);
  ctx.lineTo(0.62 * g, 0.28 * g);
  ctx.lineTo(-0.62 * g, 0.28 * g);
  ctx.closePath();
  ctx.fill();
}

/** Globe: sphere with meridian + equator. */
function glyphIp(ctx: CanvasRenderingContext2D, g: number, lw: number) {
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.arc(0, 0, 0.72 * g, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-0.72 * g, 0);
  ctx.lineTo(0.72 * g, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(0, 0, 0.34 * g, 0.72 * g, 0, 0, 2 * Math.PI);
  ctx.stroke();
}

/** House: roof + body. */
function glyphAddress(ctx: CanvasRenderingContext2D, g: number, lw: number) {
  ctx.lineWidth = lw;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(-0.8 * g, -0.04 * g);
  ctx.lineTo(0, -0.74 * g);
  ctx.lineTo(0.8 * g, -0.04 * g);
  ctx.closePath();
  ctx.fill();
  traceRoundedRect(ctx, -0.52 * g, -0.06 * g, 1.04 * g, 0.78 * g, 0.1 * g);
  ctx.stroke();
}

/** Credit card: rounded card + magnetic stripe. */
function glyphPayment(ctx: CanvasRenderingContext2D, g: number, lw: number) {
  ctx.lineWidth = lw;
  ctx.lineJoin = 'round';
  traceRoundedRect(ctx, -0.82 * g, -0.56 * g, 1.64 * g, 1.12 * g, 0.18 * g);
  ctx.stroke();
  ctx.beginPath();
  ctx.rect(-0.82 * g, -0.28 * g, 1.64 * g, 0.26 * g);
  ctx.fill();
}

/** Draw the type pictogram for a node, centred on (x, y). */
function drawGlyph(
  ctx: CanvasRenderingContext2D,
  kind: NodeKind,
  x: number,
  y: number,
  g: number,
  color: string,
  globalScale: number,
) {
  // Constant screen-space stroke weight, clamped so glyphs never turn to blobs.
  const lw = Math.min(1.5 / globalScale, g * 0.3);
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineCap = 'round';
  if (kind === 'USER') glyphUser(ctx, g);
  else if (kind === 'DEVICE') glyphDevice(ctx, g, lw);
  else if (kind === 'IP') glyphIp(ctx, g, lw);
  else if (kind === 'ADDRESS') glyphAddress(ctx, g, lw);
  else glyphPayment(ctx, g, lw);
  ctx.restore();
}

/** Andrew monotone chain convex hull. */
function convexHull(points: [number, number][]): [number, number][] {
  if (points.length < 3) return points;
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const build = (input: [number, number][]) => {
    const stack: [number, number][] = [];
    for (const p of input) {
      while (stack.length >= 2 && cross(stack[stack.length - 2], stack[stack.length - 1], p) <= 0) {
        stack.pop();
      }
      stack.push(p);
    }
    return stack;
  };
  const lower = build(sorted);
  const upper = build([...sorted].reverse());
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** O(n^2) separation force. Node counts are bounded, so this stays cheap. */
function makeCollideForce(radiusOf: (node: GNode) => number, strength = 0.85) {
  let nodes: GNode[] = [];
  const force = () => {
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = (b.x ?? 0) - (a.x ?? 0);
        let dy = (b.y ?? 0) - (a.y ?? 0);
        const min = radiusOf(a) + radiusOf(b) + 10;
        const d2 = dx * dx + dy * dy;
        if (d2 >= min * min) continue;
        const d = Math.sqrt(d2) || 0.01;
        const push = ((min - d) / d) * strength * 0.5;
        dx *= push;
        dy *= push;
        a.x = (a.x ?? 0) - dx;
        a.y = (a.y ?? 0) - dy;
        b.x = (b.x ?? 0) + dx;
        b.y = (b.y ?? 0) + dy;
      }
    }
  };
  (force as unknown as { initialize: (n: GNode[]) => void }).initialize = (n: GNode[]) => {
    nodes = n;
  };
  return force;
}

const nodeRadius = (node: GNode) => (node.isSubject ? 9.5 : node.kind === 'USER' ? 6.4 : 5.6);

/* --------------------------------------------------------------------------
   Component
   -------------------------------------------------------------------------- */

interface GraphProps {
  investigation: InvestigationResponse;
  /** Floating panels rendered over the graph, left column (top to bottom). */
  leftPanels?: ReactNode;
  /** Floating panels rendered over the graph, right column, above the inspector. */
  rightPanels?: ReactNode;
  /** Optional status line shown in the floating investigation status strip. */
  statusMeta?: ReactNode;
}



export default function GraphVisualizer({
  investigation,
  leftPanels,
  rightPanels,
}: GraphProps) {
  const navigate = useNavigate();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods | undefined>(undefined);

  const didAutoFitRef = useRef(false);

  const [size, setSize] = useState({ width: 800, height: 560 });
  const [selected, setSelected] = useState<string[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const [showLabels, setShowLabels] = useState(false);
  const [emphasis, setEmphasis] = useState(false);
  /** Side-column visibility — collapsing a column hands the space back to the graph. */
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  /** null = neighbour evidence still resolving. */
  const [neighbors, setNeighbors] = useState<NeighborEvidence | null>(null);

  const neighborIds = useMemo(
    () =>
      investigation.related_accounts
        .slice(0, MAX_NEIGHBOR_EVIDENCE_LOOKUPS)
        .map((a) => a.user_id),
    [investigation],
  );
  const neighborIdsKey = neighborIds.join(',');

  // Resolve the bridging entities between the subject and its related accounts.
  // One bounded batch request; the neighbourhood is never re-fetched per node.
  useEffect(() => {
    let cancelled = false;
    if (neighborIds.length === 0) {
      setNeighbors(new Map());
      return;
    }
    api
      .batchInvestigate(neighborIds)
      .then((results) => {
        if (cancelled) return;
        const map: NeighborEvidence = new Map();
        for (const r of results) map.set(r.user_id, r.graph_evidence);
        setNeighbors(map);
      })
      .catch(() => {
        // Fall back to relationship-type-only edges.
        if (!cancelled) setNeighbors(new Map());
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [neighborIdsKey]);

  const graph = useMemo(
    () => buildGraph(investigation, neighbors ?? new Map()),
    [investigation, neighbors],
  );
  const graphData = useMemo(() => ({ nodes: graph.nodes, links: graph.links }), [graph]);
  const adjacency = useMemo(() => buildAdjacency(graph.links), [graph]);

  // Refit once after each structural rebuild settles.
  useEffect(() => {
    didAutoFitRef.current = false;
  }, [graphData]);

  // Container sizing.
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      setSize({ width: el.clientWidth, height: el.clientHeight });
    });
    observer.observe(el);
    setSize({ width: el.clientWidth, height: el.clientHeight });
    return () => observer.disconnect();
  }, []);

  // Stable physics tuning.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    const linkForce = fg.d3Force('link') as
      | { distance: (fn: (l: unknown) => number) => unknown; strength: (fn: (l: unknown) => number) => unknown }
      | undefined;
    if (linkForce) {
      linkForce.distance((link: unknown) => {
        const kind = (link as GLink).linkKind;
        if (kind === 'subject_entity') return 62;
        if (kind === 'related_entity') return 54;
        return 110;
      });
      linkForce.strength((link: unknown) =>
        (link as GLink).linkKind === 'user_user' ? 0.22 : 0.68,
      );
    }
    const charge = fg.d3Force('charge') as
      | { strength: (v: number) => { distanceMax: (v: number) => unknown } }
      | undefined;
    if (charge) charge.strength(-190).distanceMax(460);
    fg.d3Force('collide', makeCollideForce(nodeRadius) as never);
  }, [graphData]);

  /* ---------------- derived investigation state ---------------- */

  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const segments = useMemo(() => {
    const result: { from: string; to: string; path: string[] | null }[] = [];
    for (let i = 0; i < selected.length - 1; i += 1) {
      result.push({
        from: selected[i],
        to: selected[i + 1],
        path: shortestPath(adjacency, selected[i], selected[i + 1]),
      });
    }
    return result;
  }, [selected, adjacency]);

  const pathNodeIds = useMemo(() => {
    const set = new Set<string>(selected);
    for (const segment of segments) {
      for (const id of segment.path ?? []) set.add(id);
    }
    return set;
  }, [segments, selected]);

  const pathEdgeKeys = useMemo(() => {
    const set = new Set<string>();
    for (const segment of segments) {
      const path = segment.path;
      if (!path) continue;
      for (let i = 0; i < path.length - 1; i += 1) set.add(edgeKey(path[i], path[i + 1]));
    }
    return set;
  }, [segments]);

  const incidentEdgeKeys = useMemo(() => {
    const set = new Set<string>();
    if (selectedSet.size === 0) return set;
    for (const link of graph.links) {
      // force-graph replaces source/target with node objects after the first tick.
      const a = nodeIdOf(link.source);
      const b = nodeIdOf(link.target);
      if (selectedSet.has(a) || selectedSet.has(b)) set.add(edgeKey(a, b));
    }
    return set;
  }, [graph.links, selectedSet]);

  const activeNodeIds = useMemo(() => {
    if (selectedSet.size === 0) return null;
    const set = new Set<string>(pathNodeIds);
    for (const id of selectedSet) {
      for (const edge of adjacency.get(id) ?? []) set.add(edge.to);
    }
    return set;
  }, [selectedSet, pathNodeIds, adjacency]);

  const clusterConnected =
    selected.length >= 3 && segments.length > 0 && segments.every((s) => s.path !== null);

  /* ---------------- interaction ---------------- */

  const handleNodeClick = useCallback((node: unknown) => {
    const id = nodeIdOf(node);
    if (!id) return;
    // Selection only — navigation is an explicit action in the inspector.
    setSelected((current) =>
      current.includes(id) ? current.filter((n) => n !== id) : [...current, id],
    );
  }, [setSelected]);

  const handleFit = useCallback(() => {
    fgRef.current?.zoomToFit(500, 90);
  }, []);

  // Fit once after the initial layout settles; later settles must not move the view.
  const handleEngineStop = useCallback(() => {
    if (didAutoFitRef.current) return;
    didAutoFitRef.current = true;
    fgRef.current?.zoomToFit(600, 90);
  }, []);

  const handleReset = useCallback(() => {
    setSelected([]);
    setHovered(null);
    fgRef.current?.centerAt(0, 0, 400);
    fgRef.current?.zoom(1.6, 400);
  }, [setSelected, setHovered]);

  /* ---------------- painting ---------------- */

  const paintNode = useCallback(
    (raw: unknown, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as GNode;
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const r = nodeRadius(node);
      const isSelected = selectedSet.has(node.id);
      const onPath = pathNodeIds.has(node.id);
      const dim = activeNodeIds !== null && !activeNodeIds.has(node.id) && !isSelected;
      const isUser = node.kind === 'USER';

      // Trace the node silhouette; reused for shadow, body, bevel and highlight.
      const trace = () => {
        if (isUser) {
          ctx.beginPath();
          ctx.arc(x, y, r, 0, 2 * Math.PI);
        } else if (node.kind === 'DEVICE') {
          traceRoundedSquare(ctx, x, y, r);
        } else if (node.kind === 'ADDRESS') {
          traceHexagon(ctx, x, y, r);
        } else {
          traceDiamond(ctx, x, y, r);
        }
      };

      const top = dim ? '#f2f2f4' : isUser ? C.nodeUser : C.nodeEntity;
      const bottom = dim ? C.grayFaint : isUser ? C.nodeUserShade : C.nodeEntityShade;
      const outline = dim ? C.grayDim : isUser ? C.inkSoft : C.gray;

      ctx.save();

      // --- depth pass 1: contact shadow beneath the body (screen-space blur).
      if (!dim) {
        ctx.shadowColor = C.shadow;
        ctx.shadowBlur = 5;
        ctx.shadowOffsetY = 1.6;
      }

      // --- body: vertical light-to-shade gradient reads as a lit 3D surface.
      trace();
      const body = ctx.createLinearGradient(x, y - r, x, y + r * 1.05);
      body.addColorStop(0, top);
      body.addColorStop(0.55, dim ? '#eeeef0' : isUser ? '#f6f6f8' : '#e6e6ea');
      body.addColorStop(1, bottom);
      ctx.fillStyle = body;
      ctx.fill();

      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      // --- depth pass 2: inner bevel. A light inset rim on the upper edge and a
      // soft dark inset on the lower edge, both clipped to the silhouette.
      if (!dim) {
        ctx.save();
        trace();
        ctx.clip();

        // upper-left specular bloom
        const bloom = ctx.createRadialGradient(
          x - r * 0.3,
          y - r * 0.42,
          0,
          x - r * 0.3,
          y - r * 0.42,
          r * 1.15,
        );
        bloom.addColorStop(0, C.highlight);
        bloom.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = bloom;
        ctx.globalAlpha = 0.6;
        trace();
        ctx.fill();
        ctx.globalAlpha = 1;

        // light bevel along the top edge
        trace();
        ctx.strokeStyle = C.bevel;
        ctx.lineWidth = 1.6 / globalScale;
        ctx.stroke();

        // grounded shade along the bottom edge
        const floor = ctx.createLinearGradient(x, y + r * 0.1, x, y + r);
        floor.addColorStop(0, 'rgba(10,10,12,0)');
        floor.addColorStop(1, 'rgba(10,10,12,0.16)');
        ctx.fillStyle = floor;
        trace();
        ctx.fill();

        ctx.restore();
      }

      // --- rim: crisp dark edge that separates the node from the white grid.
      trace();
      ctx.strokeStyle = outline;
      ctx.lineWidth = (node.isSubject ? 1.5 : 1.2) / globalScale;
      ctx.stroke();

      ctx.restore();

      // --- type pictogram inside the body.
      const glyphColor = dim
        ? C.grayDim
        : isSelected
          ? C.redBright
          : isUser
            ? C.glyph
            : C.glyphSoft;
      drawGlyph(ctx, node.kind, x, y, r * 0.52, glyphColor, globalScale);

      // Red treatment: subject outline, selection ring, path membership.
      if (node.isSubject || isSelected || (onPath && selectedSet.size > 1)) {
        // Soft red halo so active nodes read first on a dense white canvas.
        if (isSelected || node.isSubject) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(x, y, r + 3.4, 0, 2 * Math.PI);
          ctx.strokeStyle = isSelected ? C.redBright : C.red;
          ctx.globalAlpha = 0.16;
          ctx.lineWidth = 4.5 / globalScale;
          ctx.stroke();
          ctx.restore();
        }
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, 2 * Math.PI);
        ctx.strokeStyle = isSelected ? C.redBright : node.isSubject ? C.red : C.redDim;
        ctx.lineWidth = (isSelected ? 2 : 1.4) / globalScale;
        ctx.stroke();
      }

      if (isSelected) {
        const order = selected.indexOf(node.id) + 1;
        const fontSize = 9.5 / globalScale;
        ctx.font = `700 ${fontSize}px ui-monospace, monospace`;
        ctx.fillStyle = C.redBright;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(order), x + r + 4.5, y - r - 1);
      }

      const labelVisible =
        showLabels || isSelected || hovered === node.id || node.isSubject || isUser;
      if (!labelVisible) return;

      const fontSize = (node.isSubject ? 10 : 8.5) / globalScale;
      ctx.font = `${node.isSubject ? 600 : 500} ${fontSize}px ui-monospace, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = dim
        ? C.grayDim
        : isSelected
          ? C.redBright
          : isUser
            ? C.ink
            : C.labelSoft;
      const text = showLabels || isSelected || hovered === node.id ? node.id : shortLabel(node.id);
      ctx.fillText(text.length > 14 ? `${text.slice(0, 13)}\u2026` : text, x, y + r + 3.5);
    },
    [selectedSet, selected, pathNodeIds, activeNodeIds, showLabels, hovered],
  );

  const paintPointerArea = useCallback(
    (raw: unknown, color: string, ctx: CanvasRenderingContext2D) => {
      const node = raw as GNode;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, nodeRadius(node) + 3, 0, 2 * Math.PI);
      ctx.fill();
    },
    [],
  );

  const linkState = useCallback(
    (raw: unknown) => {
      const link = raw as GLink;
      const key = edgeKey(nodeIdOf(link.source), nodeIdOf(link.target));
      if (pathEdgeKeys.has(key)) return 'path' as const;
      if (incidentEdgeKeys.has(key)) return 'incident' as const;
      if (activeNodeIds !== null) return 'dim' as const;
      if (emphasis && link.emphasis) return 'emphasis' as const;
      return 'normal' as const;
    },
    [pathEdgeKeys, incidentEdgeKeys, activeNodeIds, emphasis],
  );

  const drawHull = useCallback(
    (ctx: CanvasRenderingContext2D, globalScale: number) => {
      if (!clusterConnected) return;
      const points: [number, number][] = [];
      for (const id of pathNodeIds) {
        const node = graph.nodeById.get(id);
        if (node && node.x !== undefined && node.y !== undefined) points.push([node.x, node.y]);
      }
      if (points.length < 3) return;
      const hull = convexHull(points);
      if (hull.length < 3) return;
      const cx = hull.reduce((sum, p) => sum + p[0], 0) / hull.length;
      const cy = hull.reduce((sum, p) => sum + p[1], 0) / hull.length;
      const inflated = hull.map(([px, py]): [number, number] => {
        const dx = px - cx;
        const dy = py - cy;
        const d = Math.hypot(dx, dy) || 1;
        return [px + (dx / d) * 14, py + (dy / d) * 14];
      });
      ctx.save();
      tracePolygon(ctx, inflated);
      ctx.setLineDash([5 / globalScale, 4 / globalScale]);
      ctx.lineWidth = 1 / globalScale;
      ctx.strokeStyle = C.redDim;
      ctx.stroke();
      ctx.restore();
    },
    [clusterConnected, pathNodeIds, graph.nodeById],
  );

  /* ---------------- panel data ---------------- */

  const focusId = selected.length > 0 ? selected[selected.length - 1] : null;
  const focusNode = focusId ? graph.nodeById.get(focusId) ?? null : null;

  const focusDegree = focusId ? (adjacency.get(focusId) ?? []).length : 0;
  const focusUserNeighbors = focusId
    ? (adjacency.get(focusId) ?? []).filter((e) => graph.nodeById.get(e.to)?.kind === 'USER').length
    : 0;

  const totalEntities =
    investigation.graph_evidence.shared_devices.length +
    investigation.graph_evidence.shared_ips.length +
    investigation.graph_evidence.shared_addresses.length +
    investigation.graph_evidence.shared_payment_instruments.length;

  const hiddenTotal = graph.hiddenEntities + graph.hiddenRelated;

  /* ---------------- render ---------------- */

  return (
    <div ref={wrapperRef} className="tech-grid absolute inset-0 overflow-hidden">
      <ForceGraph2D
        ref={fgRef as never}
        graphData={graphData as never}
        width={size.width}
        height={size.height}
        backgroundColor="rgba(0,0,0,0)"
        nodeRelSize={4}
        nodeLabel={(node: unknown) => {
          const n = node as GNode;
          return `${KIND_LABELS[n.kind]} — ${n.id}`;
        }}
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={paintPointerArea}
        linkColor={(link: unknown) => {
          const state = linkState(link);
          if (state === 'path') return C.redBright;
          if (state === 'incident') return C.red;
          if (state === 'emphasis') return C.red;
          if (state === 'dim') return C.grayDim;
          return C.grayMid;
        }}
        linkWidth={(link: unknown) => {
          const state = linkState(link);
          if (state === 'path') return 2.8;
          if (state === 'incident') return 2;
          if (state === 'emphasis') return 1.9;
          if (state === 'dim') return 0.7;
          return 1.15;
        }}
        linkLineDash={(link: unknown) =>
          (link as GLink).linkKind === 'user_user' ? [4, 3] : null
        }
        linkLabel={(link: unknown) => (link as GLink).label}
        onNodeClick={handleNodeClick}
        onNodeHover={(node: unknown) => setHovered(node ? nodeIdOf(node) : null)}
        onRenderFramePre={drawHull}
        onEngineStop={handleEngineStop}
        enableNodeDrag
        autoPauseRedraw={false}
        warmupTicks={80}
        cooldownTicks={200}
        d3AlphaDecay={0.05}
        d3VelocityDecay={0.55}
        minZoom={0.4}
        maxZoom={8}
      />

      {/* ---------------- floating overlay ----------------
           Stacking lanes: panels z-30 > controls z-20 > legend z-10 > canvas. */}
      <div className="pointer-events-none absolute inset-0">
        {/* left column */}
        {leftPanels && (
          <>
            <div
              className={clsx(
                'panel-column pointer-events-none absolute bottom-14 left-3 top-3 z-30 flex w-[44%] max-w-[252px] flex-col gap-2',
                !leftOpen && 'panel-column-collapsed panel-column-collapsed-left',
              )}
              aria-hidden={!leftOpen}
            >
              <div className="pointer-events-auto flex justify-end">
                <IconButton
                  icon={<ChevronLeft className="h-3.5 w-3.5" strokeWidth={1.8} />}
                  label="Collapse left panels"
                  onClick={() => setLeftOpen(false)}
                />
              </div>
              {leftPanels}
            </div>
            {!leftOpen && (
              <IconButton
                icon={<ChevronRight className="h-3.5 w-3.5" strokeWidth={1.8} />}
                label="Show left panels"
                onClick={() => setLeftOpen(true)}
                className="absolute left-3 top-3 z-30"
              />
            )}
          </>
        )}

        {/* right column: optional panels + inspector */}
        <div
          className={clsx(
            'panel-column pointer-events-none absolute bottom-14 right-3 top-3 z-30 flex w-[46%] max-w-[288px] flex-col gap-2',
            !rightOpen && 'panel-column-collapsed panel-column-collapsed-right',
          )}
          aria-hidden={!rightOpen}
        >
          <div className="pointer-events-auto flex justify-start">
            <IconButton
              icon={<ChevronRight className="h-3.5 w-3.5" strokeWidth={1.8} />}
              label="Collapse right panels"
              onClick={() => setRightOpen(false)}
            />
          </div>

          {rightPanels}

          <Panel
            strong
            title={selected.length > 1 ? 'Selected Network' : 'Node Inspector'}
            meta={selected.length > 0 ? `${selected.length} selected` : undefined}
            className="min-h-[180px] flex-1"
          >
            {!focusNode && (
              <div className="space-y-2">
                <p className="body-text">
                  Click a node to inspect it. Select a second node to trace the relationship
                  between them through the loaded neighborhood.
                </p>
                <div className="border-t border-black/[0.09] pt-1">
                  <Row label="Subject" value={investigation.user_id} strong wrap mono />
                  <Row label="Connected Entities" value={String(totalEntities)} mono />
                  <Row
                    label="Related Accounts"
                    value={String(investigation.related_accounts.length)}
                    mono
                  />
                  <Row
                    label="Nodes / Edges"
                    value={`${graph.nodes.length} / ${graph.links.length}`}
                    mono
                  />
                  {neighbors === null && (
                    <Row label="Status" value="Resolving shared entities" red />
                  )}
                </div>
              </div>
            )}

            {focusNode && (
              <div className="space-y-1">
                <Row label="Node" value={KIND_LABELS[focusNode.kind]} strong />
                <Row label="ID" value={focusNode.id} strong wrap mono />

                {focusNode.isSubject && (
                  <>
                    <Row
                      label="Risk Score"
                      value={`${(investigation.risk_score * 100).toFixed(1)}%`}
                      red
                      mono
                    />
                    <Row label="Risk Level" value={investigation.risk_level.toUpperCase()} red />
                    <Row label="Connected Entities" value={String(totalEntities)} mono />
                    <Row
                      label="Related Accounts"
                      value={String(investigation.related_accounts.length)}
                      mono
                    />
                    {investigation.community_context && (
                      <>
                        <Row
                          label="Community Size"
                          value={String(investigation.community_context.community_size)}
                          mono
                        />
                        <Row
                          label="Community Density"
                          value={investigation.community_context.community_density.toFixed(3)}
                          mono
                        />
                        <Row
                          label="Projection Degree"
                          value={String(investigation.community_context.user_degree)}
                          mono
                        />
                      </>
                    )}
                  </>
                )}

                {focusNode.related && (
                  <>
                    <Row
                      label="Shared Entities"
                      value={String(focusNode.related.shared_entity_count)}
                      mono
                    />
                    <Row
                      label="Relationships"
                      value={focusNode.related.relationship_types.map(relationshipLabel).join(' · ')}
                      wrap
                    />
                    <Row label="Connections In View" value={String(focusDegree)} mono />
                  </>
                )}

                {focusNode.kind !== 'USER' && (
                  <>
                    <Row label="Connected Users" value={String(focusUserNeighbors)} mono />
                    <Row label="Relationship" value={focusNode.relationshipLabel ?? '—'} />
                    <Row label="Connections In View" value={String(focusDegree)} mono />
                  </>
                )}

                <div className="flex flex-col gap-1.5 pt-2">
                  {focusNode.kind === 'USER' && !focusNode.isSubject && (
                    <ActionButton primary onClick={() => navigate(`/users/${focusNode.id}`)}>
                      Investigate User
                    </ActionButton>
                  )}
                  {focusNode.isSubject && (
                    <div className="label-caps">Current investigation subject</div>
                  )}
                  <ActionButton onClick={() => setSelected([])}>Clear Selection</ActionButton>
                </div>
              </div>
            )}

            {selected.length > 0 && (
              <div className="mt-3 space-y-2 border-t border-black/[0.09] pt-2">
                <div className="text-[12px] font-semibold uppercase leading-[1.4] tracking-[0.045em] text-ink-50">
                  {clusterConnected ? 'Investigation Cluster' : 'Investigation Path'}
                </div>
                <ol className="space-y-[3px]">
                  {selected.map((id, i) => {
                    const node = graph.nodeById.get(id);
                    return (
                      <li
                        key={id}
                        className="flex items-baseline gap-1.5 text-[13px] leading-[1.5]"
                      >
                        <span className="tech-num font-semibold text-signal-bright">{i + 1}</span>
                        <span className="tech-id truncate font-medium text-ink-50">{id}</span>
                        <span className="label-caps ml-auto shrink-0">
                          {node ? KIND_LABELS[node.kind] : ''}
                        </span>
                      </li>
                    );
                  })}
                </ol>
                {segments.length > 0 && (
                  <div className="space-y-1 pt-0.5">
                    {segments.map((segment) => (
                      <div
                        key={`${segment.from}->${segment.to}`}
                        className="text-[12px] leading-[1.55]"
                      >
                        <span className="tech-id font-medium text-ink-350">
                          {shortLabel(segment.from)} → {shortLabel(segment.to)}:{' '}
                        </span>
                        {segment.path ? (
                          <span className="tech-id font-semibold text-signal-bright">
                            {segment.path.map(shortLabel).join(' → ')}
                          </span>
                        ) : (
                          <span className="font-normal text-ink-350">
                            no relationship in loaded neighborhood
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                <p className="body-muted">
                  Selection is an investigator hypothesis, not a determination.
                </p>
              </div>
            )}

            {(graph.unresolvedTypes.size > 0 || hiddenTotal > 0) && (
              <div className="body-muted mt-3 space-y-1 border-t border-black/[0.09] pt-2">
                {graph.unresolvedTypes.size > 0 && (
                  <div>
                    Dashed edges: backend reports the relationship type but the specific shared
                    entity is not uniquely identifiable.
                  </div>
                )}
                {hiddenTotal > 0 && (
                  <div>
                    {hiddenTotal} further neighbor{hiddenTotal === 1 ? '' : 's'} not rendered
                    (bounded view).
                  </div>
                )}
              </div>
            )}
          </Panel>
        </div>
        {!rightOpen && (
          <IconButton
            icon={<ChevronLeft className="h-3.5 w-3.5" strokeWidth={1.8} />}
            label="Show right panels"
            onClick={() => setRightOpen(true)}
            className="absolute right-3 top-3 z-30"
          />
        )}

        {/* legend — own glass chip on the graph layer (z-10), centred at the
            bottom and stacked directly above the control bar. Kept clear of the
            left/right panel columns horizontally and of the controls vertically,
            so it can never collide with Graph Connectivity. */}
        <div className="pointer-events-none absolute bottom-[3.75rem] left-1/2 z-10 hidden w-[calc(100%-1.5rem)] max-w-[520px] -translate-x-1/2 justify-center xl:flex">
          <div className="glass-chip body-muted pointer-events-none px-2.5 py-1.5">
            <div className="flex flex-wrap justify-center gap-x-3 gap-y-0.5 text-[12px] font-medium text-ink-50">
              <span>&#9675; User</span>
              <span>&#9633; Device</span>
              <span>&#9671; IP</span>
              <span>&#11040; Address</span>
              <span>&#9670; Payment</span>
            </div>
            <div className="mt-0.5 text-center">
              <span className="font-semibold text-signal-bright">RED</span> selection / path ·{' '}
              <span>GRAY</span> relationship · <span>DASHED</span> type only
            </div>
          </div>
        </div>

        {/* floating control bar — bottom center, own layer above the legend */}
        <div className="glass-chip pointer-events-auto absolute bottom-3 left-1/2 z-20 flex -translate-x-1/2 items-center gap-1.5 px-2 py-1.5">
          <span className="tech-num mr-1 hidden text-[11.5px] font-semibold text-ink-350 md:inline">
            {graph.nodes.length}N / {graph.links.length}E
          </span>
          <ControlButton
            icon={<RotateCcw className="h-3 w-3" strokeWidth={1.6} />}
            label="Reset"
            onClick={handleReset}
          />
          <ControlButton
            icon={<Maximize2 className="h-3 w-3" strokeWidth={1.6} />}
            label="Fit"
            onClick={handleFit}
          />
          <ControlButton
            icon={<Eraser className="h-3 w-3" strokeWidth={1.6} />}
            label={selected.length > 0 ? `Clear (${selected.length})` : 'Clear'}
            onClick={() => setSelected([])}
            active={selected.length > 0}
          />
          <ControlButton
            icon={<Tag className="h-3 w-3" strokeWidth={1.6} />}
            label="Labels"
            onClick={() => setShowLabels((v) => !v)}
            active={showLabels}
          />
          <ControlButton
            icon={<Highlighter className="h-3 w-3" strokeWidth={1.6} />}
            label="Emphasis"
            onClick={() => setEmphasis((v) => !v)}
            active={emphasis}
          />
        </div>
      </div>
    </div>
  );
}
