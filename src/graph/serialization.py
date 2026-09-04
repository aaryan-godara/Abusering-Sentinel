"""Deterministic graph serialization.

Persists a heterogeneous or projected graph to a compact JSON node-link format
and reloads it exactly. We use a sorted node/edge ordering so the on-disk file is
byte-stable for a given graph, which makes reconstruction reproducible and
diff-friendly.

We deliberately avoid pickle (not portable / not safe) and GraphML (verbose,
type-coercion quirks). Node-link JSON keeps node/edge type attributes intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def save_graph(graph: nx.Graph, path: Path) -> Path:
    """Write ``graph`` to ``path`` as sorted node-link JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes = [
        {"id": n, **{k: v for k, v in sorted(d.items())}}
        for n, d in sorted(graph.nodes(data=True), key=lambda x: str(x[0]))
    ]
    edges = []
    for u, v, d in graph.edges(data=True):
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        edges.append({"source": a, "target": b, **{k: val for k, val in sorted(d.items())}})
    edges.sort(key=lambda e: (str(e["source"]), str(e["target"])))

    payload = {"directed": False, "multigraph": False, "nodes": nodes, "edges": edges}
    path.write_text(json.dumps(payload, indent=0, default=str), encoding="utf-8")
    return path


def load_graph(path: Path) -> nx.Graph:
    """Reload a graph written by :func:`save_graph`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = nx.Graph()
    for node in payload["nodes"]:
        node = dict(node)
        node_id = node.pop("id")
        graph.add_node(node_id, **node)
    for edge in payload["edges"]:
        edge = dict(edge)
        s = edge.pop("source")
        t = edge.pop("target")
        graph.add_edge(s, t, **edge)
    return graph


__all__ = ["save_graph", "load_graph"]
