"""Graph subpackage: heterogeneous relationship graph + user-centric features.

See :mod:`src.graph.schema` for the representation-choice rationale. The graph
uses only observable relationships; ground-truth labels never influence
construction, projection, features, or communities.
"""

from __future__ import annotations

from src.graph.builder import build_graph, edges_of_type, nodes_of_type
from src.graph.communities import detect_communities
from src.graph.features import compute_centrality, compute_user_graph_features
from src.graph.projection import build_user_projection
from src.graph.schema import EdgeType, NodeType

__all__ = [
    "NodeType",
    "EdgeType",
    "build_graph",
    "nodes_of_type",
    "edges_of_type",
    "build_user_projection",
    "compute_user_graph_features",
    "compute_centrality",
    "detect_communities",
]
