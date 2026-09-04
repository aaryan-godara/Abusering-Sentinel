"""Feature-engineering subpackage (Phase 2).

Provides the non-graph baseline feature builder and the combined feature
pipeline that emits baseline and graph-augmented datasets per split.
"""

from __future__ import annotations

from src.features.baseline import BASELINE_FEATURES, build_baseline_features

__all__ = ["BASELINE_FEATURES", "build_baseline_features"]
