import logging as _logging

from .matcher import DuckDBMapMatcher
from .dtw import dtw_align
from .logging_utils import setup_logging, get_logger

# Quiet by default; call setup_logging() to emit to a file/console.
_logging.getLogger("network_matching").addHandler(_logging.NullHandler())
from .graph_dtw import (
    GraphDTWMatcher,
    build_local_digraph,
    graph_dtw_align,
    match_edge_to_bgraph,
)
from .dag_dtw import NotADAG, NotATree, match_dag_to_bgraph, topological_order
from .tree_dtw import match_tree_to_bgraph
from .dag_conditioning import conditioned_labels, min_feedback_vertex_set
from .bgraph_prep import validate_b_geometry
from .thresholds import suggest_thresholds, isolation_forest_flags

__all__ = [
    "DuckDBMapMatcher",
    "dtw_align",
    "GraphDTWMatcher",
    "build_local_digraph",
    "graph_dtw_align",
    "match_edge_to_bgraph",
    "match_dag_to_bgraph",
    "match_tree_to_bgraph",
    "topological_order",
    "NotADAG",
    "NotATree",
    "conditioned_labels",
    "min_feedback_vertex_set",
    "validate_b_geometry",
    "suggest_thresholds",
    "isolation_forest_flags",
    "setup_logging",
    "get_logger",
]
