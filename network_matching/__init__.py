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
from .tree_dtw import NotATree, match_tree
from .bgraph_prep import validate_b_geometry
from .thresholds import suggest_thresholds, isolation_forest_flags

__all__ = [
    "DuckDBMapMatcher",
    "dtw_align",
    "GraphDTWMatcher",
    "build_local_digraph",
    "graph_dtw_align",
    "match_edge_to_bgraph",
    "match_tree",
    "NotATree",
    "validate_b_geometry",
    "suggest_thresholds",
    "isolation_forest_flags",
    "setup_logging",
    "get_logger",
]
