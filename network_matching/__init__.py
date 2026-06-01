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
from .bgraph_prep import validate_b_geometry

__all__ = [
    "DuckDBMapMatcher",
    "dtw_align",
    "GraphDTWMatcher",
    "build_local_digraph",
    "graph_dtw_align",
    "match_edge_to_bgraph",
    "validate_b_geometry",
    "setup_logging",
    "get_logger",
]
