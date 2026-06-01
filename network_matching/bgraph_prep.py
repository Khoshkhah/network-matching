"""
Pre-flight geometry validation for graph-DTW.

Graph-DTW infers B-network connectivity by snapping B-edge endpoints that are within a
tolerance. Before matching, it helps to look at how B's endpoints actually meet: are connected
edges sharing *exactly* the same endpoint, or are they a few centimetres / metres apart? That
distribution tells you what ``snap_tolerance_m`` to use (above the gap noise floor, below real
road spacing) and flags data defects (zero-length edges, duplicate vertices).

Main entry point: :func:`validate_b_geometry`.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point

Coord = Tuple[float, float]
logger = logging.getLogger("network_matching.bgraph_prep")


def _collect_endpoints(b_edges: Sequence[Tuple[Any, LineString]]) -> Tuple[np.ndarray, List[Any]]:
    pts: List[Coord] = []
    owner: List[Any] = []
    for eid, geom in b_edges:
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        pts.append((coords[0][0], coords[0][1]))
        owner.append(eid)
        pts.append((coords[-1][0], coords[-1][1]))
        owner.append(eid)
    return np.asarray(pts, float), owner


def _nearest_other_gaps(pts: np.ndarray) -> np.ndarray:
    """For each endpoint, distance to the nearest *other* endpoint."""
    if len(pts) < 2:
        return np.asarray([], float)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)  # col 0 = self (0), col 1 = nearest other
        return d[:, 1]
    except Exception:
        gaps = np.empty(len(pts))
        for i in range(len(pts)):
            dd = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            dd[i] = np.inf
            gaps[i] = dd.min()
        return gaps


def validate_b_geometry(
    b_edges: Sequence[Tuple[Any, LineString]],
    snap_candidates: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    report: bool = True,
) -> Dict[str, Any]:
    """Report B-network geometry quality to help choose ``snap_tolerance_m``.

    Parameters
    ----------
    b_edges:
        List of ``(edge_id, shapely LineString)`` in a projected CRS (meters).
    snap_candidates:
        Tolerances (m) to tabulate: how many endpoint-to-nearest-other-endpoint gaps fall in
        ``(0, t]`` for each ``t`` (i.e. would newly connect at that tolerance).
    report:
        Print a readable summary.

    Returns
    -------
    dict with keys ``n_edges, n_endpoints, exact_shared, near_within (per-tolerance counts),
    gap_percentiles_m, degenerate``.
    """
    degenerate: List[Tuple[Any, str]] = []
    for eid, geom in b_edges:
        coords = list(geom.coords)
        if len(coords) < 2 or geom.length == 0:
            degenerate.append((eid, "zero_length"))
            continue
        dups = sum(1 for k in range(1, len(coords))
                   if Point(coords[k - 1]).distance(Point(coords[k])) < 1e-9)
        if dups:
            degenerate.append((eid, f"{dups}_duplicate_vertices"))

    pts, _owner = _collect_endpoints(b_edges)
    gaps = _nearest_other_gaps(pts)
    exact = int((gaps <= 1e-9).sum())
    nonzero = gaps[gaps > 1e-9]

    near_within = {float(t): int(((gaps > 1e-9) & (gaps <= t)).sum()) for t in snap_candidates}
    if nonzero.size:
        p = np.percentile(nonzero, [50, 75, 90, 95])
        percentiles = {"p50": float(p[0]), "p75": float(p[1]), "p90": float(p[2]), "p95": float(p[3])}
    else:
        percentiles = {"p50": float("nan"), "p75": float("nan"),
                       "p90": float("nan"), "p95": float("nan")}

    summary = {
        "n_edges": len(list(b_edges)),
        "n_endpoints": int(len(pts)),
        "exact_shared": exact,
        "near_within": near_within,
        "gap_percentiles_m": percentiles,
        "degenerate": degenerate,
    }

    logger.info("validate_b_geometry: %d edges, %d endpoints, %d exact-shared, "
                "gap p50=%.2fm; degenerate=%d",
                summary["n_edges"], summary["n_endpoints"], exact,
                percentiles["p50"], len(degenerate))

    if report:
        print("=" * 56)
        print("   B-NETWORK GEOMETRY VALIDATION (graph-DTW pre-flight)")
        print("=" * 56)
        print(f"edges                 : {summary['n_edges']}")
        print(f"endpoints              : {summary['n_endpoints']}")
        print(f"exact-shared endpoints : {exact}  (gap == 0)")
        print("nearest-other-endpoint gap percentiles (m):", percentiles)
        print("endpoints with a near (but unequal) neighbour within tolerance:")
        for t in snap_candidates:
            print(f"   <= {t:>5} m : {near_within[float(t)]}")
        if degenerate:
            print(f"degenerate edges       : {len(degenerate)} (e.g. {degenerate[:5]})")
        else:
            print("degenerate edges       : 0")
        # crude suggestion: smallest candidate above the noise floor (p50 of nonzero gaps)
        print("-" * 56)
        print("Tip: pick snap_tolerance_m above typical connected-endpoint gaps but well below "
              "road spacing.")
    return summary
