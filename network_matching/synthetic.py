"""
Synthetic graph-DTW test cases and geometric perturbations.

Two building blocks for debugging / stress-testing the algorithm without any real data:

- :data:`SCENARIOS` -- named hand-built (A-edge, local B-network) cases in a plain meter CRS,
  each isolating one algorithmic behaviour (route stitching, parallel-road trap, junction choice,
  cycle termination, U-turn prohibition, overhang trimming, connectivity gaps, curvature).
  ``get_scenario(name)`` returns ``{"coords_a", "b_edges", "description", "defaults"}`` where
  ``defaults`` are the ``match_edge_to_bgraph`` kwargs the case is designed around.

- :data:`PERTURBATIONS` -- named geometric distortions of the A-edge (lateral / longitudinal
  shift, Gaussian noise, rotation, end cropping / stretching, resampling), each a family
  ``magnitude -> perturbed copy`` with a unit and a default magnitude grid. Apply one with
  :func:`apply_perturbation`; re-matching the perturbed edge against the *unchanged* B-network
  probes the algorithm's robustness (does the route survive? how does drift grow?).

Used by ``scripts/graph_dtw_debug_viz.py`` (single-case algorithm deep-dive) and
``scripts/graph_dtw_perturb_test.py`` (families x magnitudes robustness sweep), and directly
unit-tested in ``tests/test_graph_dtw_perturbations.py``.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString

Coord = Tuple[float, float]


# --------------------------------------------------------------------------------------
# Scenarios: hand-built A-edge + local B-network, each isolating one behaviour
# --------------------------------------------------------------------------------------
def _quarter_arc(radius: float, offset: float, n: int, t0: float = 0.0,
                 t1: float = 90.0) -> List[Coord]:
    """Points on a quarter circle (centre at origin) between angles ``t0``..``t1`` degrees,
    at ``radius + offset``."""
    t = np.radians(np.linspace(t0, t1, n))
    r = radius + offset
    return [(float(r * np.cos(a)), float(r * np.sin(a))) for a in t]


def _scenarios() -> Dict[str, Dict[str, Any]]:
    S: Dict[str, Dict[str, Any]] = {}

    S["split"] = dict(
        description="One straight A-edge spans three connected B-edges (0.2 m offset) -> "
                    "the stitched route B1-B2-B3.",
        coords_a=[(0.0, 0.0), (30.0, 0.0)],
        b_edges=[("B1", LineString([(0.0, 0.2), (10.0, 0.2)])),
                 ("B2", LineString([(10.0, 0.2), (20.0, 0.2)])),
                 ("B3", LineString([(20.0, 0.2), (30.0, 0.2)]))],
    )

    S["parallel_trap"] = dict(
        description="The split chain (0.2 m off) plus an ISOLATED full-length parallel road "
                    "8 m south. Correct route is the chain; shifting A south (negative lateral "
                    "shift) pulls it toward the trap.",
        coords_a=[(0.0, 0.0), (30.0, 0.0)],
        b_edges=[("B1", LineString([(0.0, 0.2), (10.0, 0.2)])),
                 ("B2", LineString([(10.0, 0.2), (20.0, 0.2)])),
                 ("B3", LineString([(20.0, 0.2), (30.0, 0.2)])),
                 ("B_trap", LineString([(0.0, -8.0), (30.0, -8.0)]))],
    )

    S["junction_turn"] = dict(
        description="A turns 90 deg north at a junction where a straight continuation also "
                    "exits. The route must follow the turn (B_west -> B_north), never B_east.",
        coords_a=[(0.0, 0.0), (15.0, 0.0), (15.0, 15.0)],
        b_edges=[("B_west", LineString([(0.0, 0.0), (15.0, 0.0)])),
                 ("B_north", LineString([(15.0, 0.0), (15.0, 15.0)])),
                 ("B_east", LineString([(15.0, 0.0), (30.0, 0.0)]))],
    )

    S["cycle"] = dict(
        description="A loop (B1 -> B_up -> B_back to start) hangs off A's straight path; the DP "
                    "must terminate and route straight through (B1 -> B4), not around the loop.",
        coords_a=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        b_edges=[("B1", LineString([(0.0, 0.0), (10.0, 0.0)])),
                 ("B_up", LineString([(10.0, 0.0), (10.0, 5.0)])),
                 ("B_back", LineString([(10.0, 5.0), (0.0, 0.0)])),
                 ("B4", LineString([(10.0, 0.0), (20.0, 0.0)]))],
    )

    S["stub"] = dict(
        description="A perpendicular edge ENDS at the junction (digitized into it). Forward-only "
                    "arcs make a dip-and-return U-turn onto it impossible: route is "
                    "B_main -> B_cont only.",
        coords_a=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        b_edges=[("B_main", LineString([(0.0, 0.0), (10.0, 0.0)])),
                 ("B_cont", LineString([(10.0, 0.0), (20.0, 0.0)])),
                 ("B_stub", LineString([(10.0, -8.0), (10.0, 0.0)]))],
    )

    S["overhang"] = dict(
        description="A (0..40 m) overhangs the B corridor (5..35 m) on both ends: route is "
                    "B1-B2 with overlap < 100% (the overhang is trimmed, not matched).",
        coords_a=[(0.0, 0.0), (40.0, 0.0)],
        b_edges=[("B1", LineString([(5.0, 0.2), (20.0, 0.2)])),
                 ("B2", LineString([(20.0, 0.2), (35.0, 0.2)]))],
    )

    S["gap"] = dict(
        description="The B chain has a 2 m gap mid-corridor (B1 ends at x=14, B2 starts at "
                    "x=16). With snap_tolerance < 2 m they never connect, so the route cannot "
                    "stitch across -- a connectivity failure case to inspect in the DP table.",
        coords_a=[(0.0, 0.0), (30.0, 0.0)],
        b_edges=[("B1", LineString([(0.0, 0.2), (14.0, 0.2)])),
                 ("B2", LineString([(16.0, 0.2), (30.0, 0.2)]))],
    )

    S["curve"] = dict(
        description="A quarter-circle road (r=40 m) split into three B-edges 0.3 m outside "
                    "A's arc; exercises bearing handling and non-axis-aligned geometry.",
        coords_a=_quarter_arc(40.0, 0.0, 13),
        b_edges=[("B_arc1", LineString(_quarter_arc(40.0, 0.3, 5, 0.0, 30.0))),
                 ("B_arc2", LineString(_quarter_arc(40.0, 0.3, 5, 30.0, 60.0))),
                 ("B_arc3", LineString(_quarter_arc(40.0, 0.3, 5, 60.0, 90.0)))],
    )

    for sc in S.values():
        sc.setdefault("defaults", dict(snap_tolerance_m=0.5, step_meters=2.0))
    return S


SCENARIOS: Dict[str, Dict[str, Any]] = _scenarios()


def list_scenarios() -> List[str]:
    return sorted(SCENARIOS)


def get_scenario(name: str) -> Dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {', '.join(list_scenarios())}")
    return SCENARIOS[name]


# --------------------------------------------------------------------------------------
# Perturbations: magnitude -> distorted copy of the A-edge (B stays untouched)
# --------------------------------------------------------------------------------------
def as_array(coords: Sequence[Coord]) -> np.ndarray:
    return np.asarray([(float(x), float(y)) for x, y in coords], float)


def resample(coords: Sequence[Coord], step: float) -> np.ndarray:
    """Uniform arc-length resampling at ``step`` meters (endpoints kept)."""
    P = as_array(coords)
    line = LineString(P)
    if step <= 0 or line.length == 0:
        return P
    n = max(int(np.ceil(line.length / step)), 1)
    s = np.linspace(0.0, line.length, n + 1)
    return np.asarray([(line.interpolate(float(t)).x, line.interpolate(float(t)).y) for t in s])


def _local_frames(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit tangent and left normal at every vertex (central differences)."""
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    L = np.hypot(T[:, 0], T[:, 1])
    L[L == 0] = 1.0
    T = T / L[:, None]
    Nrm = np.stack([-T[:, 1], T[:, 0]], axis=1)
    return T, Nrm


def lateral_shift(coords: Sequence[Coord], meters: float) -> np.ndarray:
    """Move every vertex ``meters`` along its local LEFT normal (negative = right)."""
    P = as_array(coords)
    _, Nrm = _local_frames(P)
    return P + Nrm * meters


def longitudinal_shift(coords: Sequence[Coord], meters: float) -> np.ndarray:
    """Slide every vertex ``meters`` along its local tangent (a lengthwise misregistration)."""
    P = as_array(coords)
    T, _ = _local_frames(P)
    return P + T * meters


def gaussian_noise(coords: Sequence[Coord], sigma: float, seed: int = 0,
                   densify_step: float = 5.0) -> np.ndarray:
    """IID Gaussian jitter of ``sigma`` meters per coordinate. The line is first resampled at
    ``densify_step`` so a sparse (e.g. 2-vertex) edge still gets a genuinely wiggly shape rather
    than a rigid displacement."""
    P = resample(coords, densify_step) if densify_step else as_array(coords)
    if sigma <= 0:
        return P
    rng = np.random.default_rng(seed)
    return P + rng.normal(0.0, sigma, P.shape)


def rotate(coords: Sequence[Coord], degrees: float) -> np.ndarray:
    """Rotate the whole edge about its centroid (a bearing/registration error)."""
    P = as_array(coords)
    c = P.mean(axis=0)
    a = np.radians(degrees)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (P - c) @ R.T + c


def crop_ends(coords: Sequence[Coord], pct: float) -> np.ndarray:
    """Remove ``pct`` percent of total arc length, half from each end (a partial edge)."""
    P = as_array(coords)
    line = LineString(P)
    cut = line.length * min(max(pct, 0.0), 90.0) / 100.0 / 2.0
    if cut <= 0:
        return P
    s = np.linspace(cut, line.length - cut, max(len(P), 5))
    return np.asarray([(line.interpolate(float(t)).x, line.interpolate(float(t)).y) for t in s])


def stretch_ends(coords: Sequence[Coord], meters: float) -> np.ndarray:
    """Extrapolate both ends outward ``meters`` along the end tangents (edge overhang)."""
    P = as_array(coords)
    if meters <= 0:
        return P
    t0 = P[0] - P[1]
    t1 = P[-1] - P[-2]
    t0 = t0 / (np.hypot(*t0) or 1.0)
    t1 = t1 / (np.hypot(*t1) or 1.0)
    return np.vstack([P[0] + t0 * meters, P, P[-1] + t1 * meters])


def reverse(coords: Sequence[Coord]) -> np.ndarray:
    """Reverse the digitized direction (matches against a directed B table should degrade)."""
    return as_array(coords)[::-1]


# Family registry: name -> how to apply one magnitude, its unit, and a default sweep grid.
PERTURBATIONS: Dict[str, Dict[str, Any]] = {
    "shift": dict(
        fn=lambda P, m, seed=0: lateral_shift(P, m), unit="m",
        description="lateral offset along the local normal (+ = left of travel)",
        grid=[0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]),
    "longitudinal": dict(
        fn=lambda P, m, seed=0: longitudinal_shift(P, m), unit="m",
        description="lengthwise slide along the local tangent",
        grid=[0.0, 2.0, 4.0, 6.0, 8.0, 12.0]),
    "noise": dict(
        fn=lambda P, m, seed=0: gaussian_noise(P, m, seed=seed), unit="m sigma",
        description="per-vertex Gaussian jitter (edge densified to 5 m first)",
        grid=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0]),
    "rotate": dict(
        fn=lambda P, m, seed=0: rotate(P, m), unit="deg",
        description="rotation about the edge centroid",
        grid=[0.0, 2.0, 5.0, 10.0, 15.0, 20.0]),
    "crop": dict(
        fn=lambda P, m, seed=0: crop_ends(P, m), unit="%",
        description="shorten: remove this % of length (half per end)",
        grid=[0.0, 10.0, 20.0, 30.0, 40.0]),
    "stretch": dict(
        fn=lambda P, m, seed=0: stretch_ends(P, m), unit="m",
        description="extend both ends outward along the end tangents",
        grid=[0.0, 2.0, 5.0, 10.0, 15.0]),
}


def apply_perturbation(coords: Sequence[Coord], name: str, magnitude: float,
                       seed: int = 0) -> np.ndarray:
    """Apply one named perturbation at ``magnitude`` and return the distorted copy of the edge."""
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; "
                       f"available: {', '.join(sorted(PERTURBATIONS))}")
    return PERTURBATIONS[name]["fn"](as_array(coords), float(magnitude), seed=seed)
