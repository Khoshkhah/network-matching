"""
Synthetic **DAG** test cases for DAG-DTW (docs/dag_dtw_matching.md §4).

Each scenario is a hand-built source DAG (a topologically-ordered, loop-free set of A-edges) plus
a target B-network it should match, in a plain meter CRS. They isolate one behaviour at a time --
the ladder ``chain`` (must reduce to graph-DTW), ``y_split`` (a branch), ``merge`` (two approaches),
``diamond`` (split + re-merge) -- for debugging the DP and for the playground.

``a_edges`` / ``b_edges`` are ``[(id, [(x, y), ...]), ...]``; ``get_dag(name)`` returns them as
shapely LineStrings ready for :func:`network_matching.dag_dtw.match_dag_to_bgraph`.
"""

from typing import Any, Dict, List, Sequence, Tuple

from shapely.geometry import LineString

Coord = Tuple[float, float]

# B sits a small constant offset off A so the intended match is unambiguous.
_OFF = 0.4


def _off(pts: Sequence[Coord]) -> List[Coord]:
    return [(float(x), float(y) + _OFF) for x, y in pts]


def _scenarios() -> Dict[str, Dict[str, Any]]:
    S: Dict[str, Dict[str, Any]] = {}

    # chain: one path split into two A-edges -> must behave like graph-DTW on a two-edge corridor.
    S["chain"] = dict(
        description="A single corridor as two A-edges (A1->A2) over a matching B chain. A "
                    "one-path DAG: DAG-DTW must reproduce graph-DTW here.",
        a_edges=[("A1", [(0, 0), (15, 0)]), ("A2", [(15, 0), (30, 0)])],
        b_edges=[("B1", _off([(0, 0), (15, 0)])), ("B2", _off([(15, 0), (30, 0)]))],
    )

    # y_split: main corridor branches into two exits at a junction; B branches the same way.
    S["y_split"] = dict(
        description="A_main branches into A_left and A_right at one junction; B_main branches "
                    "into B_left / B_right. The junction must map to ONE B-vertex (φ shared).",
        a_edges=[("A_main", [(0, 0), (15, 0)]),
                 ("A_left", [(15, 0), (28, 9)]),
                 ("A_right", [(15, 0), (28, -9)])],
        b_edges=[("B_main", _off([(0, 0), (15, 0)])),
                 ("B_left", _off([(15, 0), (28, 9)])),
                 ("B_right", _off([(15, 0), (28, -9)]))],
    )

    # merge: two approaches join into one outgoing corridor; both must agree on the junction.
    S["merge"] = dict(
        description="A_top and A_bot merge into A_out at one junction; B merges the same way. "
                    "The sum over the two incoming A-branches happens at the merge vertex.",
        a_edges=[("A_top", [(0, 9), (15, 0)]),
                 ("A_bot", [(0, -9), (15, 0)]),
                 ("A_out", [(15, 0), (30, 0)])],
        b_edges=[("B_top", _off([(0, 9), (15, 0)])),
                 ("B_bot", _off([(0, -9), (15, 0)])),
                 ("B_out", _off([(15, 0), (30, 0)]))],
    )

    # diamond: split then re-merge -- exercises the conserved-flow split factor at reconvergence.
    S["diamond"] = dict(
        description="A splits at one junction and re-merges at another (a diamond); the 1/outdeg "
                    "split factor keeps the shared parts counted once.",
        a_edges=[("A_in", [(0, 0), (10, 0)]),
                 ("A_up", [(10, 0), (20, 7)]),
                 ("A_dn", [(10, 0), (20, -7)]),
                 ("A_up2", [(20, 7), (30, 0)]),
                 ("A_dn2", [(20, -7), (30, 0)]),
                 ("A_out", [(30, 0), (40, 0)])],
        b_edges=[("B_in", _off([(0, 0), (10, 0)])),
                 ("B_up", _off([(10, 0), (20, 7)])),
                 ("B_dn", _off([(10, 0), (20, -7)])),
                 ("B_up2", _off([(20, 7), (30, 0)])),
                 ("B_dn2", _off([(20, -7), (30, 0)])),
                 ("B_out", _off([(30, 0), (40, 0)]))],
    )

    for sc in S.values():
        sc.setdefault("defaults", dict(snap_tolerance_m=0.5, step_meters=2.0))
    return S


DAG_SCENARIOS: Dict[str, Dict[str, Any]] = _scenarios()


def list_dags() -> List[str]:
    return sorted(DAG_SCENARIOS)


def get_dag(name: str) -> Dict[str, Any]:
    """Return ``{a_edges, b_edges, defaults, description}`` with geometries as LineStrings."""
    if name not in DAG_SCENARIOS:
        raise KeyError(f"unknown DAG {name!r}; available: {', '.join(list_dags())}")
    sc = DAG_SCENARIOS[name]
    return dict(
        a_edges=[(eid, LineString(pts)) for eid, pts in sc["a_edges"]],
        b_edges=[(eid, LineString(pts)) for eid, pts in sc["b_edges"]],
        defaults=dict(sc["defaults"]),
        description=sc["description"],
    )
