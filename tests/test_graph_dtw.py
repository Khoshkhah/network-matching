"""
Tests for the graph-DTW algorithm (network_matching/graph_dtw.py).

These exercise the algorithm directly on small hand-built B-edge lists in a local meter CRS
-- no DuckDB needed. Runnable two ways:

    pytest tests/test_graph_dtw.py          # assertion-based
    python tests/test_graph_dtw.py          # prints a readable report

Scenarios:
  1. SPLIT          -- one A-edge spans three connected B-edges -> single stitched route.
  2. PARALLEL/ISOLATED -- a connected chain beats an isolated full-length parallel road
                       (the core "reduce wrong matches" win vs edge-to-edge).
  3. SNAP TOLERANCE -- near-but-unequal endpoints connect only when within the tolerance.
  4. CYCLE          -- a loop in GB terminates and returns a sensible route.
  5. NO U-TURN      -- the directed graph cannot leave a junction on one edge and return
                       (the perpendicular-stub artifact a shared-junction vertex would allow).
  6. ZERO-TRAVERSAL -- an edge A only *touches* at a junction (never walks) is dropped from the
                       route (the OSM-1251 phantom-tail case).

The graph is built from a **directed** edge table (forward arcs only): a B-edge digitized
against A is matched via its reverse twin, not a synthesized backward arc.
"""

import os
import sys

from shapely.geometry import LineString

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import match_edge_to_bgraph  # noqa: E402


def _route_ids(result):
    return [eid for (eid, _direction, _seq) in result["route"]]


# --------------------------------------------------------------------------------------
# 1. Split: one A-edge across three connected B-edges
# --------------------------------------------------------------------------------------
def test_split_stitches_three_edges():
    coords_a = [(0.0, 0.0), (30.0, 0.0)]
    b_edges = [
        ("B1", LineString([(0.0, 0.2), (10.0, 0.2)])),
        ("B2", LineString([(10.0, 0.2), (20.0, 0.2)])),
        ("B3", LineString([(20.0, 0.2), (30.0, 0.2)])),
    ]
    res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0)

    assert _route_ids(res) == ["B1", "B2", "B3"]
    assert all(d == "forward" for (_e, d, _s) in res["route"])
    assert res["metrics"]["n_edges"] == 3
    assert res["avg_distance"] < 0.5          # ~0.2 m drift
    assert res["metrics"]["overlap_pct"] == 100


# --------------------------------------------------------------------------------------
# 2. Parallel / isolated: connected chain must beat an isolated full-length parallel road
# --------------------------------------------------------------------------------------
def test_isolated_parallel_road_rejected():
    coords_a = [(0.0, 0.0), (30.0, 0.0)]
    b_edges = [
        # connected chain, slightly offset (0.2 m) -> low drift
        ("B1", LineString([(0.0, 0.2), (10.0, 0.2)])),
        ("B2", LineString([(10.0, 0.2), (20.0, 0.2)])),
        ("B3", LineString([(20.0, 0.2), (30.0, 0.2)])),
        # isolated full-length parallel road, farther (1.0 m); edge-to-edge would pick this
        ("B4", LineString([(0.0, -1.0), (30.0, -1.0)])),
    ]
    res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0)

    assert _route_ids(res) == ["B1", "B2", "B3"]
    assert "B4" not in _route_ids(res)


# --------------------------------------------------------------------------------------
# 3. Snap tolerance: near-but-unequal endpoints connect only within tolerance
# --------------------------------------------------------------------------------------
def test_snap_tolerance_controls_connectivity():
    coords_a = [(0.0, 0.0), (20.0, 0.0)]
    # B1 ends at x=10.0, B2 starts at x=10.5 -> 0.5 m gap between the two endpoints
    b_edges = [
        ("B1", LineString([(0.0, 0.0), (10.0, 0.0)])),
        ("B2", LineString([(10.5, 0.0), (20.0, 0.0)])),
    ]

    connected = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.75, step_meters=2.0)
    assert _route_ids(connected) == ["B1", "B2"]   # 0.5 m < 0.75 m tolerance -> joined

    broken = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.1, step_meters=2.0)
    # 0.5 m > 0.1 m tolerance -> not joined; cannot stitch both, so only one edge is routed
    assert _route_ids(broken) != ["B1", "B2"]


# --------------------------------------------------------------------------------------
# 4. Cycle: a loop in GB must terminate and produce a sensible route
# --------------------------------------------------------------------------------------
def test_cycle_terminates():
    # A small loop B1->B2->B3 back to start, plus B4 leaving the loop along A's direction.
    coords_a = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    b_edges = [
        ("B1", LineString([(0.0, 0.0), (10.0, 0.0)])),     # along A
        ("B2", LineString([(10.0, 0.0), (10.0, 5.0)])),    # up
        ("B3", LineString([(10.0, 5.0), (0.0, 0.0)])),     # back to start (closes the loop)
        ("B4", LineString([(10.0, 0.0), (20.0, 0.0)])),    # continue along A
    ]
    res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0)

    assert res["route"], "expected a non-empty route"
    assert _route_ids(res)[0] == "B1"     # follows A straight through the loop
    assert "B4" in _route_ids(res)
    assert res["avg_distance"] < 1.0      # straight low-drift path, not a detour up the loop


# --------------------------------------------------------------------------------------
# 5. No U-turn: a perpendicular edge that only ENDS at the junction is never entered+left
# --------------------------------------------------------------------------------------
def test_no_uturn_onto_perpendicular_stub():
    # A runs straight east. The main road is B_main -> B_cont. B_stub is a perpendicular edge
    # that *ends* at the junction (digitized into it). With forward-only directed arcs there is
    # no arc leaving the junction onto B_stub, so the warping path cannot dip onto it and back
    # (the OSM-1278 artifact). The route must be the clean B_main -> B_cont.
    coords_a = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    b_edges = [
        ("B_main", LineString([(0.0, 0.0), (10.0, 0.0)])),     # into the junction, along A
        ("B_cont", LineString([(10.0, 0.0), (20.0, 0.0)])),    # out of the junction, along A
        ("B_stub", LineString([(10.0, -8.0), (10.0, 0.0)])),   # perpendicular, ENDS at junction
    ]
    res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0)

    assert _route_ids(res) == ["B_main", "B_cont"]
    assert "B_stub" not in _route_ids(res)
    assert all(d == "forward" for (_e, d, _s) in res["route"])
    assert res["avg_distance"] < 0.5


# --------------------------------------------------------------------------------------
# 6. Zero-traversal touch: A ending at a junction must not list the next edge it never walks
# --------------------------------------------------------------------------------------
def test_zero_traversal_end_edge_dropped():
    # A runs along B_main and ends exactly at the junction, where a continuation edge starts.
    # A only *touches* that edge's start vertex (overhang) and never traverses it, so the route
    # must be the single B_main -- not [B_main, continuation] (the OSM-1251 phantom tail).
    coords_a = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    b_edges = [
        ("B_main", LineString([(0.0, 0.0), (10.0, 0.0)])),   # A walks this fully
        ("B_next", LineString([(10.0, 0.0), (20.0, 0.0)])),  # starts at A's end; never entered
    ]
    res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0)

    assert res["metrics"]["n_edges"] == 1
    # no route edge may have zero B traversal (a pure boundary touch)
    assert all(re["b_cover_pct"] > 0 for re in res["metrics"]["route_edges"])


# --------------------------------------------------------------------------------------
# Readable __main__ report
# --------------------------------------------------------------------------------------
def _report(name, coords_a, b_edges, **kw):
    res = match_edge_to_bgraph(coords_a, b_edges, **kw)
    m = res["metrics"]
    print(f"\n[{name}]")
    print(f"  route       : {res['route']}")
    print(f"  avg drift   : {res['avg_distance']:.3f} m   (max {m['max']:.3f}, min {m['min']:.3f})")
    print(f"  matched_len : {m['matched_len']:.1f} m   n_edges={m['n_edges']}   "
          f"overlap={m['overlap_pct']}%   bearing_diff={m['bearing_diff']:.1f}")
    return res


if __name__ == "__main__":
    print("=" * 60)
    print("     GRAPH-DTW ALGORITHM TEST REPORT")
    print("=" * 60)

    _report("SPLIT (B1->B2->B3)",
            [(0.0, 0.0), (30.0, 0.0)],
            [("B1", LineString([(0, 0.2), (10, 0.2)])),
             ("B2", LineString([(10, 0.2), (20, 0.2)])),
             ("B3", LineString([(20, 0.2), (30, 0.2)]))],
            snap_tolerance_m=0.5, step_meters=2.0)

    _report("ISOLATED PARALLEL (B4 rejected)",
            [(0.0, 0.0), (30.0, 0.0)],
            [("B1", LineString([(0, 0.2), (10, 0.2)])),
             ("B2", LineString([(10, 0.2), (20, 0.2)])),
             ("B3", LineString([(20, 0.2), (30, 0.2)])),
             ("B4", LineString([(0, -1.0), (30, -1.0)]))],
            snap_tolerance_m=0.5, step_meters=2.0)

    _report("NO U-TURN (B_stub never entered)",
            [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
            [("B_main", LineString([(0, 0), (10, 0)])),
             ("B_cont", LineString([(10, 0), (20, 0)])),
             ("B_stub", LineString([(10, -8), (10, 0)]))],
            snap_tolerance_m=0.5, step_meters=2.0)

    # Run the assertion tests too.
    test_split_stitches_three_edges()
    test_isolated_parallel_road_rejected()
    test_snap_tolerance_controls_connectivity()
    test_cycle_terminates()
    test_no_uturn_onto_perpendicular_stub()
    test_zero_traversal_end_edge_dropped()
    print("\n" + "=" * 60)
    print("     ALL GRAPH-DTW TESTS PASSED")
    print("=" * 60)


# --------------------------------------------------------------------------------------
# 7. Step weights alpha/beta (docs/weighted_emission.md §12), both emission modes
# --------------------------------------------------------------------------------------
def _weights_case():
    """A straight A-edge over a connected, slightly offset B-chain (both modes align it)."""
    coords_a = [(0.0, 0.0), (30.0, 0.0)]
    b_edges = [
        ("B1", LineString([(0.0, 0.3), (10.0, 0.3)])),
        ("B2", LineString([(10.0, 0.3), (20.0, 0.3)])),
        ("B3", LineString([(20.0, 0.3), (30.0, 0.3)])),
    ]
    return coords_a, b_edges


def test_alpha_beta_defaults_are_identity():
    """alpha=1, beta=1 passed explicitly must reproduce the default call exactly -- route,
    warping, reported drift, and the DP's decision cost -- in BOTH emission modes."""
    coords_a, b_edges = _weights_case()
    for emission in ("point", "segment"):
        base = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0,
                                    emission=emission, bearing_weight=2.0, debug=True)
        expl = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=2.0,
                                    emission=emission, bearing_weight=2.0,
                                    alpha=1.0, beta=1.0, debug=True)
        assert _route_ids(base) == _route_ids(expl)
        assert base["warping_path"] == expl["warping_path"]
        assert base["avg_distance"] == expl["avg_distance"]
        assert base["debug"]["final_cost"] == expl["debug"]["final_cost"]


def test_alpha_beta_monotone_decision_cost():
    """Every candidate alignment's weighted cost is non-decreasing in beta and non-increasing
    as alpha drops, so the DP minimum must be too -- in BOTH emission modes."""
    coords_a, b_edges = _weights_case()
    for emission in ("point", "segment"):
        def cost(alpha=1.0, beta=1.0):
            res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5,
                                       step_meters=2.0, emission=emission,
                                       alpha=alpha, beta=beta, debug=True)
            return res["debug"]["final_cost"]

        c11 = cost()
        assert cost(alpha=0.5) <= c11 + 1e-9, f"{emission}: alpha discount raised the cost"
        assert cost(alpha=0.2) <= cost(alpha=0.5) + 1e-9, emission
        assert cost(beta=2.0) >= c11 - 1e-9, f"{emission}: beta penalty lowered the cost"
        assert cost(beta=5.0) >= cost(beta=2.0) - 1e-9, emission


def test_beta_penalizes_route_collapse():
    """A LONGER dense A over a SHORT B edge forces stalls -- the overhang's projections clamp
    onto B's end vertex, so several A-points share one B-state. Raising beta must raise the
    weighted decision cost, while the reported drift stays the raw geometry of the (here
    forced) alignment."""
    coords_a = [(float(x), 0.0) for x in range(0, 31, 3)]      # 11 points over 30 m
    b_edges = [("B1", LineString([(0.0, 0.3), (10.0, 0.3)]))]  # B covers only the first 10 m
    for emission in ("point", "segment"):
        res1 = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=30.0,
                                    emission=emission, alpha=1.0, beta=1.0, debug=True)
        res5 = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5, step_meters=30.0,
                                    emission=emission, alpha=1.0, beta=5.0, debug=True)
        assert _route_ids(res1) == ["B1"] and _route_ids(res5) == ["B1"]
        assert res5["debug"]["final_cost"] > res1["debug"]["final_cost"], \
            f"{emission}: stalls are forced here, beta must bite"
        # the reported drift stays RAW geometry of whichever alignment was chosen (beta may
        # legitimately move the stalls, so no equality across weights -- just sanity):
        assert 0.0 < res5["avg_distance"] < 25.0
