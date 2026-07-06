"""
Tests for the graph-DTW debug instrumentation (``debug=True``) and the perturbation-robustness
behaviour of the algorithm on the synthetic scenarios (``network_matching/synthetic.py``).

Two groups:

1. DEBUG PAYLOAD -- the internals returned by ``match_edge_to_bgraph(..., debug=True)`` must be
   self-consistent: the cost table decomposes exactly into predecessor cost + emission along the
   backtracked path, the alignment is monotone in A, and failures carry a reason.
2. ROBUSTNESS -- perturbing the A-edge (noise / shift / rotation / crop / reverse) and re-matching
   against the unchanged B-network degrades the match the way the algorithm is designed to:
   routes survive small distortions, lateral drift tracks lateral shift, a strong pull captures
   the parallel trap, and a reversed edge (directed table) is NO_MATCH.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import match_edge_to_bgraph  # noqa: E402
from network_matching.synthetic import (SCENARIOS, apply_perturbation,  # noqa: E402
                                        crop_ends, get_scenario, lateral_shift,
                                        resample, reverse, rotate, translate)


def _match(name, coords_a=None, **overrides):
    sc = get_scenario(name)
    kw = dict(sc["defaults"])
    kw.update(overrides)
    pts = sc["coords_a"] if coords_a is None else coords_a
    return match_edge_to_bgraph(list(map(tuple, pts)), sc["b_edges"], **kw)


def _routes(res):
    return [eid for (eid, _d, _s) in res["route"]]


# --------------------------------------------------------------------------------------
# 1. Debug payload invariants
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("emission", ["point", "segment", "midpoint"])
def test_debug_payload_decomposes_costs(emission):
    res = _match("split", emission=emission, debug=True)
    dbg = res["debug"]
    D, E = np.asarray(dbg["D"]), np.asarray(dbg["E"])
    assert D.shape == E.shape
    path = dbg["arc_path"] if emission in ("segment", "midpoint") else dbg["path"]
    assert path[0][2] == "START"
    for t, (i, s, move) in enumerate(path):
        if move == "START":
            assert np.isclose(D[i][s], E[i][s])
        else:
            pi, ps, _ = path[t - 1]
            # every DP state = its predecessor's accumulated cost + its own emission
            assert np.isclose(D[i][s], D[pi][ps] + E[i][s])
    assert np.isclose(dbg["final_cost"], D[-1][dbg["terminal_state"]])


@pytest.mark.parametrize("emission", ["point", "segment", "midpoint"])
def test_debug_alignment_is_monotone_in_a(emission):
    res = _match("split", emission=emission, debug=True)
    dbg = res["debug"]
    a_idx = [i for (i, _v) in dbg["pairs_all"]]
    assert a_idx == sorted(a_idx)
    lo, hi = dbg["kept_span"]
    assert 0 <= lo <= hi < len(dbg["pairs_all"])
    assert len(dbg["drift_all"]) == len(dbg["pairs_all"])
    assert all(len(p) == 3 for p in dbg["a_pool"])  # (x, y, is_node)


def test_debug_off_by_default_and_failure_reason():
    res = _match("split")
    assert "debug" not in res and "debug" not in res["metrics"]
    # an A-edge that only touches B at one boundary vertex never traverses it -> reasoned failure
    # (note: distance alone never fails graph-DTW -- candidate search culls by distance upstream)
    from shapely.geometry import LineString
    far = match_edge_to_bgraph([(0.0, 0.0), (1.0, 0.0)],
                               [("B_far", LineString([(500.0, 500.0), (510.0, 500.0)]))],
                               debug=True, snap_tolerance_m=0.5, step_meters=2.0)
    assert not np.isfinite(far["avg_distance"])
    assert far["debug"]["reason"] in {"no_finite_alignment", "zero_b_traversal"}


def test_every_scenario_matches_unperturbed():
    for name, sc in SCENARIOS.items():
        res = match_edge_to_bgraph(sc["coords_a"], sc["b_edges"], **sc["defaults"])
        assert res["route"], f"scenario {name!r} produced no route"


# --------------------------------------------------------------------------------------
# 2. Perturbation robustness of the algorithm
# --------------------------------------------------------------------------------------
def test_route_survives_small_noise():
    base = _routes(_match("split"))
    for seed in range(3):
        noisy = apply_perturbation(get_scenario("split")["coords_a"], "noise", 0.5, seed=seed)
        assert _routes(_match("split", coords_a=noisy)) == base


def test_lateral_shift_drift_tracks_shift():
    # a purely lateral offset should register (almost exactly) as that much drift
    for s in (2.0, 4.0, 8.0):
        shifted = lateral_shift(get_scenario("split")["coords_a"], s)
        res = _match("split", coords_a=shifted)
        assert _routes(res) == ["B1", "B2", "B3"]
        assert res["avg_distance"] == pytest.approx(s, abs=0.5)


def test_shift_toward_parallel_road_gets_captured():
    # the connected chain wins nearby, but shifting A onto the trap must eventually flip the
    # route -- graph-DTW prefers connectivity, not magic
    sc = get_scenario("parallel_trap")
    near = _match("parallel_trap", coords_a=lateral_shift(sc["coords_a"], -2.0))
    assert _routes(near) == ["B1", "B2", "B3"]
    captured = _match("parallel_trap", coords_a=lateral_shift(sc["coords_a"], -8.0))
    assert _routes(captured) == ["B_trap"]


def test_small_rotation_keeps_route_on_curve():
    sc = get_scenario("curve")
    res = _match("curve", coords_a=rotate(sc["coords_a"], 5.0))
    assert _routes(res) == ["B_arc1", "B_arc2", "B_arc3"]


def test_cropped_edge_matches_sub_route():
    # keeping only the middle 6 m of A (x = 12..18, strictly inside B2's 10..20 span) must
    # shrink the route to a contiguous sub-route of the full B1-B2-B3 chain: just B2
    full = _routes(_match("split"))
    cropped = crop_ends(get_scenario("split")["coords_a"], 80.0)
    sub = _routes(_match("split", coords_a=cropped))
    assert sub == ["B2"]
    start = full.index(sub[0])
    assert full[start:start + len(sub)] == sub


def test_reversed_edge_is_no_match_on_directed_table():
    # the B table is directed (a reversed road matches its reverse twin, absent here), so a
    # reversed A must not stitch the forward chain
    rev = reverse(get_scenario("split")["coords_a"])
    res = _match("split", coords_a=rev)
    assert not np.isfinite(res["avg_distance"])
    assert res["route"] == []


# --------------------------------------------------------------------------------------
# Midpoint emission + sliver-free segment pools
# --------------------------------------------------------------------------------------
def test_midpoint_mode_reports_middle_to_middle_distance():
    # a pure 0.2 m lateral offset: every A-segment middle is exactly 0.2 m from its matched
    # B-segment middle, so avg == max == min == 0.2 (the metric IS that one distance)
    res = _match("split", emission="midpoint")
    assert _routes(res) == ["B1", "B2", "B3"]
    assert res["avg_distance"] == pytest.approx(0.2, abs=1e-6)
    assert res["metrics"]["max"] == pytest.approx(0.2, abs=1e-6)
    assert res["metrics"]["min"] == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("emission", ["segment", "midpoint"])
def test_segment_states_are_never_point_like(emission):
    # segment modes must pair genuine segments: with the default min_pool_gap (step/2) no
    # matched state may sit on a sliver shorter than that on either side
    sc = get_scenario("curve")
    step = sc["defaults"]["step_meters"]
    noisy = apply_perturbation(sc["coords_a"], "noise", 1.5, seed=3)
    res = _match("curve", coords_a=noisy, emission=emission, debug=True)
    dbg, gb = res["debug"], res["graph"]
    ap, arcs, rid = dbg["a_pool"], dbg["arcs"], dbg["ridable"]
    lo, hi = dbg["kept_span"]
    for t, (i, k, _mv) in enumerate(dbg["arc_path"]):
        if not (lo <= t + 1 <= hi) or not rid[k]:
            continue
        u, w = arcs[k]
        a_len = np.hypot(ap[i + 1][0] - ap[i][0], ap[i + 1][1] - ap[i][1])
        b_len = np.hypot(gb.vx[w] - gb.vx[u], gb.vy[w] - gb.vy[u])
        assert a_len >= step / 2 - 1e-6
        assert b_len >= step / 2 - 1e-6


def test_gap_fill_is_even():
    # fill points spread evenly over each gap: consecutive spacing in (step/2, step], never a
    # leftover sliver next to the following pool point
    from network_matching.graph_dtw import _node_projection_pool
    pool = _node_projection_pool([(0.0, 0.0), (31.0, 0.0)], [], step_meters=2.0)
    xs = [p[0] for p in pool]
    gaps = np.diff(xs)
    assert gaps.max() <= 2.0 + 1e-9
    assert gaps.min() > 1.0 - 1e-9


# --------------------------------------------------------------------------------------
# Perturbation geometry sanity
# --------------------------------------------------------------------------------------
def test_perturbation_geometry():
    P = [(0.0, 0.0), (30.0, 0.0)]
    np.testing.assert_allclose(lateral_shift(P, 3.0), [(0, 3), (30, 3)])
    # translate is absolute: east/west move x, north/south move y, any bearing works
    np.testing.assert_allclose(translate(P, 5.0, 90.0), [(5, 0), (35, 0)], atol=1e-12)
    np.testing.assert_allclose(translate(P, 5.0, 270.0), [(-5, 0), (25, 0)], atol=1e-12)
    np.testing.assert_allclose(translate(P, 5.0, 0.0), [(0, 5), (30, 5)], atol=1e-12)
    np.testing.assert_allclose(apply_perturbation(P, "translate", 5.0, bearing=180.0),
                               [(0, -5), (30, -5)], atol=1e-12)
    R = rotate(P, 90.0)
    assert np.hypot(*(R[-1] - R[0])) == pytest.approx(30.0)      # rigid: length preserved
    C = crop_ends(P, 20.0)
    assert np.hypot(*(C[-1] - C[0])) == pytest.approx(24.0)      # 10% removed per end
    S = resample(P, 5.0)
    steps = np.hypot(*np.diff(S, axis=0).T)
    assert steps == pytest.approx(np.full(len(S) - 1, 5.0))
    assert reverse(P)[0][0] == 30.0
    with pytest.raises(KeyError):
        apply_perturbation(P, "wobble", 1.0)
