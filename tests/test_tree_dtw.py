"""
Tests for Tree-DTW (network_matching/tree_dtw.py) -- the standalone tree matcher.

Exercised on the hand-built synthetic source trees (chain, y_split, merge) from
network_matching/dag_synthetic.py; reconvergent DAGs (diamond) must be rejected. Small edge lists
in a plain meter CRS -- no DuckDB, no real data, and no call into the DAG matcher.
"""

import os
import sys

import numpy as np
import pytest
from shapely.geometry import LineString

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.tree_dtw import (  # noqa: E402
    NotATree, check_tree_rules, match_tree_to_bgraph)
from network_matching.dag_playground import perturb_dag  # noqa: E402
from network_matching.dag_synthetic import get_dag  # noqa: E402

TREES = ["chain", "y_split", "merge"]
RECONVERGENT = ["diamond", "double_diamond"]
_KW = dict(snap_tolerance_m=0.5, step_meters=2.0)


def _run(name, **kw):
    sc = get_dag(name)
    return match_tree_to_bgraph(sc["a_edges"], sc["b_edges"], **{**_KW, **kw})


# --- 1. the matching relation M is a valid warping (V1-V4) by construction ---------------
@pytest.mark.parametrize("name", TREES)
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.3])
def test_matching_is_v1_v4_clean(name, alpha):
    res = _run(name, horizontal_weight=alpha, validate=True)
    r = res["rules"]
    assert r["ok"], (f"{name} alpha={alpha}: v1={r['v1_cross']} v2={r['v2_predecessor']} "
                     f"v3={r['v3_successor']} v4={r['v4_uncovered']}")


# --- 2. the expected per-A-edge B-edge routes -------------------------------------------
@pytest.mark.parametrize("name,expect", [
    ("chain", {"A1": ["B1"], "A2": ["B2"]}),
    ("y_split", {"A_main": ["B_main"], "A_left": ["B_left"], "A_right": ["B_right"]}),
    ("merge", {"A_top": ["B_top"], "A_bot": ["B_bot"], "A_out": ["B_out"]}),
])
def test_routes_expected(name, expect):
    assert _run(name)["routes"] == expect


# --- 3. a reconvergent source (undirected loop) is rejected -- Tree-DTW is tree-only -----
@pytest.mark.parametrize("name", RECONVERGENT)
def test_rejects_reconvergence(name):
    sc = get_dag(name)
    with pytest.raises(NotATree):
        match_tree_to_bgraph(sc["a_edges"], sc["b_edges"], **_KW)


# --- 4. M is a relation; every source point is covered (V4); drifts finite --------------
@pytest.mark.parametrize("name", TREES)
def test_M_is_a_covering_relation(name):
    res = _run(name)
    ga = res["GA"]
    matched = {a for a, _v in res["M"]}
    assert matched == set(range(ga.n_vertices))                 # (V4) every point matched
    assert all(m["run"] and m["drift"] >= 0 for m in res["a_match"])
    # M agrees with the per-point runs
    assert res["M"] == {(m["a"], w) for m in res["a_match"] for w in m["run"]}


# --- 5. coverage weight: alpha=1 is the default; a 1:N run appears only under alpha<1 ----
@pytest.mark.parametrize("name", TREES)
def test_horizontal_weight_one_is_default(name):
    assert _run(name)["M"] == _run(name, horizontal_weight=1.0)["M"]


def test_check_tree_rules_flags_a_hole():
    # a hand-built INVALID matching: a covers v0 and v2 but skips the arc-connected v1 in between.
    res = _run("chain")
    ga, gb = res["GA"], res["GB"]
    # find a B 3-chain v0->v1->v2 and an A-point, then omit the middle -> a hole
    v0 = next(v for v in range(gb.n_vertices) if gb.succ_arcs[v]
              and gb.succ_arcs[gb.succ_arcs[v][0]])
    v1 = gb.succ_arcs[v0][0]
    v2 = gb.succ_arcs[v1][0]
    a = 0
    bad = {(a, v0), (a, v2)}                                    # hole at v1
    rules = check_tree_rules(bad, ga, gb)
    assert not rules["ok"]                                      # V2/V3 (or V4) must fire


# --- 6. accepts raw coordinate lists as well as LineStrings -----------------------------
def test_accepts_coord_lists():
    a_edges = [("A1", [(0, 0), (15, 0)]), ("A2", [(15, 0), (30, 0)])]
    b_edges = [("B1", LineString([(0, 0.4), (15, 0.4)])),
               ("B2", LineString([(15, 0.4), (30, 0.4)]))]
    res = match_tree_to_bgraph(a_edges, b_edges, validate=True, **_KW)
    assert res["rules"]["ok"]
    assert res["routes"] == {"A1": ["B1"], "A2": ["B2"]}


# --- 7. segment-to-segment matching (docs §8) -------------------------------------------
def test_emission_point_is_default_and_unchanged():
    # emission="point" is the default and bit-for-bit today's result on every tree.
    for name in TREES:
        r_def = _run(name)
        r_pt = _run(name, emission="point")
        assert r_def["M"] == r_pt["M"]
        assert r_def["routes"] == r_pt["routes"]
        assert "segment_pairs" not in r_pt                      # arc records are segment-mode only


# --- 7a. VALIDATION: the segment-state matching relation is a valid warping (V1-V4) ------
@pytest.mark.parametrize("name", TREES)
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.3])
@pytest.mark.parametrize("lam", [0.0, 3.0])
def test_segment_matching_is_v1_v4_clean(name, alpha, lam):
    # the arc-state DP must produce a V1-V4-clean M at every coverage weight / bearing weight.
    res = _run(name, emission="segment", bearing_weight=lam, horizontal_weight=alpha, validate=True)
    r = res["rules"]
    assert r["ok"], (f"{name} alpha={alpha} lam={lam}: v1={r['v1_cross']} v2={r['v2_predecessor']} "
                     f"v3={r['v3_successor']} v4={r['v4_uncovered']}")


@pytest.mark.parametrize("name", TREES)
def test_segment_M_covers_every_point_and_agrees_with_runs(name):
    # (V4) every source point matched, and M is exactly the union of the per-point coverage runs.
    res = _run(name, emission="segment", bearing_weight=3.0)
    ga = res["GA"]
    assert {a for a, _v in res["M"]} == set(range(ga.n_vertices))
    assert res["M"] == {(m["a"], w) for m in res["a_match"] for w in m["run"]}


@pytest.mark.parametrize("name", TREES)
def test_segment_pairs_are_two_real_segments(name):
    # §8.1/§8.3: every scored state is a pair of real segments, so a heading is always defined
    # (nothing degenerates to a bearingless point). One record per source segment.
    res = _run(name, emission="segment", bearing_weight=3.0)
    pairs = res["segment_pairs"]
    assert pairs, "segment mode must expose the matched (A-arc, B-arc) pairs"
    for p in pairs:
        assert set(p) == {"a_mid", "b_mid", "a_bear", "b_bear", "cost"}
        assert 0.0 <= p["a_bear"] < 360.0 and 0.0 <= p["b_bear"] < 360.0
        assert p["cost"] >= 0.0


def _narrow_fork(halfwidth, dy=0.0):
    # a stem that forks into two arms only `halfwidth` apart at the far end, optionally shifted +dy.
    return [("stem", [(-10, dy), (0, dy)]),
            ("up", [(0, dy), (20, halfwidth + dy)]),
            ("dn", [(0, dy), (20, -halfwidth + dy)])]


def test_segment_bearing_fixes_narrow_fork_under_shift():
    # a lateral shift larger than the fork's half-width makes the `dn` arm's far end sit nearer the
    # target's `up` arm, so point mode collapses onto the wrong (nearer) arm; the segment-to-segment
    # matcher keeps each arm on the same-heading target arm. (Trees can't reconverge, so this narrow
    # fork -- not a diamond -- is the tree analog of the DAG diamond test.) Both stay V1-V4 clean.
    b_edges = _narrow_fork(1.0)
    a_edges = _narrow_fork(1.0, dy=1.5)
    want = {"stem": ["stem"], "up": ["up"], "dn": ["dn"]}
    r_pt = match_tree_to_bgraph(a_edges, b_edges, emission="point", validate=True, **_KW)
    r_seg = match_tree_to_bgraph(a_edges, b_edges, emission="segment", bearing_weight=3.0,
                                 validate=True, **_KW)
    assert r_pt["routes"] != want, "expected point mode to mis-route the shifted narrow fork"
    assert r_seg["routes"] == want, f"segment should fix it, got {dict(r_seg['routes'])}"
    assert r_pt["rules"]["ok"] and r_seg["rules"]["ok"]         # both are valid warpings
    assert set(r_pt) <= set(r_seg)                              # segment output is a superset


def test_segment_unknown_emission_raises():
    sc = get_dag("chain")
    with pytest.raises(ValueError):
        match_tree_to_bgraph(sc["a_edges"], sc["b_edges"], emission="bogus", **_KW)


# --- 8. coverage weights: alpha (1:N) and beta (N:1), docs §4.1 --------------------------
@pytest.mark.parametrize("emission", ["point", "segment"])
def test_vertical_weight_one_is_default(emission):
    # beta = 1 (default) leaves the matching bit-for-bit unchanged, in both emission modes.
    for name in TREES:
        base = _run(name, emission=emission)
        b1 = _run(name, emission=emission, vertical_weight=1.0)
        assert base["M"] == b1["M"]
        assert base["routes"] == b1["routes"]


@pytest.mark.parametrize("emission", ["point", "segment"])
@pytest.mark.parametrize("name", TREES)
@pytest.mark.parametrize("beta", [1.0, 0.5, 0.2])
def test_beta_matching_is_v1_v4_clean(emission, name, beta):
    # the N:1 coverage weight must keep M a valid warping across (0, 1] in both modes.
    res = _run(name, emission=emission, vertical_weight=beta, validate=True)
    r = res["rules"]
    assert r["ok"], (f"{name} {emission} beta={beta}: v1={r['v1_cross']} v2={r['v2_predecessor']} "
                     f"v3={r['v3_successor']} v4={r['v4_uncovered']}")


def _max_b_stack(res):
    # the most source points pinned to a single target vertex (the depth of an N:1 stack).
    from collections import Counter
    return max(Counter(m["anchor"] for m in res["a_match"]).values())


@pytest.mark.parametrize("bearing_weight", [0.0, 1.0, 3.0, 5.0])
@pytest.mark.parametrize("rotate", [0, 15, 25, 35])
def test_segment_stays_valid_under_perturbation(bearing_weight, rotate):
    # regression: segment mode pins A-VERTICES (not arcs), so a split's arms always leave the one
    # committed B-vertex -- even when a strong bearing pull + rotation would tempt one arm onto a
    # non-adjacent B-edge. Previously the per-arc pin produced a V1-V4-INVALID warping here.
    a_edges = [("stem", [(0, 0), (10, 0)]), ("up", [(10, 0), (20, 6)]), ("down", [(10, 0), (20, -6)])]
    b_edges = [("stem", [(0, 0.5), (10, 0.5)]), ("up", [(10, 0.5), (20, 6.5)]),
               ("down", [(10, 0.5), (20, -5.5)])]
    A = perturb_dag(a_edges, shift=5, rotate=rotate, noise=0.1, seed=10)
    res = match_tree_to_bgraph(A, b_edges, emission="segment", bearing_weight=bearing_weight,
                               snap_tolerance_m=0.5, step_meters=1.5, validate=True)
    r = res["rules"]
    assert r["ok"], (f"rot={rotate} bw={bearing_weight}: v1={r['v1_cross']} v2={r['v2_predecessor']} "
                     f"v3={r['v3_successor']} v4={r['v4_uncovered']}")


def test_beta_induces_n_to_1_coverage():
    # a source sampled finer than the target: beta < 1 makes several source points share one target
    # point (an N:1 stack), which beta = 1 (full per-point pricing) avoids.
    a_edges = [("A", [(x, 0.0) for x in np.linspace(0, 20, 21)])]   # dense source
    b_edges = [("B", [(0, 0.3), (20, 0.3)])]                        # coarse target
    hi = match_tree_to_bgraph(a_edges, b_edges, vertical_weight=1.0, validate=True, **_KW)
    lo = match_tree_to_bgraph(a_edges, b_edges, vertical_weight=0.1, validate=True, **_KW)
    assert hi["rules"]["ok"] and lo["rules"]["ok"]                  # both valid warpings
    assert _max_b_stack(lo) > _max_b_stack(hi)                      # low beta stacks more (N:1)
