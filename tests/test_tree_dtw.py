"""
Tests for Tree-DTW (network_matching/tree_dtw.py) -- the standalone tree matcher.

Exercised on the hand-built synthetic source trees (chain, y_split, merge) from
network_matching/dag_synthetic.py; reconvergent DAGs (diamond) must be rejected. Small edge lists
in a plain meter CRS -- no DuckDB, no real data, and no call into the DAG matcher.
"""

import os
import sys

import pytest
from shapely.geometry import LineString

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.tree_dtw import (  # noqa: E402
    NotATree, check_tree_rules, match_tree_to_bgraph)
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
