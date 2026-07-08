"""
Tests for DAG-DTW (network_matching/dag_dtw.py) on the hand-built synthetic DAGs
(network_matching/dag_synthetic.py). See docs/dag_dtw_matching.md §4.

Exercised on small edge lists in a plain meter CRS -- no DuckDB, no real data.
"""

import os
import sys
from collections import deque

import numpy as np
import pytest
from shapely.geometry import LineString

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_dtw import (  # noqa: E402
    NotADAG, NotATree, check_matching_rules, forward_successor_dp, match_dag_to_bgraph,
    topological_order)
from network_matching.dag_dtw import build_local_digraph  # noqa: E402
from network_matching.dag_synthetic import DAG_SCENARIOS, get_dag  # noqa: E402
from network_matching.dag_playground import perturb_dag, _edges_to_ls  # noqa: E402

_OFF = 0.4  # the B offset used in the scenarios


def _match(name):
    sc = get_dag(name)
    return match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], debug=True, **sc["defaults"])


# --------------------------------------------------------------------------------------
# Each A-edge maps to its intended B-edge; drift ~ the offset
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,expect", [
    ("chain", {"A1": ["B1"], "A2": ["B2"]}),
    ("y_split", {"A_main": ["B_main"], "A_left": ["B_left"], "A_right": ["B_right"]}),
    ("merge", {"A_top": ["B_top"], "A_bot": ["B_bot"], "A_out": ["B_out"]}),
])
def test_each_a_edge_routes_to_its_b_edge(name, expect):
    res = _match(name)
    routes = res["routes"]
    for aeid, want in expect.items():
        got = routes[aeid]
        # the intended B-edge must be the primary match (allow a trailing junction spill)
        assert got[0] == want[0], f"{name}:{aeid} -> {got}, expected to start with {want}"
    assert res["avg_drift"] == pytest.approx(_OFF, abs=0.2)


def test_diamond_covers_all_six_edges():
    res = _match("diamond")
    for aeid in ("A_in", "A_up", "A_dn", "A_up2", "A_dn2", "A_out"):
        assert aeid in res["routes"] and res["routes"][aeid], f"{aeid} unmatched"
    assert res["avg_drift"] == pytest.approx(_OFF, abs=0.25)


# --------------------------------------------------------------------------------------
# Junction consistency: coincident A-vertices at a junction share one B-location
# --------------------------------------------------------------------------------------
def test_branch_junction_is_consistent():
    # the coincident A-vertices at the y_split junction (15, 0) must all map into the SAME
    # junction REGION of B (within a sample step). v1 point-to-point resolves the per-edge
    # junction vertices independently at backtrack, so it is region-consistent, not vertex-exact.
    res = _match("y_split")
    ga, gb, phi = res["GA"], res["GB"], res["phi"]
    jl = [a for a in range(ga.n_vertices)
          if abs(ga.vx[a] - 15.0) < 1e-6 and abs(ga.vy[a]) < 1e-6]
    assert len(jl) >= 2, "expected coincident A-vertices at the junction"
    locs = np.array([(gb.vx[phi[a]], gb.vy[phi[a]]) for a in jl if a in phi])
    spread = float(np.max(np.ptp(locs, axis=0))) if len(locs) else 0.0
    assert spread < 0.6, f"junction A-vertices spread {spread:.2f} m apart in B"


# --------------------------------------------------------------------------------------
# Sequence rules: the matched result must be a monotone forward walk in B + connected route
# --------------------------------------------------------------------------------------
def _fwd_reachable(gb, src, dst):
    if src == dst:
        return True
    seen, q = {src}, deque([src])
    while q:
        u = q.popleft()
        for w in gb.succ_arcs[u]:
            if w == dst:
                return True
            if w not in seen:
                seen.add(w)
                q.append(w)
    return False


@pytest.mark.parametrize("name", ["chain", "y_split", "merge", "diamond"])
def test_matched_sequence_obeys_rules(name):
    # on the clean scenarios: every GA arc a->a' maps to a forward B-step φ(a)->φ(a')
    # (monotone, no backward / disconnected jump), and each route's B-edges are graph-connected.
    res = _match(name)
    ga, gb, phi = res["GA"], res["GB"], res["phi"]
    for a in range(ga.n_vertices):
        if a not in phi:
            continue
        for a2 in ga.succ_arcs[a]:
            if a2 in phi:
                assert _fwd_reachable(gb, phi[a], phi[a2]), (
                    f"{name}: A arc {a}->{a2} maps to a non-forward B step "
                    f"{phi[a]}->{phi[a2]}")
    econ = set()
    for u in range(gb.n_vertices):
        for w in gb.succ_arcs[u]:
            eu, ew = gb.edge_ids[gb.vert_edge[u]], gb.edge_ids[gb.vert_edge[w]]
            if eu != ew:
                econ.add((eu, ew))
    for aeid, route in res["routes"].items():
        for i in range(1, len(route)):
            assert (route[i - 1], route[i]) in econ, (
                f"{name}: route {aeid} has disconnected B-edges {route[i-1]}->{route[i]}")
    # no teleport: a B-step may not exceed the A-step by more than ~a sample gap (no junction jump)
    for a in range(ga.n_vertices):
        if a not in phi:
            continue
        for a2 in ga.succ_arcs[a]:
            if a2 not in phi:
                continue
            bdist = np.hypot(gb.vx[phi[a]] - gb.vx[phi[a2]], gb.vy[phi[a]] - gb.vy[phi[a2]])
            adist = np.hypot(ga.vx[a] - ga.vx[a2], ga.vy[a] - ga.vy[a2])
            assert bdist - adist < 3.0, (
                f"{name}: A arc {a}->{a2} teleports in B ({bdist-adist:.1f} m jump)")


# --------------------------------------------------------------------------------------
# Topological order + acyclicity guard
# --------------------------------------------------------------------------------------
def test_topological_order_respects_arcs():
    sc = get_dag("diamond")
    b_pts = [p for _e, g in sc["b_edges"] for p in g.coords]
    ga = build_local_digraph(sc["a_edges"], b_pts, 0.5, 2.0)
    order = topological_order(ga)
    pos = {v: i for i, v in enumerate(order)}
    for u in range(ga.n_vertices):
        for w in ga.succ_arcs[u]:
            assert pos[u] < pos[w], "arc points backward in the topological order"


def test_cyclic_source_raises_not_a_dag():
    # two edges head-to-tail forming a loop -> GA has a cycle -> must be rejected
    a_edges = [("e1", LineString([(0, 0), (10, 0)])),
               ("e2", LineString([(10, 0), (0, 0)]))]
    b_edges = [("B", LineString([(0, 0.4), (10, 0.4)]))]
    with pytest.raises(NotADAG):
        match_dag_to_bgraph(a_edges, b_edges, snap_tolerance_m=0.5, step_meters=2.0)


# --------------------------------------------------------------------------------------
# Total cost = Σ over sinks (docs §3.3); every scenario produces a finite match
# --------------------------------------------------------------------------------------
def test_total_cost_and_all_scenarios_match():
    for name in DAG_SCENARIOS:
        res = _match(name)
        assert np.isfinite(res["total_cost"]) and res["total_cost"] > 0
        assert np.isfinite(res["avg_drift"])
        assert res["phi"], f"{name} produced no φ"


# --------------------------------------------------------------------------------------
# Horizontal emission weight α (docs §3.4)
# --------------------------------------------------------------------------------------
def test_horizontal_weight_one_is_unchanged():
    # α = 1 (default) must be bit-for-bit today's result on every scenario.
    for name in DAG_SCENARIOS:
        sc = get_dag(name)
        r_default = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], **sc["defaults"])
        r_one = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], horizontal_weight=1.0,
                                    **sc["defaults"])
        assert r_default["phi"] == r_one["phi"]
        assert r_default["total_cost"] == pytest.approx(r_one["total_cost"], abs=1e-9)


def test_horizontal_weight_coverage_cost_and_monotone():
    # 1:N coverage cost over an N-vertex B-run at drift δ is δ·(1 + α·(N-1)), and D is monotone
    # along the run (no laundering) -- checked directly on the α relaxation.
    from network_matching.dag_dtw import _relax_alpha
    delta = 0.4
    for N in (1, 6, 30):
        ei = np.full(N, delta)
        acc = np.zeros(N); acc[1:] = 1e9            # A-advance only at the entry v0; rest via horizontal
        gb_pred = [[] if v == 0 else [v - 1] for v in range(N)]
        for alpha in (1.0, 0.5, 0.0):
            row = np.full(N, np.inf)
            _relax_alpha(row, ei, acc, gb_pred, list(range(N)), alpha)
            assert row[N - 1] == pytest.approx(delta * (1 + alpha * (N - 1)), abs=1e-9)
        row = np.full(N, np.inf)
        _relax_alpha(row, ei, acc, gb_pred, list(range(N)), 0.3)
        assert all(row[v] >= row[v - 1] - 1e-9 for v in range(1, N)), "coverage cost must not decrease"


def test_emission_point_is_default_and_unchanged():
    # emission="point" is the default and bit-for-bit today's result on every scenario.
    for name in DAG_SCENARIOS:
        sc = get_dag(name)
        r_def = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], **sc["defaults"])
        r_pt = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], emission="point", **sc["defaults"])
        assert r_def["phi"] == r_pt["phi"]
        assert r_def["total_cost"] == pytest.approx(r_pt["total_cost"], abs=1e-9)


def test_segment_bearing_fixes_diamond_under_shift():
    # under a lateral shift the point mode collapses a diamond branch onto the nearer wrong-direction
    # B-edge; the segment+bearing emission resolves it to the corresponding B-edge.
    sc = get_dag("diamond")
    A = _edges_to_ls(perturb_dag(sc["a_edges"], shift=8))
    want = {f"A_{k}": f"B_{k}" for k in ("in", "up", "dn", "up2", "dn2", "out")}
    r_pt = match_dag_to_bgraph(A, sc["b_edges"], emission="point", **sc["defaults"])
    r_seg = match_dag_to_bgraph(A, sc["b_edges"], emission="segment", bearing_weight=3.0, **sc["defaults"])
    pt_ok = all(r_pt["routes"].get(a, [None])[0] == b for a, b in want.items())
    seg_ok = all(r_seg["routes"].get(a, [None])[0] == b for a, b in want.items())
    assert not pt_ok, "expected point mode to mis-route the shifted diamond"
    assert seg_ok, f"segment+bearing should resolve the diamond, got {dict(r_seg['routes'])}"
    # same output schema as point mode
    assert set(r_seg) == set(r_pt)


def test_segment_unknown_emission_raises():
    sc = get_dag("chain")
    with pytest.raises(ValueError):
        match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], emission="bogus", **sc["defaults"])


@pytest.mark.parametrize("name", ["chain", "y_split", "merge"])
def test_require_tree_accepts_forests(name):
    # a tree / polytree source passes require_tree=True and matches identically to the default.
    sc = get_dag(name)
    r_req = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], require_tree=True, **sc["defaults"])
    r_def = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], **sc["defaults"])
    assert r_req["phi"] == r_def["phi"]


@pytest.mark.parametrize("name", ["diamond", "double_diamond"])
def test_require_tree_rejects_reconvergence(name):
    # a reconvergent DAG (undirected loop) must raise under require_tree=True, but match by default.
    sc = get_dag(name)
    with pytest.raises(NotATree):
        match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], require_tree=True, **sc["defaults"])
    assert match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], **sc["defaults"])["phi"]  # default ok


def test_routes_detail_full_coverage():
    # every A-edge gets a detail entry; on a clean scenario each B-edge is fully covered (t 0->1).
    res = _match("diamond")
    assert set(res["routes_detail"]) == set(res["routes"])
    for aid, d in res["routes_detail"].items():
        assert d["route"] == list(res["routes"][aid])
        assert d["start"]["t"] == pytest.approx(0.0, abs=1e-6)
        assert d["end"]["t"] == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(d["avg_drift"]) and d["n_points"] > 0
        assert d["covered_len_m"] > 0


def test_routes_detail_partial_coverage_fractions():
    # a short A-edge over a long B-edge: the route starts/ends mid-B, captured as 0..1 fractions.
    A = [("Ax", LineString([(10, 0), (25, 0)]))]
    B = [("Bx", LineString([(0, 0.4), (40, 0.4)]))]
    res = match_dag_to_bgraph(A, B, snap_tolerance_m=0.5, step_meters=2.0)
    d = res["routes_detail"]["Ax"]
    assert d["route"] == ["Bx"]
    assert d["start"]["t"] == pytest.approx(10 / 40, abs=0.03)   # enters ~25 % along Bx
    assert d["end"]["t"] == pytest.approx(25 / 40, abs=0.03)     # exits ~62.5 %
    assert d["start"]["xy"][0] == pytest.approx(10.0, abs=0.6)
    assert d["end"]["xy"][0] == pytest.approx(25.0, abs=0.6)


def test_routes_detail_partial_first_and_last_edge():
    # a route spanning a B junction: first edge covered entry->end, last edge start->exit.
    A = [("A1", LineString([(7, 0), (23, 0)]))]
    B = [("B1", LineString([(0, 0.4), (15, 0.4)])), ("B2", LineString([(15, 0.4), (30, 0.4)]))]
    res = match_dag_to_bgraph(A, B, snap_tolerance_m=0.5, step_meters=2.0)
    d = res["routes_detail"]["A1"]
    assert d["route"] == ["B1", "B2"]
    assert d["edges"][0]["t_to"] == pytest.approx(1.0, abs=1e-6)     # first edge runs to its end
    assert d["edges"][-1]["t_from"] == pytest.approx(0.0, abs=1e-6)  # last edge starts at its start
    assert d["start"]["t"] == pytest.approx(7 / 15, abs=0.06)
    assert d["end"]["t"] == pytest.approx(8 / 15, abs=0.06)


def test_horizontal_weight_extends_coverage():
    # α < 1 makes 1:N coverage cheaper, so it should extend at least one route on a case where
    # coverage is a genuine choice (the shifted diamond).
    sc = get_dag("diamond")
    A = _edges_to_ls(perturb_dag(sc["a_edges"], shift=-6))
    r1 = match_dag_to_bgraph(A, sc["b_edges"], horizontal_weight=1.0, **sc["defaults"])
    r_low = match_dag_to_bgraph(A, sc["b_edges"], horizontal_weight=0.3, **sc["defaults"])
    assert dict(r1["routes"]) != dict(r_low["routes"]), "α<1 should change the coverage here"


# ----------------------------------------------------------------------------------------------
# check_matching_rules: the exact structural (V1)-(V4) validator (docs "The problem ...").
# Built on tiny hand-wired graphs so the matching M and its arcs are fully controlled -- these pin
# the two counterexamples that fixed (V3): an orphan run-entry (only V2 catches) and an orphan
# run-exit (only V3 catches), plus a cross (V1) and a clean 1:N warping that must pass.
# ----------------------------------------------------------------------------------------------
class _G:
    """Minimal directed graph exposing the fields check_matching_rules reads."""
    def __init__(self, n, arcs):
        self.n_vertices = n
        self.succ_arcs = [[] for _ in range(n)]
        self.pred_arcs = [[] for _ in range(n)]
        for u, w in arcs:
            self.succ_arcs[u].append(w)
            self.pred_arcs[w].append(u)


def test_matching_rules_valid_1toN_passes():
    # a0->v0 ; a1 covers the run v1->v2 (a genuine 1:N).  Fully valid.
    ga = _G(2, [(0, 1)])
    gb = _G(3, [(0, 1), (1, 2)])
    M = {0: {0}, 1: {1, 2}}
    r = check_matching_rules(M, ga, gb)
    assert r["ok"], r


def test_matching_rules_v2_catches_orphan_entry():
    # arc a'->a ; M(a')={u}, M(a)={v1,v2,v3} run v1->v2->v3, but a' feeds v2 (mid-run):
    # v1 is an entry reachable by nothing -> ONLY the predecessor rule (V2) may flag it.
    ga = _G(2, [(0, 1)])                      # a'=0 -> a=1
    gb = _G(4, [(1, 2), (2, 3), (0, 2)])      # u=0->v2=2 ; v1=1->v2=2->v3=3
    M = {0: {0}, 1: {1, 2, 3}}
    r = check_matching_rules(M, ga, gb)
    assert not r["ok"]
    assert (1, 1) in r["v2_predecessor"], r       # (a=1, v1=1) orphan entry flagged
    assert r["v1_cross"] == [] and r["v3_successor"] == []   # nothing else fires


def test_matching_rules_v3_catches_orphan_exit():
    # mirror: arc a->a' ; M(a)={v1,v2,v3}, M(a')={w}, a' fed from v2 (mid-run):
    # v3 is an exit that continues into nothing -> ONLY the successor rule (V3) may flag it.
    ga = _G(2, [(0, 1)])                      # a=0 -> a'=1
    gb = _G(4, [(0, 1), (1, 2), (1, 3)])      # v1=0->v2=1->v3=2 ; v2=1->w=3
    M = {0: {0, 1, 2}, 1: {3}}
    r = check_matching_rules(M, ga, gb)
    assert not r["ok"]
    assert (0, 2) in r["v3_successor"], r         # (a=0, v3=2) orphan exit flagged
    assert r["v1_cross"] == [] and r["v2_predecessor"] == []


def test_matching_rules_v1_catches_cross():
    # a0->a1 (later in A) but a0->v1 (later in B) while a1->v0 (earlier in B): an inversion.
    ga = _G(2, [(0, 1)])
    gb = _G(2, [(0, 1)])                      # v0=0 -> v1=1
    M = {0: {1}, 1: {0}}
    r = check_matching_rules(M, ga, gb)
    assert not r["ok"]
    assert any(c[:2] == (1, 0) for c in r["v1_cross"]), r   # (a1=1, v0=0) crosses a0 on v1


def test_matching_rules_v4_catches_uncovered():
    ga = _G(2, [(0, 1)])
    gb = _G(2, [(0, 1)])
    M = {0: {0}}                              # a1 left unmatched
    r = check_matching_rules(M, ga, gb)
    assert not r["ok"] and r["v4_uncovered"] == [1]


def test_matching_rules_merge_needs_every_approach():
    # merge a0->a2, a1->a2 ; both approaches must feed a2's cell (the ∀ over Apred in V2).
    ga = _G(3, [(0, 2), (1, 2)])              # a0=0, a1=1 -> a2=2
    gb = _G(4, [(0, 2), (1, 2)])              # v0=0->v2=2 ; v1=1->v2=2 ; x=3 isolated
    good = {0: {0}, 1: {1}, 2: {2}}           # both approaches feed the merge point v2
    assert check_matching_rules(good, ga, gb)["ok"]
    bad = {0: {0}, 1: {3}, 2: {2}}            # a1 sits on the isolated x=3 -> does NOT feed the merge
    r = check_matching_rules(bad, ga, gb)
    assert not r["ok"]
    assert (2, 2) in r["v2_predecessor"]      # a2's entry is not fed by every approach


def test_forward_successor_dp_is_v3_clean():
    # y_split is a source OUT-TREE (one source, a branch, no merges). The successor-DP forces BOTH
    # branches to leave the junction from the SAME point, so the (V3) successor rule holds by
    # construction -- no post-hoc repair (docs §3.0a).
    sc = get_dag("y_split")
    ga, gb, phi, M = forward_successor_dp(
        sc["a_edges"], sc["b_edges"],
        snap_tolerance_m=sc["defaults"].get("snap_tolerance_m", 0.5),
        step_meters=sc["defaults"].get("step_meters", 2.0))
    r = check_matching_rules(M, ga, gb)
    assert r["v3_successor"] == [], r      # branch left at one point -> V3 clean by construction
    assert r["ok"], r                      # out-tree + single-valued -> V1/V2/V4 also clean
