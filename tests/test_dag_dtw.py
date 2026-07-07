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
from network_matching.dag_dtw import NotADAG, match_dag_to_bgraph, topological_order  # noqa: E402
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


def test_horizontal_weight_extends_coverage():
    # α < 1 makes 1:N coverage cheaper, so it should extend at least one route on a case where
    # coverage is a genuine choice (the shifted diamond).
    sc = get_dag("diamond")
    A = _edges_to_ls(perturb_dag(sc["a_edges"], shift=-6))
    r1 = match_dag_to_bgraph(A, sc["b_edges"], horizontal_weight=1.0, **sc["defaults"])
    r_low = match_dag_to_bgraph(A, sc["b_edges"], horizontal_weight=0.3, **sc["defaults"])
    assert dict(r1["routes"]) != dict(r_low["routes"]), "α<1 should change the coverage here"
