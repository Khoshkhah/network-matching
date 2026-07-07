"""
Tests for the exact conditioning solvers (network_matching/dag_conditioning.py, docs §3.2b).

The recursive minimum-vertex-cut and the one-shot minimum-feedback-vertex-set share an exact
min-sum BP forest solver, so on any DAG they must return **equal-cost** labellings. These tests pin
that invariant (clean + perturbed), the minimum-FVS sizes, exactness vs brute force, and validity.
"""
import itertools
import os
import sys
from collections import deque

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_conditioning import (conditioned_labels,  # noqa: E402
                                               min_feedback_vertex_set)
from network_matching.dag_dtw import build_local_digraph  # noqa: E402
from network_matching.dag_synthetic import DAG_SCENARIOS, get_dag  # noqa: E402
from network_matching.dag_playground import perturb_dag, _edges_to_ls  # noqa: E402


def _backward(ga, gb, phi):
    def reach(s, d):
        if s == d:
            return True
        seen, q = {s}, deque([s])
        while q:
            u = q.popleft()
            for w in gb.succ_arcs[u]:
                if w not in seen:
                    seen.add(w); q.append(w)
        return d in seen
    return sum(1 for a in range(ga.n_vertices) for a2 in ga.succ_arcs[a]
               if not reach(phi[a], phi[a2]))


@pytest.mark.parametrize("name,size", [
    ("chain", 0), ("y_split", 0), ("merge", 0), ("diamond", 1), ("double_diamond", 2)])
def test_min_feedback_vertex_set_size(name, size):
    sc = get_dag(name)
    b_pts = [p for _e, g in sc["b_edges"] for p in g.coords]
    ga = build_local_digraph(sc["a_edges"], b_pts, 0.5, 2.0)
    assert len(min_feedback_vertex_set(ga)) == size


@pytest.mark.parametrize("name", list(DAG_SCENARIOS))
def test_recursive_equals_fvs_clean(name):
    # the two exact decompositions must agree on cost (global optimum) and both be valid.
    sc = get_dag(name)
    g, gb, phr, cr = conditioned_labels(sc["a_edges"], sc["b_edges"], method="recursive",
                                        **sc["defaults"])
    _, _, phf, cf = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs", **sc["defaults"])
    assert cr == pytest.approx(cf, abs=1e-6), f"{name}: recursive {cr} != fvs {cf}"
    assert _backward(g, gb, phr) == 0
    assert _backward(g, gb, phf) == 0


@pytest.mark.parametrize("name", ["diamond", "double_diamond"])
def test_recursive_equals_fvs_perturbed(name):
    sc = get_dag(name)
    for shift, rot, noise, seed in [(6, 0, 0.5, 1), (-4, 15, 0, 0), (8, -20, 1.5, 3)]:
        A = _edges_to_ls(perturb_dag(sc["a_edges"], shift=shift, rotate=rot, noise=noise, seed=seed))
        _, _, _pr, cr = conditioned_labels(A, sc["b_edges"], method="recursive", **sc["defaults"])
        _, _, _pf, cf = conditioned_labels(A, sc["b_edges"], method="fvs", **sc["defaults"])
        assert cr == pytest.approx(cf, abs=1e-6), \
            f"{name} shift={shift} rot={rot}: recursive {cr} != fvs {cf}"


def test_bp_equals_brute_force_chain():
    # exact BP must match an independent full enumeration on a tiny graph.
    sc = get_dag("chain")
    g, gb, _phi, cr = conditioned_labels(sc["a_edges"], sc["b_edges"], method="recursive",
                                         step_meters=15.0)
    NA, NB = g.n_vertices, gb.n_vertices
    assert float(NB) ** NA <= 5000, "chain@step15 should be tiny"
    fwd = []
    for v in range(NB):
        s = {v}; st = [v]
        while st:
            u = st.pop()
            for w in gb.succ_arcs[u]:
                if w not in s:
                    s.add(w); st.append(w)
        fwd.append(s)
    best = float("inf")
    for combo in itertools.product(range(NB), repeat=NA):
        if all(combo[a2] in fwd[combo[a]] for a in range(NA) for a2 in g.succ_arcs[a]):
            best = min(best, sum(float(np.hypot(g.vx[a] - gb.vx[combo[a]],
                                                g.vy[a] - gb.vy[combo[a]])) for a in range(NA)))
    assert cr == pytest.approx(best, abs=1e-6)


def test_candidate_restriction_safe_on_single_loop():
    # for |F| = 1 the K-nearest candidate set must not miss the all-candidates optimum.
    sc = get_dag("diamond")
    nb = build_local_digraph(sc["b_edges"], [], 0.5, 2.0).n_vertices
    _, _, _p, ck = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs", **sc["defaults"])
    _, _, _p, call = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs",
                                        cand_k=nb, **sc["defaults"])
    assert ck == pytest.approx(call, abs=1e-6)


def test_reduces_to_forest_solver_on_trees():
    # a tree has an empty FVS, so both methods just run the forest solver -> identical labels.
    for name in ("chain", "y_split", "merge"):
        sc = get_dag(name)
        _, _, phr, _cr = conditioned_labels(sc["a_edges"], sc["b_edges"], method="recursive",
                                            **sc["defaults"])
        _, _, phf, _cf = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs",
                                            **sc["defaults"])
        assert phr == phf, f"{name}: tree labellings differ"
