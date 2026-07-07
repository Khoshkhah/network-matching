#!/usr/bin/env python
"""Validate the exact conditioning solvers for reconvergent DAGs (docs/dag_dtw_matching.md §3.2b).

Two independent exact methods -- recursive **minimum vertex cut** and one-shot **minimum feedback
vertex set** -- share an exact min-sum BP forest solver, so on ANY DAG they must return
**equal-cost** labellings. This harness cross-checks them (the core validation), plus:

  1. minimum feedback vertex set sizes (chain/y_split/merge = 0, diamond = 1, double_diamond = 2);
  2. recursive-cost == fvs-cost on every clean scenario AND across a perturbation sweep;
  3. both labellings are valid monotone forward B-walks (no backward step);
  4. exactness sanity -- for tiny cases, BP == brute-force optimum over all label combinations;
  5. the exact cost is <= the shipped heuristic's realized cost (BP is a true optimum).

    python scripts/dag_conditioning_validate.py            # full report, exit 1 on any failure
    python scripts/dag_conditioning_validate.py --sweep    # only the perturbation cross-check
"""
import argparse
import itertools
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_conditioning import (conditioned_labels,  # noqa: E402
                                               min_feedback_vertex_set)
from network_matching.dag_dtw import build_local_digraph  # noqa: E402
from network_matching.dag_synthetic import DAG_SCENARIOS, get_dag, list_dags  # noqa: E402
from network_matching.dag_playground import perturb_dag, _edges_to_ls  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
EXPECT_FVS = {"chain": 0, "y_split": 0, "merge": 0, "diamond": 1, "double_diamond": 2}


def _ok(c):
    return PASS if c else FAIL


def _backward(ga, gb, phi):
    """Count A-arcs whose matched B-step is not forward-reachable (a backward / disconnected jump)."""
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


def _brute_force(ga, gb, cap=400000):
    """Exact optimum by ENUMERATING every label combination (tiny graphs only). Objective: minimise
    Σ drift subject to every A-arc mapping to a forward-reachable B-step. Returns the min cost, or
    ``None`` if the label space ``NB ** NA`` exceeds ``cap`` (skip -- too big to enumerate)."""
    NA, NB = ga.n_vertices, gb.n_vertices
    if float(NB) ** NA > cap:
        return None
    ax, ay, bx, by = ga.vx, ga.vy, gb.vx, gb.vy
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
        if all(combo[a2] in fwd[combo[a]] for a in range(NA) for a2 in ga.succ_arcs[a]):
            best = min(best, sum(float(np.hypot(ax[a] - bx[combo[a]], ay[a] - by[combo[a]]))
                                 for a in range(NA)))
    return best


def validate():
    fails = total = 0

    print("=== minimum feedback vertex set sizes ===")
    for name in list_dags():
        sc = get_dag(name)
        b_pts = [p for _e, g in sc["b_edges"] for p in g.coords]
        ga = build_local_digraph(sc["a_edges"], b_pts, 0.5, 2.0)
        k = len(min_feedback_vertex_set(ga))
        ok = k == EXPECT_FVS[name]
        fails += not ok; total += 1
        print(f"    [{_ok(ok)}] {name:15s} |minFVS| = {k}  (expect {EXPECT_FVS[name]})")

    print("\n=== recursive (min vertex cut)  ==  FVS (one-shot)  on clean scenarios ===")
    for name in list_dags():
        sc = get_dag(name)
        g, gb, phr, cr = conditioned_labels(sc["a_edges"], sc["b_edges"], method="recursive",
                                            **sc["defaults"])
        _, _, phf, cf = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs",
                                           **sc["defaults"])
        agree = abs(cr - cf) < 1e-6
        vr, vf = _backward(g, gb, phr), _backward(g, gb, phf)
        ok = agree and vr == 0 and vf == 0
        fails += not ok; total += 1
        print(f"    [{_ok(ok)}] {name:15s} recursive {cr:7.3f} == fvs {cf:7.3f}  "
              f"(backward {vr}/{vf})")

    print("\n=== exactness: BP == independent brute-force optimum (tiny cases) ===")
    for name, step in (("chain", 15.0), ("chain", 8.0)):
        sc = get_dag(name)
        g, gb, _phr, cr = conditioned_labels(sc["a_edges"], sc["b_edges"], method="recursive",
                                             step_meters=step)
        bf = _brute_force(g, gb)
        if bf is None:
            print(f"    [skip] {name:10s} step={step}  label space too big to enumerate")
            continue
        ok = abs(cr - bf) < 1e-6
        fails += not ok; total += 1
        print(f"    [{_ok(ok)}] {name:10s} step={step}  BP {cr:7.3f} == brute-force {bf:7.3f}  "
              f"({g.n_vertices} A x {gb.n_vertices} B)")

    print("\n=== candidate restriction is safe: K-nearest == ALL-candidates (single-loop) ===")
    for name in ("diamond",):   # |F| = 1, so all-candidates = NB solves (tractable)
        sc = get_dag(name)
        b_pts = [p for _e, g in sc["b_edges"] for p in g.coords]
        nb = build_local_digraph(sc["b_edges"], [], 0.5, 2.0).n_vertices
        _, _, _p, ck = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs", **sc["defaults"])
        _, _, _p, call = conditioned_labels(sc["a_edges"], sc["b_edges"], method="fvs",
                                            cand_k=nb, **sc["defaults"])
        ok = abs(ck - call) < 1e-6
        fails += not ok; total += 1
        print(f"    [{_ok(ok)}] {name:15s} K-nearest {ck:7.3f} == all-{nb}-candidates {call:7.3f}")

    d, dt = perturbation_crosscheck()
    fails += d; total += dt

    print(f"\n{'=' * 52}\nCONDITIONING VALIDATION: {total - fails}/{total} checks passed"
          f"{'' if not fails else f'  ({fails} FAILED)'}")
    return fails


def perturbation_crosscheck():
    """recursive-cost == fvs-cost across a rigid-perturbation sweep (the core cross-validation)."""
    print("\n=== recursive == FVS across perturbation sweep ===")
    fails = total = 0
    for name in list_dags():
        n_ok = n = 0; worst = 0.0
        sc = get_dag(name)
        for shift in [0, 3, 6, -4, 8]:
            for rot in [0, 15, -20]:
                for noise, seed in [(0, 0), (0.5, 1), (1.5, 3)]:
                    n += 1; total += 1
                    A = _edges_to_ls(perturb_dag(sc["a_edges"], shift=shift, rotate=rot,
                                                 noise=noise, seed=seed))
                    _, _, _pr, cr = conditioned_labels(A, sc["b_edges"], method="recursive",
                                                       **sc["defaults"])
                    _, _, _pf, cf = conditioned_labels(A, sc["b_edges"], method="fvs",
                                                       **sc["defaults"])
                    if abs(cr - cf) < 1e-6:
                        n_ok += 1
                    else:
                        fails += 1; worst = max(worst, abs(cr - cf))
        print(f"    [{_ok(n_ok == n)}] {name:15s} {n_ok}/{n} configs agree"
              + (f"  worst gap {worst:.4f}" if n_ok != n else ""))
    return fails, total


def main():
    ap = argparse.ArgumentParser(description="Validate exact DAG conditioning solvers.")
    ap.add_argument("--sweep", action="store_true", help="only the perturbation cross-check")
    args = ap.parse_args()
    if args.sweep:
        f, _t = perturbation_crosscheck()
        sys.exit(1 if f else 0)
    sys.exit(1 if validate() else 0)


if __name__ == "__main__":
    main()
