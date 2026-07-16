"""§4.1a role-aware forbidding — a forbidden cell is "not a valid run END", not "not a valid place".

The ``forbidden`` flag is set by the split coupling because some child cannot ATTACH at the cell —
a statement about the cell as the split's **run end** only. The §3 rules agree: V3 is vacuous for a
cell the vertex itself continues past (run entry/interior). Enforcing the flag in every role —
advance/stall target AND same-row coverage source AND extraction row membership — killed runs that
merely PASS THROUGH the cell, turning feasible instances infeasible (adversarial differential vs
the full-space brute: 4/120 displaced optima, 1 spurious raise; pre-existing under the old
optimality rule too, which forbade a strict superset).

The canonical counterexample: continuity from the parent forces the split's run to ENTER at a cell
no child can follow; the run then covers on to the cell where every child attaches.
"""
import copy
import itertools
import random

import networkx as nx
import pytest

from network_matching.dag_dtw import (INF, check_rules, check_split_exits, digraph, extract_cell,
                                      forward, prepare, _cost_of)


def _run_through_trap():
    """A: a0 -> p -> {c1, c2};  B: x0 -> x -> u -> {w1, w2}.

    cand(p) = {x, u}. Children attach only after u, so the coupling flags x (correct: x cannot be
    p's run END). But a0 ends on x0, and p's run must START at a successor of x0 — that is x. The
    ONLY valid matching is p -> run (x, u): enter at the flagged cell, cover to u, children attach
    at u. Role-blind enforcement deletes x everywhere and the instance dies."""
    A = digraph({"a0": (0, 0), "p": (10, 0), "c1": (20, 4), "c2": (20, -4)},
                [("a0", "p"), ("p", "c1"), ("p", "c2")])
    B = digraph({"x0": (0, 1), "x": (8, 1), "u": (12, 1), "w1": (20, 5), "w2": (20, -3)},
                [("x0", "x"), ("x", "u"), ("u", "w1"), ("u", "w2")])
    return A, B


def test_run_may_pass_through_a_forbidden_cell():
    A, B = _run_through_trap()
    prepare(A, B, r=4.0)
    forward(A, B)
    assert A.nodes["p"]["cand"]["x"]["forbidden"]        # x correctly flagged: no child attaches there
    assert not A.nodes["p"]["cand"]["u"]["forbidden"]
    assert check_split_exits(A, B) == []
    M, committed = extract_cell(A, B)                    # role-blind enforcement raised here
    assert ("p", "x") in M and ("p", "u") in M           # the run ENTERS at the flagged cell...
    assert ("c1", "w1") in M and ("c2", "w2") in M       # ...and the children attach at its END
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)


def _rand_split_case(seed):
    """Random out-tree A (guaranteed >= 1 split) over a random small target B; tight radius so
    coupling flags actually fire. Deterministic per seed."""
    rng = random.Random(seed)
    na, nb = rng.randint(4, 6), rng.randint(6, 9)
    A = nx.DiGraph()
    for i in range(na):
        A.add_node(i, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(1, na):
        A.add_edge(rng.randrange(i), i)
    if max(d for _, d in A.out_degree()) < 2:            # force a split
        leaf = max(A.nodes)
        A.add_node(na, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
        A.add_edge(next(iter(A.predecessors(leaf))), na)
    B = nx.DiGraph()
    vs = [f"v{i}" for i in range(nb)]
    for v in vs:
        B.add_node(v, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(nb - 1):
        B.add_edge(vs[i], vs[i + 1])
    for _ in range(rng.randint(1, nb // 2)):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return A, B


def _flagless_brute(A, B, alpha, beta, run_cap=3, cap=300_000):
    """Ground-truth optimum over ALL runs of D-finite cells — flags deliberately ignored (validity
    is judged by check_rules alone, which is role-aware by construction: V3 binds only run ends)."""
    def runs(a, e):
        out, frontier = [(e,)], [(e,)]
        for _ in range(run_cap - 1):
            nxt = []
            for r in frontier:
                for w in B.successors(r[-1]):
                    if w in A.nodes[a]["cand"] and w not in r:
                        nxt.append(r + (w,))
            out += nxt
            frontier = nxt
        return out

    per_vertex = []
    for a in A.nodes:
        opts = [r for e, c in A.nodes[a]["cand"].items() if c["D"] < INF for r in runs(a, e)]
        if not opts:
            return None
        per_vertex.append(opts)
    n = 1
    for o in per_vertex:
        n *= len(o)
        if n > cap:
            return "too-big"
    best, verts = None, list(A.nodes)
    for combo in itertools.product(*per_vertex):
        M = {(a, v) for a, run in zip(verts, combo) for v in run}
        if any(check_rules(M, A, B)):
            continue
        cost = _cost_of(A, B, M, alpha, beta)
        if best is None or cost < best - 1e-12:
            best = cost
    return best


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_differential_vs_flagless_brute_random_sweep(alpha, beta):
    """extract_cell must equal the flag-free full-space optimum on every feasible case in THIS
    sweep — the flags may prune, never displace the optimum or fabricate infeasibility. (Pre-fix:
    4/120 displaced optima, worst +24%, and 1 spurious raise on exactly this class.)

    KNOWN RESIDUAL outside this sweep (open; see scripts/repro_contraction_eviction/): the
    cheapest-per-pending-signature contraction in extract_cell is validity-blind, so on CYCLIC
    targets a cheap-but-V1-invalid row can evict the valid row sharing its signature before the
    terminal judge -- ~2% spurious raises / rare displaced optima on adversarial random cases,
    for role-blind and role-aware enforcement alike (the role change is a net improvement:
    18+1 vs 22+3 per 900). Fix direction: validity-aware or top-K contraction."""
    checked = 0
    for seed in range(40):
        A, B = _rand_split_case(seed)
        prepare(A, B, r=12.0)
        try:
            forward(A, B, alpha=alpha, beta=beta)
        except ValueError:
            continue                                     # genuinely no shared run-end: out of scope
        bf = _flagless_brute(A, B, alpha, beta)
        if bf in (None, "too-big"):
            continue
        M, _ = extract_cell(A, B, alpha, beta)
        got = _cost_of(A, B, M, alpha, beta)
        assert got <= bf + 1e-9, f"seed {seed}: extract_cell {got} above flag-free optimum {bf}"
        v1, v2, v3 = check_rules(M, A, B)
        assert not (v1 or v2 or v3), f"seed {seed}: invalid matching"
        checked += 1
    assert checked >= 10, f"sweep too thin ({checked} feasible cases) -- loosen the generator"
