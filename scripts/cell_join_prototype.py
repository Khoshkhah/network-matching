"""Prototype + verification of the CELL-LEVEL JOIN (docs/junction_join_extraction.md §8).

Full-resolution engine: cells (entry + run) are the propagating object; the E-multiplier ledger is
{1 advance/source, beta stall, alpha cover}; a vertex's entry-E is DEFERRED to its parent step (and
at a merge to the root-join, beta if ANY arm stalls); a merge child is absorbed by one parent line
(consumed-once), the other lines carry a PENDING separator (child, entry-cell, stall-flag) matched
at the root join. No stored history is consulted -- only prepare()'s E and the forbidden flags.

Ground truth: FULL-SPACE brute force -- every (entry, run) combination per vertex, filtered by
check_rules, costed by C(M) -- on tiny cases; three-way cross-checks (cell <= branching, cell <=
vertex-join, valid) on larger shapes.

Run:  python scripts/cell_join_prototype.py
"""
from __future__ import annotations

import itertools
import random
import sys

sys.path.insert(0, ".")
import networkx as nx

from network_matching.tree_dtw import (digraph, prepare, forward, extract, extract_join,
                                       check_rules, _cost_of, INF)

RUNCAP = 3                                                      # max cover cells per run (prototype)


def sink_reachable(A, B):
    """The pruning pre-pass (docs §8.2): one reverse search from ALL sink cells over the cell-move
    graph -- cover reversed inside a vertex, advance/stall reversed across edges. Every cell the
    exploration never sees cannot appear on any chain to a sink and is removed up front, in every
    role (entries, run ends, intermediates alike)."""
    seen, stack = set(), []
    for X in A.nodes:
        if A.out_degree(X) == 0:
            for v, c in A.nodes[X]["cand"].items():
                if not c.get("forbidden"):
                    seen.add((X, v))
                    stack.append((X, v))
    while stack:
        X, v = stack.pop()
        cand = A.nodes[X]["cand"]
        for u in B.predecessors(v):                             # cover reversed (same vertex)
            if u in cand and not cand[u].get("forbidden") and (X, u) not in seen:
                seen.add((X, u))
                stack.append((X, u))
        for P in A.predecessors(X):                             # advance/stall reversed (to the parent)
            pc = A.nodes[P]["cand"]
            for u in [v] + list(B.predecessors(v)):
                if u in pc and not pc[u].get("forbidden") and (P, u) not in seen:
                    seen.add((P, u))
                    stack.append((P, u))
    return seen


def _usable(A, a):
    return [v for v, c in A.nodes[a]["cand"].items() if not c.get("forbidden") and c["D"] < INF]


def _runs(A, B, a, e, seen=None):
    """All directed cover paths from entry ``e`` inside cand(a) (simple, length <= RUNCAP);
    removed (never-seen) cells cannot be covered either."""
    cand = A.nodes[a]["cand"]
    out, stack = [], [[e]]
    while stack:
        path = stack.pop()
        out.append(tuple(path))
        if len(path) > RUNCAP:
            continue
        for w in B.successors(path[-1]):
            if (w in cand and not cand[w].get("forbidden") and w not in path
                    and (seen is None or (a, w) in seen)):
                stack.append(path + [w])
    return out


# ------------------------------------------------------------------------------------------------
# the cell-level join
# ------------------------------------------------------------------------------------------------
def cell_join(A, B, alpha, beta):
    """(total, M) exact over the full cell-level space, or None if infeasible."""
    total, M_all = 0.0, set()
    seen = sink_reachable(A, B)
    if any(not any((X, v) in seen for v in A.nodes[X]["cand"]) for X in A.nodes):
        return None                                             # some vertex has no surviving cell
    for comp in nx.weakly_connected_components(A):
        comp = set(comp)
        order = [n for n in nx.topological_sort(A) if n in comp]
        tables: dict = {}                                       # vertex -> list of rows
        absorbed_by: dict = {}
        # row = (entry, value, pending{(c,ce): flag}, cells{vertex: run tuple})
        for X in reversed(order):
            cand = A.nodes[X]["cand"]
            E = lambda v: cand[v]["E"]
            rows = {}
            for e in _usable(A, X):
                if (X, e) not in seen:
                    continue                                    # never seen from the sinks -- removed
                for R in _runs(A, B, X, e, seen):
                    u = R[-1]
                    base_val = sum(alpha * E(c) for c in R[1:])  # cover cells; entry deferred
                    combos = [(base_val, {}, {X: R})]
                    dead = False
                    for c in A.successors(X):
                        if c not in comp:
                            continue
                        if c in absorbed_by and absorbed_by[c] != X:
                            # another line absorbed c: connect through a pending separator
                            child_rows = [(ce, 0.0, {(c, ce): False}, {})
                                          for ce in _usable(A, c) if (c, ce) in seen]
                            deferred = True
                        else:
                            absorbed_by[c] = X
                            child_rows = tables[c]
                            deferred = len(list(A.predecessors(c))) > 1
                        opts = []
                        for (ce, cval, cpend, ccells) in child_rows:
                            stall = (ce == u)
                            if not stall and ce not in B.successors(u):
                                continue                        # children connect at the run end only
                            add = cval if deferred else cval + (beta if stall else 1.0) * A.nodes[c]["cand"][ce]["E"]
                            pend = dict(cpend)
                            if deferred:
                                key = (c, ce)
                                pend[key] = pend.get(key, False) or stall
                            opts.append((add, pend, ccells))
                        if not opts:
                            dead = True
                            break
                        combos = [(v0 + av, _merge_pend(p0, ap), {**c0, **ac})
                                  for (v0, p0, c0) in combos for (av, ap, ac) in opts
                                  if _merge_pend(p0, ap) is not None]
                        if not combos:
                            dead = True
                            break
                    if dead:
                        continue
                    for (val, pend, cells) in combos:
                        key = (e, frozenset(pend.items()))
                        if key not in rows or val < rows[key][1] - 1e-12:
                            rows[key] = (e, val, pend, cells)
            if not rows:
                return None
            tables[X] = list(rows.values())
        # roots: sources pay their own entry (free entry, full E); then join on pending separators
        roots = []
        for X in comp:
            if A.in_degree(X) == 0:
                roots.append([(e, val + A.nodes[X]["cand"][e]["E"], pend, cells)
                              for (e, val, pend, cells) in tables[X]])
        joined = [(0.0, {}, {})]
        for root in roots:
            nxt = []
            for (v0, p0, c0) in joined:
                for (_e, val, pend, cells) in root:
                    p = _merge_pend(p0, pend)
                    if p is None:
                        continue
                    nxt.append((v0 + val, p, {**c0, **cells}))
            joined = nxt
            if not joined:
                return None
        best = None
        for (val, pend, cells) in joined:
            for (c, ce), flag in pend.items():                  # pay every deferred merge entry once
                val += (beta if flag else 1.0) * A.nodes[c]["cand"][ce]["E"]
            if best is None or val < best[0] - 1e-12:
                best = (val, cells)
        if best is None:
            return None
        total += best[0]
        M_all |= {(a, v) for a, run in best[1].items() for v in run}
    return total, M_all


def _merge_pend(p0, p1):
    """Union two pending dicts; None on a separator cell conflict; flags OR."""
    out = dict(p0)
    for (c, ce), flag in p1.items():
        for (c2, ce2) in out:
            if c2 == c and ce2 != ce:
                return None
        out[(c, ce)] = out.get((c, ce), False) or flag
    return out


# ------------------------------------------------------------------------------------------------
# full-space brute force (tiny cases only)
# ------------------------------------------------------------------------------------------------
def brute_full(A, B, alpha, beta, cap=300_000):
    per_vertex = []
    for a in A.nodes:
        choices = [run for e in _usable(A, a) for run in _runs(A, B, a, e)]
        if not choices:
            return None
        per_vertex.append(choices)
    n = 1
    for c in per_vertex:
        n *= len(c)
        if n > cap:
            return "TOO_BIG"
    best = None
    verts = list(A.nodes)
    for combo in itertools.product(*per_vertex):
        M = {(a, v) for a, run in zip(verts, combo) for v in run}
        v1, v2, v3 = check_rules(M, A, B)
        if v1 or v2 or v3:
            continue
        cost = _cost_of(A, B, M, alpha, beta)
        if best is None or cost < best - 1e-12:
            best = cost
    return best


# ------------------------------------------------------------------------------------------------
def tiny_cases():
    # dense-B chain: |A|=3 over a 7-cell B chain -- the coverage regime that beat the vertex join
    yield "dense_chain", digraph({0: (0, 0), 1: (9, 0), 2: (18, 0)}, [(0, 1), (1, 2)]), \
        digraph({f"b{i}": (3 * i, .4) for i in range(7)},
                [(f"b{i}", f"b{i+1}") for i in range(6)])
    # dense-B y-split: |A|=4
    yield "dense_ysplit", digraph({0: (0, 0), 1: (8, 0), 2: (16, 5), 3: (16, -5)},
                                  [(0, 1), (1, 2), (1, 3)]), \
        digraph({"s0": (0, .4), "s1": (4, .4), "j": (8, .4), "u1": (12, 2.9), "u2": (16, 5.4),
                 "d1": (12, -2.6), "d2": (16, -4.6)},
                [("s0", "s1"), ("s1", "j"), ("j", "u1"), ("u1", "u2"), ("j", "d1"), ("d1", "d2")])
    # mini merge: two sources join, one sink -- full brute feasible
    yield "mini_merge", digraph({"s1": (0, 4), "x": (6, 2), "s2": (0, -4), "z": (6, -2),
                                 "m": (12, 0), "d": (18, 0)},
                                [("s1", "x"), ("x", "m"), ("s2", "z"), ("z", "m"), ("m", "d")]), \
        digraph({"B1": (0, 4.4), "Bx": (6, 2.4), "B2": (0, -3.6), "Bz": (6, -1.6),
                 "Bm": (12, .4), "Bd": (18, .4)},
                [("B1", "Bx"), ("Bx", "Bm"), ("B2", "Bz"), ("Bz", "Bm"), ("Bm", "Bd")])


def run():
    GRID = [(1.0, 1.0), (0.5, 1.0), (0.3, 1.5)]
    print(f"{'case':14} {'a,b':10} {'cell-join':>10} {'full brute':>10} {'branching':>10} {'vtx-join':>10}  verdict")
    bad = 0
    for name, A0, B0 in tiny_cases():
        for ab in GRID:
            A, B = A0.copy(), B0.copy()
            prepare(A, B, r=25.0)
            forward(A, B, *ab)
            cj = cell_join(A, B, *ab)
            bf = brute_full(A, B, *ab)
            try:
                Mb, _ = extract(A, B, *ab, max_states=100000)
                cb = _cost_of(A, B, Mb, *ab)
            except ValueError:
                cb = None
            try:
                Mv, _ = extract_join(A, B, *ab)
                cv = _cost_of(A, B, Mv, *ab)
            except ValueError:
                cv = None
            cjc = None if cj is None else cj[0]
            ok = True
            if isinstance(bf, float) and cjc is not None:
                ok &= abs(cjc - bf) < 1e-6
            if cjc is not None and cb is not None:
                ok &= cjc <= cb + 1e-6
            if cjc is not None and cv is not None:
                ok &= cjc <= cv + 1e-6
            if cj is not None:
                vr = check_rules(cj[1], A, B)
                ok &= not any(vr)
                ok &= abs(_cost_of(A, B, cj[1], *ab) - cjc) < 1e-6   # value == C(M) self-check
            bad += not ok
            fmt = lambda x: "     -    " if x is None else ("  TOO_BIG " if x == "TOO_BIG" else f"{x:>10.3f}")
            print(f"{name:14} {str(ab):10} {fmt(cjc)} {fmt(bf)} {fmt(cb)} {fmt(cv)}  "
                  f"{'OK' if ok else '*** BAD ***'}")
    # random mini polytrees: three-way invariants (+ full brute when small enough)
    agree = n = 0
    for seed in range(30):
        rng = random.Random(seed)
        nA = rng.randint(3, 4)
        und = [(rng.randrange(i), i) for i in range(1, nA)]
        pos = {i: (rng.uniform(0, 20), rng.uniform(0, 20)) for i in range(nA)}
        A = nx.DiGraph()
        for i in range(nA):
            A.add_node(i, x=pos[i][0], y=pos[i][1])
        for k, (a, b) in enumerate(und):
            if rng.random() < 0.5:
                a, b = b, a
            mid = f"m{k}"
            A.add_node(mid, x=(pos[a][0] + pos[b][0]) / 2, y=(pos[a][1] + pos[b][1]) / 2)
            A.add_edge(a, mid)
            A.add_edge(mid, b)
        B = nx.DiGraph()
        nb = rng.randint(5, 7)
        vs = [f"v{i}" for i in range(nb)]
        for v in vs:
            B.add_node(v, x=rng.uniform(0, 20), y=rng.uniform(0, 20))
        for i in range(nb - 1):
            B.add_edge(vs[i], vs[i + 1])
        try:
            prepare(A, B, r=30.0)
            forward(A, B, 0.5, 1.0)
        except ValueError:
            continue
        cj = cell_join(A, B, 0.5, 1.0)
        bf = brute_full(A, B, 0.5, 1.0)
        n += 1
        if cj is None:
            agree += (bf is None or not isinstance(bf, float))
            continue
        ok = not any(check_rules(cj[1], A, B))
        if isinstance(bf, float):
            ok &= abs(cj[0] - bf) < 1e-6
        agree += ok
    print(f"\nrandom mini polytrees: {agree}/{n} exact & valid")
    print("VERDICT:", "ALL OK" if bad == 0 and agree == n else f"{bad} bad canonical, {n-agree} bad random")


if __name__ == "__main__":
    run()
