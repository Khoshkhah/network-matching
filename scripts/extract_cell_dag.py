"""Benchmark of the cell-DAG extraction (docs/cell_dag_extraction.md) vs the pre-integration
vertex-granularity engine.

The cell-DAG engine IS the library's ``extract_cell`` since 2026-07 (per-cell backward sweep,
inbox-push freeing, implicit runs, early discharge). The engine it replaced -- the per-vertex
``(entry, run)`` enumeration with ``run_cap`` -- is preserved HERE as ``extract_cell_vertex``,
verbatim, so the §7 numbers in the design doc stay reproducible.

Run:  python scripts/extract_cell_dag.py            # correctness parity + benchmark table
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx

from network_matching.dag_dtw import (digraph, prepare, forward, extract_cell, check_rules,
                                      _cost_of, _cell_reachable, _pend_union, _b_order, INF)


# =========================================================================================
# the BASELINE: the pre-2026-07 vertex-granularity engine, verbatim (was extract_cell)
# =========================================================================================
def _cell_runs(A, B, a, e, seen, cap, border):
    """All directed cover paths from entry ``e`` inside cand(a) (simple, <= ``cap`` cover cells)."""
    cand = A.nodes[a]["cand"]
    out, stack = [], [(e,)]
    while stack:
        path = stack.pop()
        out.append(path)
        if len(path) > cap:
            continue
        for w in sorted(B.successors(path[-1]), key=lambda t: border.get(t, 0)):
            if (w in cand and not cand[w].get("forbidden") and w not in path
                    and (a, w) in seen):
                stack.append(path + (w,))
    return out


def extract_cell_vertex(A, B, alpha=1.0, beta=1.0, run_cap=8, max_rows=50000):
    """The replaced engine: per-vertex (entry, run) enumeration, tables kept to component end."""
    border = _b_order(B)
    seen = _cell_reachable(A, B)
    for X in A.nodes:
        if not any((X, v) in seen and A.nodes[X]["cand"][v]["D"] < INF
                   for v in A.nodes[X]["cand"]):
            raise ValueError(f"vertex {X!r} has no surviving cell (sink-search + D-filter) -- "
                             "increase match_radius_m")
    M_all, committed_all = set(), {}
    for comp in nx.weakly_connected_components(A):
        comp = set(comp)
        order = [n for n in nx.topological_sort(A) if n in comp]
        tables, absorbed_by = {}, {}
        for X in reversed(order):
            cand = A.nodes[X]["cand"]
            rows = {}
            entries = sorted((v for v in cand
                              if (X, v) in seen and cand[v]["D"] < INF), key=lambda t: border[t])
            for e in entries:
                for R in _cell_runs(A, B, X, e, seen, run_cap, border):
                    u = R[-1]
                    combos = [(sum(alpha * cand[c]["E"] for c in R[1:]), {}, {X: R})]
                    dead = False
                    for c in sorted(A.successors(X), key=str):
                        if c not in comp:
                            continue
                        if c in absorbed_by and absorbed_by[c] != X:
                            child_rows = [(ce, 0.0, {(c, ce): False}, {})
                                          for ce in sorted(A.nodes[c]["cand"],
                                                           key=lambda t: border[t])
                                          if (c, ce) in seen]
                            deferred = True
                        else:
                            absorbed_by[c] = X
                            child_rows = tables[c]
                            deferred = A.in_degree(c) > 1
                        opts = []
                        for (ce, cval, cpend, ccells) in child_rows:
                            stall = (ce == u)
                            if not stall and ce not in B.successors(u):
                                continue
                            add = cval if deferred else \
                                cval + (beta if stall else 1.0) * A.nodes[c]["cand"][ce]["E"]
                            pend = dict(cpend)
                            if deferred:
                                pend[(c, ce)] = pend.get((c, ce), False) or stall
                            opts.append((add, pend, ccells))
                        if not opts:
                            dead = True
                            break
                        nxt = []
                        for (v0, p0, c0) in combos:
                            for (av, ap, ac) in opts:
                                p = _pend_union(p0, ap)
                                if p is not None:
                                    nxt.append((v0 + av, p, {**c0, **ac}))
                        combos = nxt
                        if not combos:
                            dead = True
                            break
                    if dead:
                        continue
                    for (val, pend, cells) in combos:
                        key = (e, frozenset(pend.items()))
                        if key not in rows or val < rows[key][1] - 1e-12:
                            rows[key] = (e, val, pend, cells)
                    if len(rows) > max_rows:
                        raise ValueError(f"cell-join table at {X!r} exceeded {max_rows} rows -- "
                                         "raise max_rows")
            if not rows:
                raise ValueError(f"vertex {X!r}: no feasible row -- increase match_radius_m")
            tables[X] = sorted(rows.values(), key=lambda r: (r[1], border[r[0]]))
        roots = []
        for X in sorted(comp, key=str):
            if A.in_degree(X) == 0:
                roots.append([(e, val + A.nodes[X]["cand"][e]["E"], pend, cells)
                              for (e, val, pend, cells) in tables[X]])
        joined = [(0.0, {}, {})]
        for root in roots:
            folded = {}
            for (v0, p0, c0) in joined:
                for (_e, val, pend, cells) in root:
                    p = _pend_union(p0, pend)
                    if p is None:
                        continue
                    key = frozenset(p.items())
                    v = v0 + val
                    if key not in folded or v < folded[key][0] - 1e-12:
                        folded[key] = (v, p, {**c0, **cells})
            joined = list(folded.values())
            if len(joined) > max_rows:
                raise ValueError(f"cell-join root join exceeded {max_rows} rows -- raise max_rows")
        finals = []
        for (val, pend, cells) in joined:
            for (c, ce), flag in pend.items():
                val += (beta if flag else 1.0) * A.nodes[c]["cand"][ce]["E"]
            finals.append((val, cells))
        best = None
        for val, cells in sorted(finals, key=lambda t: (t[0], str(sorted(t[1].items(), key=str)))):
            Mc = {(a, v) for a, run in cells.items() for v in run}
            if {a for a, _ in Mc} != comp:
                continue
            if any(check_rules(Mc, A, B)):
                continue
            best = (Mc, {a: run[0] for a, run in cells.items()})
            break
        if best is None:
            raise ValueError(f"cell-join: no valid root row in component of {order[0]!r} -- "
                             "increase match_radius_m")
        M_all |= best[0]
        committed_all.update(best[1])
    return M_all, committed_all


# =========================================================================================
# benchmark families (all subdivided, all congruent-B so cases stay feasible)
# =========================================================================================
def fam_dense_chain(n):
    """A: n-vertex chain, 10 m spacing.  B: 3x denser congruent chain -- coverage-run heavy."""
    An = {i: (10.0 * i, 0.0) for i in range(n)}
    Ae = [(i, i + 1) for i in range(n - 1)]
    nb = 3 * (n - 1) + 1
    Bn = {f"b{j}": (10.0 * (n - 1) * j / (nb - 1), 0.4) for j in range(nb)}
    Be = [(f"b{j}", f"b{j+1}") for j in range(nb - 1)]
    return digraph(An, Ae), digraph(Bn, Be)


def fam_btree(depth):
    """A: subdivided binary out-tree (splits everywhere, no merges).  B: congruent, +0.4 m."""
    nodes, edges = {"r": (0.0, 0.0)}, []

    def grow(name, x, y, d):
        if d == 0:
            return
        dy = 40.0 / (2 ** (depth - d + 1))
        for tag, sy in (("u", dy), ("d", -dy)):
            child, mid = f"{name}{tag}", f"{name}{tag}_m"
            nodes[mid] = (x + 5.0, (y + y + sy) / 2)
            nodes[child] = (x + 10.0, y + sy)
            edges.extend([(name, mid), (mid, child)])
            grow(child, x + 10.0, y + sy, d - 1)

    grow("r", 0.0, 0.0, depth)
    Bn = {k + "'": (x, y + 0.4) for k, (x, y) in nodes.items()}
    Be = [(a + "'", b + "'") for a, b in edges]
    return digraph(nodes, edges), digraph(Bn, Be)


def fam_diamond_chain(k):
    """A: k diamonds in series (a merge per diamond -- pending live).  B: congruent, +0.4 m."""
    nodes, edges = {"s": (0.0, 0.0)}, []
    prev = "s"
    for i in range(k):
        x0 = 4.0 + 20.0 * i
        J, xu, zd, m, t = f"J{i}", f"x{i}", f"z{i}", f"m{i}", f"t{i}"
        nodes[J] = (x0, 0.0)
        nodes[xu] = (x0 + 7.0, 3.0)
        nodes[zd] = (x0 + 7.0, -3.0)
        nodes[m] = (x0 + 14.0, 0.0)
        nodes[t] = (x0 + 17.0, 0.0)
        edges.extend([(prev, J), (J, xu), (J, zd), (xu, m), (zd, m), (m, t)])
        prev = t
    Bn = {kk + "'": (x, y + 0.4) for kk, (x, y) in nodes.items()}
    Be = [(a + "'", b + "'") for a, b in edges]
    return digraph(nodes, edges), digraph(Bn, Be)


# =========================================================================================
# harness: correctness parity, then time + peak-memory
# =========================================================================================
def _run(fn, A, B, alpha, beta):
    try:
        M, _ = fn(A, B, alpha, beta)
        return M, _cost_of(A, B, M, alpha, beta), ""
    except ValueError as e:
        return None, None, str(e)


def parity_suite():
    """Small-case parity incl. merges, cyclic B and the two-cycle refusal (main doc §7)."""
    import random
    cases = []
    for seed in range(12):                                      # random polytrees over cyclic B
        rng = random.Random(seed)
        n = rng.randint(4, 7)
        und = [(rng.randrange(i), i) for i in range(1, n)]
        pos = {i: (rng.uniform(0, 30), rng.uniform(0, 30)) for i in range(n)}
        A = nx.DiGraph()
        for i in range(n):
            A.add_node(i, x=pos[i][0], y=pos[i][1])
        for kk, (a, b) in enumerate(und):
            if rng.random() < 0.5:
                a, b = b, a
            mid = f"m{kk}"
            A.add_node(mid, x=(pos[a][0] + pos[b][0]) / 2, y=(pos[a][1] + pos[b][1]) / 2)
            A.add_edge(a, mid)
            A.add_edge(mid, b)
        Bg = nx.DiGraph()
        nb = rng.randint(6, 10)
        vs = [f"v{i}" for i in range(nb)]
        for v in vs:
            Bg.add_node(v, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
        for i in range(nb - 1):
            Bg.add_edge(vs[i], vs[i + 1])
        for _ in range(rng.randint(1, nb // 2)):
            u, v = rng.choice(vs), rng.choice(vs)
            if u != v:
                Bg.add_edge(u, v)
        cases.append((f"polytree{seed}", A, Bg, 40.0, 0.5, 1.0))
    cases.append(("two-cycle", digraph({0: (0, 0), 1: (10, 0)}, [(0, 1)]),
                  digraph({"p": (0, 1), "q": (10, 1)}, [("p", "q"), ("q", "p")]), 20.0, 1.0, 1.0))
    cases.append(("diamonds3", *fam_diamond_chain(3), 10.0, 0.5, 1.0))
    cases.append(("dense8", *fam_dense_chain(8), 15.0, 0.5, 1.0))
    cases.append(("btree3", *fam_btree(3), 15.0, 0.5, 1.5))

    ok = 0
    for name, A, Bg, r, al, be in cases:
        prepare(A, Bg, r=r)
        try:
            forward(A, Bg, alpha=al, beta=be)
        except ValueError:
            continue
        Mv, cv, why_v = _run(extract_cell_vertex, A, Bg, al, be)
        Md, cd, why_d = _run(extract_cell, A, Bg, al, be)
        if (Mv is None) != (Md is None):
            print(f"  PARITY FAIL {name}: vtx[{why_v or 'ok'}] dag[{why_d or 'ok'}]")
            continue
        if Mv is not None:
            assert not any(check_rules(Md, A, Bg)), f"{name}: dag returned invalid M"
            if abs(cv - cd) > 1e-6:
                print(f"  PARITY FAIL {name}: C(vtx)={cv:.6f}  C(dag)={cd:.6f}")
                continue
        ok += 1
    print(f"parity: {ok}/{len(cases)} cases agree (cost & refusals)")
    return ok == len(cases)


def bench():
    fams = ([("dense-chain", fam_dense_chain, n, 15.0, 0.5, 1.0) for n in (50, 150, 400, 800)]
            + [("btree", fam_btree, d, 15.0, 0.5, 1.0) for d in (4, 6, 8)]
            + [("diamonds", fam_diamond_chain, k, 10.0, 0.5, 1.0) for k in (4, 40, 120, 400)])
    print(f"\n{'family':<12} {'size':>5} {'|A|':>5} {'cells':>7}   "
          f"{'t_vtx':>8} {'t_dag':>8} {'x':>5}   {'mem_vtx':>9} {'mem_dag':>9} {'x':>5}   eq")
    for fam, build, size, r, al, be in fams:
        A, Bg = build(size)
        prepare(A, Bg, r=r)
        forward(A, Bg, alpha=al, beta=be)
        ncell = sum(len(A.nodes[a]["cand"]) for a in A.nodes)
        res = {}
        for tag, fn in (("vtx", extract_cell_vertex), ("dag", extract_cell)):
            t0 = time.perf_counter()
            M, cost, why = _run(fn, A, Bg, al, be)
            t = time.perf_counter() - t0
            tracemalloc.start()
            _run(fn, A, Bg, al, be)
            _cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            res[tag] = (t, peak, cost, why)
        tv, mv, cv, _ = res["vtx"]
        td, md, cd, _ = res["dag"]
        if cv is None and cd is None:
            eq = "REFUSED"
        elif cv is None:
            eq = "dag-only"                                     # baseline hit its cap
        elif cd is None:
            eq = "**vtx-only**"
        else:
            eq = "YES" if abs(cv - cd) < 1e-6 else "**NO**"
        print(f"{fam:<12} {size:>5} {A.number_of_nodes():>5} {ncell:>7}   "
              f"{tv:8.3f} {td:8.3f} {tv/td:5.2f}   {mv/1e6:8.2f}M {md/1e6:8.2f}M "
              f"{mv/max(md,1):5.2f}   {eq}")


if __name__ == "__main__":
    if parity_suite():
        bench()
    else:
        print("parity failures -- fix before benchmarking")
