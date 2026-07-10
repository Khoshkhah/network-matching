"""Prototype + brute-force verification of the JUNCTION-JOIN extraction
(docs/junction_join_extraction.md) -- forward-only, exact, no search.

Implements the recursion literally: every table is a sink-type table (label -> through-cost, with
the row's pinned sink/split labels AND the cells recorded along its induce-walks); splits are
processed deepest-first; each branch's terminal is found by walking down to the first table-owned
vertex, and the row's recorded cell there starts the induce-walk up to the split. A collapsed table
can therefore serve as a LATER split's branch terminal through any recorded interior vertex (e.g. a
merge) -- which is exactly the doc's open "merge bookkeeping" question, settled here numerically.

Verification: root-table minimum vs BRUTE FORCE over all sink-label combinations (reconstruct M by
the bp up-flood, cost with the honest C(M)) on canonical shapes + random subdivided polytrees.

Run:  python scripts/junction_join_prototype.py
"""
from __future__ import annotations

import itertools
import random
import sys

sys.path.insert(0, ".")
import networkx as nx

from network_matching.tree_dtw import digraph, prepare, forward, _cost_of, check_rules, INF


# ----------------------------------------------------------------------------------------------
# shared: reconstruct M from pinned sink labels (bp up-flood; None on conflict/forbidden/severed)
# ----------------------------------------------------------------------------------------------
def reconstruct(A, sink_labels):
    cells = {}                                                  # vertex -> set of covered cells
    pin = {}                                                    # vertex -> its entry cell (walk pin)
    stack = list(sink_labels.items())
    while stack:
        a, v = stack.pop()
        if a in pin:
            if pin[a] != v:
                return None                                     # two walks disagree -> inconsistent
            continue
        cand = A.nodes[a]["cand"]
        if v not in cand or cand[v].get("forbidden") or cand[v]["D"] >= INF:
            return None
        pin[a] = v
        cells.setdefault(a, set()).add(v)
        x = v
        while True:                                             # own cover chain -> run cells
            bp = cand[x]["bpD"]
            if len(bp) == 1 and bp[0][0] == a:
                x = bp[0][1]
                if x is None or cand[x].get("forbidden"):
                    return None
                cells[a].add(x)
            else:
                break
        for (p, xp) in cand[x]["bpD"]:                          # advance list: every arm of a merge
            if xp is None:
                return None
            stack.append((p, xp))
    return {(a, c) for a, cs in cells.items() for c in cs}


def brute_force(A, B, alpha, beta):
    sinks = [n for n in A.nodes if A.out_degree(n) == 0]
    options = []
    for s in sinks:
        opts = [v for v, c in A.nodes[s]["cand"].items() if not c.get("forbidden") and c["D"] < INF]
        if not opts:
            return None
        options.append(opts)
    best = None
    for combo in itertools.product(*options):
        M = reconstruct(A, dict(zip(sinks, combo)))
        if M is None or {a for a, _ in M} != set(A.nodes):
            continue
        cost = _cost_of(A, B, M, alpha, beta)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, dict(zip(sinks, combo)), M)
    return best


# ----------------------------------------------------------------------------------------------
# the junction join
# ----------------------------------------------------------------------------------------------
def junction_join(A, B, alpha, beta):
    """Root tables per component -> global (cost, sink_labels, M). None if infeasible."""
    total_cost, sink_labels = 0.0, {}
    for comp in nx.weakly_connected_components(A):
        comp = set(comp)
        # --- leaf tables: one per sink; rows = (cost, cells{vertex:cell}, pins{vertex:label}) ---
        tables, owner, consumed = {}, {}, set()                 # table id = its owner vertex
        alias = {}                                              # consumed table -> the table that absorbed it

        def find(t):
            while t in alias:
                t = alias[t]
            return t
        for s in comp:
            if A.out_degree(s) == 0:
                rows = {}
                for v, c in A.nodes[s]["cand"].items():
                    if not c.get("forbidden") and c["D"] < INF:
                        rows[v] = (c["D"], {s: v}, {s: v})
                if not rows:
                    return None
                tables[s] = rows
                owner[s] = s
        # --- splits deepest-first ---
        splits = [n for n in nx.topological_sort(A) if n in comp and A.out_degree(n) > 1]
        for U in reversed(splits):
            branches = []
            for child in A.successors(U):
                path = [U, child]                               # walk down to the first owned vertex
                cur = child
                while cur not in owner:
                    nxts = list(A.successors(cur))
                    assert len(nxts) == 1, "un-collapsed split below?"
                    cur = nxts[0]
                    path.append(cur)
                branches.append((find(owner[cur]), cur, path))   # (table id, entry vertex, U..entry)
            new_rows = {}
            per_branch_best = []
            for tid, entry, path in branches:
                best_at = {}                                    # induced U-label -> best row
                for label, (cost, cells, pins) in tables[tid].items():
                    got = _induce(A, cells, path)               # walk entry-cell up to U along path
                    if got is None:
                        continue
                    u, walked = got
                    row = (cost, {**cells, **walked}, pins)
                    if u not in best_at or cost < best_at[u][0] - 1e-12 or \
                       (abs(cost - best_at[u][0]) <= 1e-12 and str(label) < str(best_at[u][3])):
                        best_at[u] = (cost, row[1], row[2], label)
                per_branch_best.append(best_at)
            shared = set.intersection(*[set(b) for b in per_branch_best]) if per_branch_best else set()
            for u in shared:
                cost, cells, pins = 0.0, {}, {}
                ok = True
                for b in per_branch_best:
                    c_, cl_, pn_, _lab = b[u]
                    cost += c_
                    for k, v in cl_.items():                    # cells may overlap only consistently
                        if cells.get(k, v) != v:
                            ok = False
                            break
                        cells[k] = v
                    if not ok:
                        break
                    pins.update(pn_)
                if not ok:
                    continue
                pins[U] = u
                new_rows[u] = (cost, cells, pins)
            if not new_rows:
                return None                                     # no shared exit -> infeasible
            tables[U] = new_rows
            for tid, _e, path in branches:
                consumed.add(tid)
                alias[tid] = U
                for w in path:
                    owner[w] = U
            owner[U] = U
        roots = [tid for tid in tables if tid not in consumed]
        assert len(roots) == 1, f"expected one root table, got {roots}"
        rows = tables[roots[0]]
        u_best = min(rows, key=lambda u: (rows[u][0], str(u)))
        cost, cells, pins = rows[u_best]
        total_cost += cost
        sink_labels.update({a: v for a, v in pins.items() if A.out_degree(a) == 0})
    M = reconstruct(A, sink_labels)
    return total_cost, sink_labels, M


def _induce(A, cells, path):
    """Walk bp from the row's recorded cell at the deepest path vertex up to path[0] (the split).
    Returns (induced label, {vertex: cell walked}) or None (forbidden/severed/no-arm)."""
    idx = max((i for i, w in enumerate(path) if w in cells), default=None)
    if idx is None:
        return None
    cur_a, cur_v = path[idx], cells[path[idx]]
    walked = {}
    for step in range(idx - 1, -1, -1):
        nxt = path[step]                                        # the parent on THIS branch
        cand = A.nodes[cur_a]["cand"]
        x = cur_v
        while True:                                             # own cover chain to the head
            bp = cand[x]["bpD"]
            if len(bp) == 1 and bp[0][0] == cur_a:
                x = bp[0][1]
                if x is None or cand[x].get("forbidden"):
                    return None
            else:
                break
        hop = [xp for (p, xp) in cand[x]["bpD"] if p == nxt]    # the arm toward the split
        if not hop or hop[0] is None:
            return None
        cur_a, cur_v = nxt, hop[0]
        if A.nodes[cur_a]["cand"][cur_v].get("forbidden"):
            return None
        walked[cur_a] = cur_v
    return cur_v, walked


# ----------------------------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------------------------
def canonical_cases():
    yield "chain", digraph({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)]), \
        digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")])
    yield "y_split", digraph({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)},
                             [(0, 1), (1, 2), (1, 3)]), \
        digraph({"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
                [("s", "j"), ("j", "u"), ("j", "d")])
    # split under split (subdivided)
    yield "split2", digraph({0: (0, 0), 1: (8, 0), 2: (16, 6), 3: (16, -4), 4: (24, -1),
                             5: (24, -8), 6: (32, 0), 7: (32, -14)},
                            [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (5, 7)]), \
        digraph({"s": (0, .5), "j1": (8, .5), "u": (16, 6.5), "j2": (16, -3.5),
                 "a": (24, -.5), "b": (24, -7.5), "e": (32, .5), "f": (32, -13.5)},
                [("s", "j1"), ("j1", "u"), ("j1", "j2"), ("j2", "a"), ("j2", "b"),
                 ("a", "e"), ("b", "f")])
    # the canonical MERGE shape:  U -> x -> m <- z <- V ; sinks below m and on U's/V's other branches
    yield "merge_shape", digraph(
        {"sU": (0, 10), "U": (6, 10), "x": (12, 8), "sV": (0, -10), "V": (6, -10), "z": (12, -8),
         "m": (18, 0), "d1": (24, 0), "T": (30, 0),
         "y1": (12, 16), "T2": (18, 16), "w1": (12, -16), "T3": (18, -16)},
        [("sU", "U"), ("U", "x"), ("x", "m"), ("sV", "V"), ("V", "z"), ("z", "m"),
         ("m", "d1"), ("d1", "T"), ("U", "y1"), ("y1", "T2"), ("V", "w1"), ("w1", "T3")]), \
        digraph({"BsU": (0, 10.5), "BU": (6, 10.5), "Bx": (12, 8.5), "BsV": (0, -9.5),
                 "BV": (6, -9.5), "Bz": (12, -7.5), "Bm": (18, .5), "Bd": (24, .5),
                 "BT": (30, .5), "By": (12, 16.5), "BT2": (18, 16.5), "Bw": (12, -15.5),
                 "BT3": (18, -15.5)},
                [("BsU", "BU"), ("BU", "Bx"), ("Bx", "Bm"), ("BsV", "BV"), ("BV", "Bz"),
                 ("Bz", "Bm"), ("Bm", "Bd"), ("Bd", "BT"), ("BU", "By"), ("By", "BT2"),
                 ("BV", "Bw"), ("Bw", "BT3")])


def random_polytree(seed):
    """Random undirected tree -> random edge orientation -> SUBDIVIDE every edge (interior point) --
    guaranteed subdivided polytree with natural splits AND merges."""
    rng = random.Random(seed)
    n = rng.randint(4, 7)
    und = [(rng.randrange(i), i) for i in range(1, n)]
    pos = {i: (rng.uniform(0, 30), rng.uniform(0, 30)) for i in range(n)}
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, x=pos[i][0], y=pos[i][1])
    for k, (a, b) in enumerate(und):
        if rng.random() < 0.5:
            a, b = b, a
        mid = f"m{k}"
        G.add_node(mid, x=(pos[a][0] + pos[b][0]) / 2, y=(pos[a][1] + pos[b][1]) / 2)
        G.add_edge(a, mid)
        G.add_edge(mid, b)
    B = nx.DiGraph()
    nb = rng.randint(6, 10)
    vs = [f"v{i}" for i in range(nb)]
    for v in vs:
        B.add_node(v, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(nb - 1):
        B.add_edge(vs[i], vs[i + 1])
    for _ in range(rng.randint(1, nb // 2)):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return G, B


# ----------------------------------------------------------------------------------------------
def run():
    GRID = [(1.0, 1.0), (0.5, 1.0), (0.3, 1.5), (1.0, 2.0)]     # in-domain: alpha <= 1 <= beta
    agree = differ = infeasible = skipped = 0
    print(f"{'case':14} {'a,b':10} {'junction-join':>14} {'brute force':>12}  verdict")
    for name, A0, B0 in canonical_cases():
        for ab in GRID:
            A, B = A0.copy(), B0.copy()
            prepare(A, B, r=40.0)
            forward(A, B, *ab)
            jj = junction_join(A, B, *ab)
            bf = brute_force(A, B, *ab)
            _report(name, ab, A, B, jj, bf, *ab)
            agree, differ, infeasible = _tally(jj, bf, agree, differ, infeasible, *((A, B) + ab))
    for seed in range(40):
        A0, B0 = random_polytree(seed)
        for ab in GRID[:2]:
            A, B = A0.copy(), B0.copy()
            try:
                prepare(A, B, r=40.0)
                forward(A, B, *ab)
            except ValueError:
                skipped += 1
                continue
            jj = junction_join(A, B, *ab)
            bf = brute_force(A, B, *ab)
            agree, differ, infeasible = _tally(jj, bf, agree, differ, infeasible, *((A, B) + ab))
            if _mismatch(jj, bf, A, B, *ab):
                _report(f"rnd{seed}", ab, A, B, jj, bf, *ab)
    print(f"\nagree={agree}  differ={differ}  both-infeasible={infeasible}  skipped(build)={skipped}")


def _mismatch(jj, bf, A, B, alpha, beta):
    if jj is None or bf is None:
        return (jj is None) != (bf is None)
    return abs(_cost_of(A, B, jj[2], alpha, beta) - bf[0]) > 1e-6


def _tally(jj, bf, agree, differ, infeasible, A, B, alpha, beta):
    if jj is None and bf is None:
        return agree, differ, infeasible + 1
    if _mismatch(jj, bf, A, B, alpha, beta):
        return agree, differ + 1, infeasible
    return agree + 1, differ, infeasible


def _report(name, ab, A, B, jj, bf, alpha, beta):
    js = "infeasible" if jj is None else f"{_cost_of(A, B, jj[2], alpha, beta):>10.3f}"
    bs = "infeasible" if bf is None else f"{bf[0]:>10.3f}"
    ok = "AGREE" if not _mismatch(jj, bf, A, B, alpha, beta) else "*** DIFFER ***"
    extra = ""
    if jj is not None:
        v1, v2, v3 = check_rules(jj[2], A, B)
        extra = f"  (root through={jj[0]:.3f}, M valid={not (v1 or v2 or v3)})"
    print(f"{name:14} {str(ab):10} {js:>14} {bs:>12}  {ok}{extra}")


if __name__ == "__main__":
    run()
