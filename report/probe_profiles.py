"""Probe for docs/consistent_forward_table.md §6 — measures, WITHOUT implementing the recurrence:

  1. |S|                 the profiled set = splits (outdeg >= 2)              -- S empty => design is a no-op
  2. profile multiplicity  distinct profiles reachable per cell               -- ==1 everywhere => R1 exact (§4.1)
  3. profile width         |S n ancestors(a)| per cell                        -- per-cell memory
  4. memory                total stored profile entries + tracemalloc peak

Keeps ALL profiles per cell (that is the measurement); R1 would keep one.
Run:  python probe_profiles.py
"""
from __future__ import annotations

import os
import sys
import math
import tracemalloc
from collections import Counter
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx

from network_matching.dag_dtw import prepare, forward, line_digraph, layer_order, INF
from scripts.extract_cell_dag import fam_dense_chain, fam_btree, fam_diamond_chain

CAP = 20000          # per-cell profile cap; hitting it is itself a result


def postdom_drop(A, S):
    """`drop[a]` = the splits that can be discharged at `a`: those `a` post-dominates.

    Once every path out of `s` passes through `a`, no later cell can contradict `s` -- exactly one
    branch carries it from here on. Forward mirror of cell_dag_extraction.md §3.5's early discharge
    (first common ancestor going backward == post-dominator going forward).
    """
    R = A.reverse(copy=True)
    R.add_node("__T__")
    for t in [n for n in A.nodes if A.out_degree(n) == 0]:
        R.add_edge("__T__", t)
    idom = nx.immediate_dominators(R, "__T__")          # dominators on reverse == post-dominators
    drop = {a: set() for a in A.nodes}
    for s in S:
        x = idom.get(s)
        while x is not None and x != "__T__" and x != s:
            drop[x].add(s)                              # x post-dominates s
            nxt = idom.get(x)
            if nxt == x:
                break
            x = nxt
    return drop


def profile_sets(A, B, cap=CAP, discharge=True):
    """Per cell (a,v), the SET of distinct profiles reachable under the legal-move relation.

    A profile is a frozenset of (s, cell) for s in S, i.e. the assignment a configuration commits
    the splits to. Mirrors the §2.2 recurrence but keeps every profile instead of the argmin's.
    Returns (P, S, capped) with P[(a,v)] = set of frozensets.
    """
    S = {n for n in A.nodes if A.out_degree(n) >= 2}
    order, _ = layer_order(A)
    drop = postdom_drop(A, S) if discharge else {a: set() for a in A.nodes}
    P, capped = {}, False

    def fin(a):                                     # cells that survived the forward pass
        return {v: c for v, c in A.nodes[a]["cand"].items() if math.isfinite(c["D"])}

    def merge(profs):
        """Union a tuple of profiles; None if they disagree on any shared split."""
        out = {}
        for pr in profs:
            for s, v in pr:
                if out.setdefault(s, v) != v:
                    return None
        return out

    for a in order:
        cells = fin(a)
        preds = list(A.predecessors(a))
        base = {}                                   # (D)/(V) contribution, before coverage
        for v in cells:
            if not preds:                           # source: free entry
                base[v] = {frozenset()}
                continue
            # entry cells per parent: stall at v, or advance from a B-predecessor of v
            opts = []
            for p in preds:
                pc = fin(p)
                xs = [x for x in ([v] + list(B.predecessors(v))) if x in pc]
                opts.append([(p, x) for x in xs])
            if any(not o for o in opts):
                base[v] = set()
                continue
            acc = set()
            for tup in product(*opts):
                psets = [P.get((p, x), set()) for (p, x) in tup]
                if any(not ps for ps in psets):
                    continue
                for combo in product(*psets):
                    m = merge(combo)                # <- the consistency test (§2.2)
                    if m is None:
                        continue
                    acc.add(frozenset(m.items()))
                    if len(acc) >= cap:
                        break
                if len(acc) >= cap:
                    capped = True
                    break
            base[v] = acc

        d = drop[a]

        def emit(prs):
            """§2.2 overwrite of a's own split cell, then discharge everything a post-dominates."""
            out = set()
            for pr in prs:
                m = dict(pr)
                if a in S:
                    m[a] = v
                out.add(frozenset((s, x) for s, x in m.items() if s not in d))
            return out

        for v in cells:
            P[(a, v)] = emit(base[v])

        # (H) coverage: within-row, iterate to fixed point
        changed = True
        while changed:
            changed = False
            for v in cells:
                add = set()
                for v2 in B.predecessors(v):
                    if v2 in cells:
                        add |= P[(a, v2)]
                add = emit(add)
                new = P[(a, v)] | add
                if len(new) > len(P[(a, v)]):
                    P[(a, v)] = set(list(new)[:cap])
                    changed = True
                    if len(new) >= cap:
                        capped = True
    return P, S, capped


def run(name, A, B, r=20.0):
    prepare(A, B, r=r)
    forward(A, B)
    tracemalloc.start()
    P, S, capped = profile_sets(A, B)
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    live = {k: v for k, v in P.items() if v}
    mult = Counter(len(v) for v in live.values())
    width = Counter(max((len(pr) for pr in v), default=0) for v in live.values())
    entries = sum(len(pr) for v in live.values() for pr in v)
    one = mult.get(1, 0)
    tot = sum(mult.values()) or 1
    print(f"{name:28s} |A|={A.number_of_nodes():5d} |S|={len(S):4d} cells={tot:6d} "
          f"mult=1:{100*one/tot:5.1f}% max={max(mult, default=0):5d} "
          f"width_max={max(width, default=0):3d} entries={entries:8d} peak={peak:6.2f}MB"
          f"{'  CAPPED' if capped else ''}")
    return mult


CASES = [
    ("dense_chain(50)", lambda: fam_dense_chain(50)),
    ("diamond_chain(4)", lambda: fam_diamond_chain(4)),
    ("diamond_chain(10)", lambda: fam_diamond_chain(10)),
    ("btree(3)", lambda: fam_btree(3)),
    ("btree(4)", lambda: fam_btree(4)),
]

if __name__ == "__main__":
    for label, build in CASES:
        for mode in ("point", "segment"):
            A, B = build()
            if mode == "segment":
                A, B = line_digraph(A), line_digraph(B)
            try:
                run(f"{label} [{mode}]", A, B)
            except Exception as e:                  # noqa: BLE001 - probe, report and continue
                print(f"{label} [{mode}]: {type(e).__name__}: {e}")
