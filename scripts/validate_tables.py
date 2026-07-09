"""Validate EVERY cell of the forward table `D` and backward table `B` against the warping rules,
by **following the real stored back-pointers** (`bp_D`/`bp_B` from `_forward_table`).

For a cell to be correct, the partial matching it represents must be a legal warping:
  * forward  `D[a][v]` (upstream cone of `a`, pinned at `v`)   -> V1 + V2 + V3
  * backward `B[a][v]` (downstream cone of `a`, pinned at `v`) -> V1 + V2 + V3

Each cell's partial matching is reconstructed by WALKING its back-pointer list to the sources/sinks --
NOT by re-deriving the move (that would re-impose the rule and be circular). Rules are checked
**restricted to neighbours present in the cell's cone**, so V3 on the forward table (and V2 on the
backward) holds vacuously there yet a real violation would still be caught. Works for point mode
(vertex tables) and segment mode (arc line-graph tables, docs §8).

Run:  python scripts/validate_tables.py
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from network_matching.tree_dtw import (build_local_digraph, _as_linestrings, _emission,
                                       _forward_table, _topological_order, _segment_tables)


def _reconstruct(bp, a: int, v: int) -> Set[Tuple[int, int]]:
    """The partial matching cell (a, v) stands for -- follow the stored back-pointer lists."""
    out: Set[Tuple[int, int]] = set()
    stack = [(a, v)]
    while stack:
        x, w = stack.pop()
        if (x, w) in out:
            continue
        out.add((x, w))
        for (x2, w2) in bp[x][w]:
            stack.append((x2, w2))
    return out


def _check(M, pred, succ, bpred, bsucc):
    """Check V1, V2 AND V3 on a partial matching, restricted to neighbours in the matching (the cell's
    cone). ``pred``/``succ`` index source nodes; ``bpred``/``bsucc`` index target cells. Returns
    ``(v1, v2, v3)``."""
    has = M.__contains__
    inM = {a for (a, _v) in M}
    v1, v2, v3 = [], [], []
    for (a, v) in M:
        if any((am in inM) and has((am, vp)) for am in pred[a] for vp in bsucc[v]):
            v1.append((a, v))
        if not any(has((a, vm)) for vm in bpred[v]):
            for am in pred[a]:
                if am in inM and not (has((am, v)) or any(has((am, vm)) for vm in bpred[v])):
                    v2.append((a, v)); break
        if not any(has((a, vp)) for vp in bsucc[v]):
            for ap in succ[a]:
                if ap in inM and not (has((ap, v)) or any(has((ap, vp)) for vp in bsucc[v])):
                    v3.append((a, v)); break
    return v1, v2, v3


def _scan(D, bp_D, B, bp_B, pred, succ, bpred, bsucc, nodes, cells, label):
    """Validate every finite cell of D (via bp_D, source=pred cone) and B (via bp_B, source=succ cone).
    For B, the roles of pred/succ and bpred/bsucc swap (downstream cone)."""
    rep = {"n_fwd": 0, "n_bwd": 0, "fwd_bad": [], "bwd_bad": []}
    for a in nodes:
        for v in cells:
            if np.isfinite(D[a][v]):
                rep["n_fwd"] += 1
                v1, v2, v3 = _check(_reconstruct(bp_D, a, v), pred, succ, bpred, bsucc)
                if v1 or v2 or v3:
                    rep["fwd_bad"].append(f"D[{a}][{v}]  V1={v1} V2={v2} V3={v3}")
            if np.isfinite(B[a][v]):
                rep["n_bwd"] += 1
                v1, v2, v3 = _check(_reconstruct(bp_B, a, v), succ, pred, bsucc, bpred)
                if v1 or v2 or v3:
                    rep["bwd_bad"].append(f"B[{a}][{v}]  V1={v1} V2={v2} V3={v3}")
    return rep


def _build(a_edges, b_edges, snap, step):
    a_edges, b_edges = _as_linestrings(a_edges), _as_linestrings(b_edges)
    a_pts = [(float(x), float(y)) for _i, g in a_edges for (x, y) in g.coords]
    b_pts = [(float(x), float(y)) for _i, g in b_edges for (x, y) in g.coords]
    ga = build_local_digraph(a_edges, b_pts, snap, step)
    gb = build_local_digraph(b_edges, a_pts, snap, step)
    return ga, gb


def validate_point_tables(a_edges, b_edges, alpha=1.0, beta=1.0, snap=0.5, step=2.0):
    ga, gb = _build(a_edges, b_edges, snap, step)
    order = _topological_order(ga)
    NA, NB = ga.n_vertices, gb.n_vertices
    emit = _emission(ga, gb)
    outdeg_f = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)
    indeg_b = np.array([max(1, len(ga.pred_arcs[a])) for a in range(NA)], float)
    D, bp_D = _forward_table(ga.pred_arcs, ga.succ_arcs, outdeg_f, gb.succ_arcs, gb.pred_arcs,
                             order, emit, alpha, beta=beta)
    B, bp_B = _forward_table(ga.succ_arcs, ga.pred_arcs, indeg_b, gb.pred_arcs, gb.succ_arcs,
                             order[::-1], emit, alpha, beta=beta)
    return _scan(D, bp_D, B, bp_B, ga.pred_arcs, ga.succ_arcs, gb.pred_arcs, gb.succ_arcs,
                 range(NA), range(NB), "point")


def validate_segment_tables(a_edges, b_edges, bearing_weight=3.0, alpha=1.0, beta=1.0, snap=0.5, step=2.0):
    ga, gb = _build(a_edges, b_edges, snap, step)
    T = _segment_tables(ga, gb, bearing_weight, alpha, beta)
    return _scan(T["D"], T["bp_D"], T["B"], T["bp_B"], T["pred_list"], T["succ_list"],
                 T["barc_pred"], T["barc_succ"], T["real_a"], range(T["NBA"]), "segment")


def run(rep, title):
    nbad = len(rep["fwd_bad"]) + len(rep["bwd_bad"])
    print(f"[{title}]  D cells={rep['n_fwd']} B cells={rep['n_bwd']}  "
          f"-> {'ALL VALID' if nbad == 0 else f'{nbad} BAD'}")
    for m in rep["fwd_bad"][:4]:
        print("   FORWARD  bad:", m)
    for m in rep["bwd_bad"][:4]:
        print("   BACKWARD bad:", m)
    return nbad == 0


if __name__ == "__main__":
    from network_matching.dag_synthetic import get_dag
    ok = True
    print("=== POINT-mode tables ===")
    for name in ["chain", "y_split", "merge"]:
        sc = get_dag(name)
        for al, be in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5)]:
            ok &= run(validate_point_tables(sc["a_edges"], sc["b_edges"], alpha=al, beta=be),
                      f"{name} a={al} b={be}")
    print("\n=== SEGMENT-mode (arc) tables ===")
    for name in ["chain", "y_split", "merge"]:
        sc = get_dag(name)
        for bw in [0.0, 3.0]:
            ok &= run(validate_segment_tables(sc["a_edges"], sc["b_edges"], bearing_weight=bw),
                      f"{name} bw={bw}")
    print("\nSUMMARY:", "all tables valid" if ok else "SOME TABLES INVALID")
