"""Thorough POINT-mode test of the tree-DTW matcher (network_matching/tree_dtw.py).

Sweeps structure (chain / y-split / merge / deep tree) x density (A vs B sampling) x lateral shift x
noise x (alpha, beta), runs the full pipeline (prepare -> forward -> extract), and validates
the final matching M against V1-V4 with check_rules. Reports the pass envelope.

Run:  python scripts/test_nx_point.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from network_matching.tree_dtw import prepare, forward, extract, extract_join, extract_cell, check_rules, _cost_of


def densify(waypoints, step):
    """Points along a polyline at ~`step` spacing (endpoints included)."""
    pts = np.asarray(waypoints, float)
    seg = np.diff(pts, axis=0)
    slen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(slen)])
    n = max(2, int(round(cum[-1] / step)) + 1)
    out = []
    for t in np.linspace(0, cum[-1], n):
        i = min(int(np.searchsorted(cum, t, side="right") - 1), len(seg) - 1)
        i = max(i, 0)
        f = (t - cum[i]) / (slen[i] if slen[i] > 0 else 1.0)
        out.append(tuple(pts[i] + seg[i] * f))
    return out


def build(polylines, step, shift=0.0, noise=0.0, seed=0):
    """A DiGraph from named polylines (dict name -> waypoints) that share endpoints. Junctions become a
    single shared node (keyed on the *original* coordinate). `shift` moves everything north; `noise`
    jitters each node; both applied after keying so junctions stay shared."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    node_at: dict = {}

    def node(pt):
        key = (round(pt[0], 3), round(pt[1], 3))
        if key not in node_at:
            nid = len(node_at)
            x = pt[0] + (rng.normal(0, noise) if noise else 0.0)
            y = pt[1] + shift + (rng.normal(0, noise) if noise else 0.0)
            G.add_node(nid, x=float(x), y=float(y))
            node_at[key] = nid
        return node_at[key]

    for wp in polylines.values():
        pts = densify(wp, step)
        prev = node(pts[0])
        for p in pts[1:]:
            cur = node(p)
            if cur != prev:
                G.add_edge(prev, cur)
            prev = cur
    return G


STRUCTURES = {
    "chain":  {"c": [(0, 0), (30, 0)]},
    "ysplit": {"stem": [(0, 0), (10, 0)], "up": [(10, 0), (25, 10)], "dn": [(10, 0), (25, -10)]},
    "merge":  {"up": [(0, 10), (10, 0)], "dn": [(0, -10), (10, 0)], "out": [(10, 0), (25, 0)]},
    "deep":   {"stem": [(0, 0), (10, 0)], "up": [(10, 0), (20, 8)], "dn": [(10, 0), (20, -8)],
               "uu": [(20, 8), (30, 12)], "ud": [(20, 8), (30, 4)]},
}


def run_case(struct, a_step, b_step, shift, noise, alpha, beta, seed):
    poly = STRUCTURES[struct]
    A = build(poly, a_step, shift=shift, noise=noise, seed=seed)
    B = build(poly, b_step, shift=0.0, noise=0.0, seed=0)
    prepare(A, B, r=20.0)
    forward(A, B, alpha, beta)

    def run_engine(fn):
        try:
            M, _ = fn(A, B, alpha, beta)
        except ValueError as e:
            return None, f"infeasible: {e}"
        v1, v2, v3 = check_rules(M, A, B)
        v4 = [a for a in A.nodes if not any(x == a for (x, _w) in M)]
        if v1 or v2 or v3 or v4:
            return None, f"V1={len(v1)} V2={len(v2)} V3={len(v3)} V4={len(v4)}"
        return M, ""

    Mb, why_b = run_engine(extract)
    Mj, why_j = run_engine(extract_join)
    Mc, why_c = run_engine(extract_cell)
    cost = lambda M: _cost_of(A, B, M, alpha, beta)
    cross = True                                                # THE invariant: cell <= both, always
    if Mc is not None:
        if Mb is not None:
            cross &= cost(Mc) <= cost(Mb) + 1e-6
        if Mj is not None:
            cross &= cost(Mc) <= cost(Mj) + 1e-6
    ok_all = Mb is not None and Mj is not None and Mc is not None
    why = "" if ok_all and cross else f"branch[{why_b}] join[{why_j}] cell[{why_c}] cell<=both={cross}"
    return (Mb is not None, Mj is not None, Mc is not None, cross), why


if __name__ == "__main__":
    nb = nj = nc = nx_ = n_tot = 0
    fails = []
    for struct in STRUCTURES:
        for a_step, b_step in [(2.0, 2.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.5)]:   # equal / B-fine / A-fine
            for shift in (0.0, 0.5, 2.0, 5.0):
                for noise in (0.0, 0.3):
                    for alpha, beta in ((1.0, 1.0), (0.7, 1.0), (0.5, 1.5)):
                        (ok_b, ok_j, ok_c, cross), why = run_case(struct, a_step, b_step, shift,
                                                                  noise, alpha, beta, seed=7)
                        n_tot += 1
                        nb += ok_b
                        nj += ok_j
                        nc += ok_c
                        nx_ += cross
                        if why:
                            fails.append((struct, a_step, b_step, shift, noise, alpha, beta, why))
    for f in fails[:12]:
        print("  FAIL", f)
    print(f"POINT-mode sweep: branching {nb}/{n_tot} | vertex-join {nj}/{n_tot} | "
          f"CELL-join {nc}/{n_tot} valid | cell<=both {nx_}/{n_tot}")
