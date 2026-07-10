"""Thorough POINT-mode test of the tree-DTW matcher (network_matching/tree_dtw.py).

Sweeps structure (chain / y-split / merge / deep tree) x density (A vs B sampling) x lateral shift x
noise x (alpha, beta), runs the full pipeline (prepare -> forward -> backward -> extract), and validates
the final matching M against V1-V4 with check_rules. Reports the pass envelope.

Run:  python scripts/test_nx_point.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from network_matching.tree_dtw import prepare, forward, backward, extract, check_rules


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
    backward(A, B, alpha, beta)
    try:
        M, _ = extract(A, B)
    except ValueError as e:
        return False, f"infeasible: {e}"
    v1, v2, v3 = check_rules(M, A, B)
    v4 = [a for a in A.nodes if not any(x == a for (x, _w) in M)]
    ok = not (v1 or v2 or v3 or v4)
    return ok, "" if ok else f"V1={len(v1)} V2={len(v2)} V3={len(v3)} V4={len(v4)}"


if __name__ == "__main__":
    n_ok = n_tot = 0
    fails = []
    for struct in STRUCTURES:
        for a_step, b_step in [(2.0, 2.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.5)]:   # equal / B-fine / A-fine
            for shift in (0.0, 0.5, 2.0, 5.0):
                for noise in (0.0, 0.3):
                    for alpha, beta in [(1.0, 1.0), (0.7, 0.7), (0.5, 0.5)]:
                        ok, msg = run_case(struct, a_step, b_step, shift, noise, alpha, beta, seed=7)
                        n_tot += 1
                        n_ok += ok
                        if not ok:
                            fails.append((struct, a_step, b_step, shift, noise, alpha, beta, msg))
    print(f"POINT-mode sweep: {n_ok}/{n_tot} valid (V1-V4)")
    for f in fails[:40]:
        print("  FAIL", f)
    if not fails:
        print("  ALL VALID")
