"""EXPERIMENT: restrict extract_cell's inbox routing to the parent cells bpD actually links
(`_links`) instead of every feasible transition (`_feasible_links`).

Verdict rule: any cost divergence refutes it. Equal costs everywhere = the forward table's
back-pointers are enough to route the extraction.

Run:  python3 -u probe_bplinks.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import (prepare, forward, line_digraph, extract_cell, _cost_of,
                                      check_rules)
from network_matching.dag_dtw import digraph
from scripts.extract_cell_dag import (fam_dense_chain, fam_btree, fam_diamond_chain)

import networkx as nx


def parity_cases():
    """The 16 correctness cases of scripts/extract_cell_dag.py:parity_suite, rebuilt as data."""
    import random
    cases = []
    for seed in range(12):                                       # random polytrees over cyclic B
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
    return cases


def run_one(A, B, alpha, beta, bp):
    try:
        M, _ = extract_cell(A, B, alpha, beta, bp_links=bp)
        v1, v2, v3 = check_rules(M, A, B)
        if v1 or v2 or v3:
            return None, "INVALID"
        return _cost_of(A, B, M, alpha, beta), ""
    except Exception as e:                                       # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:44]}"


def compare(name, A, B, alpha=1.0, beta=1.0, r=20.0):
    prepare(A, B, r=r)
    forward(A, B, alpha, beta)
    t = time.perf_counter(); base, be = run_one(A, B, alpha, beta, False); tb = time.perf_counter() - t
    t = time.perf_counter(); exp, ee = run_one(A, B, alpha, beta, True); te = time.perf_counter() - t
    if be or ee:
        verdict = f"base={be or f'{base:.4f}'}  bp={ee or f'{exp:.4f}'}"
        status = "DIVERGE" if (be == "") != (ee == "") else "both-error"
    elif abs(base - exp) < 1e-9:
        verdict = f"{base:.4f}"
        status = "same"
    else:
        verdict = f"base={base:.4f}  bp={exp:.4f}  ({100*(exp-base)/max(base,1e-9):+.1f}%)"
        status = "DIVERGE"
    print(f"{name:36s} {status:11s} {verdict}   [{tb*1000:.0f}ms -> {te*1000:.0f}ms]")
    return status


if __name__ == "__main__":
    tally = {}
    print("=== parity suite (the 16 correctness cases) ===")
    for (label, A, Bg, r, al, be) in parity_cases():
        try:
            s = compare(label, A, Bg, al, be, r)
        except Exception as e:                                   # noqa: BLE001
            s = "setup-error"
            print(f"{label:36s} setup-error {type(e).__name__}: {str(e)[:40]}")
        tally[s] = tally.get(s, 0) + 1

    print("\n=== scaling families ===")
    fams = [("dense_chain(50)", lambda: fam_dense_chain(50)),
            ("diamond_chain(4)", lambda: fam_diamond_chain(4)),
            ("diamond_chain(10)", lambda: fam_diamond_chain(10)),
            ("btree(3)", lambda: fam_btree(3)),
            ("btree(4)", lambda: fam_btree(4))]
    for label, build in fams:
        for mode in ("point", "segment"):
            A, B = build()
            if mode == "segment":
                A, B = line_digraph(A), line_digraph(B)
            try:
                s = compare(f"{label} [{mode}]", A, B)
            except Exception as e:                               # noqa: BLE001
                s = "setup-error"
                print(f"{label} [{mode}]: {type(e).__name__}: {str(e)[:40]}")
            tally[s] = tally.get(s, 0) + 1

    print(f"\nTALLY: {tally}")
    print("VERDICT:", "REFUTED — bpD routing loses matchings"
          if tally.get("DIVERGE") else "no divergence on these cases")
