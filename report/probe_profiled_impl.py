"""Acceptance test for network_matching/profiled.py.

Two questions, in order of importance:
  1. Is the profiled table V3-VALID?  (the point of the design -- docs §5.0)
  2. Does its cost match extract_cell's? (exactness -- docs §7 gate)

Run:  python3 -u report/probe_profiled_impl.py
      PYTHONPATH=/home/kaveh/projects/map-conflation/src \
      /home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/probe_profiled_impl.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import (prepare, forward, line_digraph, extract_cell, _cost_of,
                                      check_rules, check_forward_v3, digraph)
from network_matching.profiled import forward_profiled, extract_profiled
from scripts.extract_cell_dag import fam_dense_chain, fam_btree, fam_diamond_chain

import networkx as nx


def parity_cases():
    import random
    cases = []
    for seed in range(12):
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
            A.add_edge(a, mid); A.add_edge(mid, b)
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


def check(name, A, B, r, al, be):
    prepare(A, B, r=r)
    try:
        forward(A, B, alpha=al, beta=be)
    except ValueError as e:
        print(f"{name:28s} forward refused: {str(e)[:40]}")
        return "refused"

    v3_before = len(check_forward_v3(A, B))
    try:
        base_M, _ = extract_cell(A, B, al, be)
        base = _cost_of(A, B, base_M, al, be)
    except ValueError as e:
        base, base_M = None, None
        base_err = str(e)[:34]

    t = time.perf_counter()
    try:
        forward_profiled(A, B, al, be)
        M, _committed, cost = extract_profiled(A, B, al, be)
    except ValueError as e:
        print(f"{name:28s} v3_before={v3_before}  profiled REFUSED: {str(e)[:40]}")
        return "refused" if base is None else "DIVERGE"
    secs = time.perf_counter() - t

    v1, v2, v3 = check_rules(M, A, B)
    valid = not (v1 or v2 or v3)
    covers = {a for a, _ in M} == set(A.nodes)
    recost = _cost_of(A, B, M, al, be) if valid and covers else None

    if base is None:
        print(f"{name:28s} v3_before={v3_before}  base=REFUSED({base_err})  profiled={recost}")
        return "DIVERGE"
    tag = "same" if (recost is not None and abs(recost - base) < 1e-6) else "DIVERGE"
    flags = ("" if valid else f" INVALID(v1={len(v1)},v2={len(v2)},v3={len(v3)})") + \
            ("" if covers else " PARTIAL")
    print(f"{name:28s} v3_before={v3_before:2d}  base={base:9.4f}  "
          f"profiled={'None' if recost is None else f'{recost:9.4f}'}  "
          f"join={cost:9.4f}  {tag}{flags}  [{secs*1000:.0f}ms]")
    return tag if valid and covers else "DIVERGE"


if __name__ == "__main__":
    tally = {}
    print("=== parity cases ===")
    for (label, A, Bg, r, al, be) in parity_cases():
        try:
            s = check(label, A, Bg, r, al, be)
        except Exception as e:                                   # noqa: BLE001
            s = "ERROR"
            print(f"{label:28s} ERROR {type(e).__name__}: {str(e)[:50]}")
        tally[s] = tally.get(s, 0) + 1

    print("\n=== families ===")
    for label, build in [("dense_chain(50)", lambda: fam_dense_chain(50)),
                         ("diamond_chain(4)", lambda: fam_diamond_chain(4)),
                         ("diamond_chain(10)", lambda: fam_diamond_chain(10)),
                         ("btree(3)", lambda: fam_btree(3)),
                         ("btree(4)", lambda: fam_btree(4))]:
        for mode in ("point", "segment"):
            A, Bg = build()
            if mode == "segment":
                A, Bg = line_digraph(A), line_digraph(Bg)
            try:
                s = check(f"{label} [{mode}]", A, Bg, 20.0, 1.0, 1.0)
            except Exception as e:                               # noqa: BLE001
                s = "ERROR"
                print(f"{label} [{mode}]: ERROR {type(e).__name__}: {str(e)[:50]}")
            tally[s] = tally.get(s, 0) + 1

    print(f"\nTALLY: {tally}")
