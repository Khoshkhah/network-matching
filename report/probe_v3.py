"""Baseline for the profiled forward table: how often is today's forward table V3-INVALID?

`check_forward_v3` reads the forward table on its own (seed each sink at its arg-min D, follow bpD)
and returns the V3-violating (a, v) pairs. Empty == the table is already split-consistent.
That is exactly the acceptance test the profiled design must turn green everywhere.

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python probe_v3.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import (prepare, forward, line_digraph, check_forward_v3,
                                      check_split_exits, extract_cell, _cost_of)
from scripts.extract_cell_dag import fam_dense_chain, fam_btree, fam_diamond_chain


def report(name, LA, LB, alpha=1.0, beta=1.0):
    bad = check_forward_v3(LA, LB)
    exits = check_split_exits(LA, LB)
    splits = sum(1 for n in LA.nodes if LA.out_degree(n) >= 2)
    try:
        M, _ = extract_cell(LA, LB, alpha, beta)
        cost = f"{_cost_of(LA, LB, M, alpha, beta):.3f}"
    except Exception as e:                                  # noqa: BLE001
        cost = f"{type(e).__name__}"
    flag = "V3-INVALID" if bad else "ok"
    print(f"{name:34s} splits={splits:3d}  v3_violations={len(bad):4d}  {flag:11s} "
          f"split_exits_bad={len(exits)}  extract_cost={cost}")
    return len(bad)


SYNTH = [
    ("dense_chain(50)", lambda: fam_dense_chain(50)),
    ("diamond_chain(4)", lambda: fam_diamond_chain(4)),
    ("diamond_chain(10)", lambda: fam_diamond_chain(10)),
    ("btree(3)", lambda: fam_btree(3)),
    ("btree(4)", lambda: fam_btree(4)),
]

if __name__ == "__main__":
    print("=== synthetic families (alpha=beta=1) ===")
    for label, build in SYNTH:
        for mode in ("point", "segment"):
            A, B = build()
            if mode == "segment":
                A, B = line_digraph(A), line_digraph(B)
            try:
                prepare(A, B, r=20.0)
                forward(A, B)
                report(f"{label} [{mode}]", A, B)
            except Exception as e:                          # noqa: BLE001
                print(f"{label} [{mode}]: {type(e).__name__}: {e}")

    print("\n=== real hourglass edges ===")
    try:
        from mapconflation.clean import load_network
        from mapconflation.match import load_reference, Reference, local_dag as ld
        from mapconflation.match.window import build_window
        from mapconflation.config import load_hyperparams
        from network_matching.dag_dtw import edges_to_digraph
    except Exception as e:                                  # noqa: BLE001
        print(f"map-conflation unavailable ({e}); synthetic only")
        sys.exit(0)

    hp = load_hyperparams("vancouver_city").hp
    G = load_network("/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl")
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    ref = Reference(load_reference("/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"))

    for lid in (102752, 100042, 100341, 100350):
        arcs, waist, _ = ld.build_hourglass(lid, geoms, adj, snap_m=hp.snap_m, max_hops=hp.hops)
        b_edges = build_window(arcs, ref, buf=hp.buf)[0]
        A = ld.tree_to_digraph(arcs, hp.step)
        B = edges_to_digraph(b_edges, hp.step, 1)
        LA, LB = line_digraph(A), line_digraph(B)
        LB.remove_edges_from([(x, y) for x, y in LB.edges() if y == (x[1], x[0])])
        for (u, v) in LA.nodes:
            LA.nodes[(u, v)]["road_id"], LA.nodes[(u, v)]["seq"] = A[u][v]["road_id"], A[u][v]["seq"]
        for (u, v) in LB.nodes:
            LB.nodes[(u, v)]["road_id"], LB.nodes[(u, v)]["seq"] = B[u][v]["road_id"], B[u][v]["seq"]
        for r_try in hp.rladder:
            try:
                prepare(LA, LB, r=r_try, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
                forward(LA, LB, alpha=hp.alpha, beta=hp.beta)
                break
            except Exception:                               # noqa: BLE001
                continue
        report(f"line {lid}", LA, LB, hp.alpha, hp.beta)
