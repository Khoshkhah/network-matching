"""Dissect the actual V3 violations in the forward table on 102752 / 100350.

For each violating split: which cell each child linked back to, whether those cells survived
§4.1a's forbid (i.e. were 'feasible' for every child), and what the coupling therefore did or
did not prevent.

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python probe_v3_detail.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.match.window import build_window
from mapconflation.config import load_hyperparams

from network_matching.dag_dtw import (edges_to_digraph, line_digraph, prepare, forward,
                                      check_forward_v3, _one_sided, _feasible_links, INF)


def build(lid, geoms, adj, ref, hp):
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
            return LA, LB
        except Exception:                                        # noqa: BLE001
            continue
    raise RuntimeError("no feasible r")


def dissect(lid, LA, LB):
    bad = check_forward_v3(LA, LB)
    print(f"\n===== line {lid}: {len(bad)} V3 violations =====")
    if not bad:
        return
    M = _one_sided(LA, [n for n in LA.nodes if LA.out_degree(n) == 0], "D", "bpD")
    placed = {}
    for (a, v) in M:
        placed.setdefault(a, set()).add(v)

    for (a, v) in bad:
        print(f"\n  violation at cell ({a}, {v})")
        print(f"    vertex {a} is placed on {len(placed.get(a, ()))} cell(s): {sorted(placed.get(a, ()), key=str)}")
        print(f"    outdeg={LA.out_degree(a)}  indeg={LA.in_degree(a)}   <- split iff outdeg>=2")
        cand = LA.nodes[a]["cand"]
        alive = [c for c in cand if not cand[c]["forbidden"] and cand[c]["D"] < INF]
        print(f"    surviving (non-forbidden, finite) exits of {a}: {len(alive)}")
        for ch in LA.successors(a):
            feas = _feasible_links(LA, LB, ch, a)
            linked = {x for cell in LA.nodes[ch]["cand"].values() if cell["D"] < INF
                      for (q, x) in cell["bpD"] if q == a and x is not None}
            chosen = {x for cc in placed.get(ch, ()) for (q, x) in LA.nodes[ch]["cand"][cc]["bpD"]
                      if q == a and x is not None}
            print(f"      child {ch}: feasible_exits={len(feas)}  bpD-linked={len(linked)}  "
                  f"chosen_in_M={sorted(chosen, key=str)}")
        inter = None
        for ch in LA.successors(a):
            f = _feasible_links(LA, LB, ch, a)
            inter = f if inter is None else inter & f
        print(f"    §4.1a allowed = intersection of children's feasible exits: "
              f"{len(inter or ())} cells -> coupling kept them ALL (feasibility, not choice)")


if __name__ == "__main__":
    hp = load_hyperparams("vancouver_city").hp
    G = load_network("/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl")
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    ref = Reference(load_reference("/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"))
    for lid in (102752, 100350):
        LA, LB = build(lid, geoms, adj, ref, hp)
        dissect(lid, LA, LB)
