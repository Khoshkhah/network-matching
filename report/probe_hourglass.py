"""Run the §6 profile probe on the REAL map-conflation hourglass edges (docs §8.5's slow four).

Builds LA/LB exactly as mapconflation.match.direction.match_task does, then measures:
  - profile multiplicity / width / entries, with post-dominator discharge  (this design)
  - the pending product  PROD over merges of |cand(m)|                     (what extract_cell pays today)

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python probe_hourglass.py
"""
from __future__ import annotations

import math
import signal
import os
import sys
import time
import tracemalloc
from collections import Counter

PER_EDGE_S = 180                                             # per-edge wall guard
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/tmp/claude-1000/-home-kaveh-projects-network-matching/"
                   "4e0cf963-160c-45e9-886b-a406fdbb0d8a/scratchpad")

from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.match.window import build_window
from mapconflation.config import load_hyperparams

from network_matching.dag_dtw import (edges_to_digraph, line_digraph, prepare, forward)
from probe_profiles import profile_sets

NET = "/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl"
REF = "/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"
LIDS = (102752, 100042, 100341, 100350)


def build_LA_LB(lid, geoms, adj, ref, hp):
    """The direction.py:191-206 block, verbatim in effect: A/B -> LA/LB with attrs copied."""
    arcs, waist, _ = ld.build_hourglass(lid, geoms, adj, snap_m=hp.snap_m, max_hops=hp.hops)
    b_edges, _ow, _onm, _win = build_window(arcs, ref, buf=hp.buf)[:4]
    A = ld.tree_to_digraph(arcs, hp.step)
    B = edges_to_digraph(b_edges, hp.step, 1)
    LA, LB = line_digraph(A), line_digraph(B)
    LB.remove_edges_from([(x, y) for x, y in LB.edges() if y == (x[1], x[0])])   # drop bounce
    for (u, v) in LA.nodes:
        LA.nodes[(u, v)]["road_id"], LA.nodes[(u, v)]["seq"] = A[u][v]["road_id"], A[u][v]["seq"]
    for (u, v) in LB.nodes:
        LB.nodes[(u, v)]["road_id"], LB.nodes[(u, v)]["seq"] = B[u][v]["road_id"], B[u][v]["seq"]
    return LA, LB


def pending_product(LA):
    """PROD over merge vertices (indeg >= 2) of |finite cand| -- the §8.5 'full product' column."""
    prod, parts = 1, []
    for m in LA.nodes:
        if LA.in_degree(m) >= 2:
            n = sum(1 for c in LA.nodes[m]["cand"].values() if math.isfinite(c["D"]))
            if n:
                prod *= n
                parts.append(n)
    return prod, sorted(parts, reverse=True)


def main():
    hp = load_hyperparams("vancouver_city").hp
    print(f"hp: step={hp.step} rladder={hp.rladder} k_min={hp.k_min} buf={hp.buf} "
          f"alpha={hp.alpha} beta={hp.beta} bw={hp.bearing_weight}")
    G = load_network(NET)
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    t = time.perf_counter()
    ref = Reference(load_reference(REF))
    print(f"reference built in {time.perf_counter()-t:.1f}s\n")

    for lid in LIDS:
        try:
            LA, LB = build_LA_LB(lid, geoms, adj, ref, hp)
        except Exception as e:                               # noqa: BLE001
            print(f"{lid}: build failed: {type(e).__name__}: {e}")
            continue
        for r_try in hp.rladder:                             # escalate r until forward is feasible
            try:
                prepare(LA, LB, r=r_try, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
                forward(LA, LB, alpha=hp.alpha, beta=hp.beta)
                break
            except Exception:                                # noqa: BLE001
                continue
        else:
            print(f"{lid}: no feasible r in {hp.rladder}")
            continue

        prod, parts = pending_product(LA)
        splits = [n for n in LA.nodes if LA.out_degree(n) >= 2]
        merges = [n for n in LA.nodes if LA.in_degree(n) >= 2]

        print(f"--- line {lid} --- |LA|={LA.number_of_nodes()} "
              f"splits={sum(1 for n in LA.nodes if LA.out_degree(n) >= 2)} "
              f"merges={len(merges)} pending_prod={prod:,} ... measuring", flush=True)
        tracemalloc.start()
        t = time.perf_counter()
        signal.alarm(PER_EDGE_S)                             # a timeout here IS a result
        try:
            P, S, capped = profile_sets(LA, LB, discharge=True)
        except TimeoutError:
            signal.alarm(0)
            tracemalloc.stop()
            print(f"  TIMEOUT after {PER_EDGE_S}s -> profile blows up on this edge\n")
            continue
        signal.alarm(0)
        secs = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()

        live = {k: v for k, v in P.items() if v}
        mult = Counter(len(v) for v in live.values())
        width = max((max((len(pr) for pr in v), default=0) for v in live.values()), default=0)
        entries = sum(len(pr) for v in live.values() for pr in v)
        tot = sum(mult.values()) or 1
        # the sink join (§6): profiles still live at each sink, and the distinct global keys
        sinks = [n for n in LA.nodes if LA.out_degree(n) == 0]
        per_sink, global_keys = [], set()
        for t in sinks:
            prof = set()
            for v in LA.nodes[t]["cand"]:
                prof |= P.get((t, v), set())
            per_sink.append(len(prof))
            global_keys |= prof
        live_at_sinks = sorted({s for pr in global_keys for s, _ in pr}, key=str)
        print(f"  SINK JOIN: sinks={len(sinks)}  profiles/sink={sorted(per_sink, reverse=True)[:8]}"
              f"  distinct_keys={len(global_keys)}  splits_live_at_sinks={len(live_at_sinks)}")
        print(f"  cells={tot}  r={r_try}  splits={len(splits)}")
        print(f"  pending product (PROD over merges |cand|) = {prod:,}   parts={parts[:6]}")
        print(f"  profile: max_mult={max(mult, default=0):,}  mult=1:{100*mult.get(1,0)/tot:.1f}%  "
              f"width_max={width}  entries={entries:,}  peak={peak:.2f}MB  {secs:.2f}s"
              f"{'  CAPPED' if capped else ''}")
        print(f"  ratio pending/profile_max = {prod / max(max(mult, default=1), 1):,.1f}x\n")


if __name__ == "__main__":
    main()
