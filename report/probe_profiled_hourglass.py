"""Time and memory of the profiled forward table and its extraction, on the real hourglass edges.

Measures the two phases SEPARATELY, each against its current-engine counterpart:

  phase 1   forward()          vs  forward_profiled()
  phase 2   extract_cell()     vs  extract_profiled()

Memory is tracemalloc peak around that phase alone. Cost is _cost_of on the reconstructed matching,
so the two engines are compared on the same footing.

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/probe_profiled_hourglass.py
"""
from __future__ import annotations

import os
import sys
import resource
import time
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.match.window import build_window
from mapconflation.config import load_hyperparams

from network_matching.dag_dtw import (edges_to_digraph, line_digraph, prepare, forward,
                                      extract_cell, _cost_of, check_rules, check_forward_v3)
from network_matching.profiled import forward_profiled, extract_profiled

NET = "/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl"
REF = "/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"


_PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_mb() -> float:
    """Current resident set size -- the physical memory the process actually holds, what `top` shows."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE / 1e6


def peak_rss_mb() -> float:
    """High-water RSS since process start (monotonic), so a delta across a phase is that phase's
    contribution to the peak."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # KB on Linux


def timed(fn):
    """Run fn, return (result, seconds, stats, error).

    ``stats`` carries BOTH measures, because they answer different questions and disagree wildly:
      tm    -- tracemalloc peak: Python objects allocated INSIDE this window only. Misses anything
               allocated earlier (e.g. the Dp table an extraction reads) and all C/numpy buffers.
      rss   -- resident growth and high-water, from the OS. This is the number that OOMs a machine.
    """
    r0, p0 = rss_mb(), peak_rss_mb()
    tracemalloc.start()
    t = time.perf_counter()
    try:
        out, err = fn(), None
    except Exception as e:                                       # noqa: BLE001
        out, err = None, f"{type(e).__name__}: {str(e)[:60]}"
    secs = time.perf_counter() - t
    tm = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    stats = {"tm": tm, "rss_grow": rss_mb() - r0, "rss_now": rss_mb(),
             "rss_peak_delta": peak_rss_mb() - p0}
    return out, secs, stats, err


def build_graphs(lid, geoms, adj, hp, ref):
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
    return LA, LB


def run_edge(lid, geoms, adj, hp, ref):
    LA, LB = build_graphs(lid, geoms, adj, hp, ref)

    # ---- phase 1a: today's forward (baseline) -- includes the §4.1a coupling
    r_used = None
    for r_try in hp.rladder:
        prepare(LA, LB, r=r_try, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
        _o, t_fwd, s_fwd, err = timed(lambda: forward(LA, LB, alpha=hp.alpha, beta=hp.beta))
        if err is None:
            r_used = r_try
            break
    if r_used is None:
        print(f"  line {lid}: no feasible r in {hp.rladder}")
        return
    cells = sum(len(LA.nodes[n]["cand"]) for n in LA.nodes)
    v3_before = len(check_forward_v3(LA, LB))

    # ---- phase 2a: today's extraction (baseline)
    base, t_ext, s_ext, e_ext = timed(lambda: extract_cell(LA, LB, hp.alpha, hp.beta))
    base_cost = _cost_of(LA, LB, base[0], hp.alpha, hp.beta) if base else None

    # ---- phase 1b: the profiled forward table
    _o, t_pfwd, s_pfwd, e_pfwd = timed(lambda: forward_profiled(LA, LB, hp.alpha, hp.beta))
    rows = prof_max = 0
    if e_pfwd is None:
        for n in LA.nodes:
            for c in LA.nodes[n]["cand"].values():
                k = len(c.get("Dp", {}))
                rows += k
                prof_max = max(prof_max, k)

    # ---- phase 2b: the profiled extraction
    out, t_pext, s_pext, e_pext = timed(lambda: extract_profiled(LA, LB, hp.alpha, hp.beta))
    prof_cost = v3_after = covers = None
    if out is not None:
        M, _c, _join = out
        v1, v2, v3 = check_rules(M, LA, LB)
        covers = {a for a, _ in M} == set(LA.nodes)
        v3_after = len(v3)
        prof_cost = _cost_of(LA, LB, M, hp.alpha, hp.beta) if not (v1 or v2 or v3) and covers else None

    same = (prof_cost is not None and base_cost is not None and abs(prof_cost - base_cost) < 1e-6)
    print(f"\n  line {lid}   |LA|={LA.number_of_nodes()}  cells={cells}  r={r_used}  "
          f"V3 violations: before={v3_before}  after={v3_after}")
    print(f"    {'phase':<26} {'time':>9} {'tracemalloc':>12} {'RSS grow':>10} {'RSS peak+':>10}   result")
    print(f"    {'-'*26} {'-'*9} {'-'*12} {'-'*10} {'-'*10}   {'-'*26}")

    def line(tag, secs, st, result):
        print(f"    {tag:<26} {secs*1000:>7.0f}ms {st['tm']:>10.2f}MB {st['rss_grow']:>8.1f}MB "
              f"{st['rss_peak_delta']:>8.1f}MB   {result}")

    line("1a forward (current)", t_fwd, s_fwd, f"{cells} cells")
    line("1b forward_profiled", t_pfwd, s_pfwd,
         e_pfwd or f"{rows} rows, max {prof_max}/cell")
    line("2a extract_cell (current)", t_ext, s_ext,
         f"cost {base_cost:.3f}" if base_cost is not None else (e_ext or "?"))
    line("2b extract_profiled", t_pext, s_pext,
         (f"cost {prof_cost:.3f}" if prof_cost is not None else (e_pext or "invalid/partial"))
         + ("  [MATCH]" if same else "  [DIVERGE]"))
    print(f"    live footprint:  current engine ~{s_fwd['tm'] + s_ext['tm']:.1f}MB tm / "
          f"{max(s_fwd['rss_peak_delta'], s_ext['rss_peak_delta']):.1f}MB rss-peak   |   "
          f"profiled ~{s_pfwd['tm'] + s_pext['tm']:.1f}MB tm / "
          f"{max(s_pfwd['rss_peak_delta'], s_pext['rss_peak_delta']):.1f}MB rss-peak")


if __name__ == "__main__":
    hp = load_hyperparams("vancouver_city").hp
    G = load_network(NET)
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    ref = Reference(load_reference(REF))
    print(f"hp: step={hp.step} rladder={hp.rladder} k_min={hp.k_min} "
          f"alpha={hp.alpha} beta={hp.beta} bw={hp.bearing_weight}")
    for lid in (102752, 100042, 100341, 100350):
        try:
            run_edge(lid, geoms, adj, hp, ref)
        except Exception as e:                                   # noqa: BLE001
            print(f"\n  line {lid}: FAILED {type(e).__name__}: {str(e)[:70]}")
