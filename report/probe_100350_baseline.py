"""Definitive baseline attempt: run extract_cell on line 100350 to completion, no time limit.

100350 is the one hourglass edge with no baseline -- extract_cell has never finished on it (>15 min,
and 23 GB before a memory cap was imposed). The profiled engine answers it in 0.44 s / 16 MB with
cost 308.924 and check_rules clean, but "agrees with the existing engine" is unverified there.

This settles it one of three ways, all conclusive:
  - a cost  -> compare against 308.924
  - MemoryError under the cap -> extract_cell is memory-unbounded on this edge
  - still running -> the elapsed line in the log says how long

Launch detached, with a memory cap so it can never take the machine:
  (ulimit -v 8388608; PYTHONPATH=/home/kaveh/projects/map-conflation/src setsid nohup \
   /home/kaveh/projects/osm-dra-conflation/.venv/bin/python -u report/probe_100350_baseline.py \
   >> logs/extract_cell_100350.log 2>&1 &)
"""
from __future__ import annotations

import os
import resource
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_profiled_hourglass import build_graphs, rss_mb, peak_rss_mb

from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.config import load_hyperparams

from network_matching.dag_dtw import prepare, forward, extract_cell, _cost_of, check_rules

PROFILED_COST = 308.924          # what extract_profiled returned, cost == join, check_rules clean


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


if __name__ == "__main__":
    log(f"START pid={os.getpid()}  cap={resource.getrlimit(resource.RLIMIT_AS)[0] / 2**30:.1f}GB")
    hp = load_hyperparams("vancouver_city").hp
    G = load_network("/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl")
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    ref = Reference(load_reference("/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"))
    LA, LB = build_graphs(100350, geoms, adj, hp, ref)
    for r in hp.rladder:
        prepare(LA, LB, r=r, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
        try:
            forward(LA, LB, alpha=hp.alpha, beta=hp.beta)
            break
        except Exception:                                        # noqa: BLE001
            continue
    cells = sum(len(LA.nodes[n]["cand"]) for n in LA.nodes)
    log(f"graphs ready: |LA|={LA.number_of_nodes()} cells={cells} r={r} rss={rss_mb():.0f}MB")
    log("extract_cell: running, no time limit ...")

    t = time.perf_counter()
    try:
        M, _committed = extract_cell(LA, LB, hp.alpha, hp.beta)
        secs = time.perf_counter() - t
        cost = _cost_of(LA, LB, M, hp.alpha, hp.beta)
        v1, v2, v3 = check_rules(M, LA, LB)
        log(f"DONE in {secs:.1f}s ({secs/60:.1f} min)  peakRSS={peak_rss_mb():.0f}MB")
        log(f"  extract_cell cost = {cost:.6f}   v1={len(v1)} v2={len(v2)} v3={len(v3)}")
        log(f"  extract_profiled  = {PROFILED_COST:.6f}")
        log(f"  VERDICT: {'MATCH' if abs(cost - PROFILED_COST) < 1e-3 else 'DIVERGE'}"
            f"  (delta {cost - PROFILED_COST:+.6f})")
    except MemoryError:
        log(f"MemoryError after {time.perf_counter() - t:.1f}s at peakRSS={peak_rss_mb():.0f}MB "
            f"-- extract_cell is memory-unbounded on this edge; no baseline exists")
    except Exception as e:                                       # noqa: BLE001
        log(f"FAILED after {time.perf_counter() - t:.1f}s: {type(e).__name__}: {e}")
