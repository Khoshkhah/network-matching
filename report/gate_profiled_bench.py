"""Benchmark-parity arm of the §7 gate: extract_profiled vs extract_cell vs extract_cell_vertex.

Mirrors scripts/extract_cell_dag.py:bench() -- the same families, sizes, r/alpha/beta -- so these
numbers sit alongside the §7 table in cell_dag_extraction.md. extract_cell_vertex is the
pre-2026-07 engine preserved verbatim in that script.

Costs must agree to 1e-6 wherever two engines both answer; a refusal by one is reported, never
silently treated as agreement. btree is expected to defeat the profiled engine at depth >= 4 (no
merges => nothing post-dominates => no key discharges, docs §5.1) -- it should raise on
max_profiles, not OOM, and that IS the documented result.

Run:  python3 -u report/gate_profiled_bench.py [max_seconds_per_engine]
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import prepare, forward, extract_cell, _cost_of, check_rules
from network_matching.profiled import forward_profiled, extract_profiled
from scripts.extract_cell_dag import (extract_cell_vertex, fam_dense_chain, fam_btree,
                                      fam_diamond_chain)

FAMS = ([("dense-chain", fam_dense_chain, n, 15.0, 0.5, 1.0) for n in (50, 150, 400)]
        + [("btree", fam_btree, d, 15.0, 0.5, 1.0) for d in (3, 4)]
        + [("diamonds", fam_diamond_chain, k, 10.0, 0.5, 1.0) for k in (4, 40, 120)])


def run(fn, A, B, al, be, profiled=False):
    """(seconds, peak_MB, cost, why)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        if profiled:
            forward_profiled(A, B, al, be)
            M = extract_profiled(A, B, al, be)[0]
        else:
            M = fn(A, B, al, be)[0]
        secs = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        v1, v2, v3 = check_rules(M, A, B)
        v4 = [a for a in A.nodes if not any(x == a for (x, _w) in M)]
        if v1 or v2 or v3 or v4:
            return secs, peak, None, f"INVALID V1={len(v1)} V3={len(v3)} V4={len(v4)}"
        return secs, peak, _cost_of(A, B, M, al, be), ""
    except Exception as e:                                       # noqa: BLE001
        secs = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        return secs, peak, None, f"{type(e).__name__}: {str(e)[:38]}"


if __name__ == "__main__":
    print(f"{'family':<12} {'size':>5} {'|A|':>5} {'cells':>6}   "
          f"{'t_vtx':>8} {'t_cell':>8} {'t_prof':>8}   "
          f"{'m_vtx':>8} {'m_cell':>8} {'m_prof':>8}   parity")
    verdict = []
    for fam, build, size, r, al, be in FAMS:
        A, B = build(size)
        prepare(A, B, r=r)
        try:
            forward(A, B, alpha=al, beta=be)
        except ValueError as e:
            print(f"{fam:<12} {size:>5}   forward refused: {str(e)[:40]}")
            continue
        ncell = sum(len(A.nodes[a]["cand"]) for a in A.nodes)
        tv, mv, cv, wv = run(extract_cell_vertex, A, B, al, be)
        tc, mc, cc, wc = run(extract_cell, A, B, al, be)
        tp, mp, cp, wp = run(None, A, B, al, be, profiled=True)

        got = [c for c in (cv, cc, cp) if c is not None]
        if not got:
            par = "ALL REFUSED"
        elif max(got) - min(got) < 1e-6:
            par = "YES" if len(got) == 3 else f"YES ({len(got)}/3 answered)"
        else:
            par = "**NO**"
        verdict.append(par.startswith("YES") or par == "ALL REFUSED")
        print(f"{fam:<12} {size:>5} {A.number_of_nodes():>5} {ncell:>6}   "
              f"{tv:8.3f} {tc:8.3f} {tp:8.3f}   "
              f"{mv:8.2f} {mc:8.2f} {mp:8.2f}   {par}")
        for tag, w in (("vtx", wv), ("cell", wc), ("prof", wp)):
            if w:
                print(f"{'':<12} {'':>5}   {tag}: {w}")
    print("\nVERDICT:", "GREEN" if all(verdict) else "RED")
