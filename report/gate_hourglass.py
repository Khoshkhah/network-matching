"""The real-data gate: every map-conflation hourglass edge through `match_dag(engine="auto")`.

The other gates run synthetic populations. This one pins the actual sources the library exists for,
and it exists because those sources are an EXTERNAL INPUT: map-conflation's `local_dag.build_hourglass`
constructs them, so a change there silently changes what this library is handed. Commit `d72c09b`
("flip near-reversal stubs into the junction, TURN_MAX = 160") did exactly that, and it was noticed
only because a hand-run scratch script started failing.

Baseline below is pinned to map-conflation `d173727` (2026-07-19). A mismatch means one of:
  - a regression in this library                      -> investigate here
  - the hourglass construction changed upstream       -> re-baseline, deliberately

Every cost is cross-validated against `extract_cell`, which is exact over the full space, so the
baseline is not merely "what auto produced".

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/gate_hourglass.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_profiled_hourglass import build_graphs, NET, REF
from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.config import load_hyperparams
from network_matching.dag_dtw import (prepare, forward, extract_by_engine, extract_cell,
                                      check_rules, _cost_of)
from network_matching.profiled import profiled_width, merge_pressure

# edge -> (cost, W, Mo, engine)   pinned to map-conflation d173727
BASELINE = {
    102752: (496.2937, 2, 3, "profiled"),
    100042: (420.8484, 3, 2, "cell"),
    100341: (454.4490, 3, 2, "cell"),
    100350: (304.2849, 1, 1, "profiled"),
    100935: (524.8200, 1, 1, "profiled"),
}


if __name__ == "__main__":
    hp = load_hyperparams("vancouver_city").hp
    G = load_network(NET)
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    ref = Reference(load_reference(REF))

    ok = drift = 0
    print(f"{'edge':>8} {'W':>2} {'Mo':>3} {'engine':>9} {'cost':>12} {'baseline':>12}  verdict")
    for lid, (want, wantW, wantMo, wantE) in BASELINE.items():
        LA, LB = build_graphs(lid, geoms, adj, hp, ref)
        for r in hp.rladder:
            prepare(LA, LB, r=r, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
            try:
                forward(LA, LB, alpha=hp.alpha, beta=hp.beta)
                break
            except Exception:
                continue
        W, Mo = profiled_width(LA), merge_pressure(LA)
        eng = "profiled" if W <= 2 else ("rebase" if Mo >= W else "cell")
        try:
            M = extract_by_engine(LA, LB, hp.alpha, hp.beta, "auto")[0]
            v1, v2, v3 = check_rules(M, LA, LB)
            cov = {a for a, _ in M} == set(LA.nodes)
            cost = _cost_of(LA, LB, M, hp.alpha, hp.beta) if cov and not (v1 or v2 or v3) else None
        except Exception as e:                                  # noqa: BLE001
            cost, v1, v2, v3 = None, [], [], []
            print(f"{lid:>8} {W:>2} {Mo:>3} {eng:>9} {'--':>12} {want:>12.4f}  "
                  f"{type(e).__name__}: {str(e)[:40]}")
            drift += 1
            continue
        shape_ok = (W, Mo, eng) == (wantW, wantMo, wantE)
        cost_ok = cost is not None and abs(cost - want) < 1e-3
        if cost_ok and shape_ok:
            ok += 1
            verdict = "OK"
        else:
            drift += 1
            verdict = ("SHAPE DRIFT — upstream hourglass changed?" if not shape_ok
                       else "COST REGRESSION")
        print(f"{lid:>8} {W:>2} {Mo:>3} {eng:>9} "
              f"{'--' if cost is None else f'{cost:12.4f}'} {want:>12.4f}  {verdict}")

    # cross-validate against extract_cell: the baseline must not drift to whatever auto happens to do
    mism = 0
    for lid in BASELINE:
        LA, LB = build_graphs(lid, geoms, adj, hp, ref)
        for r in hp.rladder:
            prepare(LA, LB, r=r, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
            try:
                forward(LA, LB, alpha=hp.alpha, beta=hp.beta)
                break
            except Exception:
                continue
        try:
            Mc = extract_cell(LA, LB, hp.alpha, hp.beta)[0]
            if abs(_cost_of(LA, LB, Mc, hp.alpha, hp.beta) - BASELINE[lid][0]) > 1e-3:
                mism += 1
                print(f"  CROSS-CHECK {lid}: extract_cell disagrees with the baseline")
        except Exception:
            print(f"  CROSS-CHECK {lid}: extract_cell refused (baseline rests on auto alone)")

    print(f"\n{ok}/{len(BASELINE)} at baseline   ({drift} drifted)")
    print(f"cross-check vs extract_cell: {len(BASELINE) - mism}/{len(BASELINE)} agree")
    print("\nVERDICT:", "GREEN" if ok == len(BASELINE) and mism == 0 else "RED")
