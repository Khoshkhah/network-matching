"""Is D the minimum over Dp?  (docs/profiled_forward_table.md §2.1)

D minimises over ALL upstream configurations, Dp over the CONSISTENT ones only, so in theory
D <= min_pi Dp. This measures the gap per cell on the four hourglass edges. Measured result: equal
on every cell, including the two edges that are V3-invalid -- because a single cell's D cannot be a
phantom (that needs two branches disagreeing about one split, which needs a merge whose arms share a
split ancestor; the hourglass in-side is tree-shaped). The phantom lives in the COMBINATION of cells,
not in any one value.

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/probe_D_vs_Dp.py
"""
import os, sys, math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_profiled_hourglass import build_graphs
from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.config import load_hyperparams
from network_matching.dag_dtw import prepare, forward, check_forward_v3, INF
from network_matching.profiled import forward_profiled

hp = load_hyperparams("vancouver_city").hp
G = load_network("/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl")
geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
ref = Reference(load_reference("/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"))

print(f"{'edge':>8} {'cells':>6} {'V3':>3} {'D==minDp':>9} {'D<minDp':>8} {'max gap':>9} {'mean gap':>9}")
for lid in (102752, 100042, 100341, 100350):
    LA, LB = build_graphs(lid, geoms, adj, hp, ref)
    for r in hp.rladder:
        prepare(LA, LB, r=r, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
        try: forward(LA, LB, alpha=hp.alpha, beta=hp.beta); break
        except Exception: continue
    v3 = len(check_forward_v3(LA, LB))
    forward_profiled(LA, LB, hp.alpha, hp.beta)
    eq = lt = gt = 0; gaps = []
    for n in LA.nodes:
        for v, c in LA.nodes[n]["cand"].items():
            dp = c.get("Dp", {})
            if not dp or not math.isfinite(c["D"]): continue
            m = min(x[0] for x in dp.values())
            d = c["D"]
            if abs(d-m) < 1e-9: eq += 1
            elif d < m: lt += 1; gaps.append(m-d)
            else: gt += 1
    print(f"{lid:>8} {eq+lt+gt:>6} {v3:>3} {eq:>9} {lt:>8} "
          f"{max(gaps) if gaps else 0:>9.4f} {sum(gaps)/len(gaps) if gaps else 0:>9.4f}"
          + (f"   !! {gt} cells where D > minDp (would break the bound)" if gt else ""))
