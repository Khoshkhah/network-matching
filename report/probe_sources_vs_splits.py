"""Source cells vs split cells as the profile key (docs/profiled_forward_table.md §1.1b).

The design grew from the idea of labelling cells by which SOURCE cells they came from; it uses
SPLIT cells instead. This measures how different those are on the real hourglass edges.

Result: overlap is 0 -- no source is ever a split -- so every split lies strictly downstream of its
source ancestors and BOTH CHILDREN of a split inherit the identical source-cell assignment. A source
profile therefore cannot distinguish which cell of the split each child left from, which is exactly
what V3 binds. Splits are also no wider, and narrower on 100350 (2 vs 4).

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  /home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/probe_sources_vs_splits.py
"""
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networkx as nx
from probe_profiled_hourglass import build_graphs
from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.config import load_hyperparams
from network_matching.dag_dtw import prepare, forward
from network_matching.profiled import profiled_splits, postdom_drop

hp = load_hyperparams("vancouver_city").hp
G = load_network("/home/kaveh/projects/map-conflation/cache/vancouver_city_clean_network.pkl")
geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
ref = Reference(load_reference("/home/kaveh/projects/duckOSM/data/db/vancouver_city.duckdb"))

print(f"{'edge':>8} {'|LA|':>5} {'sources':>8} {'splits':>7} {'overlap':>8} "
      f"{'split width':>12} {'source width':>13}")
for lid in (102752, 100042, 100341, 100350):
    LA, LB = build_graphs(lid, geoms, adj, hp, ref)
    for r in hp.rladder:
        prepare(LA, LB, r=r, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
        try: forward(LA, LB, alpha=hp.alpha, beta=hp.beta); break
        except Exception: continue
    srcs   = {n for n in LA.nodes if LA.in_degree(n) == 0}
    splits = profiled_splits(LA)
    dropS  = postdom_drop(LA, splits)
    dropR  = postdom_drop(LA, srcs)
    # width = max over vertices of |keys live at that vertex|
    def width(keys, drop):
        w = 0
        for a in LA.nodes:
            live = (keys & (nx.ancestors(LA, a) | {a})) - drop[a]
            w = max(w, len(live))
        return w
    print(f"{lid:>8} {LA.number_of_nodes():>5} {len(srcs):>8} {len(splits):>7} "
          f"{len(srcs & splits):>8} {width(splits, dropS):>12} {width(srcs, dropR):>13}")
