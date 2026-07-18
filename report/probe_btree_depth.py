"""How far does the profiled engine scale on a pure out-tree? (docs §7.1)

btree has no merges, so nothing post-dominates, no profile key is ever discharged, and width equals
the tree depth -- a factor over `depth` keys costs the product of |cand| over them. This measures the
ceiling, and counts splits with a single surviving exit (whose key is a constant and could be dropped
from S).

Run with a memory cap so a blow-up dies cleanly:
  (ulimit -v 6291456; python3 -u report/probe_btree_depth.py)
"""
import os, sys, time, tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_dtw import prepare, forward, extract_cell, _cost_of, INF
from network_matching.profiled import forward_profiled, extract_profiled, profiled_splits
from scripts.extract_cell_dag import fam_btree

print(f"{'depth':>5} {'|A|':>5} {'splits':>6} {'1-exit':>7} {'t_cell':>9} {'t_prof':>9} {'mem_prof':>9}  parity")
for d in (3,4,5,6):
    A,B = fam_btree(d); prepare(A,B,r=15.0); forward(A,B,0.5,1.0)
    S = profiled_splits(A)
    # splits whose surviving candidate set has exactly ONE usable exit
    one = 0
    for s in S:
        alive = [v for v,c in A.nodes[s]["cand"].items()
                 if not c.get("forbidden") and c["D"] < INF]
        if len(alive) == 1: one += 1
    t=time.perf_counter(); 
    try:
        Mc = extract_cell(A,B,0.5,1.0)[0]; cc=_cost_of(A,B,Mc,0.5,1.0)
    except Exception as e: cc=None
    tc=time.perf_counter()-t
    tracemalloc.start(); t=time.perf_counter()
    try:
        forward_profiled(A,B,0.5,1.0); Mp = extract_profiled(A,B,0.5,1.0)[0]
        cp=_cost_of(A,B,Mp,0.5,1.0); tp=time.perf_counter()-t
        mp=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
        par = "YES" if cc is not None and abs(cc-cp)<1e-6 else "**NO**"
        print(f"{d:>5} {A.number_of_nodes():>5} {len(S):>6} {one:>7} {tc:>8.3f}s {tp:>8.3f}s {mp:>8.1f}MB  {par}")
    except BaseException as e:
        tp=time.perf_counter()-t; tracemalloc.stop()
        print(f"{d:>5} {A.number_of_nodes():>5} {len(S):>6} {one:>7} {tc:>8.3f}s {tp:>8.3f}s  ---      {type(e).__name__}")
