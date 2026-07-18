"""The life of a profile key (docs/profiled_forward_table.md §1.1d).

Walks A in layer order and prints, per vertex, its role and the profiles held at one of its cells --
so a key can be watched being BORN at a split, CARRIED down, MERGED where arms must agree, and
DISCHARGED at the split's post-dominator.

Run:  python3 -u report/probe_profile_life.py [k_diamonds]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import prepare, forward, layer_order
from network_matching.profiled import forward_profiled, profiled_splits, postdom_drop
from scripts.extract_cell_dag import fam_diamond_chain

if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    A, B = fam_diamond_chain(k)
    prepare(A, B, r=10.0)
    forward(A, B, 0.5, 1.0)
    forward_profiled(A, B, 0.5, 1.0)
    S = profiled_splits(A)
    drop = postdom_drop(A, S)
    order, _ = layer_order(A)
    print(f"splits S = {sorted(map(str, S))}\n")
    print(f"{'vertex':>6} {'in/out':>7} {'role':>38} {'profiles at one cell':>44}")
    for a in order:
        role = []
        if a in S:
            role.append("SPLIT - key born")
        if A.in_degree(a) > 1:
            role.append("MERGE - arms agree")
        if drop[a]:
            role.append("DISCHARGE " + ",".join(sorted(map(str, drop[a]))))
        if not role:
            role.append("carries parent's profile")
        v = list(A.nodes[a]["cand"])[0]
        shown = sorted(("{" + ",".join(f"{s}@{c}" for s, c in sorted(p, key=str)) + "}") if p else "{}"
                       for p in A.nodes[a]["cand"][v]["Dp"])
        txt = ", ".join(shown[:3]) + (f"  (+{len(shown)-3})" if len(shown) > 3 else "")
        print(f"{str(a):>6} {A.in_degree(a)}/{A.out_degree(a):<5} {'; '.join(role):>38} {txt:>44}")
