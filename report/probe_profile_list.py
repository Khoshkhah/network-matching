"""What is the list of profiles at one cell? (docs/profiled_forward_table.md §1.1c)

Dumps every row of Dp at the cell holding the most profiles: the cost, and the split placement it
assumes. Same cell throughout -- the rows differ only in what they assume happened upstream.

Run:  python3 -u report/probe_profile_list.py [btree_depth]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import prepare, forward
from network_matching.profiled import forward_profiled, profiled_splits
from scripts.extract_cell_dag import fam_btree

if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    A, B = fam_btree(depth)
    prepare(A, B, r=15.0)
    forward(A, B, 0.5, 1.0)
    forward_profiled(A, B, 0.5, 1.0)
    print("splits S =", sorted(map(str, profiled_splits(A))), "\n")

    n, v, k = max(((n, v, len(A.nodes[n]["cand"][v]["Dp"]))
                   for n in A.nodes for v in A.nodes[n]["cand"]), key=lambda t: t[2])
    print(f"CELL ({n}, {v})  holds {k} profiles "
          f"— same cell, different upstream split placements:\n")
    rows = sorted(A.nodes[n]["cand"][v]["Dp"].items(), key=lambda kv: kv[1][0])
    for pi, (cost, _bp) in rows:
        placed = ", ".join(f"{s}->{c}" for s, c in sorted(pi, key=str)) or "(none live)"
        print(f"   cost {cost:8.4f}   when   {placed}")
    print(f"\n   min over the list = {rows[0][1][0]:.4f}   <- exactly what D holds (§2.1)")
    print(f"   D[{n}][{v}]        = {A.nodes[n]['cand'][v]['D']:.4f}")
