"""What is the list of profiles at one cell? (docs/profiled_forward_table.md §1.1c)

The smallest case that shows it -- one split, one chain, names chosen to be readable:

    A:  a -> J -> c        J is the only split (outdeg 2)
                -> d
    B:  u -> v -> w -> x

Dumps every row of Dp at each cell of `c`: the cost, and the placement of J it assumes. Same cell
throughout; the rows differ only in what happened upstream.

Run:  python3 -u report/probe_profile_list.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import digraph, prepare, forward
from network_matching.profiled import forward_profiled

if __name__ == "__main__":
    A = digraph({"a": (0, 0), "J": (10, 0), "c": (20, 6), "d": (20, -6)},
                [("a", "J"), ("J", "c"), ("J", "d")])
    B = digraph({"u": (0, 1), "v": (7, 1), "w": (14, 1), "x": (21, 1)},
                [("u", "v"), ("v", "w"), ("w", "x")])
    prepare(A, B, r=12.0)
    forward(A, B, 1.0, 1.0)
    forward_profiled(A, B, 1.0, 1.0)

    print("A:  a -> J -> {c, d}        J is the only split")
    print("B:  u -> v -> w -> x\n")
    print("candidate cells of J :", sorted(A.nodes["J"]["cand"]))
    print("candidate cells of c :", sorted(A.nodes["c"]["cand"]), "\n")
    for v in sorted(A.nodes["c"]["cand"]):
        dp = A.nodes["c"]["cand"][v]["Dp"]
        print(f"cell (c, {v})  holds {len(dp)} profile(s):")
        for pi, (cost, _bp) in sorted(dp.items(), key=lambda kv: kv[1][0]):
            placed = ", ".join(f"J ends on {c}" for _s, c in sorted(pi, key=str)) or "(no live split)"
            print(f"     cost {cost:7.3f}   when   {placed}")
        print(f"     min = {min(x[0] for x in dp.values()):.3f}"
              f"   D = {A.nodes['c']['cand'][v]['D']:.3f}\n")
