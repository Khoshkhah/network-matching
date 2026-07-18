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

    # --- width 2: add a second split downstream, so profiles carry one pair PER live split
    A2 = digraph({"a": (0, 0), "J1": (8, 0), "d": (16, -8), "c": (16, 4), "J2": (24, 4),
                  "e": (32, 9), "f": (32, 0)},
                 [("a", "J1"), ("J1", "c"), ("J1", "d"), ("c", "J2"), ("J2", "e"), ("J2", "f")])
    B2 = digraph({"u": (0, 1), "v": (8, 1), "w": (16, 1), "x": (24, 1), "y": (32, 1)},
                 [("u", "v"), ("v", "w"), ("w", "x"), ("x", "y")])
    prepare(A2, B2, r=14.0)
    forward(A2, B2, 1.0, 1.0)
    forward_profiled(A2, B2, 1.0, 1.0)
    print("=" * 62)
    print("A:  a -> J1 -> {c, d} ;  c -> J2 -> {e, f}     two splits, none discharged")
    print("B:  u -> v -> w -> x -> y\n")
    for tgt in ("c", "e"):
        v = sorted(A2.nodes[tgt]["cand"])[0]
        dp = A2.nodes[tgt]["cand"][v]["Dp"]
        live = sorted({s for pi in dp for s, _ in pi})
        print(f"cell ({tgt}, {v})   live splits: {live or 'none'}   width = {len(live)}")
        for pi, (cost, _bp) in sorted(dp.items(), key=lambda kv: kv[1][0])[:4]:
            txt = ", ".join(f"{s} ends on {c}" for s, c in sorted(pi, key=str)) or "(none)"
            print(f"     cost {cost:7.3f}   when   {txt}")
        print()
