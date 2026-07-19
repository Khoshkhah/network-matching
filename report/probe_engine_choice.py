"""Which engine wins on which shape -- and the search for re-basing's niche.

RESULT: no shape tested makes `rebase` the best engine. It is second on chains and diamonds, and
FAILS on this ladder. The niche predicted for it -- deep nesting plus many merges -- does not exist,
because two assumptions behind that prediction are wrong:

  - "a wide merge hurts cell". It does not. pending's cost is the product over CONCURRENTLY-OPEN
    merges; one merge is one factor however large its in-degree. The hourglass wall was THREE merges
    at 45x45x14, not one wide one. cell does ladder(9) -- a 10-way merge -- in 5 ms.
  - "re-basing holds width 1 under nesting". Only when the arms of a merge share a last split. A
    fan-in merge gives each arm a DIFFERENT last split, so the frontier is as wide as the fan-in and
    re-basing blows up exactly like the profiled engine.

Measured (r=14, alpha=0.5, beta=1.0):

    ladder k   splits  merge indeg   w |  profiled     cell    rebase | best
    ladder(3)       3            4   3 |    0.026s   0.003s     FAIL  | cell
    ladder(5)       5            6   5 |    4.757s   0.003s     FAIL  | cell
    ladder(7)       7            8   7 |     FAIL    0.006s     FAIL  | cell
    ladder(9)       9           10   9 |     FAIL    0.005s     FAIL  | cell

A source with BOTH pressures: deeply NESTED splits AND a wide merge.

    spine:  s0 -> s1 -> ... -> sk        every si is a split (outdeg 2)
    branch: si -> bi                     each spawns a side branch
    merge:  every bi -> m                one merge of in-degree k
            m -> t

Nested splits: si is an ancestor of s(i+1), and none is post-dominated until `m`, so at s_i the live
set is {s0..si} -- width grows with k. That kills the profiled engine.
Wide merge: `m` has in-degree k, so pending's product over concurrently-open merges is large. That is
what kills the cell engine.
Re-basing keys on the LAST split only, so its width stays 1 while still handling the merge.
"""
import sys
sys.path.insert(0,"/home/kaveh/projects/network-matching")
from network_matching.dag_dtw import digraph

def fam_ladder(k, spacing=12.0):
    nodes, edges = {}, []
    for i in range(k + 1):                                   # spine, subdivided
        nodes[f"s{i}"] = (spacing * i, 0.0)
        nodes[f"s{i}_m"] = (spacing * i + spacing / 2, 0.0)
        if i < k:
            edges += [(f"s{i}", f"s{i}_m"), (f"s{i}_m", f"s{i+1}")]
    mx = spacing * (k + 1)
    nodes["m"] = (mx, 0.0); nodes["t"] = (mx + spacing, 0.0)
    for i in range(k):                                       # side branches into the shared merge
        nodes[f"b{i}"] = (spacing * i + spacing / 2, -8.0)
        nodes[f"b{i}_m"] = ((spacing * i + spacing / 2 + mx) / 2, -8.0)
        edges += [(f"s{i}", f"b{i}"), (f"b{i}", f"b{i}_m"), (f"b{i}_m", "m")]
    edges += [(f"s{k}", "m"), ("m", "t")]
    A = digraph(nodes, edges)
    Bn = {f"{n}'": (x, y + 0.4) for n, (x, y) in nodes.items()}
    Be = [(f"{u}'", f"{v}'") for u, v in edges]
    return A, digraph(Bn, Be)

if __name__ == "__main__":
    import copy, time, statistics
    from network_matching.dag_dtw import match_dag, _cost_of, check_rules
    from network_matching.profiled import profiled_width
    print(f"{'ladder k':<10} {'|A|':>4} {'splits':>6} {'merge indeg':>11} {'w':>3} | "
          f"{'profiled':>10} {'cell':>10} {'rebase':>10} | best")
    for k in (3, 5, 7, 9):
        A, B = fam_ladder(k)
        w = profiled_width(A)
        indeg = max(A.in_degree(n) for n in A.nodes)
        res, costs = {}, {}
        for eng in ("profiled", "cell", "rebase"):
            ts = []
            for _ in range(2):
                Ax, Bx = copy.deepcopy(A), copy.deepcopy(B)
                t = time.perf_counter()
                try:
                    M, _ = match_dag(Ax, Bx, r=14.0, alpha=0.5, beta=1.0, engine=eng)
                    ts.append(time.perf_counter() - t)
                    costs[eng] = round(_cost_of(Ax, Bx, M, 0.5, 1.0), 4)
                except BaseException as e:
                    ts.append(float("inf")); costs[eng] = None; break
            res[eng] = statistics.median(ts)
        best = min(res, key=res.get)
        f = lambda v: "     FAIL" if v == float("inf") else f"{v:9.3f}s"
        agree = len({c for c in costs.values() if c is not None}) <= 1
        print(f"ladder({k}) {A.number_of_nodes():>8} {len(   [n for n in A.nodes if A.out_degree(n)>=2]):>6} "
              f"{indeg:>11} {w:>3} | {f(res['profiled'])} {f(res['cell'])} {f(res['rebase'])} | "
              f"{best}{'' if agree else '  **COSTS DIFFER**'}", flush=True)
