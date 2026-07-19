"""The quadrant neither `profiled` nor `cell` covers -- and why the dispatch is 2-D.

`cell` and `profiled` are NOT complements. They fail on INDEPENDENT pressures, so a source can trip
both at once, and then only re-basing is usable:

                      few open merges          many open merges
  shallow splits      all fast                 profiled  (hourglass)
  nested splits       cell  (btree, ladder)    BOTH BAD -> rebase  (this file)

Measured (r=14, alpha=0.5, beta=1.0):

    braid k  splits  merges  W | profiled     cell   rebase | best
    braid(2)      2       2  2 |   0.003s   0.020s   0.002s | rebase
    braid(3)      3       3  3 |   0.011s   0.042s   0.020s | profiled
    braid(4)      4       4  4 |   0.072s   0.277s   0.018s | rebase   4.0x / 15x
    braid(5)      5       5  5 |   1.384s   2.044s   0.229s | rebase   6.0x / 8.9x

The earlier `ladder` family looked like this quadrant but was not: one merge of in-degree 10 is still
ONE concurrently-open merge, so pending stayed cheap. It takes SEPARATE merges, each open at once,
to load `cell` -- the hourglass wall was three.

BOTH pressures, the shape the ladder missed: many NESTED splits AND many SEPARATE merges.

    spine:   s0 -> s1 -> ... -> sk -> m0
    branch:  si -> bi -> mi           (each split's branch rejoins far downstream)
    merges:  m0 -> m1 -> ... -> mk -> t,  with bi feeding mi

Each si is an ancestor of s(i+1) (nesting) and is not post-dominated until mi, which is near the end
-- so many splits are live at once (profiled's pressure). And there are k SEPARATE merges, each
concurrently open (pending's pressure -- the hourglass wall was only three).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_dtw import digraph

def fam_braid(k, sp=12.0):
    n, e = {}, []
    for i in range(k + 1):                                   # spine (subdivided)
        n[f"s{i}"] = (sp * i, 0.0); n[f"s{i}_m"] = (sp * i + sp / 2, 0.0)
        if i < k: e += [(f"s{i}", f"s{i}_m"), (f"s{i}_m", f"s{i+1}")]
    base = sp * (k + 1)
    for i in range(k + 1):                                   # merge chain
        n[f"m{i}"] = (base + sp * i, 0.0); n[f"m{i}_m"] = (base + sp * i + sp / 2, 0.0)
        if i < k: e += [(f"m{i}", f"m{i}_m"), (f"m{i}_m", f"m{i+1}")]
    n["t"] = (base + sp * (k + 1), 0.0)
    e += [(f"s{k}", f"s{k}_x"), (f"s{k}_x", "m0")] ; n[f"s{k}_x"] = (sp*k + sp/2, 0.0)
    e += [(f"m{k}", "t")]
    for i in range(k):                                       # branches rejoining far downstream
        n[f"b{i}"] = (sp * i + sp / 2, -9.0)
        n[f"b{i}_m"] = ((sp * i + base + sp * i) / 2, -9.0)
        e += [(f"s{i}", f"b{i}"), (f"b{i}", f"b{i}_m"), (f"b{i}_m", f"m{i}")]
    A = digraph(n, e)
    Bn = {f"{v}'": (x, y + 0.4) for v, (x, y) in n.items()}
    return A, digraph(Bn, [(f"{u}'", f"{v}'") for u, v in e])

if __name__ == "__main__":
    import copy, time, statistics
    from network_matching.dag_dtw import match_dag, _cost_of
    from network_matching.profiled import profiled_width
    print(f"{'braid k':<10} {'|A|':>4} {'splits':>6} {'merges':>6} {'w':>3} | "
          f"{'profiled':>10} {'cell':>10} {'rebase':>10} | best")
    for k in (2, 3, 4, 5):
        A, B = fam_braid(k)
        w = profiled_width(A)
        ns = sum(1 for x in A.nodes if A.out_degree(x) >= 2)
        nm = sum(1 for x in A.nodes if A.in_degree(x) > 1)
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
                except BaseException:
                    ts.append(float("inf")); costs[eng] = None; break
            res[eng] = statistics.median(ts)
        f = lambda v: "     FAIL" if v == float("inf") else f"{v:9.3f}s"
        cs = {c for c in costs.values() if c is not None}
        print(f"braid({k}) {A.number_of_nodes():>9} {ns:>6} {nm:>6} {w:>3} | "
              f"{f(res['profiled'])} {f(res['cell'])} {f(res['rebase'])} | "
              f"{min(res, key=res.get)}{'' if len(cs) <= 1 else '  **COSTS DIFFER**'}", flush=True)
