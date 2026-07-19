"""Where is the merge-pressure threshold? Sweep the two axes INDEPENDENTLY.

A fixed threshold (Mo >= 2) is far too low -- it misroutes 4 of 11 cases here. `rebase` wins only
when Mo >= W: when EVERY nested split's branch rejoins, so both other engines are maximally loaded
at once. Below that there is always a merge-free path through the nesting and `cell` stays cheap.

    k,j   W  Mo | profiled     cell   rebase |   best
    4,2   4   2 |   0.075s   0.006s   0.007s |   cell     <- Mo>=2 would pick rebase
    4,3   4   3 |   0.076s   0.045s   0.052s |   cell     <- Mo>=2 would pick rebase
    4,4   4   4 |   0.088s   0.277s   0.037s | rebase
    5,4   5   4 |   1.204s   0.341s   0.383s |   cell     <- Mo>=2 would pick rebase
    5,5   5   5 |   1.673s   2.073s   0.294s | rebase

Vary the two pressures INDEPENDENTLY to find the real Mo threshold.

braid(k, j): k nested splits; j of the branches rejoin at separate chained merges, the other k-j
run to their own sinks. So W ~ k (fixed) while Mo ~ j (swept).
"""
import sys, os, copy, time, statistics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_dtw import digraph, match_dag, _cost_of
from network_matching.profiled import profiled_width, merge_pressure

def braid(k, j, sp=12.0):
    n, e = {}, []
    for i in range(k + 1):
        n[f"s{i}"] = (sp*i, 0.0); n[f"s{i}_m"] = (sp*i + sp/2, 0.0)
        if i < k: e += [(f"s{i}", f"s{i}_m"), (f"s{i}_m", f"s{i+1}")]
    base = sp*(k+1)
    for i in range(j + 1):
        n[f"m{i}"] = (base + sp*i, 0.0); n[f"m{i}_m"] = (base + sp*i + sp/2, 0.0)
        if i < j: e += [(f"m{i}", f"m{i}_m"), (f"m{i}_m", f"m{i+1}")]
    n["t"] = (base + sp*(j+1), 0.0)
    n["sx"] = (sp*k + sp/2, 0.0)
    e += [(f"s{k}", "sx"), ("sx", "m0"), (f"m{j}", "t")]
    for i in range(k):
        n[f"b{i}"] = (sp*i + sp/2, -9.0)
        n[f"b{i}_m"] = ((sp*i + base)/2, -9.0)
        e += [(f"s{i}", f"b{i}"), (f"b{i}", f"b{i}_m")]
        if i < j:  e += [(f"b{i}_m", f"m{i}")]                 # rejoins -> a merge
        else:      n[f"b{i}_e"] = (base, -9.0 - i); e += [(f"b{i}_m", f"b{i}_e")]   # own sink
    A = digraph(n, e)
    Bn = {f"{v}'": (x, y+0.4) for v,(x,y) in n.items()}
    return A, digraph(Bn, [(f"{u}'", f"{v}'") for u,v in e])

if __name__ == "__main__":
    print(f"{'k,j':>6} {'W':>3} {'Mo':>3} | {'profiled':>10} {'cell':>10} {'rebase':>10} | {'best':>8} | rule picks")
    for k in (4, 5):
        for j in range(0, k + 1):
            A, B = braid(k, j)
            W, Mo = profiled_width(A), merge_pressure(A)
            res = {}
            for eng in ("profiled", "cell", "rebase"):
                ts = []
                for _ in range(2):
                    Ax, Bx = copy.deepcopy(A), copy.deepcopy(B)
                    t = time.perf_counter()
                    try:
                        match_dag(Ax, Bx, r=14.0, alpha=0.5, beta=1.0, engine=eng)
                        ts.append(time.perf_counter()-t)
                    except BaseException:
                        ts.append(float("inf")); break
                res[eng] = statistics.median(ts)
            best = min(res, key=res.get)
            pick = "profiled" if W <= 2 else ("rebase" if Mo >= 2 else "cell")
            f = lambda v: "     FAIL" if v == float("inf") else f"{v:9.3f}s"
            flag = "" if pick == best else f"  <-- picks {pick}"
            print(f"{k},{j:>3} {W:>3} {Mo:>3} | {f(res['profiled'])} {f(res['cell'])} {f(res['rebase'])} "
                  f"| {best:>8} |{flag}", flush=True)
