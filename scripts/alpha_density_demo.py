"""The coverage discount α changes the matching RELATION (paper §9.5.2, docs/dag_dtw_matching.md §3).

One road, matched to itself drawn five times more finely: A has 6 vertices every 5 m, B has 26
cells every 1 m, 0.5 m to the side. Because B is finer, 1:N coverage is **forced** — contiguity
makes each source vertex ride a run of target cells, at *every* α. α does not decide whether to
cover. It decides *which cells each source vertex takes*, and the two ends of its range are two
different optimisation problems:

  * α = 1 — every matched pair is charged its full emission, so the objective is a sum over the
    WHOLE relation. Covering is expensive, so the matcher buys as little of B as it can get away
    with: it shrinks the covered span (cells 4..21 of 0..25) and skews the runs (1, 3, 5, 5, 3, 1).
    A0 sits at x = 0 and is anchored on a cell at x = 4 — a 4 m registration error produced by
    nothing but B's sampling density.

  * α = 0 — only each run's ENTRY is charged, so the objective collapses to Σ_a min_v E(a, v):
    choose one 1:1 anchor cell per source vertex, and fill the gaps between them at zero cost.
    Every vertex anchors on the cell directly beneath it (0, 5, 10, 15, 20, 25), the runs come out
    even (5, 5, 5, 5, 5, 1), and the cost is exactly 6 × 0.5 = 3.00.

Between them the relation slides continuously from one to the other. The cost changes too, but the
cost is not the point: the *correspondence* changes.

    python scripts/alpha_density_demo.py
"""
import math

from network_matching.dag_dtw import digraph, extract_cell, forward, prepare, _cost_of

N_A, A_STEP = 6, 5.0             # source: 6 vertices, 5 m apart
B_STEP, LAT = 1.0, 0.5           # target: the SAME road, 5x finer, 0.5 m to the side
N_B = int(A_STEP * (N_A - 1) / B_STEP) + 1
R = 8.0
ALPHAS = (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0)


def case():
    A = digraph({i: (A_STEP * i, 0.0) for i in range(N_A)},
                [(i, i + 1) for i in range(N_A - 1)])
    B = digraph({f"b{j}": (B_STEP * j, LAT) for j in range(N_B)},
                [(f"b{j}", f"b{j + 1}") for j in range(N_B - 1)])
    return A, B


def solve(alpha, beta=1.0):
    A, B = case()
    prepare(A, B, r=R)
    forward(A, B, alpha=alpha, beta=beta)
    M, _ = extract_cell(A, B, alpha, beta)
    runs = {}
    for a, v in M:
        runs.setdefault(a, []).append(int(v[1:]))
    runs = {a: sorted(js) for a, js in runs.items()}
    lens = [len(runs[a]) for a in sorted(runs)]
    lo = min(js[0] for js in runs.values())
    hi = max(js[-1] for js in runs.values())
    # each source vertex's ANCHOR is the entry of its run; how far is it from the vertex itself?
    anchor_err = max(abs(runs[a][0] * B_STEP - a * A_STEP) for a in runs)
    drift = (sum(math.dist((A.nodes[a]["x"], A.nodes[a]["y"]),
                           (B.nodes[f"b{j}"]["x"], B.nodes[f"b{j}"]["y"]))
                 for a, js in runs.items() for j in js) / sum(lens))
    return dict(M=M, runs=runs, lens=lens, span=(lo, hi), anchor_err=anchor_err,
                drift=drift, cost=_cost_of(A, B, M, alpha, beta))


def entry_only_cost():
    """The α → 0 objective: Σ over source vertices of its cheapest cell."""
    A, B = case()
    prepare(A, B, r=R)
    return sum(min(math.dist((A.nodes[a]["x"], A.nodes[a]["y"]), (B.nodes[v]["x"], B.nodes[v]["y"]))
                   for v in A.nodes[a]["cand"]) for a in A.nodes)


if __name__ == "__main__":
    print(f"A: {N_A} vertices every {A_STEP:.0f} m.  B: the SAME road, every {B_STEP:.0f} m "
          f"({N_B} cells) — {int(A_STEP / B_STEP)}x finer.")
    print("1:N coverage is FORCED by the density; it happens at every alpha. What alpha changes is "
          "WHICH cells each\nsource vertex takes -- the matching relation itself.\n")
    print(f"  {'alpha':>5}  {'cells covered':>15}  {'run lengths':>21}  {'anchor error':>12}  "
          f"{'C(M)':>6}")
    for a in ALPHAS:
        s = solve(a)
        span = f"{s['span'][0]}..{s['span'][1]} of 0..{N_B - 1}"
        print(f"  {a:5.2f}  {span:>15}  {str(s['lens']):>21}  {s['anchor_err']:9.0f} m  "
              f"{s['cost']:6.2f}")

    hi, lo = solve(1.0), solve(0.0)
    print(f"\nalpha = 1 — the objective sums E over the WHOLE relation, so covering is expensive:")
    print(f"  the matcher buys only cells {hi['span'][0]}..{hi['span'][1]}, runs {hi['lens']},")
    print(f"  and anchors A0 (at x=0) on the cell at x={hi['runs'][0][0] * B_STEP:.0f}"
          f" -- a {hi['anchor_err']:.0f} m error from B's sampling alone.")
    print(f"\nalpha = 0 — only the ENTRY of each run is charged, so the objective is Sum_a min_v E(a,v):")
    print(f"  every vertex anchors on the cell beneath it {[r[0] for r in lo['runs'].values()]},")
    print(f"  runs {lo['lens']}, gaps filled free, and")
    print(f"  C(M) = {lo['cost']:.2f}  ==  sum over A of its cheapest cell = {entry_only_cost():.2f}")
