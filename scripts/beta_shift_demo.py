"""The stall penalty β on a rigidly-shifted source (paper §9.5, docs/dag_dtw_matching.md §3).

Two maps of the *same* road, one drawn 9 m further along the direction of travel — a registration
shift. The nearest-neighbour reading of that is a lie: A's leading vertices have no B cell before
B's start, so the cheapest-emission matching collapses them onto B's first cell (an N:1 pile) and
reports a *small* drift for a correspondence that is almost entirely wrong. β prices exactly that
collapse. Raising it flips the matcher to the 1:1 correspondence — the geometrically true reading
of a shift — whose cost is β-independent because it contains no stall cell.

    python scripts/beta_shift_demo.py
"""
import math
from collections import Counter

from network_matching.dag_dtw import digraph, prepare, forward, extract_cell, _cost_of

SHIFT, LAT, N, STEP, R = 9.0, 0.5, 11, 2.0, 15.0
BETAS = (1.0, 2.0, 4.0, 5.0, 5.5, 6.0, 8.0)


def case():
    """A: N vertices, STEP apart, along x. B: the same chain rigidly shifted (SHIFT, LAT)."""
    A = digraph({i: (STEP * i, 0.0) for i in range(N)}, [(i, i + 1) for i in range(N - 1)])
    B = digraph({f"b{i}": (STEP * i + SHIFT, LAT) for i in range(N)},
                [(f"b{i}", f"b{i + 1}") for i in range(N - 1)])
    return A, B


def solve(beta, alpha=1.0):
    A, B = case()
    prepare(A, B, r=R)
    forward(A, B, alpha=alpha, beta=beta)
    M, _ = extract_cell(A, B, alpha, beta)
    per_b = Counter(v for (_a, v) in M)
    drift = sum(math.dist((A.nodes[a]["x"], A.nodes[a]["y"]), (B.nodes[v]["x"], B.nodes[v]["y"]))
                for (a, v) in M) / len(M)
    return dict(
        M=M,
        stalls=sum(c - 1 for c in per_b.values() if c > 1),   # A-vertices piled beyond the first
        pile=max(per_b.values()),                             # worst N:1 cell
        correct=sum(1 for (a, v) in M if v == f"b{a}"),       # truth under a rigid shift: a_i <-> b_i
        drift=drift,
        cost=_cost_of(A, B, M, alpha, beta),
    )


def crossover(lo=1.0, hi=50.0, iters=60):
    """Smallest β at which the matching is stall-free (bisection; monotone in β)."""
    if solve(hi)["stalls"] > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if solve(mid)["stalls"] > 0:
            lo = mid
        else:
            hi = mid
    return hi


def overrun(beta, r=15.0, alpha=1.0):
    """The counter-case the paper must warn about: a source that GENUINELY starts before its
    target (6 vertices, x = 0..10; target cells x = 6..30). Here the head stall is *correct* —
    A's first vertices really do precede B — and a large β destroys it, pairing them 1:1 onto
    cells they never reach. Spatial overrun does not force a stall; only cardinality does."""
    A = digraph({i: (2.0 * i, 0.0) for i in range(6)}, [(i, i + 1) for i in range(5)])
    B = digraph({f"b{j}": (6.0 + 2.0 * j, 0.5) for j in range(13)},
                [(f"b{j}", f"b{j + 1}") for j in range(12)])
    prepare(A, B, r=r)
    forward(A, B, alpha=alpha, beta=beta)
    M, _ = extract_cell(A, B, alpha, beta)
    per_b = Counter(v for (_a, v) in M)
    return sorted(M), sum(c - 1 for c in per_b.values() if c > 1)


if __name__ == "__main__":
    print(f"A: {N} vertices {STEP:.0f} m apart.  B: the same chain shifted +{SHIFT:.0f} m along "
          f"travel, {LAT} m lateral.\nTruth for a rigid shift: a_i <-> b_i (all {N} pairs).  "
          f"alpha = 1, r = {R:.0f}.\n")
    print(f"  {'beta':>5}  {'N:1 stalls':>10}  {'pile':>4}  {'a_i<->b_i':>9}  "
          f"{'drift (m)':>9}  {'C(M)':>7}")
    for b in BETAS:
        s = solve(b)
        print(f"  {b:5.1f}  {s['stalls']:10d}  {s['pile']:4d}  {s['correct']:6d}/{N}  "
              f"{s['drift']:9.2f}  {s['cost']:7.2f}")

    print(f"\ncrossover: beta* = {crossover():.4f}")
    E = math.hypot(SHIFT, LAT)
    print(f"the 1:1 matching has no stall cell, so its cost does not depend on beta:")
    print(f"  C(1:1) = {N} x sqrt({SHIFT:.0f}^2 + {LAT}^2) = {N} x {E:.4f} = {N * E:.2f}")
    for b in (1.0, 8.0):
        pairs = sorted((a, int(v[1:])) for (a, v) in solve(b)["M"])
        print(f"\nbeta = {b:.0f}:  " + "  ".join(f"{a}->{j}" for a, j in pairs))

    print("\n\nthe counter-case: a source that GENUINELY overruns its target at the head")
    print("A = 6 vertices x=0..10;  B = 13 cells x=6..30.  a0,a1,a2 lie before B starts.")
    print("beta=1 reads that correctly (a head stall).  A large beta DESTROYS it:\n")
    for b in (1.0, 4.0, 5.5, 8.0):
        M, st = overrun(b)
        pairs = "  ".join(f"{a}->{v}" for a, v in M)
        note = "correct head overhang" if st else "SPURIOUS 1:1 -- the overhang is gone"
        print(f"  beta={b:<4} stalls={st}   {pairs}   <- {note}")
    print("\nSo beta does not dominate beta=1: it trades one class of error for the other.")
