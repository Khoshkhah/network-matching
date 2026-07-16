"""PROTOTYPE (not part of the library): the arc-length ledger, for paper §8.3's future work.

The shipped ledger charges each matched cell once, weighted by the move that entered it
(1 / beta / alpha). Its cost therefore counts *vertices*, which is a property of how the other
cartographer discretised their map -- and alpha is a hand-tuned patch for that.

This prototype charges each step by the LENGTH of the arc(s) it traverses instead:

    advance (both sides move)  ->  E * (len(a) + len(v))
    cover   (only B moves)     ->  E * len(v)
    stall   (only A moves)     ->  E * len(a)      [times beta, the guard that survives]

No alpha appears anywhere: every step reads its own two lengths. States are ARCS (segment mode),
which is why this needs no quadrature -- an arc has a length; a vertex does not.

    python scripts/length_weighted_prototype.py
"""
import math

INF = float("inf")


def arcs(points, y):
    """[(mid_x, mid_y, length)] for consecutive points along a straight road at height y."""
    return [((points[i] + points[i + 1]) / 2, y, points[i + 1] - points[i])
            for i in range(len(points) - 1)]


def dtw(A, B, weighted, beta=1.0):
    """Free entry and exit on B, as the engine has. weighted=False reproduces the shipped
    count-based ledger (every pair charged once); weighted=True is the arc-length ledger."""
    n, m = len(A), len(B)
    E = [[math.dist((A[i][0], A[i][1]), (B[j][0], B[j][1])) for j in range(m)] for i in range(n)]
    D = [[INF] * m for _ in range(n)]
    bp = [[None] * m for _ in range(n)]
    for j in range(m):                                   # row 0: free entry, or cover on from j-1
        entry = E[0][j] * ((A[0][2] + B[j][2]) if weighted else 1.0)
        cover = (D[0][j - 1] + E[0][j] * (B[j][2] if weighted else 1.0)) if j else INF
        D[0][j], bp[0][j] = (entry, "adv") if entry <= cover else (cover, "cov")
    for i in range(1, n):
        for j in range(m):
            la, lb = A[i][2], B[j][2]
            adv = (D[i - 1][j - 1] + E[i][j] * ((la + lb) if weighted else 1.0)) if j else INF
            stall = D[i - 1][j] + E[i][j] * beta * (la if weighted else 1.0)
            cover = (D[i][j - 1] + E[i][j] * (lb if weighted else 1.0)) if j else INF
            best = min(adv, stall, cover)
            D[i][j] = best
            bp[i][j] = "adv" if best == adv else "stall" if best == stall else "cov"
    j = min(range(m), key=lambda j: D[n - 1][j])         # free exit
    runs, i = {}, n - 1
    while True:
        runs.setdefault(i, []).append(j)
        mv = bp[i][j]
        if i == 0 and mv != "cov":
            break
        if mv == "adv":
            i, j = i - 1, j - 1
        elif mv == "stall":
            i -= 1
        else:
            j -= 1
    return {i: sorted(js) for i, js in runs.items()}


if __name__ == "__main__":
    a_pts = [0, 5, 10, 15, 20, 25]                       # source: five 5 m arcs
    print("One road. The source has five 5 m arcs (midpoints 2.5, 7.5, 12.5, 17.5, 22.5).")
    print("The target is the SAME road, 0.5 m to the side. Only its sampling changes.")
    print("Each source arc should anchor at its own place along the road, at EVERY sampling.\n")
    print(f"  {'B arc':>7} | {'shipped ledger (counts pairs)':>33} | {'arc-length ledger':>33}")
    for db in (2.5, 1.0, 0.5, 0.25, 0.1):
        b_pts = [i * db for i in range(int(round(25 / db)) + 1)]
        A, B = arcs(a_pts, 0.0), arcs(b_pts, 0.5)
        cols = []
        for w in (False, True):
            runs = dtw(A, B, w)
            cols.append(" ".join(f"{B[runs[i][0]][0]:5.1f}" for i in sorted(runs)))
        print(f"  {db:6.2f}m | {cols[0]:>33} | {cols[1]:>33}")
    print(f"\n  {'ideal':>6}  | {'':>33} | {'  2.5   7.5  12.5  17.5  22.5':>33}")
    print("\nThe shipped ledger collapses as the target is drawn more finely -- at 0.1 m every")
    print("source arc bunches into a half-metre of B. The arc-length ledger is unmoved by the")
    print("sampling, and carries no alpha at all.")
