"""Tree-DTW on networkx -- the rebuild (spec: ``docs/tree_dtw_nx.md``).

Both the source tree ``A`` and the target network ``B`` are plain ``networkx.DiGraph`` objects:
a **vertex** carries float coordinates ``x``/``y``; an **edge** is a directed segment; a **junction**
is just a vertex (split = out-degree > 1, merge = in-degree > 1). No road ids, no coincident vertices,
no stitches.

Built part by part (each independently verifiable):
  * Part 1 -- representation + radius-gated candidates stored on the node   (this file, so far)
  * Parts 2-6 -- emission, forward D, backward B, extraction, validation    (to come)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, List

import networkx as nx
import numpy as np

try:                                                            # optional: fast candidate gating
    from scipy.spatial import cKDTree as _KDTree
except Exception:                                               # pragma: no cover - fallback is exact
    _KDTree = None

INF = float("inf")


class NotATree(ValueError):
    """Raised when the source graph ``A`` is not a tree (its undirected graph has a cycle)."""


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _xy(G: nx.DiGraph, n: Hashable) -> tuple[float, float]:
    d = G.nodes[n]
    return float(d["x"]), float(d["y"])


def _emit(ax: float, ay: float, abear, bx: float, by: float, bbear, lam: float) -> float:
    """Unified emission (docs §2): ``‖pos_a − pos_v‖`` plus, when *both* nodes carry a ``bearing`` and
    ``lam`` (= bearing_weight) is set, ``lam·circ(bearing_a, bearing_v)`` with circular difference in
    ``[0, 180]``. Point-mode nodes carry no bearing, so this is a pure distance there."""
    e = math.hypot(ax - bx, ay - by)
    if abear is not None and bbear is not None and lam:
        diff = abs(float(abear) - float(bbear)) % 360.0
        e += float(lam) * min(diff, 360.0 - diff)
    return e


def _validate(A: nx.DiGraph, B: nx.DiGraph) -> None:
    """Both are DiGraphs, every node has ``x``/``y``, and ``A`` is a tree (a polytree: a DAG whose
    undirected graph is a forest -- no directed cycle, no reconvergence)."""
    for name, G in (("A", A), ("B", B)):
        if not isinstance(G, nx.DiGraph):
            raise TypeError(f"{name} must be a networkx.DiGraph, got {type(G).__name__}")
        for n in G.nodes:
            if "x" not in G.nodes[n] or "y" not in G.nodes[n]:
                raise ValueError(f"{name} node {n!r} is missing 'x'/'y' coordinates")
    if A.number_of_nodes() and not nx.is_directed_acyclic_graph(A):
        raise NotATree("source A has a directed cycle -- not a tree")
    if not nx.is_forest(A.to_undirected()):
        raise NotATree("source A has an undirected cycle (a reconvergence/diamond) -- not a tree")


# ---------------------------------------------------------------------------------------
# Part 1 -- radius-gated candidates, stored on the A node
# ---------------------------------------------------------------------------------------
def prepare(A: nx.DiGraph, B: nx.DiGraph, r: float = 20.0, k_min: int = 1,
            bearing_weight: float = 1.0) -> nx.DiGraph:
    """Validate ``A``, ``B`` and populate each A-vertex's radius-gated candidate table (docs §1-§2).

    For every A-vertex ``a`` writes::

        A.nodes[a]["cand"] = {
            v: {"E": E(a, v), "D": inf, "bpD": [], "B": inf, "bpB": []}
            for every B-vertex v whose position lies within r of a
        }

    Candidates are **gated by position** (``x, y``) within ``r`` (§1.2); the emission ``E`` is the
    unified §2 cost — plain distance for point-mode nodes, distance + ``bearing_weight·circ(bearing)``
    for segment-mode (line-graph) nodes that carry a ``bearing``. If fewer than ``k_min`` B-vertices lie
    within ``r``, the ``k_min`` nearest (by position) are included anyway (§1.3). Only ``E`` is filled;
    ``D``/``bpD``/``B``/``bpB`` are placeholders for Parts 3-4. Works unchanged on ``A, B`` (point) or
    ``line_digraph(A), line_digraph(B)`` (segment). Returns ``A`` (mutated in place).
    """
    _validate(A, B)
    b_nodes: List[Hashable] = list(B.nodes)
    if not b_nodes:
        raise ValueError("target B has no vertices")
    b_xy = np.asarray([_xy(B, v) for v in b_nodes], dtype=float)
    b_bear = [B.nodes[v].get("bearing") for v in b_nodes]
    kd = _KDTree(b_xy) if _KDTree is not None else None

    for a in A.nodes:
        ax, ay = _xy(A, a)
        abear = A.nodes[a].get("bearing")
        if kd is not None:
            idx = list(kd.query_ball_point((ax, ay), r))
        else:                                                   # exact numpy fallback
            d2 = (b_xy[:, 0] - ax) ** 2 + (b_xy[:, 1] - ay) ** 2
            idx = list(np.nonzero(d2 <= r * r)[0])
        if len(idx) < k_min:                                    # feasibility fallback: k_min nearest
            d2 = (b_xy[:, 0] - ax) ** 2 + (b_xy[:, 1] - ay) ** 2
            idx = list(np.argsort(d2)[:k_min])
        cand: Dict[Hashable, Dict[str, Any]] = {}
        for i in idx:
            v = b_nodes[i]
            e = _emit(ax, ay, abear, float(b_xy[i, 0]), float(b_xy[i, 1]), b_bear[i], bearing_weight)
            cand[v] = {"E": float(e), "D": INF, "bpD": [], "B": INF, "bpB": []}
        A.nodes[a]["cand"] = cand
    return A


# ---------------------------------------------------------------------------------------
# Part 2 -- the directed line graph (segment mode): a node per segment, carrying midpoint + bearing
# ---------------------------------------------------------------------------------------
def line_digraph(G: nx.DiGraph) -> nx.DiGraph:
    """The directed line graph ``L(G) = nx.line_graph(G)`` with each L-node ``(u, v)`` given the
    segment's **midpoint** as its ``x, y`` and its compass **bearing** (docs §2, §7). `nx.line_graph`
    gives exactly the arc adjacency (``(a,b) → (c,d)`` iff ``b == c``) but copies no attributes, so we
    attach them from ``G``'s endpoint coordinates. The result is an ordinary ``DiGraph`` that
    :func:`prepare` and the later DP treat identically to a point-mode graph."""
    L = nx.line_graph(G)
    for (u, v) in list(L.nodes):
        xu, yu = _xy(G, u)
        xv, yv = _xy(G, v)
        L.nodes[(u, v)]["x"] = 0.5 * (xu + xv)
        L.nodes[(u, v)]["y"] = 0.5 * (yu + yv)
        L.nodes[(u, v)]["bearing"] = (math.degrees(math.atan2(xv - xu, yv - yu)) + 360.0) % 360.0
    return L


# ---------------------------------------------------------------------------------------
# small builders + demo (verification of Part 1)
# ---------------------------------------------------------------------------------------
def digraph(nodes: Dict[Hashable, tuple[float, float]], edges: List[tuple]) -> nx.DiGraph:
    """Convenience builder for tests: ``nodes`` maps id -> (x, y); ``edges`` is a list of (u, v)."""
    G = nx.DiGraph()
    for n, (x, y) in nodes.items():
        G.add_node(n, x=float(x), y=float(y))
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------------------
# Part 3 -- forward table D (upstream cost), stored on the node
# ---------------------------------------------------------------------------------------
def _pass(A: nx.DiGraph, B: nx.DiGraph, order, pred, succ, bpred, bsucc,
          key: str, bpkey: str, alpha: float, beta: float) -> None:
    """One min-sum sweep filling ``cand[v][key]`` / ``cand[v][bpkey]`` for every A-vertex, in ``order``.
    Parameterised so the *same* body serves the forward pass (``pred=A.predecessors``,
    ``bpred=B.predecessors``, ``outdeg`` = A out-degree) and the backward pass (all reversed, Part 4).
    Implements the §3 three-way min: (D) advance, (V) β-stall, (H) α-coverage."""
    import heapq
    from itertools import count
    tick = count()
    deg = {n: max(1, len(list(succ(n)))) for n in A.nodes}      # the 1/outdeg split factor (§3)

    for a in order:
        cand = A.nodes[a]["cand"]
        preds = list(pred(a))
        base: Dict[Hashable, float] = {}
        base_bp: Dict[Hashable, list] = {}
        for v, c in cand.items():
            Ev = c["E"]
            if not preds:                                       # source: free entry
                base[v], base_bp[v] = Ev, []
                continue
            step = {}                                           # step_p, its arg-cell, stall_p per pred
            for p in preds:
                pc = A.nodes[p]["cand"]
                sp, spx = INF, None
                for x in bpred(v):                              # advance: one B-arc into v
                    if x in pc and pc[x][key] < sp:
                        sp, spx = pc[x][key], x
                stall = pc[v][key] if v in pc else INF          # stall: p already on v
                step[p] = (sp, spx, stall)
            # (D) every predecessor advances into v  (full E)
            adv = Ev + sum(step[p][0] / deg[p] for p in preds)
            adv_bp = [(p, step[p][1]) for p in preds]
            # (V) at least one predecessor stalls on v  (β E, force-one-stall)
            vbest, vbest_bp = INF, None
            for q in preds:
                sq, _sqx, stq = step[q]
                if stq == INF:
                    continue
                tot, bp_q, ok = stq / deg[q], [(q, v)], True
                for p in preds:
                    if p == q:
                        continue
                    sp, spx, stp = step[p]
                    m = min(stp, sp)
                    if m == INF:
                        ok = False
                        break
                    tot += m / deg[p]
                    bp_q.append((p, v) if stp <= sp else (p, spx))
                if ok and beta * Ev + tot < vbest:
                    vbest, vbest_bp = beta * Ev + tot, bp_q
            if adv <= vbest:
                base[v], base_bp[v] = adv, adv_bp
            else:
                base[v], base_bp[v] = vbest, vbest_bp

        # (H) 1:N coverage: within-row shortest path over B restricted to cand(a) (Dijkstra)
        D = dict(base)
        bp = dict(base_bp)
        pq = [(D[v], next(tick), v) for v in cand]
        heapq.heapify(pq)
        while pq:
            dv, _, v = heapq.heappop(pq)
            if dv > D[v]:
                continue
            for w in bsucc(v):                                  # v -> w a B-arc; a extends its run onto w
                if w not in cand:
                    continue
                nw = dv + alpha * cand[w]["E"]
                if nw < D[w]:
                    D[w], bp[w] = nw, [(a, v)]
                    heapq.heappush(pq, (nw, next(tick), w))
        for v in cand:
            cand[v][key], cand[v][bpkey] = D[v], bp[v]


def forward(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """Fill the forward table ``D`` and back-pointers ``bpD`` on every A-vertex's candidate table
    (docs §3). Requires :func:`prepare` to have run. Returns ``A`` (mutated in place)."""
    order = list(nx.topological_sort(A))
    _pass(A, B, order, A.predecessors, A.successors, B.predecessors, B.successors,
          "D", "bpD", alpha, beta)
    return A


def backward(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """Fill the backward table ``B`` and back-pointers ``bpB`` (docs §4) — the identical three-way
    ``min`` with A and B **reversed**: sum over **successors**, ``step`` from a B-**successor**, split
    factor = **in**-degree, swept in **reverse** topological order. Same ``α``/``β``, same emission.
    Requires :func:`prepare`. Returns ``A`` (mutated in place)."""
    order = list(reversed(list(nx.topological_sort(A))))
    _pass(A, B, order, A.successors, A.predecessors, B.successors, B.predecessors,
          "B", "bpB", alpha, beta)
    return A


# ---------------------------------------------------------------------------------------
# Part 5 -- extraction: seed once per component, then follow the back-pointers (docs §5)
# ---------------------------------------------------------------------------------------
def _is_cover(bp, c) -> bool:
    """A back-pointer list is a 1:N COVER step iff it is a single pair whose source is ``c`` itself."""
    return len(bp) == 1 and bp[0][0] == c


def extract(A: nx.DiGraph, B: nx.DiGraph):
    """Extract the matching relation ``M`` (docs §5). Seed any uncommitted vertex at its joint arg-min
    ``D+B−E`` (feasibility rule §1.3 if none is finite), then flood the stored back-pointers — commit
    each predecessor in the forward anchor's ``bpD`` and each successor in the backward anchor's
    ``bpB`` — until every vertex in the component is committed; re-seed for a disconnected forest. Each
    vertex's **coverage run is read from the FORWARD cover chain only**, so runs partition (no overlap,
    no gap-fill). Returns ``(M, committed)`` — ``M ⊆ V(A)×V(B)`` and ``committed`` = each vertex's pivot
    cell. Works identically on ``A, B`` (point) or ``line_digraph`` graphs (segment)."""
    from collections import deque
    committed: Dict[Hashable, Hashable] = {}
    M: set = set()
    q: deque = deque()

    def commit(c, v):
        if c not in committed:
            committed[c] = v
            q.append(c)

    for r in A.nodes:
        if r in committed:
            continue
        cand = A.nodes[r]["cand"]
        v_star, best = None, INF
        for v, c in cand.items():
            tot = c["D"] + c["B"] - c["E"]
            if tot < best:
                best, v_star = tot, v
        if v_star is None or math.isinf(best):
            raise ValueError(f"vertex {r!r} has no finite matching within r -- increase match_radius_m")
        commit(r, v_star)
        while q:
            c = q.popleft()
            cc = A.nodes[c]["cand"]
            v = committed[c]
            run, head = [v], v                              # coverage run = forward COVER chain only
            while _is_cover(cc[head]["bpD"], c):
                head = cc[head]["bpD"][0][1]
                run.append(head)
            for w in run:
                M.add((c, w))
            for (p, x) in cc[head]["bpD"]:                  # predecessors (advance at the forward anchor)
                commit(p, x)
            tail = v                                        # successors: past c's own backward cover
            while _is_cover(cc[tail]["bpB"], c):
                tail = cc[tail]["bpB"][0][1]
            for (s, w) in cc[tail]["bpB"]:
                commit(s, w)
    return M, committed


# ---------------------------------------------------------------------------------------
# Table validation -- V1/V2/V3 per cell, by following the REAL stored back-pointers (docs §6)
# ---------------------------------------------------------------------------------------
def _reconstruct(A: nx.DiGraph, a: Hashable, v: Hashable, bpkey: str) -> set:
    """The partial matching a cell stands for -- follow its stored back-pointer list to the ends."""
    out: set = set()
    stack = [(a, v)]
    while stack:
        x, w = stack.pop()
        if (x, w) in out:
            continue
        out.add((x, w))
        for (x2, w2) in A.nodes[x]["cand"][w][bpkey]:
            stack.append((x2, w2))
    return out


def check_rules(M: set, src: nx.DiGraph, tgt: nx.DiGraph):
    """V1, V2, V3 on a matching ``M`` over graphs ``src``/``tgt``, each rule restricted to neighbours
    present in ``M`` (docs §6). Returns ``(v1, v2, v3)`` lists of offending pairs."""
    has = M.__contains__
    inM = {a for (a, _v) in M}
    v1, v2, v3 = [], [], []
    for (a, v) in M:
        if any((am in inM) and has((am, vp)) for am in src.predecessors(a) for vp in tgt.successors(v)):
            v1.append((a, v))
        if not any(has((a, vm)) for vm in tgt.predecessors(v)):
            for am in src.predecessors(a):
                if am in inM and not (has((am, v)) or any(has((am, vm)) for vm in tgt.predecessors(v))):
                    v2.append((a, v)); break
        if not any(has((a, vp)) for vp in tgt.successors(v)):
            for ap in src.successors(a):
                if ap in inM and not (has((ap, v)) or any(has((ap, vp)) for vp in tgt.successors(v))):
                    v3.append((a, v)); break
    return v1, v2, v3


def validate_tables(A: nx.DiGraph, B: nx.DiGraph, which: str = "D"):
    """Validate every finite cell of the ``which`` table (``"D"`` forward or ``"B"`` backward) against
    V1/V2/V3, by reconstructing each cell's partial matching from its **real** back-pointers and
    checking it (docs §6). For the backward table the source/target roles reverse. Returns
    ``(n_cells, bad)`` where ``bad`` is a list of ``(a, v, v1, v2, v3)``."""
    bpkey = "bpD" if which == "D" else "bpB"
    src, tgt = (A, B) if which == "D" else (A.reverse(copy=False), B.reverse(copy=False))
    n, bad = 0, []
    for a in A.nodes:
        for v, c in A.nodes[a]["cand"].items():
            if math.isinf(c[which]):
                continue
            n += 1
            v1, v2, v3 = check_rules(_reconstruct(A, a, v, bpkey), src, tgt)
            if v1 or v2 or v3:
                bad.append((a, v, v1, v2, v3))
    return n, bad


if __name__ == "__main__":
    def dump(A: nx.DiGraph, title: str) -> None:
        print(f"\n=== {title} ===")
        for a in A.nodes:
            ax, ay = _xy(A, a)
            cand = A.nodes[a]["cand"]
            cells = ", ".join(f"{v}(E={c['E']:.2f})" for v, c in
                              sorted(cand.items(), key=lambda kv: kv[1]["E"]))
            print(f"  A[{a}] ({ax:.0f},{ay:.0f}) -> {len(cand)} cand: {cells}")

    # CHAIN 0->1->2, target chain 0.5 m north; r=20 so the far end just falls out of gate
    A = digraph({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)])
    B = digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")])
    dump(prepare(A, B, r=20.0), "chain  (r=20)")

    # SPLIT 0->1, 1->2, 1->3
    A = digraph({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)}, [(0, 1), (1, 2), (1, 3)])
    B = digraph({"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
                [("s", "j"), ("j", "u"), ("j", "d")])
    dump(prepare(A, B, r=20.0), "split  (r=20)")

    # MERGE 0->2, 1->2, 2->3
    A = digraph({0: (0, 6), 1: (0, -6), 2: (10, 0), 3: (20, 0)}, [(0, 2), (1, 2), (2, 3)])
    B = digraph({"a": (0, 6.5), "b": (0, -5.5), "m": (10, .5), "o": (20, .5)},
                [("a", "m"), ("b", "m"), ("m", "o")])
    dump(prepare(A, B, r=20.0), "merge  (r=20)")

    # PART 3-4: forward D + backward B + table validation (point mode)
    print("\n########## Parts 3-4 -- forward D & backward B + validation (point mode) ##########")
    cases = {
        "chain": (digraph({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)]),
                  digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")])),
        "split": (digraph({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)}, [(0, 1), (1, 2), (1, 3)]),
                  digraph({"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
                          [("s", "j"), ("j", "u"), ("j", "d")])),
        "merge": (digraph({0: (0, 6), 1: (0, -6), 2: (10, 0), 3: (20, 0)}, [(0, 2), (1, 2), (2, 3)]),
                  digraph({"a": (0, 6.5), "b": (0, -5.5), "m": (10, .5), "o": (20, .5)},
                          [("a", "m"), ("b", "m"), ("m", "o")])),
    }
    for name, (GA, GB) in cases.items():
        prepare(GA, GB, r=20.0)
        forward(GA, GB, alpha=1.0, beta=1.0)
        backward(GA, GB, alpha=1.0, beta=1.0)
        nD, badD = validate_tables(GA, GB, "D")
        nB, badB = validate_tables(GA, GB, "B")
        print(f"\n{name}:  D {nD} cells -> {'VALID' if not badD else badD};  "
              f"B {nB} cells -> {'VALID' if not badB else badB}")
        for a in GA.nodes:
            row = GA.nodes[a]["cand"]
            bd = min(row.items(), key=lambda kv: kv[1]["D"])
            bb = min(row.items(), key=lambda kv: kv[1]["B"])
            seed = min(row.items(), key=lambda kv: kv[1]["D"] + kv[1]["B"] - kv[1]["E"])
            print(f"   A[{a}]  minD@{bd[0]}={bd[1]['D']:.2f}  minB@{bb[0]}={bb[1]['B']:.2f}  "
                  f"seed(D+B-E)@{seed[0]}={seed[1]['D']+seed[1]['B']-seed[1]['E']:.2f}")

    # PART 5: extraction -> M, validated as a FINAL matching (point mode), incl. a coverage case
    print("\n########## Part 5 -- extraction -> M + final V1-V4 validation (point mode) ##########")
    cov = (digraph({0: (0, 0), 1: (20, 0)}, [(0, 1)]),                     # A: ONE coarse segment
           digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)},        # B: TWO fine segments
                   [("b0", "b1"), ("b1", "b2")]))
    for name, (GA, GB) in {**cases, "coverage(1:2)": cov}.items():
        prepare(GA, GB, r=25.0)
        forward(GA, GB)
        backward(GA, GB)
        M, committed = extract(GA, GB)
        v1, v2, v3 = check_rules(M, GA, GB)
        v4 = [a for a in GA.nodes if not any(x == a for (x, _w) in M)]     # every source vertex covered?
        ok = not (v1 or v2 or v3 or v4)
        am = {a: sorted((str(w) for (x, w) in M if x == a)) for a in GA.nodes}
        print(f"\n{name}: |M|={len(M)}  valid(V1-V4)={ok}"
              + ("" if ok else f"   V1={v1} V2={v2} V3={v3} V4={v4}"))
        for a in GA.nodes:
            print(f"   A[{a}] -> {am[a]}   (pivot {committed.get(a)})")

    # SEGMENT MODE (Part 2): build the directed line graphs, then the SAME prepare on them.
    A = digraph({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)}, [(0, 1), (1, 2), (1, 3)])
    B = digraph({"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
                [("s", "j"), ("j", "u"), ("j", "d")])
    LA, LB = line_digraph(A), line_digraph(B)
    print("\n=== split, SEGMENT mode: L(A) nodes (segment -> midpoint, bearing) ===")
    for s in LA.nodes:
        x, y = _xy(LA, s)
        print(f"  seg {s} -> mid=({x:.1f},{y:.1f}) bearing={LA.nodes[s]['bearing']:.1f}°")
    prepare(LA, LB, r=20.0, bearing_weight=1.0)
    print("segment candidate tables  (E = midpoint-dist + 1.0·circ(bearing)):")
    for s in LA.nodes:
        cand = LA.nodes[s]["cand"]
        cells = ", ".join(f"{e}(E={c['E']:.2f})"
                          for e, c in sorted(cand.items(), key=lambda kv: kv[1]["E"]))
        print(f"  seg {s} -> {cells}")

    # tree guard: a diamond (reconvergence) must be rejected
    try:
        prepare(digraph({0: (0, 0), 1: (1, 1), 2: (1, -1), 3: (2, 0)},
                        [(0, 1), (0, 2), (1, 3), (2, 3)]), B, r=20.0)
        print("\nERROR: diamond was NOT rejected")
    except NotATree as e:
        print(f"\ndiamond correctly rejected: {e}")
