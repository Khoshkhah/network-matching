"""DAG-DTW -- exact matcher of a directed source DAG (a tree is the loop-free special case) to a
directed network, on networkx
(spec: ``docs/dag_dtw_matching.md``; the implementation is documented there as Parts 1-6).

Both the source DAG ``A`` and the target network ``B`` are plain ``networkx.DiGraph`` objects:
a **vertex** carries float coordinates ``x``/``y``; an **edge** is a directed segment; a **junction**
is just a vertex (split = out-degree > 1, merge = in-degree > 1). No road ids, no coincident vertices,
no stitches.

Parts (each independently verifiable): representation + radius-gated candidates (Part 1), emission
(Part 2), forward ``D`` (Part 3), backward ``B`` (Part 4), extraction (Part 5), validation (Part 6).
Segment mode = the same parts run on the directed line graphs ``L(A)``, ``L(B)``.
"""
from __future__ import annotations

import math
import os
import signal
import threading
import time
from typing import Any, Dict, Hashable, List

import networkx as nx
import numpy as np

try:                                                            # optional: fast candidate gating
    from scipy.spatial import cKDTree as _KDTree
except Exception:                                               # pragma: no cover - fallback is exact
    _KDTree = None

INF = float("inf")


class NotADAG(ValueError):
    """Raised when the source graph ``A`` has a directed cycle -- the source must be a DAG."""


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
    """Both are DiGraphs, every node has ``x``/``y``, and ``A`` is a **DAG** (directed-acyclic;
    reconvergences/diamonds are fine -- a tree is the special case). On reconvergent sources only
    :func:`extract_cell` carries the exactness claim (docs/dag_dtw_matching.md §7, §10.2)."""
    for name, G in (("A", A), ("B", B)):
        if not isinstance(G, nx.DiGraph):
            raise TypeError(f"{name} must be a networkx.DiGraph, got {type(G).__name__}")
        for n in G.nodes:
            if "x" not in G.nodes[n] or "y" not in G.nodes[n]:
                raise ValueError(f"{name} node {n!r} is missing 'x'/'y' coordinates")
    if A.number_of_nodes() and not nx.is_directed_acyclic_graph(A):
        raise NotADAG("source A has a directed cycle -- not a DAG")


# ---------------------------------------------------------------------------------------
# Part 1 -- radius-gated candidates, stored on the A node
# ---------------------------------------------------------------------------------------
def prepare(A: nx.DiGraph, B: nx.DiGraph, r: float = 20.0, k_min: int = 1,
            bearing_weight: float = 1.0) -> nx.DiGraph:
    """Validate ``A``, ``B`` and populate each A-vertex's radius-gated candidate table (docs §1-§2).

    For every A-vertex ``a`` writes::

        A.nodes[a]["cand"] = {
            v: {"E": E(a, v), "D": inf, "bpD": [], "B": inf, "bpB": [],
                "forbidden": False}
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
            cand[v] = {"E": float(e), "D": INF, "bpD": [], "B": INF, "bpB": [],
                       "forbidden": False}                  # §4.1a: not a valid run END (role-aware)
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
        L.nodes[(u, v)]["length"] = math.hypot(xv - xu, yv - yu)
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
def _b_order(B: nx.DiGraph) -> Dict[Hashable, int]:
    """A fixed total order on B's vertices (sorted by id). Used to break argmin ties **identically** in
    the forward and backward passes, so equal-cost choices don't diverge between them (docs §4b)."""
    return {v: i for i, v in enumerate(sorted(B.nodes, key=str))}


def layer_order(A: nx.DiGraph) -> tuple[List[Hashable], Dict[Hashable, int]]:
    """Longest-path layering of a DAG ``A`` -- a vertex ordering ``π`` in which every split's
    children share one layer and *all* their successors sit in strictly higher layers, so no
    successor of a vertex ever precedes any of that vertex's siblings.

    Each vertex is given a depth ``L(v)`` = the longest path (in edges) from any source to ``v``:

    1. sweep ``A`` in **topological order** (so a vertex is reached only after every ancestor);
    2. a **source** (``in_degree == 0``) gets ``L = 0``;
    3. every other vertex gets ``L(v) = max(L(p) for p in predecessors(v)) + 1``;
    4. **sort** the vertices by ``L`` ascending (ties broken by ``str(id)`` for determinism).

    Returns ``(order, L)``: the sorted vertex list and the depth map.

    **The property holds on a *subdivided* DAG** -- one with at least one interior point per real
    edge -- because there a split's children each have the split as their **sole** predecessor, so
    they are pairwise incomparable (no sibling is a descendant of another). On a raw DAG where a
    sibling is also a descendant of another sibling the property is impossible for *any* ordering;
    ``L`` then merely places the descendant sibling in a later layer. ``A`` must be acyclic
    (``nx.topological_sort`` raises otherwise).
    """
    L: Dict[Hashable, int] = {}
    for v in nx.topological_sort(A):                        # ancestors before descendants
        preds = list(A.predecessors(v))
        L[v] = 0 if not preds else max(L[p] for p in preds) + 1
    order = sorted(A.nodes, key=lambda v: (L[v], str(v)))   # layer asc; id breaks ties deterministically
    return order, L


def _pass(A: nx.DiGraph, B: nx.DiGraph, order, pred, succ, bpred, bsucc,
          key: str, bpkey: str, alpha: float, beta: float, border: Dict[Hashable, int]) -> None:
    """One min-sum sweep filling ``cand[v][key]`` / ``cand[v][bpkey]`` for every A-vertex, in ``order``.
    Parameterised so the *same* body serves the forward pass (``pred=A.predecessors``,
    ``bpred=B.predecessors``, ``outdeg`` = A out-degree) and the backward pass (all reversed, Part 4)."""
    deg = {n: max(1, len(list(succ(n)))) for n in A.nodes}      # the 1/outdeg split factor (Part 3)
    for a in order:
        _fill_row(A, a, pred, bpred, bsucc, key, bpkey, alpha, beta, border, deg)


def _fill_row(A: nx.DiGraph, a: Hashable, pred, bpred, bsucc,
              key: str, bpkey: str, alpha: float, beta: float,
              border: Dict[Hashable, int], deg: Dict[Hashable, int]) -> None:
    """Fill — or **rebuild** (docs §4.1a) — ONE vertex's row: the Part 3 three-way min — (D) advance,
    (V) β-stall, (H) α-coverage — reading only the already-final neighbour rows and the row itself.
    ``border`` breaks argmin ties by a fixed B-vertex order, identically in both passes (Part 4b).
    A cell whose ``forbidden`` flag is set (§4.1a) cannot be a **run end**, so it is skipped where a
    NEIGHBOUR row attaches to it — as an advance source or a stall source; it remains a legal
    same-row coverage source (a run may pass THROUGH it; role-aware forbidding). The cell's own
    value is still computed. With no flags set (plain :func:`forward` / :func:`backward`) the
    behaviour is byte-for-byte the unconstrained recurrence."""
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
                if x in pc and not pc[x].get("forbidden"):
                    val = pc[x][key]                        # tie -> smaller B-order cell (both passes)
                    if val < sp or (val == sp < INF and border[x] < border[spx]):
                        sp, spx = val, x
            stall = pc[v][key] if (v in pc and not pc[v].get("forbidden")) else INF   # stall: p already on v
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

    # (H) 1:N coverage: within-row fixed point, iterated to convergence (docs Part 3). D[a][v] reads
    # other cells of the SAME row (D[a][v'], v'∈Bpred(v)), so relax until a full sweep changes
    # nothing -- lowering the cost and repointing its back-pointer *together*, so bp never desyncs
    # from D. α·E ≥ 0 ⇒ a monotone descent to the unique least fixed point (correct even when B is
    # cyclic, where a single pass would leave cells un-relaxed). Equal-cost coverage ties keep the
    # smaller-`border` predecessor, matching the advance step's tie-break (Part 4b). A forbidden cell
    # (§4.1a) IS a legal coverage source: the flag means "no child may attach here" (not a valid run
    # END), and a within-row cover step just passes THROUGH the cell -- the run ends elsewhere.
    D = dict(base)
    bp = dict(base_bp)
    changed = True
    while changed:
        changed = False
        for v in cand:
            dv = D[v]
            if dv == INF:
                continue
            for w in bsucc(v):                              # v -> w a B-arc; a extends its run onto w
                if w not in cand:
                    continue
                nw = dv + alpha * cand[w]["E"]
                cur, prev = D[w], bp[w]
                is_cover = len(prev) == 1 and prev[0][0] == a   # w's current value came from coverage
                if nw < cur or (nw == cur < INF and is_cover and border[v] < border[prev[0][1]]):
                    D[w], bp[w] = nw, [(a, v)]
                    changed = True
    for v in cand:
        cand[v][key], cand[v][bpkey] = D[v], bp[v]


def backward(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """Fill the backward table ``B`` and back-pointers ``bpB`` (docs §4) — the identical three-way
    ``min`` with A and B **reversed**: sum over **successors**, ``step`` from a B-**successor**, split
    factor = **in**-degree, swept in **reverse** topological order. Same ``α``/``β``, same emission.
    Requires :func:`prepare`. Returns ``A`` (mutated in place)."""
    order = list(reversed(list(nx.topological_sort(A))))
    _pass(A, B, order, A.successors, A.predecessors, B.successors, B.predecessors,
          "B", "bpB", alpha, beta, _b_order(B))
    return A


def forward(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """**The forward pass** (docs §4): fill ``D``/``bpD`` in the §4.0 longest-path layer order with the
    §4.1a split (V3) coupling built in. As each split's child is built (**including the first**), every
    cell of the split it **cannot use** — no stall or advance transition into any of the child's own
    candidates (:func:`_feasible_links`) — is marked ``forbidden``: dead as a pointer target for ALL
    siblings, past and future, in any pass; already-built siblings whose rows leaned on a
    newly-forbidden exit get a **whole-row rebuild** under the current flags, iterating to a fixed
    point. At the fixed point every surviving (non-forbidden) exit cell of every split is usable by
    ALL its children — :func:`check_split_exits` verifies exactly this. The forbid criterion is
    **feasibility, never optimality** (a child's cheapest row not linking an exit is no ground to kill
    it — the extraction optimizes over the survivors itself); multiple surviving exits are legitimate
    (the single one is chosen at extraction, which never lets a run END on a forbidden cell — though
    a run may still pass THROUGH one: the flag is role-aware, §4.1a).

    This pass owns the ``forbidden`` flags and resets them first. **Run before** :func:`backward` when
    the diagnostic backward table is wanted — it reads the flags, so its pointers never target a
    forbidden cell. Requires :func:`prepare`. Raises ``ValueError`` if a split is left with no surviving
    exit (no V3-valid warping within ``match_radius_m``) or if A is not subdivided (a split's children
    spanning layers — add an interior point on every real edge, §4.0)."""
    order, L = layer_order(A)
    border = _b_order(B)
    deg = {n: max(1, len(list(A.successors(n)))) for n in A.nodes}
    for a in A.nodes:                                           # this pass owns the flags
        for c in A.nodes[a]["cand"].values():
            c["forbidden"] = False
    for s in A.nodes:                                           # §4.0 guard: a split's children share one layer
        kids = list(A.successors(s))
        if len(kids) > 1 and len({L[k] for k in kids}) > 1:
            raise ValueError(f"source not subdivided: split {s!r} has children in layers "
                             f"{sorted({L[k] for k in kids})} -- add an interior point on every real edge (docs §4.0)")

    def refill(c):
        _fill_row(A, c, A.predecessors, B.predecessors, B.successors, "D", "bpD", alpha, beta, border, deg)

    built: set = set()
    for a in order:
        refill(a)
        built.add(a)
        _couple(A, B, a, built, refill)
    return A


def _links(A: nx.DiGraph, c: Hashable, p: Hashable) -> set:
    """The cells of ``p`` that ``c``'s finite forward cells link to — the advance/stall pairs ``(p, x)``
    in ``bpD`` (severed ``None`` references and same-source COVER pairs excluded)."""
    return {x for cell in A.nodes[c]["cand"].values() if cell["D"] < INF
            for (q, x) in cell["bpD"] if q == p and x is not None}


def _feasible_links(A: nx.DiGraph, B: nx.DiGraph, c: Hashable, p: Hashable) -> set:
    """The cells of split ``p`` that child ``c`` COULD use — a pure transition-existence test, no DP
    values: ``x`` serves ``c`` iff ``c`` can β-stall on it (``x ∈ cand(c)``) or advance out of it into
    one of its own candidates (``x ∈ Bpred(w)`` for some ``w ∈ cand(c)``); coverage only extends a row
    after one of those entries, so it adds no exits. Always a superset of :func:`_links` (a
    back-pointer is one of these transitions, chosen by cost). The §4.1a forbid criterion — pruning
    must be sound for the §5 extraction, which optimizes over the surviving cells itself, so an exit
    may only die when some child CANNOT use it, never because a child's cheapest row happened to
    prefer another (that optimality-forbidding emptied splits that had valid warpings).

    The flag this feeds is **role-aware** (docs §4.1a): it bars the cell from serving as the
    split's run END (child attachment, END-state row, sink seed) while leaving it usable as a run
    entry/interior (same-row coverage source, cover-reversed reachability) — matching exactly what
    this test judges."""
    pc = A.nodes[p]["cand"]
    out = set()
    for w in A.nodes[c]["cand"]:
        if w in pc:
            out.add(w)                                          # β-stall: p ends on w, c stays on it
        for x in B.predecessors(w):
            if x in pc:
                out.add(x)                                      # advance: p ends on x, c steps x -> w
    return out


def _couple(A: nx.DiGraph, B: nx.DiGraph, trigger: Hashable, built: set, refill) -> None:
    """The §4.1a forbid-and-rebuild step, run right after ``trigger``'s row is built. For each split
    parent ``p`` of ``trigger``: mark forbidden every non-forbidden exit cell of ``p`` that ``trigger``
    **cannot use** (:func:`_feasible_links` — transition-existence, not the row's optimal
    back-pointers); rebuild (whole row) every already-built sibling whose row linked a
    newly-forbidden exit, so no back-pointer targets a dead cell; re-examine rebuilt rows. The
    forbidden set grows monotonically, so this reaches a fixed point in at most ``|cand(p)|`` rounds;
    at the fixed point every surviving exit is usable by all built children. Raises the feasibility
    ``ValueError`` iff a split's exits empty out — then some child truly has no entry from any exit
    cell, so no V3-valid warping exists within ``r``."""
    work = [trigger]
    while work:
        c = work.pop()
        for p in A.predecessors(c):
            if len(list(A.successors(p))) < 2:                  # V3 only bites at a split
                continue
            pc = A.nodes[p]["cand"]
            usable = _feasible_links(A, B, c, p)
            newly = [v for v, cell in pc.items() if not cell["forbidden"] and v not in usable]
            if not newly:
                continue
            for v in newly:
                pc[v]["forbidden"] = True
            if all(cell["forbidden"] for cell in pc.values()):
                raise ValueError(f"split {p!r}: no surviving V3 exit within r -- increase match_radius_m")
            newset = set(newly)
            for sib in A.successors(p):
                if sib == c or sib not in built:                # later siblings build under the flags
                    continue
                if _links(A, sib, p) & newset:                  # sib's row leaned on a now-dead exit
                    refill(sib)                                 # whole-row rebuild (docs §4.1a step 3)
                    work.append(sib)                            # its links changed -> re-examine


# ---------------------------------------------------------------------------------------
# Part 5 -- extraction: seed once per component, then follow the back-pointers (docs §5)
# ---------------------------------------------------------------------------------------
def _is_cover(bp, c) -> bool:
    """A back-pointer list is a 1:N COVER step iff it is a single pair whose source is ``c`` itself."""
    return len(bp) == 1 and bp[0][0] == c


def extract_two_table(A: nx.DiGraph, B: nx.DiGraph):
    """The two-table traceback (docs §6b) — a PREVIOUS extraction, kept only for the cross-table
    diagnostics (:func:`check_reciprocity` compares its committed matching against both tables);
    the extraction is :func:`extract_cell` (docs §5). Seed any
    uncommitted vertex at its joint arg-min ``D+B−E`` (feasibility rule §1.3 if none is finite), then
    flood the stored back-pointers — commit each predecessor in the forward anchor's ``bpD`` and each
    successor in the backward anchor's ``bpB`` — until every vertex in the component is committed;
    re-seed for a disconnected forest. Each vertex's **coverage run is read from the FORWARD cover
    chain only**, so runs partition. Requires :func:`forward` **and** :func:`backward`.
    Returns ``(M, committed)``."""
    from collections import deque
    # A severed back-pointer (a cell reference of None) means the coupled optimum runs through an
    # infeasible cell -- the per-vertex feasibility check (§1.3) is not enough at a merge/split, whose
    # arms are only coupled here. Raise the feasibility error rather than dereference the missing cell.
    _unreach = "coupled matching infeasible within r (a merge/split branch is unreachable) -- increase match_radius_m"
    committed: Dict[Hashable, Hashable] = {}
    M: set = set()
    q: deque = deque()

    def commit(c, v):
        if v is None:
            raise ValueError(_unreach)
        if c not in committed:
            committed[c] = v
            q.append(c)

    for r in A.nodes:
        if r in committed:
            continue
        cand = A.nodes[r]["cand"]
        v_star, best = None, INF
        for v, c in cand.items():
            if c.get("forbidden"):
                continue                                        # §4.1a: never commit to a forbidden cell
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
                if head is None:
                    raise ValueError(_unreach)
                run.append(head)
            for w in run:
                M.add((c, w))
            for (p, x) in cc[head]["bpD"]:                  # predecessors (advance at the forward anchor)
                commit(p, x)
            tail = v                                        # successors: past c's own backward cover
            while _is_cover(cc[tail]["bpB"], c):
                tail = cc[tail]["bpB"][0][1]
                if tail is None:
                    raise ValueError(_unreach)
            for (s, w) in cc[tail]["bpB"]:
                commit(s, w)

    # Coverage gap-fill (docs §6b). A 1:N run recorded on the *backward* cover chain is missed by the
    # forward-only read above, leaving an uncovered target cell between two committed neighbours. Fill
    # it from the committed pivots, not the cover chains: for each source edge, cover the B-path between
    # the two pivots, assigning each still-uncovered cell to the downstream vertex it is a candidate of.
    covered = {w for (_a, w) in M}
    for pa, ch in A.edges:
        xa, yb = committed[pa], committed[ch]
        if xa == yb or not nx.has_path(B, xa, yb):
            continue
        for cell in nx.shortest_path(B, xa, yb)[1:]:            # strictly after the predecessor, up to ch
            if (cell not in covered and cell in A.nodes[ch]["cand"]
                    and not A.nodes[ch]["cand"][cell].get("forbidden")):
                M.add((ch, cell)); covered.add(cell)
    return M, committed


# ---------------------------------------------------------------------------------------
# Decision cost of a relation (docs §3) -- shared by match_dag's engine="all" and the engines
# ---------------------------------------------------------------------------------------
def _cost_of(A: nx.DiGraph, B: nx.DiGraph, M: set, alpha: float, beta: float) -> float:
    """The decision cost of a complete relation, read off ``M`` and the graphs alone (docs §3/§5):
    per vertex, the run's ENTRY cell (no B-predecessor inside the run) pays full ``E`` -- or ``beta*E``
    when some predecessor also holds it (an N:1 stall) -- and every further covered cell pays
    ``alpha*E``."""
    rows: Dict[Hashable, set] = {}
    for (a, v) in M:
        rows.setdefault(a, set()).add(v)
    C = 0.0
    for a, cells in rows.items():
        cand = A.nodes[a]["cand"]
        entries = sorted((v for v in cells if not any(x in cells for x in B.predecessors(v))), key=str)
        entry = entries[0] if entries else sorted(cells, key=str)[0]    # cycle-run fallback
        stall = any(entry in rows.get(p, ()) for p in A.predecessors(a))
        C += (beta if stall else 1.0) * cand[entry]["E"]
        for v in cells:
            if v != entry:
                C += alpha * cand[v]["E"]
    return C


# ---------------------------------------------------------------------------------------
# The vertex-level junction join (docs/dag_dtw_matching.md §10) -- cross-validation engine
# ---------------------------------------------------------------------------------------
def _reconstruct_from_sinks(A, sink_labels):
    """``(M, committed)`` from pinned sink labels by the ``bpD`` up-flood (cover chains -> run cells,
    advance lists -> every arm of a merge); ``None`` on a pin conflict, a forbidden touch, or a
    severed pointer."""
    cells: Dict[Hashable, set] = {}
    pin: Dict[Hashable, Hashable] = {}
    stack = list(sink_labels.items())
    while stack:
        a, v = stack.pop()
        if a in pin:
            if pin[a] != v:
                return None
            continue
        cand = A.nodes[a]["cand"]
        if v not in cand or cand[v].get("forbidden") or cand[v]["D"] >= INF:
            return None
        pin[a] = v
        cells.setdefault(a, set()).add(v)
        x = v
        while True:
            bp = cand[x]["bpD"]
            if len(bp) == 1 and bp[0][0] == a:                  # COVER pair -> run cell
                x = bp[0][1]
                if x is None or cand[x].get("forbidden"):
                    return None
                cells[a].add(x)
            else:
                break
        for (q, xq) in cand[x]["bpD"]:
            if xq is None:
                return None
            stack.append((q, xq))
    return {(a, c) for a, cs in cells.items() for c in cs}, pin


def _jj_induce(A, cells, path):
    """Walk ``bpD`` from the row's recorded cell at the deepest ``path`` vertex up to ``path[0]``
    (the split), selecting at each merge the arm toward the split. Returns
    ``(induced label, {vertex: cell})`` or ``None`` (forbidden / severed / no arm)."""
    idx = max((i for i, w in enumerate(path) if w in cells), default=None)
    if idx is None:
        return None
    cur_a, cur_v = path[idx], cells[path[idx]]
    walked: Dict[Hashable, Hashable] = {}
    for step in range(idx - 1, -1, -1):
        nxt = path[step]
        cand = A.nodes[cur_a]["cand"]
        x = cur_v
        while True:                                             # own cover chain to the head
            bp = cand[x]["bpD"]
            if len(bp) == 1 and bp[0][0] == cur_a:
                x = bp[0][1]
                if x is None or cand[x].get("forbidden"):
                    return None
            else:
                break
        hop = [xp for (q, xp) in cand[x]["bpD"] if q == nxt]
        if not hop or hop[0] is None:
            return None
        cur_a, cur_v = nxt, hop[0]
        if A.nodes[cur_a]["cand"][cur_v].get("forbidden"):
            return None
        walked[cur_a] = cur_v
    return cur_v, walked


def extract_join(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0):
    """**The vertex-level junction join** (docs/dag_dtw_matching.md §10) -- the CROSS-VALIDATION
    engine, kept only to validate :func:`extract_cell`; exact over the stored-history family: the
    optimal labels for all sinks and splits by a recursive table join over the split hierarchy.
    Every table is a sink-type table (label -> through-cost + pinned labels + recorded cells);
    splits are processed deepest-first; each branch's terminal is found by walking down to the
    first table-owned vertex (consumed-once: an absorbed table serves a later split through its
    recorded interior cells -- the polytree message flow, no cost division). The root table's
    minimum row is the family's decision-cost optimum; rows are tried cheapest-first and the first
    whose reconstructed ``M`` passes ``check_rules`` wins. Requires :func:`prepare` +
    :func:`forward`. Raises ``ValueError`` when no row of a component survives. Returns
    ``(M, committed)``.

    Cross-validation invariant vs :func:`extract_cell` (docs §10.2): whenever both succeed,
    ``C(M_cell) <= C(M_join)`` -- the cell join is exact over the FULL space, this one only over
    the stored-history family (intra-vertex run alternatives are frozen)."""
    M_all: set = set()
    committed_all: Dict[Hashable, Hashable] = {}
    for comp in nx.weakly_connected_components(A):
        comp = set(comp)
        tables: Dict[Hashable, dict] = {}
        owner: Dict[Hashable, Hashable] = {}
        consumed: set = set()
        alias: Dict[Hashable, Hashable] = {}

        def find(t):
            while t in alias:
                t = alias[t]
            return t

        for s_ in comp:                                         # leaf tables, one per sink
            if A.out_degree(s_) == 0:
                rows = {}
                for v, c in A.nodes[s_]["cand"].items():
                    if not c.get("forbidden") and c["D"] < INF:
                        rows[v] = (c["D"], {s_: v}, {s_: v})
                if not rows:
                    raise ValueError(f"sink {s_!r} has no usable cell -- increase match_radius_m")
                tables[s_] = rows
                owner[s_] = s_
        splits = [n for n in nx.topological_sort(A) if n in comp and A.out_degree(n) > 1]
        for U in reversed(splits):                              # deepest split first
            branches = []
            for child in A.successors(U):
                path, cur = [U, child], child
                while cur not in owner:
                    nxts = list(A.successors(cur))
                    cur = nxts[0]                               # out-degree 1 below (splits collapsed)
                    path.append(cur)
                branches.append((find(owner[cur]), path))
            per_branch = []
            for tid, path in branches:
                best_at: Dict[Hashable, tuple] = {}
                for label, (cost, cells, pins) in tables[tid].items():
                    got = _jj_induce(A, cells, path)
                    if got is None:
                        continue
                    u, walked = got
                    cur_best = best_at.get(u)
                    if (cur_best is None or cost < cur_best[0] - 1e-12 or
                            (abs(cost - cur_best[0]) <= 1e-12 and str(label) < str(cur_best[3]))):
                        best_at[u] = (cost, {**cells, **walked}, pins, label)
                per_branch.append(best_at)
            new_rows: Dict[Hashable, tuple] = {}
            for u in set.intersection(*[set(b) for b in per_branch]) if per_branch else set():
                cost, cells, pins, ok = 0.0, {}, {}, True
                for b in per_branch:
                    c_, cl_, pn_, _l = b[u]
                    cost += c_
                    for k, v in cl_.items():
                        if cells.get(k, v) != v:                # branches may overlap only consistently
                            ok = False
                            break
                        cells[k] = v
                    if not ok:
                        break
                    pins.update(pn_)
                if ok:
                    pins[U] = u
                    new_rows[u] = (cost, cells, pins)
            if not new_rows:
                raise ValueError(f"split {U!r}: no shared exit across its branches -- "
                                 "increase match_radius_m")
            tables[U] = new_rows
            for tid, path in branches:
                consumed.add(tid)
                alias[tid] = U
                for w in path:
                    owner[w] = U
            owner[U] = U
        roots = [t for t in tables if t not in consumed]
        rows = tables[roots[0]]                                 # exactly one per component
        best = None
        for u in sorted(rows, key=lambda u: (rows[u][0], str(u))):   # cheapest-first, judge decides
            cost, _cells, pins = rows[u]
            got = _reconstruct_from_sinks(A, {a: v for a, v in pins.items() if A.out_degree(a) == 0})
            if got is None:
                continue
            Mc, pin = got
            if {a for a, _ in Mc} != comp:
                continue
            v1, v2, v3 = check_rules(Mc, A, B)
            if v1 or v2 or v3:
                continue
            best = (Mc, pin)
            break
        if best is None:
            raise ValueError(f"no valid root row in component of {roots[0]!r} -- increase match_radius_m")
        M_all |= best[0]
        committed_all.update(best[1])
    return M_all, committed_all


# ---------------------------------------------------------------------------------------
# The cell-level join (docs/dag_dtw_matching.md §5) -- THE extraction, implemented as the
# per-cell backward sweep over the cell DAG (docs/cell_dag_extraction.md)
# ---------------------------------------------------------------------------------------
def _cell_reachable(A: nx.DiGraph, B: nx.DiGraph) -> set:
    """The §5.2 cell-removal pre-pass: one reverse search from ALL sink cells over the cell-move graph
    (cover reversed inside a vertex, advance/stall reversed across edges). Cells never seen cannot
    appear on any chain to a sink and are removed up front. Role-aware ``forbidden`` (§4.1a): the
    flag bars a cell from being a **run END** — a sink seed or an attachment target across an edge —
    but a run may pass THROUGH it, so the within-row cover-reversed step traverses flagged cells."""
    seen, stack = set(), []
    for X in A.nodes:
        if A.out_degree(X) == 0:
            for v, c in A.nodes[X]["cand"].items():
                if not c.get("forbidden"):
                    seen.add((X, v))
                    stack.append((X, v))
    while stack:
        X, v = stack.pop()
        cand = A.nodes[X]["cand"]
        for u in B.predecessors(v):                             # cover reversed (same vertex):
            if u in cand and (X, u) not in seen:                # through-role -- flags do not bar it
                seen.add((X, u))
                stack.append((X, u))
        for P in A.predecessors(X):                             # advance/stall reversed (to the parent)
            pc = A.nodes[P]["cand"]
            for u in [v] + list(B.predecessors(v)):
                if u in pc and not pc[u].get("forbidden") and (P, u) not in seen:
                    seen.add((P, u))
                    stack.append((P, u))
    return seen


def _pend_union(p0: dict, p1: dict):
    """Union two pending dicts; None on a separator-cell conflict; stall flags OR."""
    out = dict(p0)
    for (c, ce), flag in p1.items():
        for (c2, ce2) in out:
            if c2 == c and ce2 != ce:
                return None
        out[(c, ce)] = out.get((c, ce), False) or flag
    return out


def extract_cell(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0,
                 max_rows: int = 50000):
    """**THE extraction -- the cell-level join**, implemented as the per-cell backward sweep over
    the cell DAG (docs/cell_dag_extraction.md; semantics unchanged: docs/dag_dtw_matching.md §5).
    One reverse-topological sweep, sinks -> sources; per cell two states -- END (children attach
    at the run end) and ENTRY (the cover recursion: runs are IMPLICIT, one arc per cover edge, no
    ``run_cap``) -- each a table with one row ``(value, pending, cells)`` per pending signature,
    cheapest only. Rows are pushed to their readers' inboxes and the row is freed the same turn,
    so peak memory is the sweep frontier, not the whole table set. Merge coordination is
    consumed-once + ``pending``, with **early discharge**: a key is paid and dropped at the arms'
    first common ancestor (statically precomputed), which keeps chains of merges linear. Exact
    over the FULL cell-level space; built from ``E`` alone -- the forward table serves only as
    pruning (``forbidden`` = not a valid run END, §4.1a role-aware — runs may pass through;
    ``D < inf`` / sink-reach). The E-multiplier ledger is {1
    advance/source, beta stall, alpha cover}; a vertex's entry-E is deferred to its parent's
    connecting edge (a merge's to its discharge). ``M`` travels with the rows -- the winning
    row's cells map IS the relation, no traceback.

    Joined rows are tried cheapest-first; the first whose ``M`` passes ``check_rules`` wins.
    Raises ``ValueError`` on infeasibility (a vertex with no surviving cell -- located precisely),
    when no root row is valid, or when a table exceeds ``max_rows`` (loud, never a silent
    truncation). Requires :func:`prepare` + :func:`forward`. Returns ``(M, committed)``."""
    border = _b_order(B)
    seen = _cell_reachable(A, B)
    for X in A.nodes:
        if not any((X, v) in seen and A.nodes[X]["cand"][v]["D"] < INF
                   for v in A.nodes[X]["cand"]):
            raise ValueError(f"vertex {X!r} has no surviving cell (sink-search + D-filter) -- "
                             "increase match_radius_m")
    bsucc = {v: set(B.successors(v)) for v in B.nodes}
    bpred = {v: set(B.predecessors(v)) for v in B.nodes}
    M_all: set = set()
    committed_all: Dict[Hashable, Hashable] = {}
    for comp in nx.weakly_connected_components(A):
        comp = set(comp)
        order = [n for n in nx.topological_sort(A) if n in comp]
        absorbed_by: Dict[Hashable, Hashable] = {}              # child -> its absorbing arm
        for X in reversed(order):
            for c in sorted(A.successors(X), key=str):
                if c in comp and c not in absorbed_by:
                    absorbed_by[c] = X
        row_cells = {X: [v for v in A.nodes[X]["cand"] if (X, v) in seen] for X in comp}
        inbox: Dict[Hashable, dict] = {X: {} for X in comp}     # inbox[P][u][c] -> option rows
        roots: list = []

        # early discharge: a key (m, ce) is paid & dropped at the first swept vertex whose cone
        # contains ALL of m's arms (their common ancestors, static) -- there every arm is folded
        # into the row, so the assumption is a fact (docs/cell_dag_extraction.md §3.5).
        dischargeable: Dict[Hashable, set] = {X: set() for X in comp}
        for m in comp:
            arms = list(A.predecessors(m))
            if len(arms) < 2:
                continue
            anc = None
            for p in arms:
                s = nx.ancestors(A, p) | {p}
                anc = s if anc is None else anc & s
            for X in anc & comp:
                dischargeable[X].add(m)

        for X in reversed(order):                               # the backward sweep, row at a time
            cand = A.nodes[X]["cand"]
            cells_row = row_cells[X]
            rowset = set(cells_row)
            children = [c for c in sorted(A.successors(X), key=str) if c in comp]

            # END[(X,u)]: children attach at the run end u (cell_dag doc §3.2)
            END: Dict[Hashable, list] = {}
            my_inbox = inbox.pop(X)                             # consumed this turn, then freed
            for u in sorted(cells_row, key=lambda t: border[t]):
                if cand[u].get("forbidden"):                    # §4.1a role: not a valid run END --
                    continue                                    # (the run may still pass through u)
                combos = [(0.0, {}, {X: (u,)})]
                dead = False
                for c in children:
                    if absorbed_by.get(c) == X:                 # absorbed: options were pushed
                        opts = my_inbox.get(u, {}).get(c, [])
                    else:                                       # other arm of a merge: interface
                        opts = []
                        for ce in sorted(A.nodes[c]["cand"], key=lambda t: border[t]):
                            if (c, ce) not in seen:
                                continue
                            stall = (ce == u)
                            if not stall and ce not in bsucc[u]:
                                continue
                            opts.append((0.0, {(c, ce): stall}, {}))
                    if not opts:
                        dead = True
                        break
                    per_sig: dict = {}                          # contract per signature per fold
                    for (v0, p0, c0) in combos:
                        for (av, ap, ac) in opts:
                            p = _pend_union(p0, ap)
                            if p is None:
                                continue
                            key = frozenset(p.items())
                            cur = per_sig.get(key)
                            v = v0 + av
                            if cur is None or v < cur[0] - 1e-12:
                                per_sig[key] = (v, p, {**c0, **ac})
                    combos = list(per_sig.values())
                    if len(combos) > max_rows:
                        raise ValueError(f"cell-join table at {X!r} exceeded {max_rows} rows -- "
                                         "raise max_rows")
                    if not combos:
                        dead = True
                        break
                if not dead and dischargeable[X]:               # early discharge + re-contract
                    dis = dischargeable[X]
                    per_sig = {}
                    for (v0, p0, c0) in combos:
                        if any(mm in dis for (mm, _ce) in p0):
                            np, v = {}, v0
                            for (mm, ce), flag in p0.items():
                                if mm in dis:                   # all arms folded: pay once, drop
                                    v += (beta if flag else 1.0) * A.nodes[mm]["cand"][ce]["E"]
                                else:
                                    np[(mm, ce)] = flag
                            row = (v, np, c0)
                        else:
                            row = (v0, p0, c0)
                        key = frozenset(row[1].items())
                        cur = per_sig.get(key)
                        if cur is None or row[0] < cur[0] - 1e-12:
                            per_sig[key] = row
                    combos = list(per_sig.values())
                if not dead:
                    END[u] = combos

            # ENTRY[(X,v)]: the cover recursion (cell_dag doc §3.1) -- runs grow one arc per edge
            succ_row = {v: [w for w in bsucc[v] if w in rowset] for v in cells_row}
            TAIL: Dict[Hashable, dict] = {v: {} for v in cells_row}

            def _fill(v):
                table: dict = {}
                for row in END.get(v, []):
                    key = frozenset(row[1].items())
                    cur = table.get(key)
                    if cur is None or row[0] < cur[0] - 1e-12:
                        table[key] = row
                for w in succ_row[v]:
                    aE = alpha * cand[w]["E"]
                    for (val, pend, cells) in TAIL[w].values():
                        nv = val + aE
                        key = frozenset(pend.items())
                        cur = table.get(key)
                        if cur is None or nv < cur[0] - 1e-12:
                            table[key] = (nv, pend, {**cells, X: (v,) + cells[X]})
                if len(table) > max_rows:
                    raise ValueError(f"cell-join table at {X!r} exceeded {max_rows} rows -- "
                                     "raise max_rows")
                changed = table != TAIL[v]
                TAIL[v] = table
                return changed

            G_row = nx.DiGraph()
            G_row.add_nodes_from(cells_row)
            G_row.add_edges_from((v, w) for v in cells_row for w in succ_row[v])
            try:                                                # acyclic row: one visit per cell
                for v in reversed(list(nx.topological_sort(G_row))):
                    _fill(v)
            except nx.NetworkXUnfeasible:                       # cyclic row: relax to fixed point
                passes, changed = 0, True
                while changed:
                    changed = False
                    for v in cells_row:
                        changed |= _fill(v)
                    passes += 1
                    if passes > 2 * len(cells_row) + 4:
                        raise ValueError(f"cell-join row relaxation at {X!r} failed to converge")

            # push ENTRY rows to the readers, then the row retires (cell_dag doc §4)
            entry_cells = [e for e in cells_row if cand[e]["D"] < INF]
            if A.in_degree(X) == 0:                             # source: pay own entry, to root
                roots.append([(e, val + cand[e]["E"], pend, cells)
                              for e in sorted(entry_cells, key=lambda t: border[t])
                              for (val, pend, cells) in TAIL[e].values()])
            else:
                is_merge = A.in_degree(X) > 1
                p_row = set(row_cells[absorbed_by[X]])          # only the absorber is pushed to
                p_box = inbox[absorbed_by[X]]
                for e in entry_cells:
                    eE = cand[e]["E"]
                    for (val, pend, cells) in TAIL[e].values():
                        for u in ({e} | bpred[e]) & p_row:
                            stall = (u == e)
                            if is_merge:                        # deferred: tag, don't pay (§3.3)
                                opt = (val, {**pend, (X, e): pend.get((X, e), False) or stall},
                                       cells)
                            else:                               # priced by the edge type (§3.2)
                                opt = (val + (beta if stall else 1.0) * eE, pend, cells)
                            p_box.setdefault(u, {}).setdefault(X, []).append(opt)
            del END, TAIL, my_inbox                             # the row retires
        joined = [(0.0, {}, {})]
        for root in roots:
            folded: Dict[frozenset, tuple] = {}
            for (v0, p0, c0) in joined:
                for (_e, val, pend, cells) in root:
                    p = _pend_union(p0, pend)
                    if p is None:
                        continue
                    key = frozenset(p.items())                  # contract per pending-key: only the
                    v = v0 + val                                # pendings matter for future folds
                    if key not in folded or v < folded[key][0] - 1e-12:
                        folded[key] = (v, p, {**c0, **cells})
            joined = list(folded.values())
            if len(joined) > max_rows:
                raise ValueError(f"cell-join root join exceeded {max_rows} rows -- raise max_rows")
        finals = []
        for (val, pend, cells) in joined:
            for (c, ce), flag in pend.items():                  # pay every deferred merge entry once
                val += (beta if flag else 1.0) * A.nodes[c]["cand"][ce]["E"]
            finals.append((val, cells))
        best = None
        for val, cells in sorted(finals, key=lambda t: (t[0], str(sorted(t[1].items(), key=str)))):
            Mc = {(a, v) for a, run in cells.items() for v in run}
            if {a for a, _ in Mc} != comp:
                continue
            v1, v2, v3 = check_rules(Mc, A, B)
            if v1 or v2 or v3:
                continue                                        # judge: invalid rows are skipped
            best = (Mc, {a: run[0] for a, run in cells.items()})
            break
        if best is None:
            raise ValueError(f"cell-join: no valid root row in component of {order[0]!r} -- "
                             "increase match_radius_m")
        M_all |= best[0]
        committed_all.update(best[1])
    return M_all, committed_all


# ---------------------------------------------------------------------------------------
# Edge-table -> DiGraph conversion (the Mode-1/2 input system, docs §9)
# ---------------------------------------------------------------------------------------
def _densify(coords, step_meters: float):
    """Points along a polyline at ~``step_meters`` spacing (endpoints kept)."""
    pts = np.asarray(coords, float)
    if len(pts) < 2:
        return [tuple(map(float, p)) for p in pts]
    seg = np.diff(pts, axis=0)
    slen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(slen)])
    n = max(2, int(round(cum[-1] / max(step_meters, 1e-9))) + 1)
    out = []
    for t in np.linspace(0.0, cum[-1], n):
        i = min(int(np.searchsorted(cum, t, side="right") - 1), len(seg) - 1)
        i = max(i, 0)
        f = (t - cum[i]) / (slen[i] if slen[i] > 0 else 1.0)
        out.append((float(pts[i][0] + seg[i][0] * f), float(pts[i][1] + seg[i][1] * f)))
    return out


def edges_to_digraph(edges, step_meters: float = 5.0, snap_decimals: int = 3) -> nx.DiGraph:
    """Build the matcher's ``DiGraph`` from an edge table: ``edges`` is ``[(edge_id, coords), ...]``
    with ``coords`` a projected (meters) polyline, directed along its digitization. Each polyline is
    **densified** at ``step_meters`` (this is what supplies the subdivision — interior points on
    every real edge); shared endpoints become one junction node (coordinates snapped to
    ``snap_decimals``); every arc carries ``road_id`` (its input edge id) and ``seq`` (its position
    along that edge) so segment-mode matchings aggregate back to input-edge level. ``seq`` counts
    *arcs*, gapless, and **continues across rows that share an edge id** (a multipart geometry
    exported as several rows), so ``(road_id, seq)`` identifies an arc uniquely."""
    G = nx.DiGraph()
    node_at: Dict[tuple, int] = {}
    next_seq: Dict[Hashable, int] = {}

    def node(pt):
        key = (round(pt[0], snap_decimals), round(pt[1], snap_decimals))
        if key not in node_at:
            nid = len(node_at)
            G.add_node(nid, x=float(pt[0]), y=float(pt[1]))
            node_at[key] = nid
        return node_at[key]

    for eid, coords in edges:
        pts = _densify(coords, step_meters)
        prev = node(pts[0])
        k = next_seq.get(eid, 0)
        for p in pts[1:]:
            cur = node(p)
            if cur != prev:
                G.add_edge(prev, cur, road_id=eid, seq=k)
                k += 1
            prev = cur
        next_seq[eid] = k
    return G


def parts_from_matching(M: set, LA: nx.DiGraph, LB: nx.DiGraph) -> List[dict]:
    """Edge-level **parts** of a segment-mode matching (docs §11): one dict per maximal run of
    consecutive A-arcs (by ``seq`` along their input edge) matched to the same B input edge,
    ordered along each A edge. Re-entry into a B edge yields separate parts. The **route's own
    begin/end non-overlap** (terminal stall runs, where B stops advancing because the A edge
    extends past the B coverage) is emitted as separate rows with ``part_type`` ``"head"`` /
    ``"tail"``; matched parts are ``"match"`` and exclude the overhang pairs. Per row: the covered
    A span (``a_from_m``/``a_to_m``), pair/arc counts, mean/max midpoint drift, mean circular
    bearing difference, and the used B span with its non-overlapping ``b_head_m``/``b_tail_m``
    leftovers. Requires the line-graph nodes to carry ``road_id``/``seq`` (attached by the DuckDB
    pipeline) and ``length`` (attached by :func:`line_digraph`); raises ``ValueError`` otherwise
    (point-mode matchings have no arc bookkeeping to aggregate)."""
    def positions(L):
        by_road: Dict[Hashable, list] = {}
        for n, d in L.nodes(data=True):
            if "road_id" not in d or "seq" not in d or "length" not in d:
                raise ValueError("parts_from_matching needs 'road_id'/'seq'/'length' node "
                                 "attributes (segment-mode pipeline line graphs)")
            by_road.setdefault(d["road_id"], []).append((int(d["seq"]), float(d["length"])))
        pos, tot = {}, {}
        for rid, arcs in by_road.items():
            arcs.sort()
            c = 0.0
            for seq, ln in arcs:
                if (rid, seq) in pos:            # would silently corrupt every span on this road
                    raise ValueError(f"duplicate (road_id, seq) = ({rid!r}, {seq}) -- arc ids must "
                                     "be unique per road (see edges_to_digraph)")
                pos[(rid, seq)] = (c, c + ln)
                c += ln
            tot[rid] = c
        return pos, tot

    apos, atot = positions(LA)
    bpos, btot = positions(LB)
    _bcell: Dict[Hashable, list] = {}                # per B road: its cell lengths → median = the
    for (_rid, _seq), (_c0, _c1) in bpos.items():    # arc's own RESOLUTION (what one step is there)
        _bcell.setdefault(_rid, []).append(_c1 - _c0)
    bstep = {rid: sorted(v)[len(v) // 2] for rid, v in _bcell.items()}

    def circ(a, b):
        d = abs(float(a) - float(b)) % 360.0
        return min(d, 360.0 - d)

    pairs_by_road: Dict[Hashable, list] = {}
    for sa, sb in M:
        da, db = LA.nodes[sa], LB.nodes[sb]
        pairs_by_road.setdefault(da["road_id"], []).append(dict(
            a_seq=int(da["seq"]), dest_id=db["road_id"], b_seq=int(db["seq"]), b_node=sb,
            dist_m=math.hypot(da["x"] - db["x"], da["y"] - db["y"]),
            bear=circ(da["bearing"], db["bearing"])))

    def chain_order(group, prev_node):
        """Order one A-arc's pairs along B. (V1)-(V3) make the arc's matched B-arcs a directed
        chain in ``LB``, so walk it — a continuation of ``prev_node`` first, then successors.
        Ordering by ``dest_id`` instead would compare ids, not geometry (``"10" < "2"``)."""
        by_node = {p["b_node"]: p for p in group}
        if len(by_node) < len(group):                       # not a simple chain: keep it stable
            return sorted(group, key=lambda p: (str(p["dest_id"]), p["b_seq"]))
        starts = [n for n in by_node if not any(m in by_node for m in LB.predecessors(n))]
        if prev_node is not None:                           # continue the previous arc's B chain
            cont = ([n for n in by_node if n == prev_node]
                    or [n for n in by_node if LB.has_edge(prev_node, n)])
            starts = cont or starts
        if len(starts) != 1:
            return sorted(group, key=lambda p: (str(p["dest_id"]), p["b_seq"]))
        out, cur = [], starts[0]
        while cur is not None and cur in by_node:
            out.append(by_node.pop(cur))
            nxt = [m for m in LB.successors(cur) if m in by_node]
            cur = nxt[0] if len(nxt) == 1 else None
        if by_node:                                         # unreachable leftovers: not one chain
            return sorted(group, key=lambda p: (str(p["dest_id"]), p["b_seq"]))
        return out

    rows: List[dict] = []
    for rid, prs in sorted(pairs_by_road.items(), key=lambda kv: str(kv[0])):
        prs.sort(key=lambda p: (p["a_seq"], str(p["dest_id"]), p["b_seq"]))
        ordered, prev_node, i = [], None, 0
        while i < len(prs):                      # within one A-arc: walk B's chain, continuing first
            j = i
            while j < len(prs) and prs[j]["a_seq"] == prs[i]["a_seq"]:
                j += 1
            group = chain_order(prs[i:j], prev_node)
            ordered.extend(group)
            prev_node = group[-1]["b_node"]
            i = j

        # Route-terminal stall runs -> head/tail non-overlap rows (docs §11.1): consecutive
        # begin/end A-arcs all matched to ONE B-arc mean B is not advancing there; the surplus
        # arcs (all but the geometrically true one) are the A-side overhang.
        bkey = lambda p: (p["dest_id"], p["b_seq"])
        n = len(ordered)
        hi = 0
        while hi < n and bkey(ordered[hi]) == bkey(ordered[0]):
            hi += 1
        tj = n
        while tj > 0 and bkey(ordered[tj - 1]) == bkey(ordered[-1]):
            tj -= 1
        head: List[dict] = []
        tail: List[dict] = []
        if hi <= tj:                             # hi > tj: one single stall for the whole edge
            pre = sorted({p["a_seq"] for p in ordered[:hi]})
            if len(pre) > 1:
                head = [p for p in ordered[:hi] if p["a_seq"] != pre[-1]]
            suf = sorted({p["a_seq"] for p in ordered[tj:]})
            if len(suf) > 1:
                tail = [p for p in ordered[tj:] if p["a_seq"] != suf[0]]
        core = ordered[len(head):n - len(tail)]

        runs: List[list] = []
        for p in core:                           # split the core into maximal same-dest runs
            if runs and runs[-1][-1]["dest_id"] == p["dest_id"]:
                runs[-1].append(p)
            else:
                runs.append([p])

        # The rows partition the A edge: an arc whose pairs straddle two runs (a mid-arc advance
        # across a B junction) belongs to the earlier one, so a running cursor sets every a_from.
        emit = ([(head, "head")] if head else []) + [(r, "match") for r in runs] \
            + ([(tail, "tail")] if tail else [])
        cursor = min(apos[(rid, p["a_seq"])][0] for p in ordered) if ordered else 0.0
        for k, (run, ptype) in enumerate(emit, start=1):
            dest = run[0]["dest_id"]
            a_from = cursor
            a_to = max(cursor, max(apos[(rid, p["a_seq"])][1] for p in run))
            cursor = a_to
            claim = run
            if ptype == "match":
                # RESOLUTION TRIM — the B-side dual of the head/tail A-overhang (docs §11.1). At a
                # route hand-off the entry vertex "walks" along the arc via a 1:N coverage run from
                # the junction cell to where the edge actually lies; those walked-through cells are
                # route CONNECTIVITY, not correspondence — leaving them in is exactly how two lines
                # tiling one street both claimed the whole arc. Per A-vertex: a covered cell is a
                # CLAIM only if dist² ≤ dmin² + step² (step = the arc's own median cell length) —
                # along-track excess of at most one step over the vertex's nearest cell; the lateral
                # offset (real A↔B displacement) cancels, so offset streets are untouched.
                step2 = bstep[dest] ** 2
                by_a: Dict[int, list] = {}
                for p in run:
                    by_a.setdefault(p["a_seq"], []).append(p)
                claim = []
                for cells in by_a.values():
                    dmin = min(c["dist_m"] for c in cells)
                    claim.extend(c for c in cells if c["dist_m"] ** 2 <= dmin * dmin + step2)
            b_segs = tuple(sorted({int(p["b_seq"]) for p in claim}))
            b_from = min(bpos[(dest, p["b_seq"])][0] for p in claim)
            b_to = max(bpos[(dest, p["b_seq"])][1] for p in claim)
            dists = [p["dist_m"] for p in claim]
            rows.append(dict(
                source_id=rid, part=k, part_type=ptype, dest_id=dest,
                a_from_m=a_from, a_to_m=a_to, a_len_m=a_to - a_from,
                a_pct=100.0 * (a_to - a_from) / atot[rid] if atot[rid] > 0 else 0.0,
                n_pairs=len(claim),
                n_a_arcs=len({p["a_seq"] for p in claim}),
                n_b_arcs=len(b_segs),
                drift_m=float(np.mean(dists)), drift_max_m=float(max(dists)),
                bearing_diff_deg=float(np.mean([p["bear"] for p in claim])),
                b_from_m=b_from, b_to_m=b_to, b_len_m=btot[dest],
                b_head_m=b_from, b_tail_m=btot[dest] - b_to,
                # The B segments this run ACTUALLY matched. b_from/b_to are only their min/max, so a
                # run with a gap reads as claiming the whole span between — callers that need the real
                # claim (e.g. per-segment exclusivity) must use this, not the extent.
                b_segs=b_segs))
    return rows


# ---------------------------------------------------------------------------------------
# One-call pipeline entry point
# ---------------------------------------------------------------------------------------
class _Budget:
    """Wall-clock and RSS ceiling for one extraction, enforced by a repeating ``SIGALRM``.

    A periodic signal is the only mechanism that covers EVERY engine without instrumenting each
    one's inner loops -- the handler runs in the main thread between bytecodes and raises there.
    Line 100935 of vancouver_city ground for 30 s (cell) and 67 s (profiled) before dying of
    MemoryError; a pipeline over thousands of edges wants "no match for this one" in a second.

    THREADING: works in any **process** -- a joblib/multiprocessing worker runs its task in that
    process's own main thread, so SIGALRM is delivered normally, and RSS then measures that worker
    alone. Verified on line 100935 with the loky backend: budget fired at 3.07 s. It does NOT work in
    a worker **thread** -- CPython only ever delivers signals to the main thread, so the guard
    degrades to a no-op there (measured: a worker thread ran 25 s past a 3 s budget). Making threads
    work would need ctypes async-exception injection or cooperative checks inside every engine's
    inner loops; `max_work` is thread-safe and already catches the hopeless cases, so neither is
    implemented.
    """

    def __init__(self, seconds=None, memory_mb=None, tick=0.25):
        self.seconds, self.memory_mb, self.tick = seconds, memory_mb, tick
        self.t0 = self.rss0 = self.prev = None
        self.armed = False

    @staticmethod
    def _rss_mb():
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
        except Exception:                                       # noqa: BLE001 - not Linux
            return 0.0

    def __enter__(self):
        if (self.seconds is None and self.memory_mb is None) or \
           threading.current_thread() is not threading.main_thread():
            return self
        self.t0, self.rss0 = time.monotonic(), self._rss_mb()
        self.prev = signal.signal(signal.SIGALRM, self._check)
        signal.setitimer(signal.ITIMER_REAL, self.tick, self.tick)
        self.armed = True
        return self

    def __exit__(self, *exc):
        if self.armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.prev)
            self.armed = False
        return False

    def _check(self, _sig, _frm):
        if self.seconds is not None and time.monotonic() - self.t0 > self.seconds:
            raise ValueError(f"extraction exceeded max_seconds={self.seconds:g}s -- source too "
                             "complex; reduce it or raise max_seconds")
        if self.memory_mb is not None:
            used = self._rss_mb() - self.rss0
            if used > self.memory_mb:
                raise ValueError(f"extraction exceeded max_memory_mb={self.memory_mb:g} "
                                 f"(used {used:.0f} MB) -- source too complex; reduce it or raise "
                                 "max_memory_mb")


def extract_by_engine(A: nx.DiGraph, B: nx.DiGraph, alpha: float, beta: float, engine: str,
                      max_work: float = 5e7, max_seconds: float = 60.0,
                      max_memory_mb: float = 2000.0):
    """Run the chosen extraction on an already ``prepare``d + ``forward``ed pair. ``(M, committed)``.

    THE single dispatch point -- :func:`match_dag` and ``DuckDBMapMatcher.match_dag`` both call it,
    so the engine set cannot drift between them (it did once: the matcher kept a stale copy that did
    not know about ``"auto"``).

    ``"auto"`` picks on the source's **profile width** (pure topology, no cell work): ``<= 2`` uses
    the profiled engine, which is where it wins -- reconvergent sources, every real conflation edge;
    wider means splits nested without merges between them, where ``"cell"`` is faster.
    """
    if engine in ("auto", "profiled", "rebase"):
        from .profiled import (profiled_width, merge_pressure, predict_work,       # lazy: cycle
                               engine_costs, forward_profiled, extract_profiled, match_rebased)
        if engine == "auto":
            # TWO independent pressures, so the choice is 2-D, not a line:
            #   W  = nested-split pressure -- kills the PROFILED key (width grows with depth)
            #   Mo = concurrently-open merges -- kills CELL's pending (its product)
            #
            # Re-basing wins only when Mo >= W: when EVERY nested split's branch rejoins, so both
            # other engines are maximally loaded at once. Below that there is always a merge-free
            # path through the nesting and `cell` stays cheap -- measured by sweeping the two axes
            # independently (report/probe_braid.py): at W=5, `cell` wins at Mo=0..4 (0.003-0.34s)
            # and only loses at Mo=5 (2.07s vs rebase 0.29s). A fixed threshold such as Mo>=2
            # misroutes 4 of 11 swept cases.
            W = profiled_width(A)
            engine = ("profiled" if W <= 2
                      else "rebase" if merge_pressure(A) >= W
                      else "cell")

            # REFUSE FAST -- but only on the estimate of the engine actually chosen. Every engine's
            # table size is readable off the forward table in <1 ms, and a source beyond them grinds
            # for 30-70 s before dying of MemoryError, useless in a pipeline over thousands of edges.
            #
            # Gating on min(cell, profiled) was WRONG: neither describes re-basing, whose key resets
            # at every split. Hourglass line 100935 predicts 7.9e8 (cell) and 1.3e9 (profiled) and
            # was refused for it -- while re-basing predicts 3.3e5 and answers in 0.24 s. If the
            # structural pick is over budget but another engine is not, take that one; refuse only
            # when all three are. Bounds are UPPER -- reachability prunes them hard -- so the gate
            # sits well above the workable range, not at it.
            if max_work is not None:
                # BOTH PHASES. An engine's forward table and its extraction have independent memory
                # profiles, and a phase that fits does not excuse one that does not. Re-basing is why
                # this matters: its extraction is cheap BECAUSE the forward pass already paid for
                # SEG, so gating on extraction alone understates it 40x (docs §9.1).
                #
                # max_work must sit ABOVE the loosest workable prediction, not at it -- these are
                # upper bounds and reachability prunes them 10-300x. Measured on the five real
                # hourglass edges, the CHOSEN engine predicts up to 1.9e7 (100935: 18,572,678 rows
                # against 59,274 actual SEG entries). Hence 5e7. The hopeless are still nowhere
                # near: 100935's unchosen engines predict 7.9e8 (cell) and 2.6e12 (profiled).
                costs = engine_costs(A)
                worst = {e: max(f, x) for e, (f, x) in costs.items()}
                if worst[engine] > max_work:
                    engine = min(worst, key=worst.get)       # fall back to the cheapest estimate
                if worst[engine] > max_work:
                    detail = ", ".join(f"{e} ~{f:,.0f}/{x:,.0f}" for e, (f, x) in sorted(costs.items()))
                    raise ValueError(
                        f"source too complex for exact matching: cheapest engine "
                        f"{min(worst, key=worst.get)!r} would enumerate ~{min(worst.values()):,.0f} "
                        f"rows (limit {max_work:,.0f}; forward/extract rows per engine: {detail}). "
                        "Reduce the source (fewer hops / shorter stubs) or raise max_work to try "
                        "anyway.")
        with _Budget(max_seconds, max_memory_mb):
            if engine == "rebase":
                return match_rebased(A, B, alpha, beta)
            if engine == "profiled":
                forward_profiled(A, B, alpha, beta)
                M, com, _cost = extract_profiled(A, B, alpha, beta)
                return M, com
    engines = {"cell": extract_cell, "join": extract_join}
    if engine in engines:
        with _Budget(max_seconds, max_memory_mb):
            return engines[engine](A, B, alpha, beta)
    if engine == "all":                                         # the cross-validating choice
        best = None
        for fn in (extract_cell, extract_join):
            try:
                M, com = fn(A, B, alpha, beta)
            except ValueError:
                continue
            c = _cost_of(A, B, M, alpha, beta)
            if best is None or c < best[0] - 1e-12:
                best = (c, M, com)
        if best is None:
            raise ValueError("both extraction engines infeasible -- increase match_radius_m")
        return best[1], best[2]
    raise ValueError(f"unknown engine {engine!r} "
                     "(use 'auto', 'profiled', 'rebase', 'cell', 'join' or 'all')")


def match_dag(A: nx.DiGraph, B: nx.DiGraph, r: float = 20.0, alpha: float = 1.0,
              beta: float = 1.0, mode: str = "point", engine: str = "auto",
              bearing_weight: float = 1.0, k_min: int = 1, max_work: float = 5e7,
              max_seconds: float = 60.0, max_memory_mb: float = 2000.0):
    """One-call DAG-DTW pipeline (docs/dag_dtw_matching.md): ``prepare`` -> ``forward`` (the
    coupled pass) -> extraction, on ``A``/``B`` (``mode="point"``, ``M`` over vertices) or on their
    directed line graphs (``mode="segment"``, ``M`` over arcs — nodes are ``(u, v)`` edge tuples of
    the originals, so the result is self-describing).

    ``engine``: ``"auto"`` (default) dispatches on the source's **profile width** — a pure-topology
    number available before any cell work. Width ``<= 2`` uses ``"profiled"``
    (docs/profiled_forward_table.md), which on real conflation sources is 60-3000x faster than
    ``"cell"`` and fixes the V3 phantom the forward table leaves at splits; wider sources are nested
    splits, where ``"cell"`` is faster and the profiled key grows with depth. Explicit choices:
    ``"profiled"``, ``"rebase"`` (the re-based variant — costs measured *since the last split*, so
    the key stays width 1 on pure out-trees where ``"profiled"`` grows with depth; slower elsewhere,
    docs/profiled_forward_table.md §8), ``"cell"`` (the cell-level join — exact over the full
    space), ``"join"`` (the vertex-level junction join — the cross-validation engine, docs §10), or
    ``"all"`` — run both classic engines and return the **cheapest valid** matching. Weights: ``alpha ∈ (0, 1]``, ``beta ∈ [1, ∞)`` (docs §3). The source may be any
    **subdivided DAG** (a tree is the special case); on reconvergent sources only the ``"cell"``
    engine carries the exactness claim (docs/dag_dtw_matching.md §7, §10.2). Returns
    ``(M, committed)``; raises ``ValueError`` on infeasibility (increase ``r``)."""
    if mode == "segment":
        A2, B2 = line_digraph(A), line_digraph(B)
    elif mode == "point":
        A2, B2 = A, B
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'point' or 'segment')")
    prepare(A2, B2, r=r, k_min=k_min, bearing_weight=bearing_weight)
    forward(A2, B2, alpha=alpha, beta=beta)
    return extract_by_engine(A2, B2, alpha, beta, engine, max_work,
                             max_seconds, max_memory_mb)



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


def _advance_anchor(A: nx.DiGraph, c: Hashable, v: Hashable, bpkey: str) -> Hashable:
    """Walk ``c``'s own COVER chain from cell ``v`` to where its ADVANCE pointers live (the run's
    start for ``bpD``, its end for ``bpB``). A COVER step is a single same-source pair ``[(c, ·)]``."""
    x = v
    while _is_cover(A.nodes[c]["cand"][x][bpkey], c):
        x = A.nodes[c]["cand"][x][bpkey][0][1]
    return x


def check_reciprocity(A: nx.DiGraph, committed: Dict[Hashable, Hashable]):
    """Cross-table agreement on the committed matching (docs §6b): every source edge the FORWARD table
    threads must be threaded identically by the BACKWARD table. Each committed vertex ``c`` connects to
    its predecessors at its run-START ``head(c)`` (``bpD`` advance anchor) and to its successors at its
    run-END ``tail(c)`` (``bpB`` advance anchor) -- the pivot ``committed[c]`` walked along ``c``'s own
    COVER chain. The invariant, over every source edge ``p → c``::

        (p, tail(p)) ∈ bpD[c][head(c)]   ⟺   (c, head(c)) ∈ bpB[p][tail(p)]

    i.e. ``p`` feeds ``c`` from ``p``'s run-end, and ``c`` continues ``p`` back at ``c``'s run-start --
    reciprocally, at the same cells. (No coverage ⇒ every anchor equals the pivot.) Same-source COVER
    pairs are consumed by the anchor walk, not tested (they have no backward mirror). Returns a list of
    offending ``(p, c, reason)`` edges -- empty iff the two tables agree on the optimum. NOT valid off
    ``M``: table-wide reciprocity is false (see docs §6b)."""
    head = {c: _advance_anchor(A, c, v, "bpD") for c, v in committed.items()}   # run start (fwd advance)
    tail = {c: _advance_anchor(A, c, v, "bpB") for c, v in committed.items()}   # run end   (bwd advance)
    bad = []
    for c in committed:
        for (p, x) in A.nodes[c]["cand"][head[c]]["bpD"]:       # forward: c fed by predecessor p
            if p == c or p not in committed:                    # COVER guard / p outside this component
                continue
            if x != tail[p]:
                bad.append((p, c, f"bpD[{c}][{head[c]}] pins {p}@{x}, but {p}'s run-end anchor is @{tail[p]}"))
            if (c, head[c]) not in A.nodes[p]["cand"][tail[p]]["bpB"]:
                bad.append((p, c, f"forward {p}->{c} unmirrored: ({c},{head[c]}) not in bpB[{p}][{tail[p]}]"))
        for (s, w) in A.nodes[c]["cand"][tail[c]]["bpB"]:       # backward: c continues into successor s
            if s == c or s not in committed:
                continue
            if w != head[s]:
                bad.append((c, s, f"bpB[{c}][{tail[c]}] pins {s}@{w}, but {s}'s run-start anchor is @{head[s]}"))
            if (c, tail[c]) not in A.nodes[s]["cand"][head[s]]["bpD"]:
                bad.append((c, s, f"backward {c}->{s} unmirrored: ({c},{tail[c]}) not in bpD[{s}][{head[s]}]"))
    return bad


def _reach(A: nx.DiGraph, start, bpkey: str):
    """Walk ``bpkey`` back-pointers cell -> cell from ``start``, branching at every entry. Return
    ``(terminals, hit_none)``: ``terminals`` = the vertex component of every terminal cell reached (an
    empty back-pointer list -- a source for ``bpD``, a sink for ``bpB``); ``hit_none`` flags a severed
    path (a ``None`` cell reference). Cover pointers (same vertex) and advances (a new vertex) are both
    just cell -> cell moves; a vertex is tallied only where the walk terminates (docs §6c)."""
    reached, seen, hit_none = set(), set(), False
    stack = [start]
    while stack:
        a, v = stack.pop()
        if (a, v) in seen:
            continue
        seen.add((a, v))
        bp = A.nodes[a]["cand"][v][bpkey]
        if not bp:
            reached.add(a)                                  # terminal cell: a is a source (D) / sink (B)
            continue
        for (a2, v2) in bp:
            if v2 is None:
                hit_none = True                             # severed path
            else:
                stack.append((a2, v2))
    return reached, hit_none


def check_reachability(A: nx.DiGraph, which: str = "D"):
    """Per-table reachability (docs §6c): a table's back-pointers must reconstruct the tree's own
    source <-> sink reachability. ``which="D"`` -- from every finite cell of every SINK, walking ``bpD``
    (upstream, branching at each predecessor) must reach **exactly** that sink's ancestor sources.
    ``which="B"`` -- from every finite cell of every SOURCE, walking ``bpB`` (downstream) must reach
    exactly that source's descendant sinks. Returns invalid cells as ``(vertex, cell, reason)`` (empty ⇒
    the table is sound). ``∞``-cost cells are infeasible by construction and are skipped. Works on ``A``
    (point) or ``line_digraph(A)`` (segment)."""
    key = "D" if which == "D" else "B"
    bpkey = "bpD" if which == "D" else "bpB"
    sources = {n for n in A.nodes if A.in_degree(n) == 0}
    sinks = {n for n in A.nodes if A.out_degree(n) == 0}
    if which == "D":
        ends = sinks
        expected = {t: (nx.ancestors(A, t) | {t}) & sources for t in sinks}
    else:
        ends = sources
        expected = {s: (nx.descendants(A, s) | {s}) & sinks for s in sources}
    bad = []
    for a in ends:
        want = expected[a]
        for v, c in A.nodes[a]["cand"].items():
            if math.isinf(c[key]):                          # infeasible cell -- expected not to reach
                continue
            reached, hit_none = _reach(A, (a, v), bpkey)
            if hit_none:
                bad.append((a, v, "severed: back-pointer to a None cell"))
            elif reached != want:
                bad.append((a, v, f"reached {sorted(map(str, reached))} != required {sorted(map(str, want))}"))
    return bad


def _one_sided(A: nx.DiGraph, ends, key: str, bpkey: str) -> set:
    """The matching implied by ONE table on its own: seed each end (sinks for the forward table, sources
    for the backward) at its arg-min cell and follow that table's back-pointers, unioning the walks
    (docs §6d)."""
    M: set = set()
    for e in ends:
        cand = A.nodes[e]["cand"]
        finite = [v for v in cand if math.isfinite(cand[v][key])]
        if not finite:
            continue
        vstar = min(finite, key=lambda v: cand[v][key])
        stack, seen = [(e, vstar)], set()
        while stack:
            a, v = stack.pop()
            if (a, v) in seen:
                continue
            seen.add((a, v)); M.add((a, v))
            for (a2, v2) in A.nodes[a]["cand"][v][bpkey]:
                if v2 is not None:
                    stack.append((a2, v2))
    return M


def check_forward_v3(A: nx.DiGraph, B: nx.DiGraph):
    """Read the FORWARD table on its own (seed each sink at its arg-min ``D``, follow ``bpD``) and test
    **V3** (the split rule). The forward pass couples **merges** (V2) but is *optimistic at splits*, so
    its own matching **can violate V3 — by design** (main §4.2); this surfaces exactly where. Returns the
    V3-violating ``(a, v)`` pairs (empty ⇒ the forward table is also split-consistent on this input).
    Mirror of :func:`check_backward_v2`."""
    sinks = [n for n in A.nodes if A.out_degree(n) == 0]
    return check_rules(_one_sided(A, sinks, "D", "bpD"), A, B)[2]


def check_backward_v2(A: nx.DiGraph, B: nx.DiGraph):
    """Read the BACKWARD table on its own (seed each source at its arg-min ``B``, follow ``bpB``) and test
    **V2** (the merge rule) — the mirror of :func:`check_forward_v3`. Backward couples **splits** (V3),
    is optimistic at merges. Returns the V2-violating pairs."""
    sources = [n for n in A.nodes if A.in_degree(n) == 0]
    return check_rules(_one_sided(A, sources, "B", "bpB"), A, B)[1]


def check_split_exits(A: nx.DiGraph, B: nx.DiGraph):
    """The §4.1a invariant, after :func:`forward`: for every split, (i) at least one exit cell
    survives (non-``forbidden``), (ii) **every** surviving exit is **usable** by **every** child —
    the child can stall on it or advance out of it (:func:`_feasible_links`; feasibility, not
    optimality — a child's cheapest row not *linking* an exit is legitimate, the extraction chooses
    among survivors), and (iii) no child's row links to a forbidden cell. Returns the violations
    ``[(split, exit_or_None, child, why)]`` — empty ⇒ the forward table's split structure is
    (V3)-sound. *(Deliberately NOT* :func:`check_forward_v3`: *its independent per-sink decode
    wrongly flags two different-but-valid surviving options; multiple surviving exits are
    legitimate, docs §4.1a.)*"""
    bad = []
    for s in A.nodes:
        children = list(A.successors(s))
        if len(children) < 2:
            continue
        pc = A.nodes[s]["cand"]
        survivors = {v for v, cell in pc.items() if not cell.get("forbidden")}
        if not survivors:
            bad.append((s, None, None, "no surviving exit"))
            continue
        forbidden = set(pc) - survivors
        for ch in children:
            usable = _feasible_links(A, B, ch, s)
            for v in survivors - usable:
                bad.append((s, v, ch, "surviving exit unusable by child"))
            for v in _links(A, ch, s) & forbidden:
                bad.append((s, v, ch, "child links a forbidden cell"))
    return bad


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
        M, committed = extract_two_table(GA, GB)
        v1, v2, v3 = check_rules(M, GA, GB)
        v4 = [a for a in GA.nodes if not any(x == a for (x, _w) in M)]     # every source vertex covered?
        recip = check_reciprocity(GA, committed)                          # §6b cross-table agreement
        ok = not (v1 or v2 or v3 or v4)
        am = {a: sorted((str(w) for (x, w) in M if x == a)) for a in GA.nodes}
        print(f"\n{name}: |M|={len(M)}  valid(V1-V4)={ok}  reciprocity={'AGREE' if not recip else recip}"
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

    # reconvergent sources are SUPPORTED (docs §7): a subdivided diamond matches end-to-end
    An = {"S": (0, 0), "s1": (4, 0), "J": (8, 0), "x": (12, 3), "z": (12, -3),
          "m": (16, 0), "t1": (20, 0), "T": (24, 0)}
    Ae = [("S", "s1"), ("s1", "J"), ("J", "x"), ("J", "z"), ("x", "m"), ("z", "m"),
          ("m", "t1"), ("t1", "T")]
    Ad = digraph(An, Ae)
    Bd = digraph({k + "'": (v[0], v[1] + 0.4) for k, v in An.items()},
                 [(a + "'", b + "'") for a, b in Ae])
    M, _ = match_dag(Ad, Bd, r=4.0)
    print(f"\ndiamond source matched: {len(M)} pairs (reconvergent DAGs are supported)")

    # the guard that remains: a directed CYCLE in the source is rejected
    try:
        prepare(digraph({0: (0, 0), 1: (10, 0)}, [(0, 1), (1, 0)]), B, r=20.0)
        print("ERROR: cyclic source was NOT rejected")
    except NotADAG as e:
        print(f"cyclic source correctly rejected: {e}")
