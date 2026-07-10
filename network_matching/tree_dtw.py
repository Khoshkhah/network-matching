"""Tree-DTW -- exact matcher of a directed source tree to a directed network, on networkx
(spec: ``docs/tree_dtw_matching.md``; the implementation is documented there as Parts 1-6).

Both the source tree ``A`` and the target network ``B`` are plain ``networkx.DiGraph`` objects:
a **vertex** carries float coordinates ``x``/``y``; an **edge** is a directed segment; a **junction**
is just a vertex (split = out-degree > 1, merge = in-degree > 1). No road ids, no coincident vertices,
no stitches.

Parts (each independently verifiable): representation + radius-gated candidates (Part 1), emission
(Part 2), forward ``D`` (Part 3), backward ``B`` (Part 4), extraction (Part 5), validation (Part 6).
Segment mode = the same parts run on the directed line graphs ``L(A)``, ``L(B)``.
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
                       "forbidden": False}                  # §4.1a: a forbidden cell takes no pointer
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
    A cell whose ``forbidden`` flag is set (§4.1a) is skipped everywhere a neighbour cell is *linked to*
    — as an advance source, a stall source, or a same-row coverage source — so no back-pointer is ever
    created to it; the cell's own value is still computed. With no flags set (plain :func:`forward` /
    :func:`backward`) the behaviour is byte-for-byte the unconstrained recurrence."""
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
    # (§4.1a) may not be a coverage SOURCE -- no pointer [(a, v)] may be created to it.
    D = dict(base)
    bp = dict(base_bp)
    changed = True
    while changed:
        changed = False
        for v in cand:
            dv = D[v]
            if dv == INF or cand[v].get("forbidden"):
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


def forward(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """Fill the forward table ``D`` and back-pointers ``bpD`` on every A-vertex's candidate table
    (docs §3). Requires :func:`prepare` to have run. Returns ``A`` (mutated in place)."""
    order = list(nx.topological_sort(A))
    _pass(A, B, order, A.predecessors, A.successors, B.predecessors, B.successors,
          "D", "bpD", alpha, beta, _b_order(B))
    return A


def backward(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """Fill the backward table ``B`` and back-pointers ``bpB`` (docs §4) — the identical three-way
    ``min`` with A and B **reversed**: sum over **successors**, ``step`` from a B-**successor**, split
    factor = **in**-degree, swept in **reverse** topological order. Same ``α``/``β``, same emission.
    Requires :func:`prepare`. Returns ``A`` (mutated in place)."""
    order = list(reversed(list(nx.topological_sort(A))))
    _pass(A, B, order, A.successors, A.predecessors, B.successors, B.predecessors,
          "B", "bpB", alpha, beta, _b_order(B))
    return A


def forward_v3(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0) -> nx.DiGraph:
    """Fill the forward table ``D``/``bpD`` **with the split (V3) coupling** (docs §4.1a), sweeping A in
    the §4.0 longest-path layer order: a split's children are all built — grouped, one layer — before
    anything downstream reads their rows. As each child is built (**including the first**), every cell
    of the split it does *not* link to is marked ``forbidden`` — dead as a pointer target for ALL
    siblings, past and future, in any pass; already-built siblings that leaned on a newly-forbidden exit
    get a **whole-row rebuild** under the current flags, iterating to a fixed point. At the fixed point
    every surviving (non-forbidden) exit cell of every split is linked by ALL its children —
    :func:`check_split_exits` verifies exactly this. Multiple surviving exits are legitimate (the single
    one is chosen at extraction, which never commits a vertex to a forbidden cell).

    Plain :func:`forward` remains the uncoupled §4.1 sweep; this pass owns the ``forbidden`` flags and
    resets them first. **Run before** :func:`backward`: the backward pass reads the flags, so its
    pointers never target a forbidden cell (the reverse order lets the unconstrained backward table
    point into cells forbidden afterwards). Requires :func:`prepare`. Raises ``ValueError`` if a split
    is left with no surviving exit (no V3-valid warping within ``match_radius_m``) or if A is not
    subdivided (a split's children spanning layers — add an interior point on every real edge, §4.0)."""
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
        _couple(A, a, built, refill)
    return A


def _links(A: nx.DiGraph, c: Hashable, p: Hashable) -> set:
    """The cells of ``p`` that ``c``'s finite forward cells link to — the advance/stall pairs ``(p, x)``
    in ``bpD`` (severed ``None`` references and same-source COVER pairs excluded)."""
    return {x for cell in A.nodes[c]["cand"].values() if cell["D"] < INF
            for (q, x) in cell["bpD"] if q == p and x is not None}


def _couple(A: nx.DiGraph, trigger: Hashable, built: set, refill) -> None:
    """The §4.1a forbid-and-rebuild step, run right after ``trigger``'s row is built. For each split
    parent ``p`` of ``trigger``: mark forbidden every non-forbidden exit cell of ``p`` that ``trigger``
    does not link to; rebuild (whole row) every already-built sibling that linked a newly-forbidden
    exit; re-examine rebuilt rows. The forbidden set grows monotonically, so this reaches a fixed point
    in at most ``|cand(p)|`` rounds; at the fixed point every surviving exit is linked by all built
    children. Raises the feasibility ``ValueError`` if a split's exits empty out."""
    work = [trigger]
    while work:
        c = work.pop()
        for p in A.predecessors(c):
            if len(list(A.successors(p))) < 2:                  # V3 only bites at a split
                continue
            pc = A.nodes[p]["cand"]
            linked = _links(A, c, p)
            newly = [v for v, cell in pc.items() if not cell["forbidden"] and v not in linked]
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
                if _links(A, sib, p) & newset:                  # sib leaned on a now-dead exit
                    refill(sib)                                 # whole-row rebuild (docs §4.1a step 3)
                    work.append(sib)                            # its links changed -> re-examine


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

    # Coverage gap-fill (§8.6). A 1:N run recorded on the *backward* cover chain is missed by the
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


def check_split_exits(A: nx.DiGraph):
    """The §4.1a invariant, after :func:`forward_v3`: for every split, (i) at least one exit cell
    survives (non-``forbidden``), (ii) **every** child links to **every** surviving exit, and (iii) no
    child links to a forbidden cell. Returns the violations ``[(split, exit_or_None, child, why)]`` —
    empty ⇒ the forward table's split structure is (V3)-consistent. *(Deliberately NOT*
    :func:`check_forward_v3`: *its independent per-sink decode wrongly flags two different-but-valid
    surviving options; multiple surviving exits are legitimate, docs §4.1a.)*"""
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
        for ch in children:
            linked = _links(A, ch, s)
            for v in survivors - linked:
                bad.append((s, v, ch, "surviving exit not linked by child"))
            for v in linked - survivors:
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
        M, committed = extract(GA, GB)
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

    # tree guard: a diamond (reconvergence) must be rejected
    try:
        prepare(digraph({0: (0, 0), 1: (1, 1), 2: (1, -1), 3: (2, 0)},
                        [(0, 1), (0, 2), (1, 3), (2, 3)]), B, r=20.0)
        print("\nERROR: diamond was NOT rejected")
    except NotATree as e:
        print(f"\ndiamond correctly rejected: {e}")
