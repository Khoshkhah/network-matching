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
    :func:`extract_cell` carries the exactness claim (docs/dag_dtw_matching.md §10.4)."""
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
    cell of the split it does *not* link to is marked ``forbidden`` — dead as a pointer target for ALL
    siblings, past and future, in any pass; already-built siblings that leaned on a newly-forbidden exit
    get a **whole-row rebuild** under the current flags, iterating to a fixed point. At the fixed point
    every surviving (non-forbidden) exit cell of every split is linked by ALL its children —
    :func:`check_split_exits` verifies exactly this. Multiple surviving exits are legitimate (the single
    one is chosen at extraction, which never commits a vertex to a forbidden cell).

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


def extract_two_table(A: nx.DiGraph, B: nx.DiGraph):
    """The two-table traceback (docs §5) — the PREVIOUS default extraction, kept for the §6b
    cross-table diagnostics (:func:`check_reciprocity` compares its committed matching against both
    tables) and for comparison; the default is now the forward-only :func:`extract`. Seed any
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
# The extraction (docs §5) -- branching exploration over the forward table, valid-only judge
# ---------------------------------------------------------------------------------------
def _reverse_bpD(A: nx.DiGraph) -> Dict[tuple, list]:
    """The transpose ``R`` of the forward back-pointers: ``R[(p, x)]`` lists every finite cell
    ``(c, w)`` with ``(p, x) in bpD[c][w]`` -- the DOWN-walk structure (same-vertex entries are the
    reverse cover chain, other-vertex entries the child entry cells)."""
    R: Dict[tuple, list] = {}
    for c in A.nodes:
        for w, cell in A.nodes[c]["cand"].items():
            if cell["D"] >= INF:
                continue
            for (p, x) in cell["bpD"]:
                if x is not None:
                    R.setdefault((p, x), []).append((c, w))
    return R


def _pull_run(A: nx.DiGraph, st: dict, p: Hashable, y: Hashable) -> bool:
    """Pull ``p``'s cover chain from cell ``y`` back to its already-known run cells (docs §5: runs
    are not guessed -- the cells a connection references force exactly the chain between them).
    False when the chain touches a forbidden cell or never meets the run."""
    cc = A.nodes[p]["cand"]
    seen = []
    x = y
    while x not in st["runs"][p]:
        if x is None or x not in cc or cc[x].get("forbidden"):
            return False
        seen.append(x)
        bp = cc[x]["bpD"]
        if len(bp) == 1 and bp[0][0] == p:                      # COVER pair -> one B-arc back
            x = bp[0][1]
        else:
            return False                                        # ran out of chain before meeting the run
    st["runs"][p].update(seen)
    return True


def _explore_label(A: nx.DiGraph, R: Dict[tuple, list], comp: set, a0: Hashable, v0: Hashable,
                   border: Dict[Hashable, int], cap: int) -> list:
    """All complete candidate relations for ONE anchor label (docs §5). Deterministic moves are
    taken; every genuine choice point -- a child vertex with several entry cells -- **branches**,
    one continuation per option, every alternative kept alive until the judge. Touching a forbidden
    cell (or a severed pointer) kills that branch only. Returns ``[(M, committed), ...]``; empty
    when every branch dies."""
    from collections import deque
    forb = lambda a, v: v not in A.nodes[a]["cand"] or A.nodes[a]["cand"][v].get("forbidden")
    fin = lambda a, v: A.nodes[a]["cand"][v]["D"] < INF
    if forb(a0, v0) or not fin(a0, v0):
        return []
    states = [{"committed": {a0: v0}, "runs": {a0: {v0}}, "q": deque([a0])}]
    out, processed = [], 0
    seen_states: set = set()                                    # dedupe: same committed + same pending

    def copy(st):
        return {"committed": dict(st["committed"]),
                "runs": {k: set(v) for k, v in st["runs"].items()},
                "q": deque(st["q"])}

    while states:
        st = states.pop()
        skey = (frozenset(st["committed"].items()), frozenset(st["q"]))
        if skey in seen_states:                                 # identical future -- already explored
            continue
        seen_states.add(skey)
        processed += 1
        if processed > cap:
            raise ValueError(f"extraction branching exceeded {cap} states -- raise max_states")
        dead = False
        while st["q"] and not dead:
            c = st["q"].popleft()
            w = st["committed"][c]
            cc = A.nodes[c]["cand"]
            # --- UP: own cover chain to the head, then the advance list (a merge commits ALL arms) ---
            x = w
            while True:
                bp = cc[x]["bpD"]
                if len(bp) == 1 and bp[0][0] == c:
                    x = bp[0][1]
                    if x is None or forb(c, x):
                        dead = True
                        break
                    st["runs"][c].add(x)
                else:
                    break
            if dead:
                break
            for (p, xp) in cc[x]["bpD"]:
                if xp is None or forb(p, xp):
                    dead = True
                    break
                if p in st["committed"]:
                    if not _pull_run(A, st, p, xp):             # connects into p's run -> pull the chain
                        dead = True
                        break
                else:
                    st["committed"][p] = xp
                    st["runs"][p] = {xp}
                    st["q"].append(p)
            if dead:
                break
            # --- DOWN: the run's forward closure, then each uncommitted child's entry options ---
            clo = set(st["runs"][c])
            frontier = list(clo)
            while frontier:
                y = frontier.pop()
                for (c2, z) in R.get((c, y), []):
                    if c2 == c and z not in clo and not forb(c, z):
                        clo.add(z)
                        frontier.append(z)
            for s_ in A.successors(c):
                if s_ in st["committed"]:
                    continue
                sc = A.nodes[s_]["cand"]
                opts = sorted({t for y in clo for (s2, t) in R.get((c, y), [])
                               if s2 == s_ and not forb(s_, t) and sc[t]["D"] < INF},
                              key=lambda t: (border[t], str(t)))
                if not opts:
                    dead = True                                 # this branch cannot place s_
                    break
                for t in opts[1:]:                              # BRANCH: one continuation per option
                    nst = copy(st)
                    nst["committed"][s_] = t
                    nst["runs"][s_] = {t}
                    nst["q"].append(s_)
                    nst["q"].appendleft(c)                      # reprocess c there (idempotent) for its other children
                    states.append(nst)
                st["committed"][s_] = opts[0]
                st["runs"][s_] = {opts[0]}
                st["q"].append(s_)
        if not dead and set(st["committed"]) == comp:
            out.append(({(a, v) for a, cells in st["runs"].items() for v in cells},
                        st["committed"]))
    return out


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


def extract(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0,
            max_states: int = 4096):
    """**The extraction** (docs §5) -- anchored branching enumeration over the forward table; **no
    backward table**. Per weakly-connected component: the anchor is the vertex with the FEWEST usable
    (non-forbidden, finite-``D``) cells; every usable anchor cell is explored via the two pointer
    types (stored ``bpD`` up, its transpose down), and every genuine choice point **branches** -- all
    alternatives stay alive until the judge. The judge discards candidates violating (V1)-(V4) --
    validity is the definition of a matching -- and returns the cheapest valid ``C(M)``. Raises
    ``ValueError`` when no candidate of any label survives, or when branching exceeds ``max_states``
    (never a silent truncation). Requires :func:`prepare` + :func:`forward`. Returns
    ``(M, committed)``. The two-table traceback remains as :func:`extract_two_table` (docs §6b)."""
    border = _b_order(B)
    R = _reverse_bpD(A)
    M_all: set = set()
    committed_all: Dict[Hashable, Hashable] = {}
    for comp in nx.weakly_connected_components(A):
        def usable(a):
            return [v for v, c in A.nodes[a]["cand"].items()
                    if not c.get("forbidden") and c["D"] < INF]
        anchor = min(comp, key=lambda a: (len(usable(a)), str(a)))
        labels = sorted(usable(anchor), key=lambda v: border[v])
        if not labels:
            raise ValueError(f"anchor {anchor!r} has no usable cell -- increase match_radius_m")
        best = None
        seen_M = set()
        for v0 in labels:
            for M, com in _explore_label(A, R, comp, anchor, v0, border, max_states):
                key = frozenset(M)
                if key in seen_M:
                    continue
                seen_M.add(key)
                v1, v2, v3 = check_rules(M, A, B)
                if v1 or v2 or v3:
                    continue                                    # invalid candidate -- discarded
                cost = _cost_of(A, B, M, alpha, beta)
                if best is None or cost < best[0] - 1e-12:
                    best = (cost, M, com)
        if best is None:
            raise ValueError(f"no valid candidate in component of {anchor!r} -- increase match_radius_m")
        M_all |= best[1]
        committed_all.update(best[2])
    return M_all, committed_all


# ---------------------------------------------------------------------------------------
# The junction-join extraction (docs/dag_dtw_matching.md §10) -- forward-only, EXACT
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
    """**The junction-join extraction** (docs/dag_dtw_matching.md §10) -- forward-only and
    **exact**: the optimal labels for all sinks and splits by a recursive table join over the split
    hierarchy. Every table is a sink-type table (label -> through-cost + pinned labels + recorded
    cells); splits are processed deepest-first; each branch's terminal is found by walking down to
    the first table-owned vertex (consumed-once: an absorbed table serves a later split through its
    recorded interior cells -- the polytree message flow, no cost division). The root table's
    minimum row is the exact decision-cost optimum; rows are tried cheapest-first and the first
    whose reconstructed ``M`` passes ``check_rules`` wins (the judge's word is unchanged -- but note
    the contraction keeps only per-label bests, so the validity fallback is narrower than
    :func:`extract`'s candidate pool). Requires :func:`prepare` + :func:`forward`. Raises
    ``ValueError`` when no row of a component survives. Returns ``(M, committed)``.

    Cross-validation invariant vs :func:`extract` (branching): whenever both succeed,
    ``C(M_join) <= C(M_branch)`` -- the join is exact, the branching is best-of-enumerated."""
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
# The cell-level join (docs/dag_dtw_matching.md §10.2) -- full resolution, from scratch
# ---------------------------------------------------------------------------------------
def _cell_reachable(A: nx.DiGraph, B: nx.DiGraph) -> set:
    """§8.2 cell-removal pre-pass: one reverse search from ALL sink cells over the cell-move graph
    (cover reversed inside a vertex, advance/stall reversed across edges). Cells never seen cannot
    appear on any chain to a sink and are removed up front, in every role."""
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
        for u in B.predecessors(v):                             # cover reversed (same vertex)
            if u in cand and not cand[u].get("forbidden") and (X, u) not in seen:
                seen.add((X, u))
                stack.append((X, u))
        for P in A.predecessors(X):                             # advance/stall reversed (to the parent)
            pc = A.nodes[P]["cand"]
            for u in [v] + list(B.predecessors(v)):
                if u in pc and not pc[u].get("forbidden") and (P, u) not in seen:
                    seen.add((P, u))
                    stack.append((P, u))
    return seen


def _cell_runs(A: nx.DiGraph, B: nx.DiGraph, a: Hashable, e: Hashable, seen: set,
               cap: int, border: Dict[Hashable, int]):
    """All directed cover paths from entry ``e`` inside cand(a) (simple, <= ``cap`` cover cells);
    removed cells cannot be covered. Deterministic order."""
    cand = A.nodes[a]["cand"]
    out, stack = [], [(e,)]
    while stack:
        path = stack.pop()
        out.append(path)
        if len(path) > cap:
            continue
        for w in sorted(B.successors(path[-1]), key=lambda t: border.get(t, 0)):
            if (w in cand and not cand[w].get("forbidden") and w not in path
                    and (a, w) in seen):
                stack.append(path + (w,))
    return out


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
                 run_cap: int = 8, max_rows: int = 50000):
    """**The cell-level join** (docs/dag_dtw_matching.md §10.2) -- exact over the FULL
    cell-level space, runs included. Built from scratch upstream: only ``prepare``'s ``E`` and the
    ``forbidden``/``D<inf`` filters (pruning) are used -- the stored propagation is never consulted.
    A row is ``(entry, value, pending, cells)``; the E-multiplier ledger is {1 advance/source,
    beta stall, alpha cover}; a vertex's entry-E is DEFERRED to its parent step (at a merge, to the
    root join -- beta if ANY arm stalls); a merge child is absorbed by one parent line
    (consumed-once), the other lines carry pending separators matched at the root join. ``M``
    travels with the rows -- the winning row's cells map IS the relation, no traceback.

    Joined rows are tried cheapest-first; the first whose ``M`` passes ``check_rules`` wins. Raises
    ``ValueError`` on infeasibility (a vertex with no surviving cell -- located precisely), when no
    root row is valid, or when a table exceeds ``max_rows`` (loud, never a silent truncation).
    ``run_cap`` bounds cover-run length. Requires :func:`prepare` + :func:`forward`. Returns
    ``(M, committed)``."""
    border = _b_order(B)
    seen = _cell_reachable(A, B)
    for X in A.nodes:
        if not any((X, v) in seen and A.nodes[X]["cand"][v]["D"] < INF
                   for v in A.nodes[X]["cand"]):
            raise ValueError(f"vertex {X!r} has no surviving cell (sink-search + D-filter) -- "
                             "increase match_radius_m")
    M_all: set = set()
    committed_all: Dict[Hashable, Hashable] = {}
    for comp in nx.weakly_connected_components(A):
        comp = set(comp)
        order = [n for n in nx.topological_sort(A) if n in comp]
        tables: Dict[Hashable, list] = {}
        absorbed_by: Dict[Hashable, Hashable] = {}
        for X in reversed(order):
            cand = A.nodes[X]["cand"]
            rows: Dict[tuple, tuple] = {}
            entries = sorted((v for v in cand
                              if (X, v) in seen and cand[v]["D"] < INF), key=lambda t: border[t])
            for e in entries:
                for R in _cell_runs(A, B, X, e, seen, run_cap, border):
                    u = R[-1]
                    combos = [(sum(alpha * cand[c]["E"] for c in R[1:]), {}, {X: R})]
                    dead = False
                    for c in sorted(A.successors(X), key=str):
                        if c not in comp:
                            continue
                        if c in absorbed_by and absorbed_by[c] != X:
                            child_rows = [(ce, 0.0, {(c, ce): False}, {})
                                          for ce in sorted(A.nodes[c]["cand"], key=lambda t: border[t])
                                          if (c, ce) in seen]
                            deferred = True
                        else:
                            absorbed_by[c] = X
                            child_rows = tables[c]
                            deferred = A.in_degree(c) > 1
                        opts = []
                        for (ce, cval, cpend, ccells) in child_rows:
                            stall = (ce == u)
                            if not stall and ce not in B.successors(u):
                                continue                        # children connect at the run end only
                            add = cval if deferred else \
                                cval + (beta if stall else 1.0) * A.nodes[c]["cand"][ce]["E"]
                            pend = dict(cpend)
                            if deferred:
                                pend[(c, ce)] = pend.get((c, ce), False) or stall
                            opts.append((add, pend, ccells))
                        if not opts:
                            dead = True
                            break
                        nxt = []
                        for (v0, p0, c0) in combos:
                            for (av, ap, ac) in opts:
                                p = _pend_union(p0, ap)
                                if p is not None:
                                    nxt.append((v0 + av, p, {**c0, **ac}))
                        combos = nxt
                        if not combos:
                            dead = True
                            break
                    if dead:
                        continue
                    for (val, pend, cells) in combos:
                        key = (e, frozenset(pend.items()))
                        if key not in rows or val < rows[key][1] - 1e-12:
                            rows[key] = (e, val, pend, cells)
                    if len(rows) > max_rows:
                        raise ValueError(f"cell-join table at {X!r} exceeded {max_rows} rows -- "
                                         "raise max_rows")
            if not rows:
                raise ValueError(f"vertex {X!r}: no feasible row -- increase match_radius_m")
            tables[X] = sorted(rows.values(), key=lambda r: (r[1], border[r[0]]))
        roots = []
        for X in sorted(comp, key=str):                         # sources pay their own entry (full E)
            if A.in_degree(X) == 0:
                roots.append([(e, val + A.nodes[X]["cand"][e]["E"], pend, cells)
                              for (e, val, pend, cells) in tables[X]])
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
    along that edge) so segment-mode matchings aggregate back to input-edge level."""
    G = nx.DiGraph()
    node_at: Dict[tuple, int] = {}

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
        for k, p in enumerate(pts[1:]):
            cur = node(p)
            if cur != prev:
                G.add_edge(prev, cur, road_id=eid, seq=k)
            prev = cur
    return G


# ---------------------------------------------------------------------------------------
# One-call pipeline entry point
# ---------------------------------------------------------------------------------------
def match_dag(A: nx.DiGraph, B: nx.DiGraph, r: float = 20.0, alpha: float = 1.0,
              beta: float = 1.0, mode: str = "point", engine: str = "cell",
              bearing_weight: float = 1.0, k_min: int = 1):
    """One-call DAG-DTW pipeline (docs/dag_dtw_matching.md): ``prepare`` -> ``forward`` (the
    coupled pass) -> extraction, on ``A``/``B`` (``mode="point"``, ``M`` over vertices) or on their
    directed line graphs (``mode="segment"``, ``M`` over arcs — nodes are ``(u, v)`` edge tuples of
    the originals, so the result is self-describing).

    ``engine``: ``"cell"`` (the cell-level join — exact over the full space; default),
    ``"branch"`` (the branching exploration), ``"join"`` (the vertex-level junction join), or
    ``"all"`` — run all three and return the **cheapest valid** matching, the cross-validating
    choice. Weights: ``alpha ∈ (0, 1]``, ``beta ∈ [1, ∞)`` (docs §3). The source may be any
    **subdivided DAG** (a tree is the special case); on reconvergent sources only the ``"cell"``
    engine carries the exactness claim (docs/dag_dtw_matching.md §10.4). Returns
    ``(M, committed)``; raises ``ValueError`` on infeasibility (increase ``r``)."""
    if mode == "segment":
        A2, B2 = line_digraph(A), line_digraph(B)
    elif mode == "point":
        A2, B2 = A, B
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'point' or 'segment')")
    prepare(A2, B2, r=r, k_min=k_min, bearing_weight=bearing_weight)
    forward(A2, B2, alpha=alpha, beta=beta)
    engines = {"cell": extract_cell, "branch": extract, "join": extract_join}
    if engine in engines:
        return engines[engine](A2, B2, alpha, beta)
    if engine == "all":                                         # the cross-validating choice
        best = None
        for fn in (extract_cell, extract, extract_join):
            try:
                M, com = fn(A2, B2, alpha, beta)
            except ValueError:
                continue
            c = _cost_of(A2, B2, M, alpha, beta)
            if best is None or c < best[0] - 1e-12:
                best = (c, M, com)
        if best is None:
            raise ValueError("all three extraction engines infeasible -- increase match_radius_m")
        return best[1], best[2]
    raise ValueError(f"unknown engine {engine!r} (use 'cell', 'branch', 'join' or 'all')")


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
    """The §4.1a invariant, after :func:`forward`: for every split, (i) at least one exit cell
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

    # tree guard: a diamond (reconvergence) must be rejected
    try:
        prepare(digraph({0: (0, 0), 1: (1, 1), 2: (1, -1), 3: (2, 0)},
                        [(0, 1), (0, 2), (1, 3), (2, 3)]), B, r=20.0)
        print("\nERROR: diamond was NOT rejected")
    except NotADAG as e:
        print(f"\ndiamond correctly rejected: {e}")
