"""Tree-DTW -- a **self-contained** exact matcher of a directed source *tree* to a directed network.

Independent implementation of ``docs/tree_dtw_matching.md``: it does not call, import, or depend on
the DAG matcher. It builds the two local digraphs with the shared geometry builder
(:func:`build_local_digraph`) and then does everything itself -- topological order, the emission
matrix, the forward table ``D`` and backward table ``B``, the joint ``D+B-E`` traceback, coverage-run
recovery, and the V1-V4 validator.

Output is the **matching relation ``M``** (docs §3), a set of ``(a_point, b_point)`` pairs -- *not* a
single-valued ``phi``. A source point that covers a 1:N run of target points contributes several pairs
to ``M`` (its run, recovered from the horizontal ``(H)`` coverage chain); point-to-point cells are
singletons. The source ``GA`` must be a directed tree (branches + merges, no undirected loop);
:class:`NotATree` is raised otherwise (docs §7).
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import LineString

from .graph_dtw import LocalBGraph, build_local_digraph

__all__ = ["match_tree_to_bgraph", "check_tree_rules", "NotATree"]


class NotATree(Exception):
    """Raised when the source ``GA`` is not a directed tree (it has an undirected loop /
    reconvergence). Tree-DTW is exact only on a tree (docs §7)."""


# ---------------------------------------------------------------------------------------
# Source-tree structure
# ---------------------------------------------------------------------------------------
def _topological_order(ga: LocalBGraph) -> List[int]:
    """Kahn topological order of the source points (sources first); raises on a directed cycle."""
    V = ga.n_vertices
    indeg = [len(ga.pred_arcs[v]) for v in range(V)]
    q = deque(v for v in range(V) if indeg[v] == 0)
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in ga.succ_arcs[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(order) != V:
        raise ValueError("source GA has a directed cycle -- not a DAG")
    return order


def _is_tree(ga: LocalBGraph) -> bool:
    """True iff the source's *undirected* skeleton is a forest (no loop / reconvergence).
    A connected component with ``V`` vertices is a tree iff it has exactly ``V-1`` undirected edges."""
    V = ga.n_vertices
    edges = sum(len(ga.succ_arcs[v]) for v in range(V))          # each directed arc = one undirected edge
    # count weakly-connected components via union-find
    parent = list(range(V))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u in range(V):
        for w in ga.succ_arcs[u]:
            ru, rw = find(u), find(w)
            if ru != rw:
                parent[ru] = rw
    ncomp = len({find(v) for v in range(V)})
    return edges == V - ncomp                                    # forest: |E| = |V| - #components


# ---------------------------------------------------------------------------------------
# Emission and the two cost tables (docs §4.1 / §4.2)
# ---------------------------------------------------------------------------------------
def _emission(ga: LocalBGraph, gb: LocalBGraph) -> np.ndarray:
    """Point-to-point emission ``E(a, v) = dist(a, v)``, an ``(NA, NB)`` matrix (docs §1, point mode)."""
    return np.hypot(gb.vx[None, :] - ga.vx[:, None], gb.vy[None, :] - ga.vy[:, None])


def _forward_table(pred_arcs, succ_arcs, outdeg, gb_succ, gb_pred, order, emit, alpha,
                   beta=1.0, ridable=None):
    """Fill one directional cost table **and its back-pointers** (docs §4.1). Forward:
    ``pred_arcs``/``succ_arcs`` = GA's predecessor/successor lists, ``outdeg`` the split factor
    (``1/outdeg``), ``gb_succ``/``gb_pred`` the target's forward/backward adjacency. Backward ``B``
    uses the reversed arguments. The same filler drives point mode (indices = vertices) and segment
    mode (indices = arcs, docs §8).

    Each cell is the cheapest of: the full-cost **advance** (every predecessor steps into ``v``); the
    **N:1 stall** (≥1 predecessor already on ``v``, forced by ``min_q``), charged ``β·E`` once; and the
    **1:N coverage** (H) horizontal move, charged ``α·E``. ``α = β = 1`` is bit-for-bit the plain
    point-to-point recurrence. ``ridable`` (segment mode) is an ``(NB,)`` bool mask of hostable target
    states; a non-ridable state (a junction stitch) can't rest a match but is a free pass-through in
    the horizontal move. Default ``None`` = all hostable.

    Returns ``(D, bp)``: the ``(NA, NB)`` cost table, and ``bp[a][v]`` = **the list of cells this
    value was computed from** (docs §4.1) -- ``[]`` (ENTER/source), ``[(a, v')]`` (a same-source
    COVER step), or ``[(p, x_p), ...]`` over predecessors (ADVANCE, each ``x_p ∈ reach(v)``).
    """
    NA, NB = len(pred_arcs), emit.shape[1]
    D = np.full((NA, NB), np.inf)
    bp: List[List[List[Tuple[int, int]]]] = [[[] for _ in range(NB)] for _ in range(NA)]
    for a in order:
        ei = emit[a]
        preds = pred_arcs[a]
        # per-predecessor step value + its argmin cell (x ∈ Bpred(v)); stall = D[p][v] (x = v).
        step_val: Dict[int, np.ndarray] = {}
        step_arg: Dict[int, np.ndarray] = {}
        for p in preds:
            Dp = D[p]
            sv = np.full(NB, np.inf)
            sa = np.full(NB, -1, int)
            for v in range(NB):
                bx, bval = -1, np.inf
                for x in gb_pred[v]:                             # x ∈ Bpred(v)  (one arc before v)
                    if Dp[x] < bval:
                        bval, bx = Dp[x], x
                sv[v], sa[v] = bval, bx
            step_val[p], step_arg[p] = sv, sa
        # base value + advance/stall back-pointer, per cell v
        base = np.full(NB, np.inf)
        for v in range(NB):
            if not preds:
                base[v] = ei[v]                                  # ENTER: source, free entry
                bp[a][v] = []
                continue
            # (advance) full E + Σ step/outdeg ;   (stall) β E + Σ best/outdeg + force one q onto v
            val_adv, ok_adv, par_adv = ei[v], True, []
            best_sum, ok_stall, par_best, min_gap, q_star = 0.0, True, [], np.inf, None
            for p in preds:
                inv = 1.0 / outdeg[p]
                sv, xs, st = step_val[p][v], int(step_arg[p][v]), D[p][v]
                if np.isfinite(sv):
                    val_adv += sv * inv
                    par_adv.append((p, xs))
                else:
                    ok_adv = False
                if st <= sv:                                     # best of stall/step for this branch
                    bval, bx = st, v
                else:
                    bval, bx = sv, xs
                if np.isfinite(bval):
                    best_sum += bval * inv
                    par_best.append((p, bx))
                    gap = (st - bval) * inv                      # cost to FORCE this branch to stall
                    if gap < min_gap:
                        min_gap, q_star = gap, p
                else:
                    ok_stall = False
            val_adv = val_adv if ok_adv else np.inf
            val_stall = (beta * ei[v] + best_sum + min_gap) if (ok_stall and q_star is not None) else np.inf
            if val_stall < val_adv:
                base[v] = val_stall
                bp[a][v] = [(q_star, v)] + [(p, bx) for (p, bx) in par_best if p != q_star]
            else:
                base[v] = val_adv
                bp[a][v] = par_adv
        if ridable is not None:
            base = np.where(ridable, base, np.inf)               # a stitch cannot HOST a state (§8.3)
        # (H) horizontal coverage: Dijkstra over the target arcs, edge (u->v) costs α·E(a,v).
        # A cell lowered here gets a COVER back-pointer [(a, u)] (same source a, from u).
        dist = base.copy()
        heap = [(float(dist[v]), v) for v in range(NB) if np.isfinite(dist[v])]
        heapq.heapify(heap)
        while heap:
            c, u = heapq.heappop(heap)
            if c > dist[u]:
                continue
            for v in gb_succ[u]:
                cand = dist[u] + alpha * ei[v]
                if cand < dist[v]:
                    dist[v] = cand
                    bp[a][v] = [(a, u)]                          # COVER
                    heapq.heappush(heap, (cand, v))
        D[a] = dist
    return D, bp


def _bpath(gb: LocalBGraph, s: int, t: int) -> List[int]:
    """Shortest target path ``[s, ..., t]`` along GB's forward arcs (BFS), or ``None`` if none."""
    if s == t:
        return [s]
    prev: Dict[int, int] = {s: -1}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for w in gb.succ_arcs[u]:
            if w not in prev:
                prev[w] = u
                q.append(w)
    if t not in prev:
        return None
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def _reachable(gb: LocalBGraph, v: int, cache: Dict[int, Set[int]]) -> Set[int]:
    """Target points forward-reachable from ``v`` (including ``v``), memoized."""
    s = cache.get(v)
    if s is None:
        s = {v}
        stack = [v]
        while stack:
            u = stack.pop()
            for w in gb.succ_arcs[u]:
                if w not in s:
                    s.add(w)
                    stack.append(w)
        cache[v] = s
    return s


def _extract(n_nodes, D, B, emit, bp_D, bp_B, hostable=None, nodes=None):
    """The backward phase (docs §5): **seed once, then follow the stored back-pointer lists**.

    Generic over point mode (nodes = source vertices, cells = B-vertices) and segment mode (nodes =
    source arcs, cells = B-arcs). ``bp_D``/``bp_B`` are the lists from :func:`_forward_table`; ``nodes``
    is the set of seedable source nodes (default all; segment mode passes only the real arcs, so stitch
    arcs -- which host nothing -- are never seeded).

    Step 1 -- the ONLY argmin -- seeds one representative per weakly-connected source component at its
    joint optimum ``argmin_v D[r][v]+B[r][v]-E(r,v)`` (restricted to hostable cells). Step 2 floods
    outward, **reading** each node's lists: the same-node COVER pairs give its 1:N run; the ADVANCE
    pairs at the run's head/tail ARE its predecessors/successors (V2/V3). Every node is committed once
    (a tree). Returns ``(committed, M)``: node -> anchor cell, and the full matching set of pairs.
    """
    tot = D + B - emit
    if hostable is not None:
        tot = np.where(np.asarray(hostable)[None, :], tot, np.inf)
    committed: Dict[int, int] = {}
    M: Set[Tuple[int, int]] = set()
    queue: deque = deque()

    def commit(x, w):
        if x not in committed:
            committed[x] = int(w)
            queue.append(x)

    for r in (range(n_nodes) if nodes is None else nodes):
        if r in committed:
            continue                                            # already filled by an earlier component
        vr = int(np.argmin(tot[r]))                             # Step 1: the one seed argmin
        if not np.isfinite(tot[r][vr]):
            raise ValueError(f"tree-DTW extraction: no finite matching for source node {r}")
        commit(r, vr)
        while queue:                                            # Step 2: follow the lists (both ways)
            c = queue.popleft()
            v = committed[c]
            run = {v}
            head = v
            while len(bp_D[c][head]) == 1 and bp_D[c][head][0][0] == c:   # same-c COVER pair
                head = bp_D[c][head][0][1]
                run.add(head)
            tail = v
            while len(bp_B[c][tail]) == 1 and bp_B[c][tail][0][0] == c:
                tail = bp_B[c][tail][0][1]
                run.add(tail)
            for w in run:
                M.add((c, w))
            for (p, x) in bp_D[c][head]:                        # predecessors (V2)
                commit(p, x)
            for (s, w) in bp_B[c][tail]:                        # successors (V3)
                commit(s, w)
    return committed, M


# ---------------------------------------------------------------------------------------
# Segment-state DP (docs §8) -- true segment-to-segment: a state is an (A-arc, B-arc) pair
# ---------------------------------------------------------------------------------------
def _enumerate_arcs(g: LocalBGraph):
    """All directed arcs of ``g`` as ``(tail[], head[])`` index arrays, per-vertex arc adjacency
    (``arcs_from``/``arcs_to``), and a ``ridable`` mask. An arc is a **segment** (ridable) iff both
    ends lie on the *same* B-edge; an inter-edge junction **stitch** (different edges) is non-ridable
    -- free connectivity, never a hosted state (docs §8.3)."""
    au: List[int] = []
    aw: List[int] = []
    arcs_from: List[List[int]] = [[] for _ in range(g.n_vertices)]
    arcs_to: List[List[int]] = [[] for _ in range(g.n_vertices)]
    for u in range(g.n_vertices):
        for w in g.succ_arcs[u]:
            k = len(au)
            au.append(u)
            aw.append(w)
            arcs_from[u].append(k)
            arcs_to[w].append(k)
    au_a = np.asarray(au, int)
    aw_a = np.asarray(aw, int)
    ridable = (g.vert_edge[au_a] == g.vert_edge[aw_a]) if len(au) else np.zeros(0, bool)
    return au_a, aw_a, arcs_from, arcs_to, ridable


def _arc_geom(g: LocalBGraph, au: np.ndarray, aw: np.ndarray):
    """Per-arc midpoint ``(mx, my)`` and compass bearing ``(deg·atan2(Δx, Δy) + 360) mod 360``."""
    ux, uy, hx, hy = g.vx[au], g.vy[au], g.vx[aw], g.vy[aw]
    mx, my = 0.5 * (ux + hx), 0.5 * (uy + hy)
    bear = (np.degrees(np.arctan2(hx - ux, hy - uy)) + 360.0) % 360.0
    return mx, my, bear


def _segment_tables(ga: LocalBGraph, gb: LocalBGraph, bearing_weight: float, alpha: float, beta: float):
    """Build the segment-mode (A-arc, B-arc) forward ``D`` / backward ``B`` tables and their
    back-pointers over the **arc line-graph** (docs §8), with junction stitches contracted on both
    sides. Returns a dict of everything the extraction (:func:`_segment_anchors`) and the table
    validator need: ``real_a``, arc-endpoint arrays ``Au``/``Aw``/``Bu``/``Bw``, the ``ridable``
    masks, the real-arc line-graph adjacency ``pred_list``/``succ_list`` and ``order``, the target-arc
    adjacency ``barc_pred``/``barc_succ``, the arc emission ``emit``, the tables ``D``/``B`` with
    ``bp_D``/``bp_B``, and per-arc geometry."""
    Au, Aw, A_from, A_to, A_rid = _enumerate_arcs(ga)
    Bu, Bw, B_from, B_to, B_rid = _enumerate_arcs(gb)
    NAA, NBA = len(Au), len(Bu)
    real_a = [k for k in range(NAA) if A_rid[k]]
    if not real_a or NBA == 0 or not B_rid.any():
        raise ValueError("segment mode needs at least one source and one target segment "
                         "(all arcs were junction stitches)")
    amx, amy, abear = _arc_geom(ga, Au, Aw)
    bmx, bmy, bbear = _arc_geom(gb, Bu, Bw)

    # --- effective real-arc line-graph: contract one-hop junction stitches so a real segment's
    #     neighbours are real segments (a merge/split at a junction becomes several real neighbours).
    def real_preds(k: int) -> List[int]:
        t = int(Au[k])
        preds = [j for j in A_to[t] if A_rid[j]]                     # same-edge chain into t
        for st in A_to[t]:
            if not A_rid[st]:                                        # ... or through a stitch INTO t
                preds += [j for j in A_to[int(Au[st])] if A_rid[j]]
        return list(dict.fromkeys(preds))

    def real_succs(k: int) -> List[int]:
        h = int(Aw[k])
        succs = [j for j in A_from[h] if A_rid[j]]                   # same-edge chain out of h
        for st in A_from[h]:
            if not A_rid[st]:                                        # ... or through a stitch OUT of h
                succs += [j for j in A_from[int(Aw[st])] if A_rid[j]]
        return list(dict.fromkeys(succs))

    a_pred = {k: real_preds(k) for k in real_a}
    a_succ = {k: real_succs(k) for k in real_a}
    pred_list = [a_pred.get(k, []) for k in range(NAA)]
    succ_list = [a_succ.get(k, []) for k in range(NAA)]
    outdeg = np.ones(NAA, float)
    indeg = np.ones(NAA, float)
    for k in real_a:
        outdeg[k] = max(1, len(a_succ[k]))
        indeg[k] = max(1, len(a_pred[k]))

    # topological order of the real-arc line-graph (a DAG whenever GA is)
    remaining = {k: len(a_pred[k]) for k in real_a}
    q = deque(k for k in real_a if remaining[k] == 0)
    order: List[int] = []
    while q:
        k = q.popleft()
        order.append(k)
        for j in a_succ[k]:
            remaining[j] -= 1
            if remaining[j] == 0:
                q.append(j)
    if len(order) != len(real_a):
        raise ValueError("source arc line-graph has a directed cycle -- not a DAG")

    # target-arc adjacency, REAL arcs only (junction stitches contracted, like the A-side): a real
    # B-arc hands off to the real B-arc one step away, crossing a junction stitch transparently. This
    # is what keeps the DP off stitches entirely -- an A-arc can never rest on a stitch, so the split's
    # arms leave the real B-junction (not a stitch mid-way onto one branch).
    def _barc_pred(e: int) -> List[int]:                            # real arcs ending at tail(e)
        u = int(Bu[e])
        preds = [j for j in B_to[u] if B_rid[j]]
        for st in B_to[u]:
            if not B_rid[st]:                                       # ... or through a stitch INTO u
                preds += [j for j in B_to[int(Bu[st])] if B_rid[j]]
        return list(dict.fromkeys(preds))

    def _barc_succ(e: int) -> List[int]:                           # real arcs starting at head(e)
        w = int(Bw[e])
        succs = [j for j in B_from[w] if B_rid[j]]
        for st in B_from[w]:
            if not B_rid[st]:                                       # ... or through a stitch OUT of w
                succs += [j for j in B_from[int(Bw[st])] if B_rid[j]]
        return list(dict.fromkeys(succs))

    barc_pred = [_barc_pred(e) for e in range(NBA)]
    barc_succ = [_barc_succ(e) for e in range(NBA)]

    # arc emission E(α, e); stitch target arcs are free pass-through (0), never hosted (B_rid mask).
    emit = np.hypot(bmx[None, :] - amx[:, None], bmy[None, :] - amy[:, None])
    if bearing_weight:
        bd = np.abs(abear[:, None] - bbear[None, :])
        emit = emit + float(bearing_weight) * np.minimum(bd, 360.0 - bd)
    emit[:, ~B_rid] = 0.0

    # §4.1 forward D (predecessor sum) and §4.2 backward B (successor sum), on the arc line-graph,
    # each with its back-pointer lists.
    D, bp_D = _forward_table(pred_list, succ_list, outdeg, barc_succ, barc_pred, order, emit, alpha,
                             beta=beta, ridable=B_rid)
    B, bp_B = _forward_table(succ_list, pred_list, indeg, barc_pred, barc_succ, order[::-1], emit, alpha,
                             beta=beta, ridable=B_rid)
    return dict(real_a=real_a, Au=Au, Aw=Aw, Bu=Bu, Bw=Bw, A_rid=A_rid, B_rid=B_rid,
                pred_list=pred_list, succ_list=succ_list, order=order,
                barc_pred=barc_pred, barc_succ=barc_succ, emit=emit, NAA=NAA, NBA=NBA,
                D=D, bp_D=bp_D, B=B, bp_B=bp_B,
                amx=amx, amy=amy, abear=abear, bmx=bmx, bmy=bmy, bbear=bbear)


def _segment_anchors(ga: LocalBGraph, gb: LocalBGraph, bearing_weight: float,
                     alpha: float, beta: float):
    """True segment-to-segment matcher (docs §8): DP states are ``(A-arc, B-arc)`` pairs. Builds the
    arc tables (:func:`_segment_tables`), runs the §5 back-pointer extraction on the arc line-graph,
    and maps the arc matching to per-vertex outputs. Returns ``(anchor, M, segment_pairs, D, B)``."""
    T = _segment_tables(ga, gb, bearing_weight, alpha, beta)
    real_a, Au, Aw, Bu, Bw, B_rid = T["real_a"], T["Au"], T["Aw"], T["Bu"], T["Bw"], T["B_rid"]
    NAA, emit, D, bp_D, B, bp_B = T["NAA"], T["emit"], T["D"], T["bp_D"], T["B"], T["bp_B"]
    amx, amy, abear = T["amx"], T["amy"], T["abear"]
    bmx, bmy, bbear = T["bmx"], T["bmy"], T["bbear"]

    # §5 extraction over the arc line-graph: seed one real arc per component, then FOLLOW the stored
    # lists (no per-arc guess, no gap-fill). committed_arc: each real A-arc -> its B-arc anchor.
    committed_arc, _ = _extract(NAA, D, B, emit, bp_D, bp_B, hostable=B_rid, nodes=real_a)

    def _arc_run(k: int) -> List[int]:
        """The ordered B-arcs source segment ``k`` rides -- its anchor plus the 1:N coverage run,
        read straight off the same-``k`` COVER chains (docs §5). Usually just ``[anchor]``."""
        e0 = committed_arc[k]
        left, x = [e0], e0
        while len(bp_D[k][x]) == 1 and bp_D[k][x][0][0] == k:       # coverage back to the run's start
            x = bp_D[k][x][0][1]
            left.append(x)
        left.reverse()
        right, x = [], e0
        while len(bp_B[k][x]) == 1 and bp_B[k][x][0][0] == k:       # coverage on to the run's end
            x = bp_B[k][x][0][1]
            right.append(x)
        return left + right
    runs = {k: _arc_run(k) for k in real_a}

    # per-A-vertex φ from the RUN endpoints (a segment's tail lands on its run's first B-tail, its head
    # on the run's last B-head). Consecutive arcs share a vertex on the same B-vertex (back-pointer
    # coupling), so φ is a consistent monotone walk.
    anchor: Dict[int, int] = {}
    for k in real_a:
        anchor.setdefault(int(Au[k]), int(Bu[runs[k][0]]))
    for k in real_a:
        h = int(Aw[k])
        if h not in anchor:
            anchor[h] = int(Bw[runs[k][-1]])
    for a in range(ga.n_vertices):                                  # isolated 1-point edges (rare)
        anchor.setdefault(a, int(np.argmin(_emission(ga, gb)[a])))

    # per-vertex matching M: each source point at its anchor, plus each segment's B-run assigned to its
    # HEAD vertex (the downstream point, docs §4.1 convention) -- a 1:N arc coverage becomes the head's
    # run, so M is gap-free with no gap-fill.
    M: Set[Tuple[int, int]] = {(a, anchor[a]) for a in range(ga.n_vertices)}
    for k in real_a:
        h = int(Aw[k])
        for e in runs[k]:
            M.add((h, int(Bw[e])))

    # segment_pairs (viz): each source segment (t→h) middle-to-middle to the B-arc it actually rode,
    # plus both arcs' endpoints so a plot can draw the matched A-segment and B-segment themselves.
    segment_pairs = []
    for k in real_a:
        e = committed_arc[k]
        t, h, u, w = int(Au[k]), int(Aw[k]), int(Bu[e]), int(Bw[e])
        segment_pairs.append(dict(
            a_mid=(float(amx[k]), float(amy[k])),
            b_mid=(float(bmx[e]), float(bmy[e])),
            a_p0=(float(ga.vx[t]), float(ga.vy[t])), a_p1=(float(ga.vx[h]), float(ga.vy[h])),
            b_p0=(float(gb.vx[u]), float(gb.vy[u])), b_p1=(float(gb.vx[w]), float(gb.vy[w])),
            a_bear=float(abear[k]), b_bear=float(bbear[e]),
            cost=float(emit[k][e])))
    return anchor, M, segment_pairs, D, B


# ---------------------------------------------------------------------------------------
# V1-V4 validator (docs §3) -- self-contained, no dependency on the DAG matcher
# ---------------------------------------------------------------------------------------
def check_tree_rules(M: Set[Tuple[int, int]], ga: LocalBGraph, gb: LocalBGraph) -> Dict[str, Any]:
    """Check the matching relation ``M`` against the four valid-warping rules (docs §3), using only
    immediate neighbours. Returns ``{ok, v1_cross, v2_predecessor, v3_successor, v4_uncovered}``."""
    Mset = set(M)

    def hasm(a, v):
        return (a, v) in Mset

    v1: List[Tuple[int, int]] = []
    v2: List[Tuple[int, int]] = []
    v3: List[Tuple[int, int]] = []
    for (a, v) in Mset:
        # (V1) no cross: no DAG-predecessor of a sits on a B-successor of v
        for am in ga.pred_arcs[a]:
            for vp in gb.succ_arcs[v]:
                if hasm(am, vp):
                    v1.append((a, v))
                    break
            else:
                continue
            break
        # (V2) predecessor rule: continues a run at a, OR every predecessor feeds it
        cont_pred = any(hasm(a, vm) for vm in gb.pred_arcs[v])
        if not cont_pred:
            for am in ga.pred_arcs[a]:
                if not (hasm(am, v) or any(hasm(am, vm) for vm in gb.pred_arcs[v])):
                    v2.append((a, v))
                    break
        # (V3) successor rule: continues a run at a, OR every successor carries it on
        cont_succ = any(hasm(a, vp) for vp in gb.succ_arcs[v])
        if not cont_succ:
            for ap in ga.succ_arcs[a]:
                if not (hasm(ap, v) or any(hasm(ap, vp) for vp in gb.succ_arcs[v])):
                    v3.append((a, v))
                    break
    matched = {a for (a, _v) in Mset}
    v4 = [a for a in range(ga.n_vertices) if a not in matched]
    ok = not (v1 or v2 or v3 or v4)
    return {"ok": ok, "v1_cross": v1, "v2_predecessor": v2, "v3_successor": v3, "v4_uncovered": v4}


# ---------------------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------------------
def _as_linestrings(edges: Sequence[Tuple[Any, Any]]) -> List[Tuple[Any, LineString]]:
    return [(eid, g if isinstance(g, LineString) else LineString(g)) for eid, g in edges]


def match_tree_to_bgraph(
    a_edges: Sequence[Tuple[Any, Any]],
    b_edges: Sequence[Tuple[Any, Any]],
    *,
    snap_tolerance_m: float = 0.5,
    step_meters: float = 2.0,
    emission: str = "point",
    bearing_weight: float = 0.0,
    horizontal_weight: float = 1.0,
    vertical_weight: float = 1.0,
    validate: bool = False,
) -> Dict[str, Any]:
    """Match the source **tree** ``a_edges`` onto the local directed graph of ``b_edges`` and return
    the matching relation ``M`` (docs/tree_dtw_matching.md).

    ``a_edges`` / ``b_edges``: ``[(id, LineString)]`` (or ``(id, [(x, y), ...])``) in meters.
    ``emission``: ``"point"`` (default, docs §2–§7) is a **point-state** DP -- a state is one source
    point matched to one target point, scored ``E(a, v) = dist(a, v)``. ``"segment"`` (docs §8) is the
    true **segment-to-segment** matcher -- a state is an ``(A-arc, B-arc)`` pair, scored middle-to-
    middle ``+ bearing_weight·Δbearing`` on *every* move -- which resolves nearest-vs-corresponding (a
    split's arms under a lateral shift). ``"point"`` is bit-for-bit the former result; both modes
    return the same ``M``/``routes`` shape. The **call-time hyperparameters** (docs §4.1, §8): λ =
    ``bearing_weight`` (heading term, ``"segment"`` mode only, ~1-5); α = ``horizontal_weight`` ≤ 1
    discounts **1:N** coverage (one source point spanning a run of target points); β =
    ``vertical_weight`` ≤ 1 discounts **N:1** coverage (many source points stacking on one target
    point). ``α = β = 1`` (defaults) is the plain point-to-point pricing, bit-for-bit unchanged. Both
    weights apply in both emission modes. ``validate=True`` runs :func:`check_tree_rules`.

    Returns ``{M, a_match, routes, emission, GA, GB, D, B, order, anchor}`` where ``M`` is a set of
    ``(a, v)`` pairs and ``a_match`` maps each source point to the sorted list of target points it
    covers. ``"segment"`` mode also returns ``segment_pairs`` -- the matched ``(A-arc, B-arc)`` records
    (``a_mid``/``b_mid`` midpoints, ``a_bear``/``b_bear`` bearings, ``cost``), the middle-to-middle
    pairs the DP actually scored, for the correspondence view. ``D``/``B`` are vertex cost tables in
    point mode and arc cost tables in segment mode. Raises :class:`NotATree` if the source has an
    undirected loop (docs §7).
    """
    a_edges = _as_linestrings(a_edges)
    b_edges = _as_linestrings(b_edges)
    a_pts = [(float(x), float(y)) for _id, g in a_edges for (x, y) in g.coords]
    b_pts = [(float(x), float(y)) for _id, g in b_edges for (x, y) in g.coords]

    ga = build_local_digraph(a_edges, b_pts, snap_tolerance_m, step_meters)
    gb = build_local_digraph(b_edges, a_pts, snap_tolerance_m, step_meters)
    order = _topological_order(ga)
    if not _is_tree(ga):
        raise NotATree("source GA has an undirected loop (a reconvergence/diamond); "
                       "Tree-DTW is exact only on a directed tree (docs §7)")

    NA = ga.n_vertices
    alpha = float(horizontal_weight)
    beta = float(vertical_weight)

    # Pin each source point to a B-vertex anchor φ and read off the matching M (docs §4–§5). Point
    # mode: a vertex-state DP + the §5 back-pointer extraction. Segment mode (§8): the same, over the
    # arc line-graph, plus the (A-arc, B-arc) middle-to-middle pairs for the correspondence view.
    segment_pairs = None
    if emission == "point":
        emit = _emission(ga, gb)
        outdeg_f = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)
        indeg_b = np.array([max(1, len(ga.pred_arcs[a])) for a in range(NA)], float)
        # §4.1 forward D (predecessor sum, 1/outdeg) and §4.2 backward B (successor sum, 1/indeg),
        # each with its back-pointer lists; §5 extraction follows them (seed once, then flood).
        D, bp_D = _forward_table(ga.pred_arcs, ga.succ_arcs, outdeg_f,
                                 gb.succ_arcs, gb.pred_arcs, order, emit, alpha, beta=beta)
        B, bp_B = _forward_table(ga.succ_arcs, ga.pred_arcs, indeg_b,
                                 gb.pred_arcs, gb.succ_arcs, order[::-1], emit, alpha, beta=beta)
        anchor, M = _extract(NA, D, B, emit, bp_D, bp_B)
    elif emission == "segment":
        anchor, M, segment_pairs, D, B = _segment_anchors(ga, gb, float(bearing_weight), alpha, beta)
    else:
        raise ValueError(f"unknown emission {emission!r}; use 'point' or 'segment'")

    # a_match = M grouped per source point (the anchor plus any 1:N coverage run it covers).
    grouped: Dict[int, Set[int]] = {a: set() for a in range(NA)}
    for (a, w) in M:
        grouped[a].add(w)
    a_match: Dict[int, List[int]] = {a: sorted(grouped[a]) for a in range(NA)}

    a_match_out, routes = _matches_and_routes(ga, gb, anchor, a_match, order)
    out: Dict[str, Any] = dict(M=M, a_match=a_match_out, routes=routes, emission=emission,
                               GA=ga, GB=gb, D=D, B=B, order=order, anchor=anchor)
    if segment_pairs is not None:
        out["segment_pairs"] = segment_pairs
    if validate:
        out["rules"] = check_tree_rules(M, ga, gb)
    return out


def _matches_and_routes(ga, gb, anchor, a_match, order):
    """Per-point match records (anchor + covered run) and per-A-edge B-edge routes.

    The route is the ordered B-edges each A-edge's anchors run through, with a leading/trailing
    **single-vertex junction touch** trimmed (an A-edge's boundary vertex grazing the neighbour's
    B-edge at the junction) when a real edge remains -- the same boundary handling as the DAG matcher.
    """
    recs = []
    seq: Dict[Any, List[List[Any]]] = {}                        # aeid -> [[beid, count], ...] in order
    for a in order:
        va = anchor[a]
        ax, ay = float(ga.vx[a]), float(ga.vy[a])
        bx, by = float(gb.vx[va]), float(gb.vy[va])
        rec = dict(a=a, ax=ax, ay=ay, anchor=int(va), bx=bx, by=by,
                   drift=float(np.hypot(bx - ax, by - ay)), run=list(a_match[a]))
        recs.append(rec)
        ae, be = ga.vert_edge[a], gb.vert_edge[va]
        aeid = ga.edge_ids[ae] if 0 <= ae < len(ga.edge_ids) else None
        beid = gb.edge_ids[be] if 0 <= be < len(gb.edge_ids) else None
        if aeid is None or beid is None:
            continue
        run = seq.setdefault(aeid, [])
        if run and run[-1][0] == beid:
            run[-1][1] += 1
        else:
            run.append([beid, 1])
    routes: Dict[Any, List[Any]] = {}
    for aeid, run in seq.items():
        r = run[:]
        if len(r) > 1 and r[0][1] <= 1:                         # leading junction touch
            r = r[1:]
        if len(r) > 1 and r[-1][1] <= 1:                        # trailing junction touch
            r = r[:-1]
        routes[aeid] = [b for b, _c in r]
    return recs, routes
