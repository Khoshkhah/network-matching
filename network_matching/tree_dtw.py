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
    """Point-to-point emission ``E(a, v) = dist(a, v)``, an ``(NA, NB)`` matrix (docs §1)."""
    return np.hypot(gb.vx[None, :] - ga.vx[:, None], gb.vy[None, :] - ga.vy[:, None])


def _forward_table(pred_arcs, succ_arcs, outdeg, gb_succ, gb_pred, order, emit, alpha):
    """Fill one directional cost table (docs §4.1). Forward: ``pred_arcs``/``succ_arcs`` = GA's
    predecessor/successor lists, ``outdeg`` the split factor (``1/outdeg``), ``gb_succ``/``gb_pred``
    the target graph's forward/backward adjacency. Backward ``B`` uses the reversed arguments.

    Returns the ``(NA, NB)`` cost table ``D``.
    """
    NA, NB = len(pred_arcs), emit.shape[1]
    D = np.full((NA, NB), np.inf)
    for a in order:
        ei = emit[a]
        # (A) carried cost: sum over predecessors of the split-shared best hand-off into reach(v).
        acc = np.zeros(NB)
        for p in pred_arcs[a]:
            m = D[p].copy()                                      # x = v  (hold)
            for v in range(NB):
                for x in gb_pred[v]:                             # x ∈ Bpred(v)  (one arc before v)
                    if D[p][x] < m[v]:
                        m[v] = D[p][x]
            acc += m / outdeg[p]
        base = ei + acc                                          # (A) entry: FULL emission + carried
        # (H) horizontal coverage: Dijkstra over the target arcs, edge (u->v) costs α·E(a,v).
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
                    heapq.heappush(heap, (cand, v))
        D[a] = dist
    return D


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
    horizontal_weight: float = 1.0,
    validate: bool = False,
) -> Dict[str, Any]:
    """Match the source **tree** ``a_edges`` onto the local directed graph of ``b_edges`` and return
    the matching relation ``M`` (docs/tree_dtw_matching.md).

    ``a_edges`` / ``b_edges``: ``[(id, LineString)]`` (or ``(id, [(x, y), ...])``) in meters.
    ``horizontal_weight`` is ``α ≤ 1`` (§4.1); ``validate=True`` runs :func:`check_tree_rules`.

    Returns ``{M, a_match, routes, GA, GB, D, B, order}`` where ``M`` is a set of ``(a, v)`` pairs and
    ``a_match`` maps each source point to the sorted list of target points it covers. Raises
    :class:`NotATree` if the source has an undirected loop (docs §7).
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
    emit = _emission(ga, gb)
    outdeg_f = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)
    indeg_b = np.array([max(1, len(ga.pred_arcs[a])) for a in range(NA)], float)

    # §4.1 forward D (predecessor sum, 1/outdeg) and §4.2 backward B (successor sum, 1/indeg).
    D = _forward_table(ga.pred_arcs, ga.succ_arcs, outdeg_f,
                       gb.succ_arcs, gb.pred_arcs, order, emit, alpha)
    B = _forward_table(ga.succ_arcs, ga.pred_arcs, indeg_b,
                       gb.pred_arcs, gb.succ_arcs, order[::-1], emit, alpha)

    # §5 traceback: pin each point's anchor by GlobalCost = D+B-E (reverse-topo, reach guard).
    reach_cache: Dict[int, Set[int]] = {}
    anchor: Dict[int, int] = {}
    for a in reversed(order):
        tot = D[a] + B[a] - emit[a]
        succ = ga.succ_arcs[a]
        if not succ:
            anchor[a] = int(np.argmin(tot))
            continue
        targets = [anchor[s] for s in succ if s in anchor]
        chosen = None
        for v in np.argsort(tot):
            if not np.isfinite(tot[v]):
                break
            rv = _reachable(gb, int(v), reach_cache)
            if all(t in rv for t in targets):
                chosen = int(v)
                break
        anchor[a] = chosen if chosen is not None else int(np.argmin(tot))

    # Build the matching relation M gap-free (docs §3): each point's anchor, plus the target B-path
    # between an arc's two anchors, assigned as coverage to the downstream point. This fills any B-gap
    # across a step or a junction, so (V2)/(V3) hold by construction; a genuinely denser B stretch
    # simply becomes a 1:N run for that downstream point.
    run: Dict[int, Set[int]] = {a: {anchor[a]} for a in range(NA)}
    for a in order:
        for ap in ga.succ_arcs[a]:
            path = _bpath(gb, anchor[a], anchor[ap])
            if not path or len(path) < 2:
                continue
            if len(ga.pred_arcs[ap]) > 1:                        # ap is a MERGE: the incoming B-tail
                run[a].update(path[1:-1])                        #   belongs to the upstream branch a,
            else:                                                #   the merge point stays ap's anchor.
                run[ap].update(path[1:])                         # chain/split: downstream ap rides it
    a_match: Dict[int, List[int]] = {a: sorted(run[a]) for a in range(NA)}
    M: Set[Tuple[int, int]] = {(a, w) for a in range(NA) for w in run[a]}

    a_match_out, routes = _matches_and_routes(ga, gb, anchor, a_match, order)
    out: Dict[str, Any] = dict(M=M, a_match=a_match_out, routes=routes,
                               GA=ga, GB=gb, D=D, B=B, order=order, anchor=anchor)
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
        recs.append(dict(a=a, ax=ax, ay=ay, anchor=int(va), bx=bx, by=by,
                         drift=float(np.hypot(bx - ax, by - ay)), run=list(a_match[a])))
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
