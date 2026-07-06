"""
DAG-DTW: match a **source DAG** (a topologically-ordered, acyclic set of A-edges) to the local
directed B-network in ONE joint solve. The single-path graph-DTW is the special case of a
one-edge DAG.

This is the **point-to-point v1** described in ``docs/dag_dtw_matching.md``: states are
(A-vertex ``a``, B-vertex ``v``); the DP sweeps A in topological order and, per A-vertex, relaxes
B with Dijkstra (B may cycle). The recurrence has all three DTW moves, with the DAG generalizations
on the A-advance term -- a **sum** over incoming A-branches (cover every edge) with a **1/outdeg**
split factor (conserve the cost-flow so shared prefixes are counted once):

    D[a][v] = E(a,v) + min(
        min_{v' in Bpred(v)}                       D[a][v'],            # (H) B advances, A stays
        Σ_{a' in Apred(a)} 1/outdeg(a') · min_{v' in Bpred(v)∪{v}} D[a'][v']   # (A) A advances
    )

Free entry at every source (empty A-sum), total cost read at the sinks
(``Σ_sinks min_v D[t][v]`` -- exact for any DAG shape thanks to the split factor). Junction
consistency (one B-vertex ``φ(a)`` per A-vertex) is fixed at backtrack.

Both GA (source) and GB (target) are built with the same :func:`build_local_digraph` used by
graph-DTW, so the pooling / stitching / vertex-owns-its-edge machinery is shared. GA must be
acyclic; a cyclic source raises :class:`NotADAG`.
"""

import heapq
from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString

from .graph_dtw import LocalBGraph, build_local_digraph

Coord = Tuple[float, float]


class NotADAG(Exception):
    """Raised when the source graph GA contains a directed cycle (it must be acyclic)."""


# --------------------------------------------------------------------------------------
# Topological order, laid out as [sources | middle | sinks] (docs §2)
# --------------------------------------------------------------------------------------
def topological_order(ga: LocalBGraph) -> List[int]:
    """Kahn's algorithm over GA's forward arcs. Returns vertices with predecessors before
    successors; raises :class:`NotADAG` on a cycle. Sources (in-degree 0) come out first and
    sinks (out-degree 0) last, so the order is the three-block layout of docs §2."""
    V = ga.n_vertices
    indeg = np.array([len(ga.pred_arcs[v]) for v in range(V)], int)
    q = deque(int(v) for v in range(V) if indeg[v] == 0)   # all sources first
    order: List[int] = []
    left = indeg.copy()
    while q:
        u = q.popleft()
        order.append(u)
        for w in ga.succ_arcs[u]:
            left[w] -= 1
            if left[w] == 0:
                q.append(w)
    if len(order) != V:
        raise NotADAG("source graph GA has a directed cycle; it must be a DAG")
    return order


def _gb_arcs(gb: LocalBGraph) -> Tuple[np.ndarray, np.ndarray]:
    """Flat (tail, head) arrays of GB's forward arcs, for vectorized relaxation."""
    tails, heads = [], []
    for u in range(gb.n_vertices):
        for w in gb.succ_arcs[u]:
            tails.append(u)
            heads.append(w)
    return np.asarray(tails, int), np.asarray(heads, int)


# --------------------------------------------------------------------------------------
# The joint DP
# --------------------------------------------------------------------------------------
def match_dag_to_bgraph(
    a_edges: Sequence[Tuple[Any, LineString]],
    b_edges: Sequence[Tuple[Any, LineString]],
    *,
    snap_tolerance_m: float = 0.5,
    step_meters: float = 2.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """Align the source DAG made of ``a_edges`` to the local directed graph of ``b_edges``.

    Point-to-point v1: the emission is ``E(a, v) = dist(a, v)`` (no direction term).
    ``a_edges`` / ``b_edges``: lists of ``(id, shapely LineString)`` in a projected CRS (meters).
    Returns a dict with:

    - ``phi``: ``{a_vertex_index -> b_vertex_index}`` -- the junction-consistent label map;
    - ``a_vertex_match``: per A-vertex ``(x, y, b_vertex, b_edge_id, drift)``;
    - ``routes``: ``{a_edge_id -> [b_edge_id, ...]}`` the ordered B-edges each A-edge maps to;
    - ``total_cost`` (Σ over sinks) and ``avg_drift`` (mean per-A-vertex drift);
    - ``GA`` / ``GB`` (the two :class:`LocalBGraph`s), and with ``debug=True`` the full ``D`` table.
    """
    a_pts = [(float(x), float(y)) for _id, g in a_edges for (x, y) in g.coords]
    b_pts = [(float(x), float(y)) for _id, g in b_edges for (x, y) in g.coords]

    # GA (source) and GB (target) are built the same way; each is enriched with the other's nodes.
    ga = build_local_digraph(a_edges, b_pts, snap_tolerance_m, step_meters)
    gb = build_local_digraph(b_edges, a_pts, snap_tolerance_m, step_meters)
    order = topological_order(ga)                      # raises NotADAG on a cyclic source

    NA, NB = ga.n_vertices, gb.n_vertices
    ax, ay = ga.vx, ga.vy
    bx, by = gb.vx, gb.vy
    bu, bw = _gb_arcs(gb)                               # GB forward arcs (tail, head)
    outdeg = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)

    INF = float("inf")
    D = np.full((NA, NB), INF)
    hpar = np.full((NA, NB), -1, int)                  # horizontal (within-a) back-pointer

    for a in order:
        ei = np.hypot(bx - ax[a], by - ay[a])          # E(a, ·): point-to-point drift
        preds = ga.pred_arcs[a]
        if not preds:
            base = ei.copy()                           # source: free entry (empty A-sum = 0)
        else:
            acc = np.zeros(NB)
            for ap in preds:
                # m[v] = min over v' in {v} ∪ Bpred(v) of D[ap][v']  (vertical + diagonal)
                m = D[ap].copy()
                if bu.size:
                    np.minimum.at(m, bw, D[ap][bu])    # relax each arc tail into its head
                acc += m / outdeg[ap]                  # split factor: divide by the branch count
            base = ei + acc
        # (H) horizontal: within-a Dijkstra over GB, each B-step re-paying E(a, ·)
        D[a] = base.copy()
        heap = [(base[v], v) for v in range(NB)]
        heapq.heapify(heap)
        while heap:
            c, u = heapq.heappop(heap)
            if c > D[a][u]:
                continue
            for w in gb.succ_arcs[u]:
                cand = D[a][u] + ei[w]
                if cand < D[a][w]:
                    D[a][w] = cand
                    hpar[a][w] = u
                    heapq.heappush(heap, (cand, w))

    sinks = [a for a in range(NA) if len(ga.succ_arcs[a]) == 0]
    total_cost = float(sum(np.min(D[t]) for t in sinks)) if sinks else float("inf")

    # --- backtrack: fix one φ(a) per A-vertex (junction consistency) ---
    phi: Dict[int, int] = {}

    def resolve(a: int, v: int) -> None:
        if a in phi:
            return
        cur = v
        while hpar[a][cur] >= 0:                        # unwind the within-a horizontal walk
            cur = hpar[a][cur]
        phi[a] = cur                                    # entry B-vertex of this A-vertex
        preds = ga.pred_arcs[a]
        if not preds:
            return
        for ap in preds:                                # go to EVERY predecessor (cover branches)
            best_vp, best = cur, D[ap][cur]
            for u in gb.pred_arcs[cur]:
                if D[ap][u] < best:
                    best, best_vp = D[ap][u], u
            resolve(ap, best_vp)

    for t in sinks:
        resolve(t, int(np.argmin(D[t])))

    # --- per-A-vertex match + per-A-edge route ---
    a_vertex_match = []
    drifts = []
    for a in range(NA):
        v = phi.get(a)
        if v is None:
            continue
        beid = gb.edge_ids[gb.vert_edge[v]] if 0 <= gb.vert_edge[v] < len(gb.edge_ids) else None
        d = float(np.hypot(ax[a] - bx[v], ay[a] - by[v]))
        drifts.append(d)
        a_vertex_match.append((float(ax[a]), float(ay[a]), int(v), beid, d))

    routes: Dict[Any, List[Any]] = {}
    for a in order:                                     # keep topological order within each A-edge
        v = phi.get(a)
        if v is None:
            continue
        aeid = ga.edge_ids[ga.vert_edge[a]] if 0 <= ga.vert_edge[a] < len(ga.edge_ids) else None
        beid = gb.edge_ids[gb.vert_edge[v]] if 0 <= gb.vert_edge[v] < len(gb.edge_ids) else None
        seq = routes.setdefault(aeid, [])
        if beid is not None and (not seq or seq[-1] != beid):
            seq.append(beid)

    res: Dict[str, Any] = {
        "phi": phi,
        "a_vertex_match": a_vertex_match,
        "routes": routes,
        "total_cost": total_cost,
        "avg_drift": float(np.mean(drifts)) if drifts else float("inf"),
        "GA": ga,
        "GB": gb,
        "sinks": sinks,
        "sources": [a for a in range(NA) if len(ga.pred_arcs[a]) == 0],
    }
    if debug:
        res["debug"] = {"D": D, "order": order, "outdeg": outdeg, "hpar": hpar}
    return res
