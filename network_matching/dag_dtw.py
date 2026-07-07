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

**Junction resolution -- forward + backward (docs §3.2a).** Two cost tables are built with the
same :func:`_dp_table`: the **forward** ``D`` (sources -> a) and the **backward** ``B`` (a ->
sinks, on the reversed graphs, with the symmetric ``1/indeg`` split). Each A-vertex is labelled by a
reverse-topological backtrack that scores by the **joint** cost ``D[a][v] + B[a][v] - E(a,v)`` (the
value all routes through ``a`` agree on) **subject to a reachability constraint** (the chosen v must
still walk forward to every successor's φ). The joint score resolves two routes disagreeing at a
shared junction; the constraint keeps the topology consistent (a shifted source edge can't collapse
onto a nearest cross road). **Clean on all tree scenarios** (a reconvergent *diamond* is the
remaining caveat).

**Two stages.** The tables + joint labels decide the **topology** -- which B-edges each A-edge maps
to (the route). A second :func:`_arclength_rematch` pass then decides the **position** -- each
A-vertex is placed at its *arc-length* fraction along its route's B-polyline (free entry/exit at
sources/sinks). Pure point-to-point picks the nearest B-vertex per A-vertex, which under a large
offset *compresses* A onto part of a B-edge and *jumps* at junctions; arc-length re-placement makes
the B-position advance proportionally to A, so the matched sequence is jump-free. Validated by
``scripts/dag_dtw_validate.py``.
"""

import heapq
from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from .graph_dtw import LocalBGraph, build_local_digraph

Coord = Tuple[float, float]


class NotADAG(Exception):
    """Raised when the source graph GA contains a directed cycle (it must be acyclic)."""


def check_sequence_rules(res: Dict[str, Any], jump_tol: float = 3.0) -> Dict[str, Any]:
    """Verify the matched result obeys the map-matching **sequence rules** and return a report
    ``{ok, arc_viol, route_viol, jump_viol}``. Used by both the validation script and the
    playground figure so the drawing tells the truth about the match:

    - **monotone forward B-walk** (``arc_viol``): every GA arc ``a -> a'`` maps to a forward
      B-step ``φ(a) -> φ(a')`` (reachable along GB arcs); never backward / disconnected.
    - **connected route** (``route_viol``): consecutive B-edges in a route are graph-connected.
    - **no teleport** (``jump_viol``): the B-advance ``|φ(a) - φ(a')|`` may not exceed the
      A-advance ``|a - a'|`` by more than ``jump_tol`` metres -- catches a junction whose
      coincident A-vertices map to far-apart B-positions (graph-reachable, but a *jump* in B).

    ``jump_viol`` entries are ``(a, a', metres)`` so the drawing can highlight the offending arc.
    """
    ga, gb, phi = res["GA"], res["GB"], res["phi"]

    def reachable(src, dst):
        if src == dst:
            return True
        seen, q = {src}, deque([src])
        while q:
            u = q.popleft()
            for w in gb.succ_arcs[u]:
                if w == dst:
                    return True
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        return False

    arc_viol, jump_viol = [], []
    for a in range(ga.n_vertices):
        if a not in phi:
            continue
        for a2 in ga.succ_arcs[a]:
            if a2 not in phi:
                continue
            if not reachable(phi[a], phi[a2]):
                arc_viol.append((a, a2))
            bdist = float(np.hypot(gb.vx[phi[a]] - gb.vx[phi[a2]], gb.vy[phi[a]] - gb.vy[phi[a2]]))
            adist = float(np.hypot(ga.vx[a] - ga.vx[a2], ga.vy[a] - ga.vy[a2]))
            if bdist - adist > jump_tol:
                jump_viol.append((a, a2, round(bdist - adist, 1)))
    econ = set()
    for u in range(gb.n_vertices):
        for w in gb.succ_arcs[u]:
            eu, ew = gb.edge_ids[gb.vert_edge[u]], gb.edge_ids[gb.vert_edge[w]]
            if eu != ew:
                econ.add((eu, ew))
    route_viol = []
    for aeid, route in res.get("routes", {}).items():
        for i in range(1, len(route)):
            if (route[i - 1], route[i]) not in econ:
                route_viol.append((aeid, route[i - 1], route[i]))
    return {"ok": not (arc_viol or route_viol or jump_viol),
            "arc_viol": arc_viol, "route_viol": route_viol, "jump_viol": jump_viol}


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
# The DAG-DTW cost table (run forward, and backward on the reversed graphs)
# --------------------------------------------------------------------------------------
def _dp_table(pred_arcs, succ_arcs, outdeg, gb_succ, bu, bw, order, ax, ay, bx, by):
    """Fill the DAG-DTW cost table for a graph in ONE direction (docs §3, §3.2a). Called forward
    with (GA, GB) and backward with the reversed graphs; ``pred_arcs``/``succ_arcs`` are the A-graph
    adjacency in the current direction, ``outdeg`` its out-degree (the split factor -- ``1/outdeg``
    forward, ``1/indeg`` backward), and ``gb_succ``/``bu``/``bw`` the B-graph forward arcs in that
    direction. Returns the ``(NA, NB)`` cost table."""
    NA, NB = len(pred_arcs), len(bx)
    D = np.full((NA, NB), float("inf"))
    for a in order:
        ei = np.hypot(bx - ax[a], by - ay[a])          # E(a, ·): point-to-point drift
        preds = pred_arcs[a]
        if not preds:
            base = ei.copy()                           # source: free entry (empty A-sum = 0)
        else:
            acc = np.zeros(NB)
            for ap in preds:
                m = D[ap].copy()                       # v'=v (vertical)
                if bu.size:
                    np.minimum.at(m, bw, D[ap][bu])    # v'∈Bpred(v) (diagonal)
                acc += m / outdeg[ap]                  # conserved-flow split factor
            base = ei + acc
        D[a] = base.copy()                             # (H) within-a Dijkstra over the B-arcs
        heap = [(base[v], v) for v in range(NB)]
        heapq.heapify(heap)
        while heap:
            c, u = heapq.heappop(heap)
            if c > D[a][u]:
                continue
            for w in gb_succ[u]:
                cand = D[a][u] + ei[w]
                if cand < D[a][w]:
                    D[a][w] = cand
                    heapq.heappush(heap, (cand, w))
    return D


# --------------------------------------------------------------------------------------
# Arc-length re-match: place each A-vertex proportionally along its route (jump-free)
# --------------------------------------------------------------------------------------
def _arclength_rematch(ga, gb, routes, a_geoms, b_geoms, phi0):
    """The DP decides WHICH B-edges each A-edge maps to (the route/topology); this decides WHERE
    on them. Each A-vertex is placed at its **arc-length fraction** between an ``entry`` and an
    ``exit`` position on its route's B-polyline (snapped to the nearest route vertex), so the
    matched B-position advances *proportionally* to A -- no jump/compression.

    **Free entry / exit, like graph-DTW.** The A-edge's endpoint is *pinned* to the route boundary
    only when it is an interior **junction** (so branches meet consistently). At a DAG **source**
    (in-degree 0) the entry is FREE -- the A-start projects onto the route and may land in the
    *middle* of a B-edge; at a DAG **sink** (out-degree 0) the exit is FREE the same way. Vertices
    whose A-edge has no route keep their DP ``phi0``.
    """
    phi = dict(phi0)
    for e in range(len(ga.edge_ids)):
        aeid = ga.edge_ids[e]
        route = routes.get(aeid)
        if not route or aeid not in a_geoms:
            continue
        rv: List[int] = []                                   # route B-vertices, in route order
        for beid in route:
            try:
                be_idx = gb.edge_ids.index(beid)
            except ValueError:
                continue
            vids = list(np.where(gb.vert_edge == be_idx)[0])
            bg = b_geoms.get(beid)
            if bg is not None:
                vids.sort(key=lambda v: bg.project(Point(gb.vx[v], gb.vy[v])))
            rv.extend(int(v) for v in vids)
        if not rv:
            continue
        rvx = np.array([gb.vx[v] for v in rv])
        rvy = np.array([gb.vy[v] for v in rv])
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rvx), np.diff(rvy)))])
        route_len = cum[-1] or 1.0
        route_line = LineString(list(zip(rvx, rvy))) if len(rv) > 1 else None

        ageom = a_geoms[aeid]
        vids_a = list(np.where(ga.vert_edge == e)[0])
        s = np.array([ageom.project(Point(ga.vx[i], ga.vy[i])) for i in vids_a])
        oi = np.argsort(s)
        first, last = vids_a[oi[0]], vids_a[oi[-1]]          # A-edge start / end vertices
        smin, smax = float(s[oi[0]]), float(s[oi[-1]])
        span = (smax - smin) or 1.0

        # FREE at a DAG source/sink (project onto the route -> may be mid-edge); PINNED at a junction
        is_source = len(ga.pred_arcs[first]) == 0
        is_sink = len(ga.succ_arcs[last]) == 0
        if route_line is not None and is_source:
            entry = float(route_line.project(Point(ga.vx[first], ga.vy[first])))
        else:
            entry = 0.0
        if route_line is not None and is_sink:
            exit_ = float(route_line.project(Point(ga.vx[last], ga.vy[last])))
        else:
            exit_ = route_len
        if exit_ < entry:                                    # keep the span monotone
            entry, exit_ = exit_, entry

        for k, vi in enumerate(vids_a):
            target = entry + (exit_ - entry) * (float(s[k]) - smin) / span
            phi[int(vi)] = rv[int(np.argmin(np.abs(cum - target)))]
    return phi


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

    # --- FORWARD + BACKWARD tables, then JOINT junction resolution (docs §3.2a) ---
    # Forward D[a][v] = cheapest cost to align sources -> a with a at v.
    outdeg_f = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)
    D = _dp_table(ga.pred_arcs, ga.succ_arcs, outdeg_f, gb.succ_arcs, bu, bw,
                  order, ax, ay, bx, by)
    # Backward B[a][v] = cheapest cost to align a -> sinks with a at v: reverse BOTH graphs
    # (GA sinks become sources; GB arcs flip) and run the same DP with the symmetric 1/indeg split.
    outdeg_b = np.array([max(1, len(ga.pred_arcs[a])) for a in range(NA)], float)
    B = _dp_table(ga.succ_arcs, ga.pred_arcs, outdeg_b, gb.pred_arcs, bw, bu,
                  order[::-1], ax, ay, bx, by)

    sinks = [a for a in range(NA) if len(ga.succ_arcs[a]) == 0]
    total_cost = float(sum(np.min(D[t]) for t in sinks)) if sinks else float("inf")

    # φ by reverse-topological backtrack, combining BOTH ideas:
    #   * score each A-vertex by the JOINT cost D[a][v] + B[a][v] - E(a,v) -- the forward+backward
    #     value all routes through a agree on (subtract E, counted in both tables). This resolves
    #     two routes disagreeing at a shared junction (§3.2a);
    #   * subject to a REACHABILITY constraint -- the chosen v must still walk FORWARD to every
    #     successor's φ. This is the CHEAP enforcement of JOINT junction consistency (§3.2c): the
    #     junction labels are a joint decision, not per-junction argmin, and this constraint throws
    #     away the inconsistent tie-breaks that a shift creates (a shifted source edge collapsing
    #     onto a nearest cross road; two neighbours disagreeing so the chain can't run forward).
    # (Pure per-vertex argmin of D+B drops the constraint and regresses under perturbation; the
    #  greedy D-only score disagrees at shared junctions. Together they are clean on all trees.)
    reach_sets: Dict[int, set] = {}

    def _reach(v: int) -> set:
        s = reach_sets.get(v)
        if s is None:
            s = {v}
            stack = [v]
            while stack:
                u = stack.pop()
                for w in gb.succ_arcs[u]:
                    if w not in s:
                        s.add(w)
                        stack.append(w)
            reach_sets[v] = s
        return s

    phi: Dict[int, int] = {}
    for a in reversed(order):
        ei = np.hypot(bx - ax[a], by - ay[a])
        tot = D[a] + B[a] - ei
        succ = ga.succ_arcs[a]
        if not succ:                                   # sink: free choice (min joint cost)
            phi[a] = int(np.argmin(tot))
            continue
        targets = [phi[s] for s in succ if s in phi]
        chosen = None
        for v in np.argsort(tot):                      # jointly-cheapest first
            if not np.isfinite(tot[v]):
                break
            if all(t in _reach(int(v)) for t in targets):
                chosen = int(v)
                break
        phi[a] = chosen if chosen is not None else int(np.argmin(tot))

    # --- routes (topology) from the monotone backtrack, with junction-touch trim ---
    # per (A-edge, B-edge) vertex counts, in topological order, so a leading/trailing junction
    # TOUCH (a single A-vertex grazing the neighbouring B-edge at the junction) can be trimmed.
    seq_counts: Dict[Any, List[List[Any]]] = {}        # a_edge -> [[b_edge, count], ...] in order
    for a in order:
        v = phi.get(a)
        if v is None:
            continue
        aeid = ga.edge_ids[ga.vert_edge[a]] if 0 <= ga.vert_edge[a] < len(ga.edge_ids) else None
        beid = gb.edge_ids[gb.vert_edge[v]] if 0 <= gb.vert_edge[v] < len(gb.edge_ids) else None
        if beid is None:
            continue
        run = seq_counts.setdefault(aeid, [])
        if run and run[-1][0] == beid:
            run[-1][1] += 1
        else:
            run.append([beid, 1])

    routes: Dict[Any, List[Any]] = {}
    for aeid, run in seq_counts.items():
        r = run[:]
        # drop a leading / trailing single-vertex touch (the junction graze) if a real edge remains
        if len(r) > 1 and r[0][1] <= 1:
            r = r[1:]
        if len(r) > 1 and r[-1][1] <= 1:
            r = r[:-1]
        routes[aeid] = [b for b, _c in r]

    # --- assign a shared BOUNDARY B-edge to ONE A-edge: when consecutive A-edges overshoot /
    # undershoot the junction, one grabs the other's B-edge (route ends with what the next starts
    # with). Give that boundary edge to whichever A-edge covers it with MORE vertices; the other
    # yields it. Otherwise the junction-end pins to the far end of the shared edge -> a backward
    # step (the ``chain`` A1->A2 case).
    cnt: Dict[Tuple[Any, Any], int] = {}               # (a_edge, b_edge) -> vertex count
    for a in order:
        v = phi.get(a)
        if v is None:
            continue
        cnt[(ga.edge_ids[ga.vert_edge[a]], gb.edge_ids[gb.vert_edge[v]])] = \
            cnt.get((ga.edge_ids[ga.vert_edge[a]], gb.edge_ids[gb.vert_edge[v]]), 0) + 1
    a_succ_edges: Dict[Any, set] = {}                  # a_edge -> set of successor a_edges
    for u in range(NA):
        for w in ga.succ_arcs[u]:
            if ga.vert_edge[u] != ga.vert_edge[w]:
                a_succ_edges.setdefault(ga.edge_ids[ga.vert_edge[u]], set()).add(
                    ga.edge_ids[ga.vert_edge[w]])
    changed = True
    while changed:
        changed = False
        for e1, succs in a_succ_edges.items():
            r1 = routes.get(e1)
            if not r1:
                continue
            for e2 in succs:
                r2 = routes.get(e2)
                if not r2 or r1[-1] != r2[0]:
                    continue
                b = r1[-1]                              # the shared boundary B-edge
                if cnt.get((e1, b), 0) <= cnt.get((e2, b), 0) and len(r1) > 1:
                    routes[e1] = r1[:-1]                # predecessor covers less -> it yields
                    changed = True
                    break
                elif cnt.get((e1, b), 0) > cnt.get((e2, b), 0) and len(r2) > 1:
                    routes[e2] = r2[1:]                 # successor covers less -> it yields
                    changed = True
                    break

    # --- arc-length RE-MATCH: place each A-vertex proportionally along its route (jump-free) ---
    a_geoms = {eid: g for eid, g in a_edges}
    b_geoms = {eid: g for eid, g in b_edges}
    phi = _arclength_rematch(ga, gb, routes, a_geoms, b_geoms, phi)

    # --- per-A-vertex match + drift, from the re-matched φ ---
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
        res["debug"] = {"D": D, "B": B, "order": order}
    return res
