"""
Graph-DTW: generalize the segment-to-segment :func:`network_matching.dtw.dtw_align`
so the *target is a directed graph* of nearby B-edges instead of a single polyline.

Like ``dtw_align`` this is a **projection-based, continuous** DTW -- both sides are enriched with
the *projections of the other side's nodes* before the dynamic table is built:

- the **A axis** = A's own nodes + the projections of every candidate B vertex onto A
  (so projection points land on edge ``a`` too), and
- each **B-edge** = its own nodes + the projections of A's nodes onto that edge.

Where ``dtw_align`` aligns one A-edge to one B-edge, ``graph_dtw_align`` aligns one A-edge to the
whole **local directed graph** ``GB`` built from the B-edges near it. The warping path advances
monotonically along A while walking *forward through graph-connected* B-edges only, so it can
never hop onto a geometrically-close but topologically disconnected road (parallel-road /
junction disambiguation). Backtracking the dynamic table yields the ordered, connected route of
B-edges (each with a traversal direction) plus the drift metric.

The classic DTW predecessor ``j-1`` (a single point on one polyline) becomes "any
graph-predecessor ``u`` of ``v``" in ``GB``. Because ``GB`` may contain cycles (loops,
roundabouts) there is no topological sweep order for the within-row "B advances" move, so for
each fixed A-index the horizontal relaxation is solved as a non-negative-weight shortest path
(Dijkstra) -- exact regardless of cycles.

Public entry points
-------------------
- :func:`match_edge_to_bgraph` -- the atomic primitive: one A-edge + a list of B-edges -> match.
- :class:`GraphDTWMatcher` -- thin parameter-holding wrapper exposing ``.match_edge``.
- :func:`build_local_digraph` / :func:`graph_dtw_align` -- the building blocks.

Coordinates are assumed to already be in a local projected CRS in **meters** (as produced by
``DuckDBMapMatcher.generate_candidate_pairs``), so distances are plain Euclidean.
"""

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point

Coord = Tuple[float, float]
RouteEntry = Tuple[Any, str, int]  # (b_edge_id, direction, seq)
PoolPoint = Tuple[float, float, bool]  # (x, y, is_node)


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def _node_projection_pool(
    coords: Sequence[Coord],
    other_nodes: Sequence[Coord],
    step_meters: Optional[float] = 10.0,
) -> List[PoolPoint]:
    """Build a projection-enriched point pool along the polyline ``coords`` (mirrors the point
    pools in :func:`dtw_align`).

    The pool contains, ordered by arc-length:
      - every original vertex (**node**) of ``coords`` -> ``is_node=True``;
      - the projection (foot of perpendicular) of every point in ``other_nodes`` that falls
        strictly inside the line -> ``is_node=False``;
      - optional gap-fill samples wherever consecutive pool points are farther apart than
        ``step_meters`` (keeps long straight stretches from being coarse). Pass ``step_meters``
        falsy to disable.

    Returns a list of ``(x, y, is_node)``.
    """
    line = LineString(coords)
    length = line.length
    if length == 0:
        return [(float(coords[0][0]), float(coords[0][1]), True)]

    entries: List[Tuple[float, bool]] = []  # (arc_length, is_node)
    cum = 0.0
    entries.append((0.0, True))
    for k in range(1, len(coords)):
        cum += Point(coords[k - 1]).distance(Point(coords[k]))
        entries.append((min(cum, length), True))

    for on in other_nodes:
        s = line.project(Point(on))
        if 1e-9 < s < length - 1e-9:
            entries.append((s, False))

    entries.sort(key=lambda t: t[0])

    if step_meters:
        filled: List[Tuple[float, bool]] = []
        for idx, ent in enumerate(entries):
            filled.append(ent)
            if idx + 1 < len(entries):
                s0, s1 = ent[0], entries[idx + 1][0]
                gap = s1 - s0
                if gap > step_meters:
                    n = int(gap // step_meters)
                    for j in range(1, n + 1):
                        filled.append((s0 + j * step_meters, False))
        entries = sorted(filled, key=lambda t: t[0])

    # de-duplicate by arc-length, preferring is_node=True when points coincide
    pool: List[Tuple[float, bool]] = []
    for s, isn in entries:
        if pool and abs(pool[-1][0] - s) <= 1e-9:
            if isn:
                pool[-1] = (pool[-1][0], True)
            continue
        pool.append((s, isn))

    out: List[PoolPoint] = []
    for s, isn in pool:
        p = line.interpolate(float(s))
        out.append((float(p.x), float(p.y), isn))
    return out


def _bearing(p0: Coord, p1: Coord) -> float:
    """Absolute bearing (0-360 degrees) of the vector from ``p0`` to ``p1``."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def _inf_metrics() -> Dict[str, Any]:
    return {
        "average": float("inf"),
        "max": float("inf"),
        "min": float("inf"),
        "overlap_pct": 0,
        "matched_len": 0.0,
        "route": [],
        "route_edges": [],
        "n_edges": 0,
        "bearing_diff": float("inf"),
        "warp_vertices": [],
        "warp_is_node": [],
        "warp_a_is_node": [],
        "warp_edge": [],
    }


# --------------------------------------------------------------------------------------
# Local directed B-graph
# --------------------------------------------------------------------------------------
# An arc is (predecessor_or_successor_vertex, edge_index, direction) where ``direction`` is
# "forward" (along the B-edge's digitized geometry) or "backward".
Arc = Tuple[int, int, str]


@dataclass
class LocalBGraph:
    """Compact directed graph of the B-edges near one A-edge.

    Vertices are projection-enriched points (each B-edge's nodes + projections of A's nodes onto
    it). B-edges that share an endpoint (within the snap tolerance) are joined by collapsing
    those endpoints onto a single shared vertex, so connectivity is implicit through shared
    vertices (no separate junction arcs needed).
    """

    vx: np.ndarray                       # vertex x coords, length V
    vy: np.ndarray                       # vertex y coords, length V
    pred_arcs: List[List[Arc]]           # pred_arcs[v] = incoming arcs (u, edge, dir)
    succ_arcs: List[List[Arc]]           # succ_arcs[v] = outgoing arcs (w, edge, dir)
    edge_ids: List[Any]                  # edge_index -> original b_edge_id
    n_vertices: int
    is_node: np.ndarray                  # bool, length V: True = original B vertex, False = projection
    is_endpoint: np.ndarray              # bool, length V: True = an edge endpoint (junction / dead-end)
    b_raw_nodes: List[Coord]             # all raw geometry vertices of the candidate B-edges
    edge_len: List[float]                # edge_index -> total length (m) of that B-edge


def build_local_digraph(
    b_edges: Sequence[Tuple[Any, LineString]],
    a_coords: Sequence[Coord],
    snap_tolerance_m: float = 0.75,
    step_meters: float = 10.0,
    oneway_ids: Optional[Sequence[Any]] = None,
) -> LocalBGraph:
    """Build the local directed graph ``GB`` from ``(b_edge_id, LineString)`` edges, using
    projection-enriched vertices.

    Each B-edge's vertices = its own nodes + the projections of ``a_coords``' nodes onto that
    edge (continuous-DTW style). Consecutive vertices get a directed arc in the digitized
    direction ("forward") and, unless the edge id is in ``oneway_ids``, also the reverse
    ("backward") -- geometry-only conflation, so a B road digitized opposite to A is still
    walkable along A. Endpoints of different edges within ``snap_tolerance_m`` are merged onto a
    single shared vertex (how the route crosses from one B-edge to a connected next one).
    """
    oneway = set(oneway_ids or [])
    a_nodes = [(float(x), float(y)) for x, y in a_coords]
    # Sort candidate edges by id so vertex numbering (and thus DP tie-breaks) is deterministic
    # regardless of the order candidate rows arrive in (DuckDB joins are not order-stable).
    b_edges = sorted(b_edges, key=lambda t: str(t[0]))

    edge_ids: List[Any] = []
    edge_pools: List[List[PoolPoint]] = []
    b_raw_nodes: List[Coord] = []
    for eid, geom in b_edges:
        coords = list(geom.coords)
        if len(coords) < 2:
            continue  # skip degenerate edges
        edge_ids.append(eid)
        edge_pools.append(_node_projection_pool(coords, a_nodes, step_meters))
        b_raw_nodes.extend((float(c[0]), float(c[1])) for c in coords)

    # --- endpoint clustering (union-find over the 2*E endpoints) ---
    endpoints: List[Tuple[int, int, float, float]] = []  # (edge_idx, which 0/1, x, y)
    for e, pool in enumerate(edge_pools):
        endpoints.append((e, 0, pool[0][0], pool[0][1]))
        endpoints.append((e, 1, pool[-1][0], pool[-1][1]))

    parent = list(range(len(endpoints)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tol2 = snap_tolerance_m * snap_tolerance_m
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dx = endpoints[i][2] - endpoints[j][2]
            dy = endpoints[i][3] - endpoints[j][3]
            if dx * dx + dy * dy <= tol2:
                parent[find(i)] = find(j)

    # --- assign vertex ids (shared vertex per endpoint cluster, fresh id per interior pt) ---
    vx: List[float] = []
    vy: List[float] = []
    is_node: List[bool] = []
    is_endpoint: List[bool] = []
    cluster_vertex: Dict[int, int] = {}

    def endpoint_vertex(ep_index: int) -> int:
        root = find(ep_index)
        if root not in cluster_vertex:
            cluster_vertex[root] = len(vx)
            vx.append(endpoints[ep_index][2])
            vy.append(endpoints[ep_index][3])
            is_node.append(True)       # endpoints are original nodes
            is_endpoint.append(True)   # ...and they are edge endpoints (junctions/dead-ends)
        return cluster_vertex[root]

    edge_vertex_lists: List[List[int]] = []
    for e, pool in enumerate(edge_pools):
        m = len(pool)
        vids: List[int] = []
        for k, (px, py, isn) in enumerate(pool):
            if k == 0:
                vid = endpoint_vertex(e * 2 + 0)
            elif k == m - 1:
                vid = endpoint_vertex(e * 2 + 1)
            else:
                vid = len(vx)
                vx.append(px)
                vy.append(py)
                is_node.append(bool(isn))
                is_endpoint.append(False)
            vids.append(vid)
        edge_vertex_lists.append(vids)

    V = len(vx)
    pred_arcs: List[List[Arc]] = [[] for _ in range(V)]
    succ_arcs: List[List[Arc]] = [[] for _ in range(V)]

    def add_arc(u: int, w: int, e: int, direction: str) -> None:
        succ_arcs[u].append((w, e, direction))
        pred_arcs[w].append((u, e, direction))

    for e, vids in enumerate(edge_vertex_lists):
        is_oneway = edge_ids[e] in oneway
        for k in range(len(vids) - 1):
            u, w = vids[k], vids[k + 1]
            if u == w:
                continue
            add_arc(u, w, e, "forward")
            if not is_oneway:
                add_arc(w, u, e, "backward")

    # total length (m) of each B-edge (from its densified pool, which includes the endpoints)
    edge_len = []
    for pool in edge_pools:
        L = 0.0
        for t in range(1, len(pool)):
            L += float(np.hypot(pool[t][0] - pool[t - 1][0], pool[t][1] - pool[t - 1][1]))
        edge_len.append(L)

    return LocalBGraph(np.asarray(vx, float), np.asarray(vy, float),
                       pred_arcs, succ_arcs, edge_ids, V,
                       np.asarray(is_node, bool), np.asarray(is_endpoint, bool),
                       b_raw_nodes, edge_len)


# --------------------------------------------------------------------------------------
# Graph-DTW alignment (the dynamic table)
# --------------------------------------------------------------------------------------
def _collapse_route(crossed: List[Tuple[int, str]], edge_ids: List[Any]) -> List[RouteEntry]:
    """Collapse the backtracked sequence of crossed arcs (edge_index, direction) into an
    ordered route of distinct B-edge runs: ``[(b_edge_id, direction, seq), ...]``."""
    route: List[RouteEntry] = []
    seq = 0
    for e, direction in crossed:
        eid = edge_ids[e]
        if route and route[-1][0] == eid:
            continue  # same edge still being traversed
        route.append((eid, direction, seq))
        seq += 1
    return route


def graph_dtw_align(
    coords_a: Sequence[Coord],
    gb: LocalBGraph,
    step_meters: float = 10.0,
    trim_ends_m: float = 1.0,
) -> Tuple[float, List[Tuple[Coord, Coord]], Dict[str, Any]]:
    """Align A-edge ``coords_a`` to the local directed graph ``gb`` with projection-enriched
    points on both sides.

    The A axis is built here as A's nodes + projections of ``gb.b_raw_nodes`` onto A; ``gb``
    already carries each B-edge's nodes + projections of A's nodes. Mirrors
    :func:`network_matching.dtw.dtw_align`'s return contract:

    - ``average_distance``: mean drift (meters) along the warping path.
    - ``warping_path``: list of ``((ax, ay), (bx, by))`` aligned point pairs (A pool point ->
      B-graph vertex).
    - ``metrics``: ``{average, max, min, overlap_pct, matched_len, route, n_edges, bearing_diff,
      warp_vertices, warp_is_node, warp_a_is_node}``. ``warp_is_node`` / ``warp_a_is_node`` flag
      whether each matched B / A point is an original node (point-to-point) or a projection
      (point-to-projection).

    Dynamic table (monotonic in A, free in the B-graph)::

        D[i][v] = dist(a_i, v) + min(
            D[i-1][v],                          # vertical : advance A, stay at v
            min over u in pred(v) of D[i][u],   # horizontal: advance B along an arc, A stays
            min over u in pred(v) of D[i-1][u]  # diagonal : advance both
        )

    Row 0 is free-entry (``D[0][v] = dist(a_0, v)``). The horizontal term is resolved per row by
    a Dijkstra relaxation over the (non-negative-weight) arcs, exact even with cycles.
    Termination covers all of A (``min_v D[N-1][v]``); backtrack for the best matching.
    """
    a_pool = _node_projection_pool(list(coords_a), gb.b_raw_nodes, step_meters)
    N = len(a_pool)
    V = gb.n_vertices
    if N < 1 or V < 1:
        return float("inf"), [], _inf_metrics()

    ax = np.asarray([p[0] for p in a_pool], float)
    ay = np.asarray([p[1] for p in a_pool], float)
    a_is_node = [bool(p[2]) for p in a_pool]
    vx, vy = gb.vx, gb.vy
    INF = float("inf")

    D = np.full((N, V), INF)
    back: List[List[Optional[Tuple]]] = [[None] * V for _ in range(N)]

    def emit(i: int) -> np.ndarray:
        return np.hypot(ax[i] - vx, ay[i] - vy)  # length-V vector of A[i]->vertex distances

    # Row 0 -- free choice of B entry vertex.
    e0 = emit(0)
    for v in range(V):
        D[0][v] = e0[v]
        back[0][v] = ("START", -1, -1, "")

    for i in range(1, N):
        ei = emit(i)
        # base = vertical + diagonal (depend only on row i-1)
        for v in range(V):
            best = D[i - 1][v]
            bmove: Tuple = ("V", v, -1, "")
            for (u, e, direction) in gb.pred_arcs[v]:
                if D[i - 1][u] < best:
                    best = D[i - 1][u]
                    bmove = ("D", u, e, direction)
            D[i][v] = ei[v] + best
            back[i][v] = bmove

        # horizontal relaxation within row i (Dijkstra; arc weight = emission at landed vertex)
        heap = [(D[i][v], v) for v in range(V)]
        heapq.heapify(heap)
        while heap:
            c, v = heapq.heappop(heap)
            if c > D[i][v]:
                continue
            for (w, e, direction) in gb.succ_arcs[v]:
                cand = D[i][v] + ei[w]
                if cand < D[i][w]:
                    D[i][w] = cand
                    back[i][w] = ("H", v, e, direction)
                    heapq.heappush(heap, (cand, w))

    v_best = int(np.argmin(D[N - 1]))
    if not np.isfinite(D[N - 1][v_best]):
        return INF, [], _inf_metrics()

    # Backtrack -> per-step (a_index, vertex, edge-the-vertex-sits-on, direction).
    pairs: List[Tuple[int, int]] = []
    step_e: List[int] = []      # edge index of the arc INTO v (-1 for vertical/START)
    step_dir: List[str] = []
    i, v = N - 1, v_best
    while True:
        move, u, e, direction = back[i][v]
        pairs.append((i, v))
        step_e.append(e)
        step_dir.append(direction)
        if move == "START":
            break
        if move == "V":
            i -= 1
        elif move == "D":
            i -= 1
            v = u
        else:  # "H"
            v = u
    pairs.reverse(); step_e.reverse(); step_dir.reverse()

    # A vertical/START step has no arc (-1); it sits on the edge it arrived on -> forward-fill,
    # then back-fill the leading steps from the first real edge.
    last = -1
    for k in range(len(step_e)):
        if step_e[k] != -1:
            last = step_e[k]
        elif last != -1:
            step_e[k] = last
    first_valid = next((x for x in step_e if x != -1), -1)
    for k in range(len(step_e)):
        if step_e[k] == -1:
            step_e[k] = first_valid
        else:
            break

    warping_all = [((float(ax[i]), float(ay[i])), (float(vx[v]), float(vy[v]))) for (i, v) in pairs]

    # Group consecutive steps by the B-edge they sit on.
    groups: List[Tuple[int, int, int]] = []  # (start_idx, end_idx, edge_index)
    K0 = len(pairs)
    k = 0
    while k < K0:
        e = step_e[k]
        j = k
        while j + 1 < K0 and step_e[j + 1] == e:
            j += 1
        groups.append((k, j, e))
        k = j + 1

    def _a_len(gk: int, gj: int) -> float:
        # A-length covered by a group: each A-segment (t-1 -> t) attributed to its destination.
        L = 0.0
        for t in range(max(gk, 1), gj + 1):
            (pa0, _0), (pa1, _1) = warping_all[t - 1], warping_all[t]
            L += float(np.hypot(pa1[0] - pa0[0], pa1[1] - pa0[1]))
        return L

    # Trim spurious free-entry/exit fragments: leading/trailing edges that cover ~0 m of A
    # (the start/end vertex snapping onto a crossing edge). Only the ends are trimmed.
    g0, g1 = 0, len(groups) - 1
    if trim_ends_m and len(groups) > 1:
        while g0 < g1 and _a_len(groups[g0][0], groups[g0][1]) < trim_ends_m:
            g0 += 1
        while g1 > g0 and _a_len(groups[g1][0], groups[g1][1]) < trim_ends_m:
            g1 -= 1
    lo, hi = groups[g0][0], groups[g1][1]

    # Restrict the warping path to the kept span.
    pairs = pairs[lo:hi + 1]
    step_dir = step_dir[lo:hi + 1]
    step_e_kept = step_e[lo:hi + 1]
    warping = warping_all[lo:hi + 1]
    warp_vertices = [v for (_i, v) in pairs]
    warp_a_is_node = [a_is_node[i] for (i, _v) in pairs]
    warp_edge = [(gb.edge_ids[e] if 0 <= e < len(gb.edge_ids) else None) for e in step_e_kept]
    drift = [float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) for pa, pb in warping]
    average = float(np.mean(drift)) if drift else float("inf")

    # --- divide the result per B-edge the route passes through (kept groups, re-sequenced) ---
    kept_groups = [(gk - lo, gj - lo, ge) for (gk, gj, ge) in groups[g0:g1 + 1]]
    route_edges: List[Dict[str, Any]] = []
    for seq, (k, j, e) in enumerate(kept_groups):
        seg = drift[k:j + 1]
        a_len = b_len = 0.0
        for t in range(max(k, 1), j + 1):
            (pa0, pb0), (pa1, pb1) = warping[t - 1], warping[t]
            a_len += float(np.hypot(pa1[0] - pa0[0], pa1[1] - pa0[1]))
            b_len += float(np.hypot(pb1[0] - pb0[0], pb1[1] - pb0[1]))
        directions = [step_dir[t] for t in range(k, j + 1) if step_dir[t]]
        if j > k:
            (a_s, b_s) = warping[k]
            (a_e, b_e) = warping[j]
            bd = abs(_bearing(a_s, a_e) - _bearing(b_s, b_e))
            edge_bearing = float(min(bd, 360 - bd))
        else:
            edge_bearing = 0.0
        b_edge_len = gb.edge_len[e] if 0 <= e < len(gb.edge_len) else 0.0
        b_cover = 100.0 * b_len / b_edge_len if b_edge_len > 0 else 0.0
        route_edges.append({
            "dest_id": gb.edge_ids[e] if 0 <= e < len(gb.edge_ids) else None,
            "direction": directions[0] if directions else "forward",
            "seq": seq,                      # order of this B-edge along the route (0,1,2,...)
            "match_dist_avg": float(np.mean(seg)) if seg else float("inf"),
            "match_dist_max": float(np.max(seg)) if seg else float("inf"),
            "match_dist_min": float(np.min(seg)) if seg else float("inf"),
            "a_len": a_len,
            "matched_len": b_len,
            "b_edge_len": b_edge_len,
            "b_cover_pct": round(min(100.0, b_cover), 1),
            "bearing_diff": edge_bearing,
            "n_points": len(seg),
        })

    route = [(re["dest_id"], re["direction"], re["seq"]) for re in route_edges]
    matched_len = float(sum(re["matched_len"] for re in route_edges))

    total_a_len = LineString([p[:2] for p in a_pool]).length if N > 1 else 0.0
    kept_a = float(sum(re["a_len"] for re in route_edges))
    overlap_pct = int(min(100, round(100.0 * kept_a / total_a_len))) if total_a_len > 0 else 0

    a0, a1 = warping[0][0], warping[-1][0]
    b0, b1 = warping[0][1], warping[-1][1]
    bearing_diff = abs(_bearing(a0, a1) - _bearing(b0, b1))
    bearing_diff = float(min(bearing_diff, 360 - bearing_diff))

    metrics = {
        "average": average,
        "max": float(np.max(drift)) if drift else float("inf"),
        "min": float(np.min(drift)) if drift else float("inf"),
        "overlap_pct": overlap_pct,
        "matched_len": matched_len,
        "route": route,
        "route_edges": route_edges,
        "n_edges": len(route),
        "bearing_diff": bearing_diff,
        "warp_vertices": warp_vertices,
        "warp_is_node": [bool(gb.is_node[v]) for v in warp_vertices],
        "warp_a_is_node": warp_a_is_node,
        "warp_edge": warp_edge,
    }
    return average, warping, metrics


# --------------------------------------------------------------------------------------
# Public primitive + wrapper
# --------------------------------------------------------------------------------------
def match_edge_to_bgraph(
    coords_a: Sequence[Coord],
    b_edges: Sequence[Tuple[Any, LineString]],
    *,
    snap_tolerance_m: float = 0.75,
    step_meters: float = 10.0,
    oneway_ids: Optional[Sequence[Any]] = None,
    trim_ends_m: float = 1.0,
) -> Dict[str, Any]:
    """Map-match one A-edge to the local directed graph of nearby B-edges (continuous,
    projection-based DTW).

    The atomic, pure (strings/geometries in, dict out), parallelizable unit: build the local
    directed graph from ``b_edges`` (projection-enriched with A's nodes) and align ``coords_a``
    to it.

    Parameters
    ----------
    coords_a:
        Ordered ``(x, y)`` coordinates of the A-edge, in a projected CRS (meters).
    b_edges:
        List of ``(b_edge_id, shapely LineString in the same CRS)`` -- the candidate B-edges
        near this A-edge (e.g. one ``groupby('id_a')`` group of ``generate_candidate_pairs``).
    snap_tolerance_m:
        Endpoints of different B-edges within this distance are treated as a shared junction.
    step_meters:
        Optional gap-fill resolution added on top of the node+projection pools (set falsy to use
        pure node+projection pools like ``dtw_align``).
    oneway_ids:
        Optional collection of B-edge ids that may only be traversed in their digitized
        direction (no "backward" arcs).
    trim_ends_m:
        Drop leading/trailing route-edges that cover less than this many meters of A (spurious
        free-entry/exit fragments). Set 0 to disable.

    Returns
    -------
    dict with keys ``route`` (``[(b_edge_id, direction, seq), ...]``), ``warping_path``,
    ``metrics`` (see :func:`graph_dtw_align`), ``avg_distance``, and ``graph`` (the
    :class:`LocalBGraph`, handy for visualization).
    """
    gb = build_local_digraph(
        b_edges, coords_a, snap_tolerance_m=snap_tolerance_m,
        step_meters=step_meters, oneway_ids=oneway_ids,
    )
    avg, warping, metrics = graph_dtw_align(
        coords_a, gb, step_meters=step_meters, trim_ends_m=trim_ends_m)
    return {
        "route": metrics["route"],
        "warping_path": warping,
        "metrics": metrics,
        "avg_distance": avg,
        "graph": gb,
    }


class GraphDTWMatcher:
    """Thin wrapper holding graph-DTW parameters; exposes :meth:`match_edge`.

    Convenience for scripts/notebooks and a single-edge debug tool (parallels the existing
    ``scripts/diagnose_match.py``). The heavy pipeline (candidate generation, parallel
    fan-out, output tables) is intentionally *not* here -- this is the algorithm primitive.
    """

    def __init__(self, snap_tolerance_m: float = 0.75, step_meters: float = 10.0,
                 oneway_ids: Optional[Sequence[Any]] = None, trim_ends_m: float = 1.0):
        self.snap_tolerance_m = snap_tolerance_m
        self.step_meters = step_meters
        self.oneway_ids = oneway_ids
        self.trim_ends_m = trim_ends_m

    def match_edge(self, coords_a: Sequence[Coord],
                   b_edges: Sequence[Tuple[Any, LineString]]) -> Dict[str, Any]:
        return match_edge_to_bgraph(
            coords_a, b_edges,
            snap_tolerance_m=self.snap_tolerance_m,
            step_meters=self.step_meters,
            oneway_ids=self.oneway_ids,
            trim_ends_m=self.trim_ends_m,
        )
