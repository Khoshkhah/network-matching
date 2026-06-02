"""
Graph-DTW: generalize the segment-to-segment :func:`network_matching.dtw.dtw_align`
so the *target is a directed graph* of nearby B-edges instead of a single polyline.

Like ``dtw_align`` this is a **projection-based, continuous** DTW -- both sides are enriched with
the *projections of the other side's nodes* before the dynamic table is built:

- the **A axis** = A's own nodes + the projections of every candidate B vertex onto A
  (so projection points land on edge ``a`` too), and
- each **B-edge** = its own nodes + the projections of A's nodes onto that edge.

Where ``dtw_align`` aligns one A-edge to one B-edge, ``graph_dtw_align`` aligns one A-edge to the
whole **local directed graph** ``GB`` built from the B-edges near it. The B table is already
fully directed (every bidirectional road is two opposing directed edges), so ``GB`` uses
*forward arcs only*: each edge is walked start->end and connectivity is head-to-tail. The warping
path advances monotonically along A while walking forward through graph-connected B-edges only,
so it can never hop onto a geometrically-close but topologically disconnected road, and -- because
each vertex carries the edge it lies on (the DP state knows its edge) -- it can never U-turn onto
an edge that merely touches a junction. Backtracking the dynamic table yields the ordered,
connected route of B-edges plus the drift metric.

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
# The B network is already a fully **directed** edge table (every bidirectional road appears as
# two opposing directed edges -- e.g. the NVDB ``is_reverse`` twins), so the graph uses *forward
# arcs only*: each edge is traversed start->end along its digitized geometry, and connectivity is
# head-to-tail (one edge's end vertex feeding the next edge's start vertex). An arc is simply the
# neighbour vertex id; the edge a vertex lies on is carried in ``vert_edge``.
Arc = int


@dataclass
class LocalBGraph:
    """Compact directed graph of the B-edges near one A-edge.

    Vertices are projection-enriched points, and **every vertex belongs to exactly one B-edge**
    (``vert_edge``). Endpoints of different edges that meet at a junction are kept as *separate
    coincident vertices*, stitched by directed inter-edge arcs (edge ``u``'s end -> edge ``w``'s
    start). Keeping the owning edge in the vertex -- hence in the DP state -- is what makes the
    route unambiguous at junctions: a state is never "at the junction", it is "at edge ``u``'s
    end" or "at edge ``w``'s start", which are different cells of the table.
    """

    vx: np.ndarray                       # vertex x coords, length V
    vy: np.ndarray                       # vertex y coords, length V
    pred_arcs: List[List[Arc]]           # pred_arcs[v] = incoming neighbour vertices
    succ_arcs: List[List[Arc]]           # succ_arcs[v] = outgoing neighbour vertices
    edge_ids: List[Any]                  # edge_index -> original b_edge_id
    n_vertices: int
    is_node: np.ndarray                  # bool, length V: True = original B vertex, False = projection
    is_endpoint: np.ndarray              # bool, length V: True = an edge endpoint (junction / dead-end)
    vert_edge: np.ndarray                # int, length V: vertex -> the B-edge index it lies on
    b_raw_nodes: List[Coord]             # all raw geometry vertices of the candidate B-edges
    edge_len: List[float]                # edge_index -> total length (m) of that B-edge


def build_local_digraph(
    b_edges: Sequence[Tuple[Any, LineString]],
    a_coords: Sequence[Coord],
    snap_tolerance_m: float = 0.75,
    step_meters: float = 10.0,
) -> LocalBGraph:
    """Build the local **directed** graph ``GB`` from ``(b_edge_id, LineString)`` edges.

    The B table is already directed (each road's two travel directions are separate rows), so
    arcs are *forward only*: each edge contributes intra-edge arcs along its digitized direction,
    and where one edge's **end** coincides with another edge's **start** (within
    ``snap_tolerance_m``) a directed inter-edge arc joins them head-to-tail. Every vertex belongs
    to exactly one edge (``vert_edge``); junction endpoints are kept as *separate coincident
    vertices* rather than merged, so the DP state always knows which edge it is on. Each edge's
    vertices = its own nodes + the projections of ``a_coords``' nodes onto it (continuous-DTW
    style). If A runs opposite to an edge, it simply matches that edge's reverse twin (a different
    ``directed_id``) instead -- no synthesized backward arcs needed.
    """
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

    # --- vertices: every pool point is its own vertex, owned by exactly one edge ---
    # Junction endpoints are NOT merged: each edge keeps its own endpoint vertices, so a vertex
    # (and thus the DP state) always knows which edge it lies on. Connectivity is added as
    # directed inter-edge arcs below.
    vx: List[float] = []
    vy: List[float] = []
    is_node: List[bool] = []
    is_endpoint: List[bool] = []
    vert_edge: List[int] = []
    edge_vertex_lists: List[List[int]] = []
    for e, pool in enumerate(edge_pools):
        m = len(pool)
        vids: List[int] = []
        for k, (px, py, isn) in enumerate(pool):
            vids.append(len(vx))
            vx.append(px)
            vy.append(py)
            is_node.append(bool(isn))
            is_endpoint.append(k == 0 or k == m - 1)
            vert_edge.append(e)
        edge_vertex_lists.append(vids)

    V = len(vx)
    pred_arcs: List[List[Arc]] = [[] for _ in range(V)]
    succ_arcs: List[List[Arc]] = [[] for _ in range(V)]

    def add_arc(u: int, w: int) -> None:
        succ_arcs[u].append(w)
        pred_arcs[w].append(u)

    # intra-edge arcs: forward (digitized) direction only -- the reverse direction is its own row.
    for vids in edge_vertex_lists:
        for k in range(len(vids) - 1):
            if vids[k] != vids[k + 1]:
                add_arc(vids[k], vids[k + 1])

    # inter-edge arcs: one edge's END -> another edge's START where they coincide within the snap
    # tolerance (head-to-tail). This is the directed junction crossing; no backward/U-turn arcs.
    tol2 = snap_tolerance_m * snap_tolerance_m
    starts = [vids[0] for vids in edge_vertex_lists]
    ends = [vids[-1] for vids in edge_vertex_lists]
    for eu, u in enumerate(ends):
        for ew, w in enumerate(starts):
            if eu == ew:
                continue
            dx = vx[u] - vx[w]
            dy = vy[u] - vy[w]
            if dx * dx + dy * dy <= tol2:
                add_arc(u, w)

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
                       np.asarray(vert_edge, int), b_raw_nodes, edge_len)


# --------------------------------------------------------------------------------------
# Graph-DTW alignment (the dynamic table)
# --------------------------------------------------------------------------------------
def graph_dtw_align(
    coords_a: Sequence[Coord],
    gb: LocalBGraph,
    step_meters: float = 10.0,
    trim_ends_m: float = 0.0,
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
        back[0][v] = ("START", -1)

    for i in range(1, N):
        ei = emit(i)
        # base = vertical + diagonal (depend only on row i-1)
        for v in range(V):
            best = D[i - 1][v]
            bmove: Tuple = ("V", v)
            for u in gb.pred_arcs[v]:
                if D[i - 1][u] < best:
                    best = D[i - 1][u]
                    bmove = ("D", u)
            D[i][v] = ei[v] + best
            back[i][v] = bmove

        # horizontal relaxation within row i (Dijkstra; arc weight = emission at landed vertex)
        heap = [(D[i][v], v) for v in range(V)]
        heapq.heapify(heap)
        while heap:
            c, v = heapq.heappop(heap)
            if c > D[i][v]:
                continue
            for w in gb.succ_arcs[v]:
                cand = D[i][v] + ei[w]
                if cand < D[i][w]:
                    D[i][w] = cand
                    back[i][w] = ("H", v)
                    heapq.heappush(heap, (cand, w))

    v_best = int(np.argmin(D[N - 1]))
    if not np.isfinite(D[N - 1][v_best]):
        return INF, [], _inf_metrics()

    # Backtrack -> ordered (a_index, vertex) pairs.
    pairs: List[Tuple[int, int]] = []
    i, v = N - 1, v_best
    while True:
        move, u = back[i][v]
        pairs.append((i, v))
        if move == "START":
            break
        if move == "V":
            i -= 1
        elif move == "D":
            i -= 1
            v = u
        else:  # "H"
            v = u
    pairs.reverse()

    # The edge each step sits on is read DIRECTLY from the vertex (``vert_edge``) -- no arc
    # inference, no shared-junction ambiguity: every vertex belongs to exactly one edge.
    step_e: List[int] = [int(gb.vert_edge[v]) for (_i, v) in pairs]

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

    def _b_len(gk: int, gj: int) -> float:
        # B-length actually traversed by a group (physical movement along the edge).
        L = 0.0
        for t in range(max(gk, 1), gj + 1):
            (_0, pb0), (_1, pb1) = warping_all[t - 1], warping_all[t]
            L += float(np.hypot(pb1[0] - pb0[0], pb1[1] - pb0[1]))
        return L

    g0, g1 = 0, len(groups) - 1
    # Always drop a leading/trailing edge with ZERO B traversal: A merely touches that edge's
    # boundary vertex at a junction (its end overhangs onto the next edge's start) but never walks
    # it -- it is overhang, not a match. Without this the route lists an extra edge the picture
    # shows no traversal of (B-used == 0%).
    while g0 < g1 and _b_len(groups[g0][0], groups[g0][1]) <= 1e-9:
        g0 += 1
    while g1 > g0 and _b_len(groups[g1][0], groups[g1][1]) <= 1e-9:
        g1 -= 1
    # Optional further trim of leading/trailing fragments by A-length covered (off by default).
    if trim_ends_m and g1 > g0:
        while g0 < g1 and _a_len(groups[g0][0], groups[g0][1]) < trim_ends_m:
            g0 += 1
        while g1 > g0 and _a_len(groups[g1][0], groups[g1][1]) < trim_ends_m:
            g1 -= 1
    lo, hi = groups[g0][0], groups[g1][1]

    # If B is never actually traversed (A only touches boundary vertices -- e.g. a stub that ends
    # at, but never runs along, a B-edge), there is no real match -> NO_MATCH.
    if _b_len(lo, hi) <= 1e-9:
        return INF, [], _inf_metrics()

    drift_all = [float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) for pa, pb in warping_all]

    # Coverage = all of A EXCEPT the leading run on the route's entry vertex and the trailing run
    # on its terminal vertex -- i.e. only where A OVERHANGS past the route's first/last B-edge
    # endpoint. A mid-corridor stall (A denser than B, so the B vertex momentarily waits) is still
    # COVERED -- it is not overhang. ``ve`` / ``vt`` are the entry / terminal B vertices.
    ve, vt = pairs[lo][1], pairs[hi][1]

    # --- per-B-edge breakdown, on the FULL warping with absolute indices (so the A-segment that
    # bridges a trimmed end edge to the first/last KEPT edge is still counted). ---
    kept_groups = groups[g0:g1 + 1]
    route_edges: List[Dict[str, Any]] = []
    seq = 0
    for (k, j, e) in kept_groups:
        if not (0 <= e < len(gb.edge_ids)):
            continue  # degenerate group: A matched a vertex without crossing any B-arc (no B-edge)
        seg = drift_all[k:j + 1]
        # a_len = A covered by this edge: an A-segment counts UNLESS it is leading overhang (B still
        # on the entry vertex ``ve``) or trailing overhang (B already settled on the terminal vertex
        # ``vt``). b_len = B traversed (sum of B movement); 0 on a collapse, so a fully-walked B-edge
        # is still 100% used.
        a_len = b_len = 0.0
        for t in range(max(k, 1), j + 1):
            (pa0, pb0), (pa1, pb1) = warping_all[t - 1], warping_all[t]
            b_len += float(np.hypot(pb1[0] - pb0[0], pb1[1] - pb0[1]))
            is_lead = pairs[t][1] == ve            # B has not left the entry vertex yet
            is_trail = pairs[t - 1][1] == vt       # B has already settled on the terminal vertex
            if not is_lead and not is_trail:
                a_len += float(np.hypot(pa1[0] - pa0[0], pa1[1] - pa0[1]))
        if j > k:
            (a_s, b_s) = warping_all[k]
            (a_e, b_e) = warping_all[j]
            bd = abs(_bearing(a_s, a_e) - _bearing(b_s, b_e))
            edge_bearing = float(min(bd, 360 - bd))
        else:
            edge_bearing = 0.0
        b_edge_len = gb.edge_len[e] if 0 <= e < len(gb.edge_len) else 0.0
        b_cover = 100.0 * b_len / b_edge_len if b_edge_len > 0 else 0.0
        route_edges.append({
            "dest_id": gb.edge_ids[e],
            "direction": "forward",          # directed table: every edge is traversed start->end
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
        seq += 1

    route = [(re["dest_id"], re["direction"], re["seq"]) for re in route_edges]
    matched_len = float(sum(re["matched_len"] for re in route_edges))

    # A coverage: the warping spans all of A, but A-length where A OVERHANGS past the first/last
    # edge's endpoint (a run of A-points collapsing onto a single B vertex) is uncovered.
    # overlap_pct = covered A / A; it is < 100 whenever A's ends stick out past B's corridor
    # (a segmentation/overhang effect that can happen on any network -- not a dead-end).
    total_a_len = LineString([p[:2] for p in a_pool]).length if N > 1 else 0.0
    kept_a = float(sum(re["a_len"] for re in route_edges))
    overlap_pct = int(min(100, round(100.0 * kept_a / total_a_len))) if total_a_len > 0 else 0
    # per-edge A coverage as % of the WHOLE A-edge (these sum to overlap_pct)
    for re in route_edges:
        re["cover_pct"] = round(100.0 * re["a_len"] / total_a_len, 1) if total_a_len > 0 else 0.0

    # Returned warping path / per-step arrays: sliced to the kept span for clean visualization.
    pairs_k = pairs[lo:hi + 1]
    step_e_kept = step_e[lo:hi + 1]
    warping = warping_all[lo:hi + 1]
    warp_vertices = [v for (_i, v) in pairs_k]
    warp_a_is_node = [a_is_node[i] for (i, _v) in pairs_k]
    warp_edge = [(gb.edge_ids[e] if 0 <= e < len(gb.edge_ids) else None) for e in step_e_kept]
    drift = drift_all[lo:hi + 1]
    average = float(np.mean(drift)) if drift else float("inf")

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
    trim_ends_m: float = 0.0,
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
        Endpoints of different B-edges within this distance are joined head-to-tail (one edge's
        end -> another's start) into a junction crossing.
    step_meters:
        Optional gap-fill resolution added on top of the node+projection pools (set falsy to use
        pure node+projection pools like ``dtw_align``).
    trim_ends_m:
        Default ``0`` (off). Optional cleanup that *removes* a leading/trailing route edge
        covering less than this many meters of A. Not a gap-filler (use ``snap_tolerance_m`` for
        connectivity); off by default because it can delete legitimate corridor edges.

    Returns
    -------
    dict with keys ``route`` (``[(b_edge_id, direction, seq), ...]``), ``warping_path``,
    ``metrics`` (see :func:`graph_dtw_align`), ``avg_distance``, and ``graph`` (the
    :class:`LocalBGraph`, handy for visualization).
    """
    gb = build_local_digraph(
        b_edges, coords_a, snap_tolerance_m=snap_tolerance_m, step_meters=step_meters,
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
                 trim_ends_m: float = 0.0):
        self.snap_tolerance_m = snap_tolerance_m
        self.step_meters = step_meters
        self.trim_ends_m = trim_ends_m

    def match_edge(self, coords_a: Sequence[Coord],
                   b_edges: Sequence[Tuple[Any, LineString]]) -> Dict[str, Any]:
        return match_edge_to_bgraph(
            coords_a, b_edges,
            snap_tolerance_m=self.snap_tolerance_m,
            step_meters=self.step_meters,
            trim_ends_m=self.trim_ends_m,
        )
