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
    min_gap_m: float = 0.0,
) -> List[PoolPoint]:
    """Build a projection-enriched point pool along the polyline ``coords`` (mirrors the point
    pools in :func:`dtw_align`).

    The pool contains, ordered by arc-length:
      - every original vertex (**node**) of ``coords`` -> ``is_node=True``;
      - the projection (foot of perpendicular) of every point in ``other_nodes`` that falls
        strictly inside the line -> ``is_node=False``;
      - optional gap-fill samples wherever consecutive pool points are farther apart than
        ``step_meters``, distributed EVENLY within each gap (resulting spacing is always in
        ``(step_meters/2, step_meters]``, never a leftover sliver). Pass ``step_meters`` falsy
        to disable.

    ``min_gap_m`` > 0 additionally drops added (non-node) points that land closer than this to
    a kept neighbour, so no pool segment shorter than ``min_gap_m`` is created by enrichment --
    the segment-emission modes use this to guarantee genuine segment-to-segment states.
    Original geometry vertices are never dropped.

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
                    # spread the fill EVENLY over the gap (spacing in (step/2, step]) instead of
                    # stepping from s0 and leaving a leftover sliver before the next point
                    n = int(np.ceil(gap / step_meters)) - 1
                    spacing = gap / (n + 1)
                    for j in range(1, n + 1):
                        filled.append((s0 + j * spacing, False))
        entries = sorted(filled, key=lambda t: t[0])

    # de-duplicate by arc-length, preferring is_node=True when points coincide
    pool: List[Tuple[float, bool]] = []
    for s, isn in entries:
        if pool and abs(pool[-1][0] - s) <= 1e-9:
            if isn:
                pool[-1] = (pool[-1][0], True)
            continue
        pool.append((s, isn))

    # merge away enrichment slivers: a non-node closer than min_gap_m to its kept predecessor
    # (or to a following node) is dropped; real geometry nodes are always kept
    if min_gap_m > 0 and len(pool) > 2:
        kept: List[Tuple[float, bool]] = [pool[0]]
        for s, isn in pool[1:]:
            if isn:
                if not kept[-1][1] and s - kept[-1][0] < min_gap_m and len(kept) > 1:
                    kept.pop()                       # non-node crowding a real node: drop it
                kept.append((s, isn))
            elif s - kept[-1][0] >= min_gap_m:
                kept.append((s, isn))
        pool = kept

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
    min_gap_m: float = 0.0,
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
        edge_pools.append(_node_projection_pool(coords, a_nodes, step_meters, min_gap_m))
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


def _segment_dp_pairs(gb: LocalBGraph, ax: np.ndarray, ay: np.ndarray,
                      bearing_weight: float = 0.0, alpha: float = 1.0, beta: float = 1.0,
                      dbg: Optional[Dict[str, Any]] = None) -> Optional[List[Tuple[int, int]]]:
    """True segment-to-segment DP (``emission="segment"``): states are (A-segment i, B-arc u->v).

    Every state pays ONE distance between the two segment MIDDLES,
    ``|mid(a_i, a_{i+1}) - mid(u, v)|`` -- plus, with ``bearing_weight`` > 0, a per-state heading
    penalty -- so NO alignment move can bypass the local cost. (The earlier point-state
    formulation charged bearing only on diagonal moves, letting the DP escape via stalls and
    collapse the route; see ``docs/weighted_emission.md`` §9-§11.) Non-ridable stitch arcs are
    free (pure connectivity: a zero-length junction connector is not a segment, so crossing it
    carries no cost) and can never host a state. Moves: A-advance on the same arc (N segments :
    1 arc), arc-advance within a row (1 segment : N arcs; Dijkstra, each arc entered pays), or
    both. Each entered state pays its emission weighted by the move that enters it -- ``beta`` on
    the N:1 stall, ``alpha`` on the 1:N arc-advance, ``1`` on the diagonal (docs §12; defaults
    reproduce the unweighted recurrence). Returns the vertex-level ``(a_index, vertex)`` pairs of
    the best alignment -- the same format the point-state backtrack yields, so all downstream
    grouping/metrics are shared -- or ``None`` if no finite alignment exists.

    ``dbg``: optional dict populated with the DP internals (``arcs``, ``ridable``, ``D``, ``E``,
    ``arc_path``, ``terminal_state``, ``final_cost``) for algorithm debugging -- see
    :func:`graph_dtw_align`'s ``debug`` flag.
    """
    V = gb.n_vertices
    N = len(ax)
    if N < 2:
        return None
    # Enumerate the arcs once; a DP state indexes this list.
    au: List[int] = []
    av: List[int] = []
    for u in range(V):
        for w in gb.succ_arcs[u]:
            au.append(u)
            av.append(w)
    NA = len(au)
    if NA == 0:
        return None
    auv = np.asarray(au, int)
    avv = np.asarray(av, int)
    arcs_from: List[List[int]] = [[] for _ in range(V)]   # arcs starting at a vertex
    arcs_to: List[List[int]] = [[] for _ in range(V)]     # arcs ending at a vertex
    for k in range(NA):
        arcs_from[au[k]].append(k)
        arcs_to[av[k]].append(k)

    vx, vy = gb.vx, gb.vy
    ux, uy = vx[auv], vy[auv]
    hx, hy = vx[avv], vy[avv]
    arc_len = np.hypot(hx - ux, hy - uy)
    # A junction-snap stitch (sub-half-metre end->start connector) is CONNECTIVITY, not a segment:
    # it has no meaningful heading and an A-segment must never pair with it. Non-ridable arcs can
    # only be passed through within a row (H); they carry no (A-segment : arc) state of their own.
    ridable = arc_len >= 0.5
    bw = float(bearing_weight)
    al, be = float(alpha), float(beta)                # §12 step weights (1, 1 = unweighted)
    if bw > 0.0:
        arc_bear = (np.degrees(np.arctan2(hx - ux, hy - uy)) + 360.0) % 360.0
        seg_bear = (np.degrees(np.arctan2(np.diff(ax), np.diff(ay))) + 360.0) % 360.0

    NS = N - 1                        # number of A-segments
    INF = float("inf")
    D = np.full((NS, NA), INF)
    back: List[List[Optional[Tuple]]] = [[None] * NA for _ in range(NS)]
    if dbg is not None:
        dbg["arcs"] = list(zip(au, av))
        dbg["ridable"] = ridable.copy()
        _E_rows: List[np.ndarray] = []

    def emit_seg(i: int) -> np.ndarray:
        # E(i, e) = |mid(a_i, a_{i+1}) - mid(u, v)| -- one middle-to-middle distance;
        # stitches (non-ridable) are free: connectivity, not a segment to pay for.
        e = np.hypot(0.5 * (ax[i] + ax[i + 1]) - 0.5 * (ux + hx),
                     0.5 * (ay[i] + ay[i + 1]) - 0.5 * (uy + hy))
        e = np.where(ridable, e, 0.0)
        if bw > 0.0:                  # [+ lambda * circular bearing diff, ridable arcs only]
            bd = np.abs(seg_bear[i] - arc_bear)
            e = e + bw * np.where(ridable, np.minimum(bd, 360.0 - bd), 0.0)
        if dbg is not None:
            _E_rows.append(e)         # rows arrive in order: emit_seg(0), emit_seg(1), ...
        return e

    def relax_row(i: int, ei: np.ndarray) -> None:
        # 1 A-segment : N arcs (coverage) -- Dijkstra within the row; every arc entered pays
        # alpha * its emission (§12).
        heap = [(D[i][k], k) for k in range(NA)]
        heapq.heapify(heap)
        while heap:
            c, k = heapq.heappop(heap)
            if c > D[i][k]:
                continue
            for nb in arcs_from[av[k]]:
                cand = D[i][k] + al * ei[nb]
                if cand < D[i][nb]:
                    D[i][nb] = cand
                    back[i][nb] = ("H", k)
                    heapq.heappush(heap, (cand, nb))

    e0 = emit_seg(0)
    for k in range(NA):
        if ridable[k]:                              # a stitch cannot host an (A-segment : arc) state
            D[0][k] = e0[k]
            back[0][k] = ("START", -1)
    relax_row(0, e0)
    for i in range(1, NS):
        ei = emit_seg(i)
        for k in range(NA):
            if not ridable[k]:                      # stitches are reachable only via H (pass-through)
                continue
            cand = be * ei[k] + D[i - 1][k]         # V: next segment stays -- N:1 stall, beta*E
            bmove: Tuple = ("V", k)
            for p in arcs_to[au[k]]:                # D: both advance -- full E
                c2 = ei[k] + D[i - 1][p]
                if c2 < cand:
                    cand = c2
                    bmove = ("D", p)
            if np.isfinite(cand):
                D[i][k] = cand
                back[i][k] = bmove
        relax_row(i, ei)

    k_best = int(np.argmin(D[NS - 1]))
    if dbg is not None:
        dbg["D"] = D
        dbg["E"] = np.vstack(_E_rows)
    if not np.isfinite(D[NS - 1][k_best]):
        if dbg is not None:
            dbg["reason"] = "no_finite_alignment"
        return None

    states: List[Tuple[int, int]] = []
    moves: List[str] = []
    i, k = NS - 1, k_best
    while True:
        move, p = back[i][k]
        states.append((i, k))
        moves.append(move)
        if move == "START":
            break
        if move == "V":
            i -= 1
        elif move == "D":
            i -= 1
            k = p
        else:  # "H"
            k = p
    states.reverse()
    moves.reverse()
    if dbg is not None:
        dbg["arc_path"] = [(si, sk, mv) for (si, sk), mv in zip(states, moves)]
        dbg["terminal_state"] = k_best
        dbg["final_cost"] = float(D[NS - 1][k_best])

    # States -> vertex pairs: the entry arc contributes its tail, every state its head.
    pairs: List[Tuple[int, int]] = [(states[0][0], au[states[0][1]])]
    for (i, k) in states:
        pairs.append((i + 1, av[k]))
    return pairs


# --------------------------------------------------------------------------------------
# Graph-DTW alignment (the dynamic table)
# --------------------------------------------------------------------------------------
def graph_dtw_align(
    coords_a: Sequence[Coord],
    gb: LocalBGraph,
    step_meters: float = 10.0,
    trim_ends_m: float = 0.0,
    emission: str = "point",
    bearing_weight: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    debug: bool = False,
    min_gap_m: float = 0.0,
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

    Dynamic table (monotonic in A, free in the B-graph); each state pays its emission weighted
    by the move that ENTERS it (``alpha``/``beta``, docs/weighted_emission.md §12)::

        D[i][v] = min(
            beta  * dist(a_i, v) + D[i-1][v],                        # vertical  : N:1 stall
            alpha * dist(a_i, v) + min over u in pred(v) of D[i][u], # horizontal: 1:N coverage
            1     * dist(a_i, v) + min over u in pred(v) of D[i-1][u]  # diagonal: advance both
        )

    Row 0 is free-entry (``D[0][v] = dist(a_0, v)``, then a row-0 horizontal relaxation so a
    LEADING coverage run is priced like any other -- a no-op at ``alpha = 1``). The horizontal
    term is resolved per row by a Dijkstra relaxation over the (non-negative-weight) arcs, exact
    even with cycles. Termination covers all of A (``min_v D[N-1][v]``); backtrack for the best
    matching. Domain ``alpha in (0, 1]``, ``beta in [1, inf)``, defaults ``1``/``1`` -- the
    unweighted recurrence, byte-for-byte (same names and semantics as ``match_dag``, Mode 3).
    The weights shape the CHOICE; the reported ``average``/``max``/``min`` stay raw geometry of
    the chosen warping.

    ``emission`` selects the local cost MODEL. ``"point"`` (default): the recurrence above --
    states are (A-point, B-vertex), each cell adds ``dist(a_i, v)``. ``"segment"``: true
    segment-to-segment -- states are (A-segment, B-arc) and EVERY state pays ONE distance between
    the two segment MIDDLES ``|mid(a_i, a_{i+1}) - mid(u, v)|`` plus, with ``bearing_weight`` > 0,
    a per-state heading penalty ``lambda * circ_diff(bearing(seg), bearing(arc))``; no alignment
    move can bypass either term (see :func:`_segment_dp_pairs` and ``docs/weighted_emission.md``
    §11). Junction stitches are free (pure connectivity), and the reported
    ``average``/``max``/``min`` (overall and per route edge) are statistics of those
    middle-to-middle distances over the matched states. ``min_gap_m`` is forwarded to the A-axis
    pool (see :func:`_node_projection_pool`), keeping segment states sliver-free.

    ``debug=True`` attaches the raw algorithm internals under ``metrics["debug"]`` (also on
    failure returns, with a ``reason``), for the debug tooling in
    ``scripts/graph_dtw_debug_viz.py``:

    - always: ``params``, ``a_pool`` (``(x, y, is_node)`` A-axis points), and -- once the DP ran --
      ``D`` (accumulated-cost table), ``E`` (per-state emission table), ``final_cost``,
      ``terminal_state``;
    - on success: ``pairs_all`` (full untrimmed ``(a_index, vertex)`` alignment), ``kept_span``
      (``(lo, hi)`` indices into it after overhang trimming), ``groups`` (per-B-edge runs over the
      full alignment), ``drift_all`` (per-step drift, untrimmed);
    - point mode: ``path`` = ``(a_index, vertex, move)`` per backtracked state, moves in
      ``{"START", "V", "H", "D"}``; ``D``/``E`` are ``(N_A_points, V)``;
    - segment mode: ``arcs`` (state axis: ``(u, v)`` vertex pairs), ``ridable`` (which arcs may
      host a state), ``arc_path`` = ``(a_segment, arc, move)``; ``D``/``E`` are
      ``(N_A_segments, N_arcs)``.
    """
    if emission == "midpoint":                # deprecated alias -> the one segment mode
        emission = "segment"
    al, be = float(alpha), float(beta)                # §12 step weights (1, 1 = unweighted)
    a_pool = _node_projection_pool(list(coords_a), gb.b_raw_nodes, step_meters, min_gap_m)
    N = len(a_pool)
    V = gb.n_vertices
    dbg: Optional[Dict[str, Any]] = None
    if debug:
        dbg = {"params": {"emission": emission, "bearing_weight": float(bearing_weight),
                          "alpha": al, "beta": be,
                          "step_meters": step_meters, "trim_ends_m": trim_ends_m},
               "a_pool": a_pool}

    def _fail(reason: str) -> Tuple[float, List[Tuple[Coord, Coord]], Dict[str, Any]]:
        mm = _inf_metrics()
        if dbg is not None:
            dbg.setdefault("reason", reason)
            mm["debug"] = dbg
        return float("inf"), [], mm

    if N < 1 or V < 1:
        return _fail("empty_inputs")

    ax = np.asarray([p[0] for p in a_pool], float)
    ay = np.asarray([p[1] for p in a_pool], float)
    a_is_node = [bool(p[2]) for p in a_pool]
    vx, vy = gb.vx, gb.vy
    INF = float("inf")

    seg_dbg: Optional[Dict[str, Any]] = None
    if emission == "segment":
        # True segment-to-segment: states are (A-segment, B-arc); every state pays the
        # middle-to-middle distance (+ optional bearing). Yields vertex-level pairs in the same
        # format as the point DP. Always collect the state path -- the metrics are computed from
        # it (the reported distances ARE the middle-to-middle state costs).
        seg_dbg = dbg if dbg is not None else {}
        seg_pairs = _segment_dp_pairs(gb, ax, ay, float(bearing_weight), al, be, dbg=seg_dbg)
        if seg_pairs is None:
            return _fail("no_finite_alignment")
        pairs: List[Tuple[int, int]] = seg_pairs
    else:
        D = np.full((N, V), INF)
        back: List[List[Optional[Tuple]]] = [[None] * V for _ in range(N)]

        def emit(i: int) -> np.ndarray:
            return np.hypot(ax[i] - vx, ay[i] - vy)  # length-V vector of A[i]->vertex distances

        def relax_h(i: int, ei: np.ndarray) -> None:
            # horizontal (1:N coverage) relaxation within row i -- Dijkstra; every vertex
            # entered pays alpha * its emission (§12).
            heap = [(D[i][v], v) for v in range(V)]
            heapq.heapify(heap)
            while heap:
                c, v = heapq.heappop(heap)
                if c > D[i][v]:
                    continue
                for w in gb.succ_arcs[v]:
                    cand = D[i][v] + al * ei[w]
                    if cand < D[i][w]:
                        D[i][w] = cand
                        back[i][w] = ("H", v)
                        heapq.heappush(heap, (cand, w))

        # Row 0 -- free choice of B entry vertex; the relaxation prices a LEADING coverage run
        # (free entry dominates at alpha = 1, so this is a no-op on the default).
        e0 = emit(0)
        for v in range(V):
            D[0][v] = e0[v]
            back[0][v] = ("START", -1)
        relax_h(0, e0)

        for i in range(1, N):
            ei = emit(i)
            # base: V = N:1 stall (beta * E), D = both advance (full E) -- §12 step weights
            for v in range(V):
                cand = be * ei[v] + D[i - 1][v]
                bmove: Tuple = ("V", v)
                for u in gb.pred_arcs[v]:
                    c2 = ei[v] + D[i - 1][u]
                    if c2 < cand:
                        cand = c2
                        bmove = ("D", u)
                D[i][v] = cand
                back[i][v] = bmove
            relax_h(i, ei)

        v_best = int(np.argmin(D[N - 1]))
        if dbg is not None:
            dbg["D"] = D
            dbg["E"] = np.hypot(ax[:, None] - vx[None, :], ay[:, None] - vy[None, :])
            dbg["terminal_state"] = v_best
        if not np.isfinite(D[N - 1][v_best]):
            return _fail("no_finite_alignment")

        # Backtrack -> ordered (a_index, vertex) pairs.
        pairs = []
        moves: List[str] = []
        i, v = N - 1, v_best
        while True:
            move, u = back[i][v]
            pairs.append((i, v))
            moves.append(move)
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
        moves.reverse()
        if dbg is not None:
            dbg["path"] = [(pi, pv, mv) for (pi, pv), mv in zip(pairs, moves)]
            dbg["final_cost"] = float(D[N - 1][v_best])

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

    if dbg is not None:
        dbg["pairs_all"] = pairs
        dbg["groups"] = groups
        dbg["kept_span"] = (lo, hi)

    # If B is never actually traversed (A only touches boundary vertices -- e.g. a stub that ends
    # at, but never runs along, a B-edge), there is no real match -> NO_MATCH.
    if _b_len(lo, hi) <= 1e-9:
        return _fail("zero_b_traversal")

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
    # Cap covered-A by matched_len (the B-length actually traversed): you cannot cover more of A than
    # the corridor you walked. Without this, a long A stretched onto a short B via interior stalls (A
    # much denser than B) inflates coverage toward 100%, and the DTW's stall-vs-overhang choice differs
    # by traversal direction -- so forward and reverse of the SAME edge get different overlap. The cap
    # is direction-symmetric and a no-op for normal matches (covered_A ~= matched_len). Shared coverage
    # code, so it applies to BOTH "point" and "segment" emission.
    kept_a = min(kept_a, matched_len)
    overlap_pct = int(min(100, round(100.0 * kept_a / total_a_len))) if total_a_len > 0 else 0
    # per-edge A coverage as % of the WHOLE A-edge (raw a_len; sums to the uncapped coverage)
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

    # Segment mode: the reported distances ARE the segment-state costs -- one distance per
    # matched (A-segment, B-arc) state, middle to middle. Kept states are those whose produced
    # alignment pair (state t -> pair t+1) lies in the kept span; free stitches are skipped.
    part_drift, part_bearing_diff = average, bearing_diff   # default: full measures (point mode / short routes)
    mid_stats: Optional[Dict[Any, List[float]]] = None
    if emission == "segment" and seg_dbg is not None and "arc_path" in seg_dbg:
        arcs_l = seg_dbg["arcs"]
        rid = seg_dbg["ridable"]
        mid_stats = {}
        seg_bearing_diffs: List[float] = []
        seg_records: List[Tuple[Any, float, float]] = []      # (osm_edge_id, dist, bearing_diff) per state
        for t, (i, k, _mv) in enumerate(seg_dbg["arc_path"]):
            if not (lo <= t + 1 <= hi) or not rid[k]:
                continue
            u, w = arcs_l[k]
            eid = gb.edge_ids[gb.vert_edge[u]]
            d = float(np.hypot(0.5 * (ax[i] + ax[i + 1]) - 0.5 * (vx[u] + vx[w]),
                               0.5 * (ay[i] + ay[i + 1]) - 0.5 * (vy[u] + vy[w])))
            # per-segment heading diff: A micro-segment vs its matched B-arc (circular, degrees)
            _delta = abs(_bearing((ax[i], ay[i]), (ax[i + 1], ay[i + 1]))
                         - _bearing((vx[u], vy[u]), (vx[w], vy[w])))
            _bd = min(_delta, 360.0 - _delta)
            mid_stats.setdefault(eid, []).append(d)
            seg_bearing_diffs.append(_bd)
            seg_records.append((eid, d, _bd))
        all_d = [d for ds in mid_stats.values() for d in ds]
        if all_d:
            average = float(np.mean(all_d))
            for re in route_edges:
                ds = mid_stats.get(re["dest_id"])
                if ds:
                    re["match_dist_avg"] = float(np.mean(ds))
                    re["match_dist_max"] = float(np.max(ds))
                    re["match_dist_min"] = float(np.min(ds))
        # segment mode: report bearing_diff as the MEAN per-segment heading diff (parallel to the
        # mean per-segment distance above), not the single start->end chord angle used by point mode.
        if seg_bearing_diffs:
            bearing_diff = float(np.mean(seg_bearing_diffs))
        # part_* : the SAME means but DROPPING states on the route's FIRST and LAST OSM edge (interior
        # only) -- the entry/exit edges carry overhang / partial-match noise. Falls back to the full
        # measure when the route has < 3 OSM edges (no interior state remains).
        if route_edges and seg_records:
            _first, _last = route_edges[0]["dest_id"], route_edges[-1]["dest_id"]
            _mid = [(d, bd) for (eid, d, bd) in seg_records if eid != _first and eid != _last]
            if _mid:
                part_drift = float(np.mean([d for d, _ in _mid]))
                part_bearing_diff = float(np.mean([bd for _, bd in _mid]))
            else:
                part_drift, part_bearing_diff = average, bearing_diff

    metrics = {
        "average": average,
        "max": (float(np.max(all_d)) if mid_stats is not None and all_d
                else float(np.max(drift)) if drift else float("inf")),
        "min": (float(np.min(all_d)) if mid_stats is not None and all_d
                else float(np.min(drift)) if drift else float("inf")),
        "overlap_pct": overlap_pct,
        "matched_len": matched_len,
        "route": route,
        "route_edges": route_edges,
        "n_edges": len(route),
        "bearing_diff": bearing_diff,
        "part_drift": part_drift,
        "part_bearing_diff": part_bearing_diff,
        "warp_vertices": warp_vertices,
        "warp_is_node": [bool(gb.is_node[v]) for v in warp_vertices],
        "warp_a_is_node": warp_a_is_node,
        "warp_edge": warp_edge,
    }
    if dbg is not None:
        dbg["drift_all"] = drift_all
        metrics["debug"] = dbg
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
    emission: str = "point",
    bearing_weight: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    debug: bool = False,
    min_pool_gap_m: Optional[float] = None,
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
    emission:
        Local cost: ``"point"`` (default, point-to-point) or ``"segment"`` (true
        segment-to-segment: ONE distance between the two segment middles
        ``|mid(A-seg) - mid(B-arc)|``, junction stitches free, sliver-free pools, and the
        reported distances are those middle-to-middle distances). ``"midpoint"`` is accepted as
        a deprecated alias for ``"segment"``. See ``docs/weighted_emission.md``.
    bearing_weight:
        Optional λ for a length-independent heading penalty (segment mode only); ``0`` = off.
        Recommended with ``emission="segment"`` (λ ≈ 1-5): a middle-to-middle distance is blind
        to a segment rotating about its own middle, so the bearing term is what pins heading.
    alpha, beta:
        Step weights, both emission modes (``docs/weighted_emission.md`` §12; same names and
        semantics as ``match_dag``): each state pays its emission weighted by the move entering
        it -- ``alpha`` on a 1:N coverage step (B advances, A stays; domain ``(0, 1]``),
        ``beta`` on an N:1 stall (A advances, B stays; domain ``[1, inf)``), ``1`` on a 1:1
        advance. Defaults ``1``/``1`` reproduce the unweighted recurrence byte-for-byte; the
        reported distance metrics stay raw geometry either way.
    debug:
        ``True`` additionally returns the raw DP internals (cost/emission tables, backtracked
        path with move types, trim window) under the ``debug`` key -- see
        :func:`graph_dtw_align`. Off by default; no extra work when off.
    min_pool_gap_m:
        Minimum spacing of enrichment (non-node) pool points -- prevents sliver segments so
        segment states are genuinely segment-to-segment. Default ``None`` = ``0`` for
        ``emission="point"`` (unchanged behaviour) and ``step_meters / 2`` for ``"segment"``;
        pass an explicit value to override.

    Returns
    -------
    dict with keys ``route`` (``[(b_edge_id, direction, seq), ...]``), ``warping_path``,
    ``metrics`` (see :func:`graph_dtw_align`), ``avg_distance``, and ``graph`` (the
    :class:`LocalBGraph`, handy for visualization). With ``debug=True`` also ``debug``.
    """
    if emission == "midpoint":                # deprecated alias -> the one segment mode
        emission = "segment"
    if min_pool_gap_m is None:
        min_pool_gap_m = 0.0 if emission == "point" else 0.5 * step_meters
    gb = build_local_digraph(
        b_edges, coords_a, snap_tolerance_m=snap_tolerance_m, step_meters=step_meters,
        min_gap_m=min_pool_gap_m,
    )
    avg, warping, metrics = graph_dtw_align(
        coords_a, gb, step_meters=step_meters, trim_ends_m=trim_ends_m,
        emission=emission, bearing_weight=bearing_weight, alpha=alpha, beta=beta,
        debug=debug, min_gap_m=min_pool_gap_m)
    res = {
        "route": metrics["route"],
        "warping_path": warping,
        "metrics": metrics,
        "avg_distance": avg,
        "graph": gb,
    }
    if debug:
        res["debug"] = metrics.get("debug")
    return res


class GraphDTWMatcher:
    """Thin wrapper holding graph-DTW parameters; exposes :meth:`match_edge`.

    Convenience for scripts/notebooks and a single-edge debug tool (parallels the existing
    ``scripts/diagnose_match.py``). The heavy pipeline (candidate generation, parallel
    fan-out, output tables) is intentionally *not* here -- this is the algorithm primitive.
    """

    def __init__(self, snap_tolerance_m: float = 0.75, step_meters: float = 10.0,
                 trim_ends_m: float = 0.0, emission: str = "point", bearing_weight: float = 0.0):
        self.snap_tolerance_m = snap_tolerance_m
        self.step_meters = step_meters
        self.trim_ends_m = trim_ends_m
        self.emission = emission
        self.bearing_weight = bearing_weight

    def match_edge(self, coords_a: Sequence[Coord],
                   b_edges: Sequence[Tuple[Any, LineString]],
                   debug: bool = False) -> Dict[str, Any]:
        return match_edge_to_bgraph(
            coords_a, b_edges,
            snap_tolerance_m=self.snap_tolerance_m,
            step_meters=self.step_meters,
            trim_ends_m=self.trim_ends_m,
            emission=self.emission,
            bearing_weight=self.bearing_weight,
            debug=debug,
        )
