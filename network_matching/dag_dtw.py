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


class NotATree(Exception):
    """Raised (only under ``require_tree=True``) when the source GA has an undirected loop
    (a reconvergence / diamond) -- i.e. it is a DAG but not a forest / polytree."""


def _is_tree(ga) -> bool:
    """True iff GA's UNDIRECTED skeleton is a forest (no undirected cycle / reconvergence): the
    cyclomatic number ``E - V + C`` is 0. Interior sample points are degree-2 and never add a cycle,
    so this detects a diamond regardless of sampling density."""
    NA = ga.n_vertices
    E = sum(len(ga.succ_arcs[a]) for a in range(NA))     # one undirected edge per directed arc
    seen = [False] * NA
    C = 0
    for s in range(NA):
        if seen[s]:
            continue
        C += 1
        stack = [s]; seen[s] = True
        while stack:
            u = stack.pop()
            for w in ga.succ_arcs[u]:
                if not seen[w]:
                    seen[w] = True; stack.append(w)
            for w in ga.pred_arcs[u]:
                if not seen[w]:
                    seen[w] = True; stack.append(w)
    return E - NA + C == 0


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


def check_matching_rules(M, ga, gb) -> Dict[str, Any]:
    """Validate a (possibly **many-to-many**) matching ``M`` against the four local warping rules of
    docs/dag_dtw_matching.md ("The problem -- a constrained optimization over valid matchings",
    (V1)-(V4)). This is the exact *structural* definition on a general relation; :func:`check_sequence_rules`
    is the metric-tolerant validator for the shipped **single-valued** ``φ`` (whose 1:N coverage lives
    on the edge route, not on ``φ`` -- so a subsampled ``φ`` is deliberately *not* a full valid warping).

    ``M`` may be a mapping ``{a: v}`` / ``{a: iterable-of-v}`` or an iterable of ``(a, v)`` pairs.
    Returns ``{ok, v1_cross, v2_predecessor, v3_successor, v4_uncovered}`` -- each a list of offending
    cells. Uses only immediate neighbours (``Apred/Asucc = ga.pred_arcs/succ_arcs``, ``Bpred/Bsucc =
    gb.pred_arcs/succ_arcs``) and membership -- **no reachability**. (V2)/(V3) are named by the
    neighbours they inspect -- predecessors / successors -- not "backward/forward," which is ambiguous.

    - **(V1) no cross**: no DAG-predecessor of ``a`` sits on a B-successor of ``v``.
    - **(V2) predecessor rule** (every cell is *fed*): ``(a,v)`` either rode B inside ``a``'s run (has
      a matched ``Bpred`` at ``a``) or **every** ``Apred(a)`` feeds it (held at ``v`` or advanced
      ``v⁻→v``); sources satisfy the ``∀`` vacuously. Catches a **merge** entered at two points.
    - **(V3) successor rule** (every cell *continues*): the mirror of (V2) with ``Asucc``/``Bsucc``;
      sinks vacuous. Catches a **branch** left at two points.
    - **(V4) coverage**: every A-vertex is matched.
    """
    Ma: Dict[int, set] = {}
    if isinstance(M, dict):
        for a, vs in M.items():
            Ma[int(a)] = ({int(vs)} if isinstance(vs, (int, np.integer))
                          else set(int(v) for v in vs))
    else:
        for a, v in M:
            Ma.setdefault(int(a), set()).add(int(v))

    def has(a, v):
        return v in Ma.get(a, ())

    v1: List = []; v2: List = []; v3: List = []
    for a, vs in Ma.items():
        for v in vs:
            # (V1) no cross
            for am in ga.pred_arcs[a]:
                for vp in gb.succ_arcs[v]:
                    if has(am, vp):
                        v1.append((a, v, am, vp))
            # (V2) predecessor rule (fed): rode B, OR every incoming arc feeds this cell (∀ over Apred;
            #      empty Apred -> vacuously true -> a source's run-entry is free, no boundary exemption)
            if not any(has(a, vm) for vm in gb.pred_arcs[v]) and not all(
                    has(am, v) or any(has(am, vm) for vm in gb.pred_arcs[v])
                    for am in ga.pred_arcs[a]):
                v2.append((a, v))
            # (V3) successor rule (continues): continues in B, OR every outgoing arc carries this cell on
            if not any(has(a, vp) for vp in gb.succ_arcs[v]) and not all(
                    has(ap, v) or any(has(ap, vp) for vp in gb.succ_arcs[v])
                    for ap in ga.succ_arcs[a]):
                v3.append((a, v))

    v4 = [a for a in range(ga.n_vertices) if not Ma.get(a)]   # (V4) every A-vertex matched
    return {"ok": not (v1 or v2 or v3 or v4),
            "v1_cross": v1, "v2_predecessor": v2, "v3_successor": v3, "v4_uncovered": v4}


def forward_successor_dp(a_edges, b_edges, *, snap_tolerance_m=0.5, step_meters=2.0):
    """Forward DP that enforces the **(V3) successor rule by construction** (docs §3.0a): process each
    vertex **together with all its successors**, committing its single exit and expanding *every*
    successor from it, so a branch is always left at **one** point.

    ``F[a][v]`` = min cost of ``a``'s subtree with ``a`` **at** ``v``; every successor ``a_k`` expands
    from that one ``v`` (held, or one B-arc past it), so the branch cannot smear. Point-to-point, and
    exact for a source **out-tree** (branches, no merges — the predecessor/(V2) coordination is the
    mirror, the backward pass, not included here). Returns ``(ga, gb, phi, M)`` with ``M`` the set of
    ``(a, phi[a])`` pairs; ``check_matching_rules(M, ga, gb)["v3_successor"]`` is empty by construction.
    """
    a_pts = [(float(x), float(y)) for _id, g in a_edges for (x, y) in g.coords]
    b_pts = [(float(x), float(y)) for _id, g in b_edges for (x, y) in g.coords]
    ga = build_local_digraph(a_edges, b_pts, snap_tolerance_m, step_meters)
    gb = build_local_digraph(b_edges, a_pts, snap_tolerance_m, step_meters)
    order = topological_order(ga)
    NA, NB = ga.n_vertices, gb.n_vertices
    ax, ay, bx, by = ga.vx, ga.vy, gb.vx, gb.vy

    F: List = [None] * NA                                          # F[a][v] over all B-vertices v
    for a in reversed(order):                                      # successors are finalised first
        Fa = np.hypot(bx - ax[a], by - ay[a]).astype(float)       # base: a's own drift at each v
        for ak in ga.succ_arcs[a]:
            m = F[ak].copy()                                       # w = v : successor held at v
            for v in range(NB):
                for w in gb.succ_arcs[v]:                          # w one B-arc past v
                    if F[ak][w] < m[v]:
                        m[v] = F[ak][w]
            Fa = Fa + m                                            # every successor expands from v
        F[a] = Fa

    phi: Dict[int, int] = {}                                       # backtrack: source argmin, push down
    for s in (a for a in range(NA) if not ga.pred_arcs[a]):
        phi[s] = int(np.argmin(F[s]))
    for a in order:
        v = phi.get(a)
        if v is None:
            continue
        for ak in ga.succ_arcs[a]:
            cands = [v] + list(gb.succ_arcs[v])                    # all successors leave from a's ONE v
            phi[ak] = min(cands, key=lambda w: F[ak][w])
    M = {(a, phi[a]) for a in phi}
    return ga, gb, phi, M


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
def _vertex_segments(g):
    """Per-vertex ``(mid_x, mid_y, bearing_deg)`` of the segment a vertex owns on its own edge (docs
    §3.5): vertex ``v`` pairs with its same-edge successor ``w`` (``v→w``); the last vertex of an edge
    falls back to its incoming same-edge segment; a degenerate 1-vertex edge falls back to ``mid=v,
    bearing=0``. Bearing is the compass convention ``(deg·atan2(Δx, Δy) + 360) mod 360``."""
    N = g.n_vertices
    vx, vy, ve = g.vx, g.vy, g.vert_edge
    mx, my, bear = vx.astype(float).copy(), vy.astype(float).copy(), np.zeros(N)
    for v in range(N):
        e = ve[v]
        w = next((s for s in g.succ_arcs[v] if ve[s] == e), None)       # same-edge successor v→w
        if w is not None:
            dx, dy = vx[w] - vx[v], vy[w] - vy[v]
            mx[v], my[v] = 0.5 * (vx[v] + vx[w]), 0.5 * (vy[v] + vy[w])
        else:
            u = next((p for p in g.pred_arcs[v] if ve[p] == e), None)   # incoming u→v (edge end)
            if u is None:
                continue                                                # degenerate: keep mid=v, bear=0
            dx, dy = vx[v] - vx[u], vy[v] - vy[u]
            mx[v], my[v] = 0.5 * (vx[u] + vx[v]), 0.5 * (vy[u] + vy[v])
        bear[v] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    return mx, my, bear


def _emission_matrix(ga, gb, emission, bearing_weight):
    """The ``(NA, NB)`` emission ``E(a, v)`` (docs §3.5). ``"point"``: point-to-point distance
    (bit-for-bit today's inline ``hypot``). ``"segment"``: middle-to-middle distance +
    ``bearing_weight · circ(bearing(a), bearing(v))``, ``circ ∈ [0,180]`` degrees."""
    ax, ay, bx, by = ga.vx, ga.vy, gb.vx, gb.vy
    if emission == "point":
        return np.hypot(bx[None, :] - ax[:, None], by[None, :] - ay[:, None])
    if emission == "segment":
        amx, amy, abear = _vertex_segments(ga)
        bmx, bmy, bbear = _vertex_segments(gb)
        dist = np.hypot(bmx[None, :] - amx[:, None], bmy[None, :] - amy[:, None])
        bd = np.abs(abear[:, None] - bbear[None, :])
        return dist + float(bearing_weight) * np.minimum(bd, 360.0 - bd)
    raise ValueError(f"unknown emission {emission!r}; use 'point' or 'segment'")


def _dp_table(pred_arcs, succ_arcs, outdeg, gb_succ, bu, bw, order, emit,
              horizontal_weight=1.0):
    """Fill the DAG-DTW cost table for a graph in ONE direction (docs §3, §3.2a). Called forward
    with (GA, GB) and backward with the reversed graphs; ``pred_arcs``/``succ_arcs`` are the A-graph
    adjacency in the current direction, ``outdeg`` its out-degree (the split factor -- ``1/outdeg``
    forward, ``1/indeg`` backward), and ``gb_succ``/``bu``/``bw`` the B-graph forward arcs in that
    direction. ``emit`` is the ``(NA, NB)`` emission matrix ``E(a, ·)`` (point distance, or segment
    middle-to-middle + bearing, docs §3.5) -- the same for both passes. ``horizontal_weight`` (α ≤ 1,
    docs §3.4) discounts the emission on a horizontal coverage-extension step (α·E when the min comes
    from the (H) horizontal term, full E on an A-advance). Returns the ``(NA, NB)`` cost table."""
    NA, NB = len(pred_arcs), emit.shape[1]
    D = np.full((NA, NB), float("inf"))
    alpha = float(horizontal_weight)
    btopo = _btopo(gb_succ, NB) if alpha != 1.0 else None    # B-topological order for the α≠1 (H) pass
    gb_pred = None
    if alpha != 1.0:
        gb_pred = [[] for _ in range(NB)]
        for u in range(NB):
            for w in gb_succ[u]:
                gb_pred[w].append(u)
    for a in order:
        ei = emit[a]                                   # E(a, ·): point distance, or segment+bearing
        preds = pred_arcs[a]
        acc = np.zeros(NB)                             # (A) term; source has an empty sum -> 0
        for ap in preds:
            m = D[ap].copy()                           # v'=v (vertical)
            if bu.size:
                np.minimum.at(m, bw, D[ap][bu])        # v'∈Bpred(v) (diagonal)
            acc += m / outdeg[ap]                      # conserved-flow split factor

        if alpha == 1.0:
            D[a] = (ei + acc)                          # (H) within-a Dijkstra over the B-arcs
            heap = [(D[a][v], v) for v in range(NB)]
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
        else:
            # α≠1: emission depends on the winning move, so resolve (H) in B-topological order --
            #   D[a][v] = α·E(a,v) + min(h, A)  with α = horizontal_weight if the (H) term h wins
            #   (came from a covered D[a][v']), else α = 1 for the (A) A-advance term A.
            _relax_alpha(D[a], ei, acc, gb_pred, btopo, alpha)
    return D


def _btopo(gb_succ, NB):
    """A topological order of the B-graph, or ``None`` if it has a directed cycle."""
    indeg = [0] * NB
    for u in range(NB):
        for w in gb_succ[u]:
            indeg[w] += 1
    q = deque(v for v in range(NB) if indeg[v] == 0)
    order = []
    ind = indeg[:]
    while q:
        u = q.popleft(); order.append(u)
        for w in gb_succ[u]:
            ind[w] -= 1
            if ind[w] == 0:
                q.append(w)
    return order if len(order) == NB else None


def _relax_alpha(row, ei, acc, gb_pred, btopo, alpha):
    """The α≠1 (H) pass (docs §3.4). ``row`` is filled in place: for each B-vertex the emission is
    discounted (α·E) iff the horizontal term wins, else full E on the A-advance. Acyclic B -> one
    topological pass; cyclic B -> bounded iterative relaxation (α<1 contracts, so it converges)."""
    NB = len(ei)
    if btopo is not None:
        for v in btopo:
            h = min((row[u] for u in gb_pred[v]), default=float("inf"))
            row[v] = alpha * ei[v] + h if h < acc[v] else ei[v] + acc[v]
        return
    row[:] = ei + acc                                  # cyclic B: init at the A-advance upper bound
    for _ in range(NB):
        changed = False
        for v in range(NB):
            h = min((row[u] for u in gb_pred[v]), default=float("inf"))
            nv = alpha * ei[v] + h if h < acc[v] else ei[v] + acc[v]
            if nv < row[v] - 1e-12:
                row[v] = nv
                changed = True
        if not changed:
            break


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
# Per-A-edge route detail: route + score + partial-coverage start/end fractions (docs §6.1)
# --------------------------------------------------------------------------------------
def _route_detail(ga, gb, routes, phi, b_geoms):
    """For each A-edge, the mapped route with a score AND exactly where the route starts/ends on its
    boundary B-edges. ``t`` is the fractional arc-length position (0..1) along a B-edge, in the
    B-edge's own direction; only the first/last B-edge can be partial. Computed from ``φ`` and the
    B-edge geometries -- no extra DP. Returns ``{a_edge_id -> detail dict}`` (docs §6.1)."""
    detail: Dict[Any, Dict[str, Any]] = {}
    edge_verts: Dict[int, List[int]] = {}
    for a in range(ga.n_vertices):
        if a in phi:
            edge_verts.setdefault(int(ga.vert_edge[a]), []).append(a)

    for e_idx, verts in edge_verts.items():
        a_edge_id = ga.edge_ids[e_idx]
        route = list(routes.get(a_edge_id, []))
        if not route:
            continue
        per: Dict[Any, List[Tuple[float, float]]] = {b: [] for b in route}   # b_edge -> [(t, drift)]
        all_drifts: List[float] = []
        for a in verts:
            v = phi[a]
            beid = gb.edge_ids[gb.vert_edge[v]]
            drift = float(np.hypot(ga.vx[a] - gb.vx[v], ga.vy[a] - gb.vy[v]))
            all_drifts.append(drift)
            if beid in per:
                g = b_geoms.get(beid)
                t = float(g.project(Point(gb.vx[v], gb.vy[v]), normalized=True)) if g is not None else 0.0
                per[beid].append((t, drift))

        edges_out: List[Dict[str, Any]] = []
        covered_len = 0.0
        n = len(route)
        for i, b in enumerate(route):
            ts = per[b]
            if not ts:                                       # a route edge with no landed vertex
                continue
            tvals = [t for t, _d in ts]
            dvals = [d for _t, d in ts]
            lo, hi = min(tvals), max(tvals)
            if n == 1:
                t_from, t_to = lo, hi                        # single edge: entry .. exit on one B-edge
            elif i == 0:
                t_from, t_to = lo, 1.0                       # first: entry .. end of b1
            elif i == n - 1:
                t_from, t_to = 0.0, hi                       # last: start of bk .. exit
            else:
                t_from, t_to = 0.0, 1.0                      # interior: full
            g = b_geoms.get(b)
            xy_from = tuple(g.interpolate(t_from, normalized=True).coords[0]) if g is not None else None
            xy_to = tuple(g.interpolate(t_to, normalized=True).coords[0]) if g is not None else None
            covered_len += (t_to - t_from) * (g.length if g is not None else 0.0)
            edges_out.append({"b_edge": b, "t_from": t_from, "t_to": t_to,
                              "cover_pct": 100.0 * (t_to - t_from),
                              "avg_drift": float(np.mean(dvals)),
                              "xy_from": xy_from, "xy_to": xy_to})
        if not edges_out:
            continue
        detail[a_edge_id] = {
            "route": route,
            "avg_drift": float(np.mean(all_drifts)),
            "max_drift": float(np.max(all_drifts)),
            "n_points": len(verts),
            "covered_len_m": covered_len,
            "start": {"b_edge": edges_out[0]["b_edge"], "t": edges_out[0]["t_from"],
                      "xy": edges_out[0]["xy_from"]},
            "end": {"b_edge": edges_out[-1]["b_edge"], "t": edges_out[-1]["t_to"],
                    "xy": edges_out[-1]["xy_to"]},
            "edges": edges_out,
        }
    return detail


# --------------------------------------------------------------------------------------
# The joint DP
# --------------------------------------------------------------------------------------
def match_dag_to_bgraph(
    a_edges: Sequence[Tuple[Any, LineString]],
    b_edges: Sequence[Tuple[Any, LineString]],
    *,
    snap_tolerance_m: float = 0.5,
    step_meters: float = 2.0,
    emission: str = "point",
    bearing_weight: float = 0.0,
    horizontal_weight: float = 1.0,
    require_tree: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Align the source DAG made of ``a_edges`` to the local directed graph of ``b_edges``.

    ``emission`` (docs §3.5): ``"point"`` (default) scores ``E(a, v) = dist(a, v)``; ``"segment"``
    scores the middle-to-middle segment distance ``+ bearing_weight·Δbearing`` (the direction term
    that fixes the diamond). Only the emission changes -- the DP, junction machinery, and output are
    identical, so ``"segment"`` returns the same result dict as ``"point"``, just a better alignment.
    ``"point"`` is bit-for-bit today's result; ``bearing_weight`` (λ, same 1-5 scale as graph-DTW) is
    used only in ``"segment"`` mode.
    ``horizontal_weight`` (α ≤ 1, docs §3.4) discounts the emission on a horizontal 1:N
    coverage-extension step (α·E), leaving a genuine A-advance match at full E; α=1 (default) is the
    plain recurrence, bit-for-bit unchanged. ``require_tree`` (docs §7): if True, assert the source
    has no undirected loop (a forest/polytree) and raise :class:`NotATree` on a reconvergence — the
    exactly-solvable regime that side-steps the diamond limit.
    ``a_edges`` / ``b_edges``: lists of ``(id, shapely LineString)`` in a projected CRS (meters).
    Returns a dict with:

    - ``phi``: ``{a_vertex_index -> b_vertex_index}`` -- the junction-consistent label map;
    - ``a_vertex_match``: per A-vertex ``(x, y, b_vertex, b_edge_id, drift)``;
    - ``routes``: ``{a_edge_id -> [b_edge_id, ...]}`` the ordered B-edges each A-edge maps to;
    - ``routes_detail``: ``{a_edge_id -> {route, avg_drift, max_drift, n_points, covered_len_m,
      start, end, edges}}`` -- the route with a score AND where it starts/ends on its boundary
      B-edges (a 0..1 fraction ``t`` and a map ``xy``; only the first/last B-edge can be partial),
      docs §6.1;
    - ``total_cost`` (realized ``Σ E(a, φ(a))``, consistent with ``avg_drift``) and ``avg_drift``
      (mean per-A-vertex drift); ``dp_cost`` = ``Σ_sinks min D`` (the DP's discrete optimum -- a
      diagnostic, not a bound on ``total_cost``);
    - ``GA`` / ``GB`` (the two :class:`LocalBGraph`s), and with ``debug=True`` the full ``D`` table.
    """
    a_pts = [(float(x), float(y)) for _id, g in a_edges for (x, y) in g.coords]
    b_pts = [(float(x), float(y)) for _id, g in b_edges for (x, y) in g.coords]

    # GA (source) and GB (target) are built the same way; each is enriched with the other's nodes.
    ga = build_local_digraph(a_edges, b_pts, snap_tolerance_m, step_meters)
    gb = build_local_digraph(b_edges, a_pts, snap_tolerance_m, step_meters)
    order = topological_order(ga)                      # raises NotADAG on a cyclic source
    if require_tree and not _is_tree(ga):              # caller asserts a forest / polytree (§7)
        raise NotATree("source GA has an undirected loop (a reconvergence/diamond); "
                       "require_tree=True demands a forest/polytree")

    NA, NB = ga.n_vertices, gb.n_vertices
    ax, ay = ga.vx, ga.vy
    bx, by = gb.vx, gb.vy
    bu, bw = _gb_arcs(gb)                               # GB forward arcs (tail, head)
    outdeg = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)

    # Emission E(a,·): point distance, or segment middle-to-middle + bearing (docs §3.5); the SAME
    # matrix drives the forward D, the backward B, and the D+B-E backtrack, so they stay consistent.
    emit = _emission_matrix(ga, gb, emission, bearing_weight)

    # --- FORWARD + BACKWARD tables, then JOINT junction resolution (docs §3.2a) ---
    # Forward D[a][v] = cheapest cost to align sources -> a with a at v.
    outdeg_f = np.array([max(1, len(ga.succ_arcs[a])) for a in range(NA)], float)
    D = _dp_table(ga.pred_arcs, ga.succ_arcs, outdeg_f, gb.succ_arcs, bu, bw,
                  order, emit, horizontal_weight)
    # Backward B[a][v] = cheapest cost to align a -> sinks with a at v: reverse BOTH graphs
    # (GA sinks become sources; GB arcs flip) and run the same DP with the symmetric 1/indeg split.
    outdeg_b = np.array([max(1, len(ga.pred_arcs[a])) for a in range(NA)], float)
    B = _dp_table(ga.succ_arcs, ga.pred_arcs, outdeg_b, gb.pred_arcs, bw, bu,
                  order[::-1], emit, horizontal_weight)

    sinks = [a for a in range(NA) if len(ga.succ_arcs[a]) == 0]
    # DP DIAGNOSTIC, not the reported total: Σ_sinks min D is the DP's *discrete, unconstrained*
    # optimum (every sink free to pick its own cheapest sampled B-vertex). It is NOT a bound on the
    # realized cost: the continuous arc-length re-match can place a vertex between samples and beat
    # it (clean trees), while joint-consistency + shift can push the realized cost well above it. On
    # a diamond it is itself inexact (docs §3.3). The reported total_cost is the REALIZED sum of the
    # final per-vertex drifts (below), consistent with avg_drift.
    dp_cost = float(sum(np.min(D[t]) for t in sinks)) if sinks else float("inf")

    # φ by reverse-topological backtrack, combining BOTH ideas:
    #   * score each A-vertex by the JOINT cost D[a][v] + B[a][v] - E(a,v) -- the forward+backward
    #     value all routes through a agree on (subtract E, counted in both tables). This resolves
    #     two routes disagreeing at a shared junction (§3.2a);
    #   * subject to a REACHABILITY constraint -- the chosen v must still walk FORWARD to every
    #     successor's φ. This is the CHEAP enforcement of JOINT junction consistency (§3.2c): the
    #     junction labels are a joint decision, not per-junction argmin, and this constraint throws
    #     away the inconsistent tie-breaks that a shift creates.
    # (Recursive minimum-vertex-cut conditioning -- docs §3.2b -- was implemented and tested here; it
    #  correctly solves the reconvergence LOOP, but the synthetic `diamond`'s failures are NOT loop
    #  failures: even with the split junction pinned to the exact B-split, point mode collapses both
    #  branches onto the nearer B-edge because that genuinely costs less. That is the
    #  nearest-vs-corresponding limit, which needs a DIRECTION term (segment/bearing), not a cut.)
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
        ei = emit[a]                                   # same emission as D/B, to undo the double-count
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
        "routes_detail": _route_detail(ga, gb, routes, phi, b_geoms),   # per-edge score + coverage
        "total_cost": float(sum(drifts)) if drifts else float("inf"),   # REALIZED Σ E(a, φ(a))
        "dp_cost": dp_cost,                                             # Σ_sinks min D (DP optimum)
        "avg_drift": float(np.mean(drifts)) if drifts else float("inf"),
        "GA": ga,
        "GB": gb,
        "sinks": sinks,
        "sources": [a for a in range(NA) if len(ga.pred_arcs[a]) == 0],
    }
    if debug:
        res["debug"] = {"D": D, "B": B, "order": order}
    return res
