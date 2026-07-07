"""
Exact conditioning solvers for **reconvergent** source DAGs (docs/dag_dtw_matching.md §3.2b).

These are REFERENCE / VALIDATION solvers, deliberately **not** wired into
:func:`network_matching.dag_dtw.match_dag_to_bgraph` (point mode does not benefit -- the synthetic
``diamond``'s failures are the nearest-vs-corresponding limit, not a reconvergence loop). They exist
to (a) provide the exact joint labelling for a genuine loop, and (b) **cross-validate each other**.

Two methods, one shared forest base solver:

* :func:`solve_recursive` -- recursive **minimum vertex cut**: if the free part is a forest, run the
  exact BP forest solver; else pick a small vertex cut, enumerate its nearby B-candidates, pin,
  split into the **independent** components and recurse. Cost ``(#cand) ** (largest single cut)`` --
  linear in the number of loops.

The forest base solver (:func:`_forest_solve`) is **exact** -- min-sum belief propagation on the
polytree, minimising ``Σ drift`` subject to the directed reachability constraint. Conditioning is
exact only because this base solver is; a heuristic base makes the two strategies disagree (which is
exactly what the ``double_diamond`` cross-check first exposed).
* :func:`solve_fvs` -- one-shot **minimum feedback vertex set**: enumerate **all** label combinations
  of the whole set ``F`` jointly, solve the forest ``GA - F`` for each. Cost ``(#cand) ** |F|`` --
  exponential in the number of loops.

Both minimise the **same** realized cost with the **same** exact forest solver, so with a full
candidate set (``cand_k = NB``) they return labellings of **equal cost** on any DAG -- which is what
:mod:`scripts.dag_conditioning_validate` checks. Under the default K-nearest candidate restriction
they still agree on realistic conflation inputs; the one place they can diverge (both remaining
*valid*) is **adversarial anti-parallel, multi-loop** geometry, where the feasible optimum's labels
sit beyond the nearest few B-vertices and the two solvers widen their candidates differently. That
is a candidate-restriction limit, not a solver bug: raising ``cand_k`` closes it.

**Reachability feasibility.** A pinned boundary can be *infeasible* (an A-arc forced to run backward
in B -- normal when a B edge is digitized against A's flow). :func:`_forest_solve` returns ``+inf``
for such a pinning (via the BP value, not a re-summed drift), and :func:`_pins_feasible` rejects
pinnings whose *internal* arcs run backward, so an infeasible labelling is never scored as valid.
"""
from collections import deque, namedtuple
from itertools import combinations, product
from typing import Any, Dict, List, Tuple

import numpy as np

from .dag_dtw import build_local_digraph, topological_order

CAND_K = 12   # nearby B-candidates enumerated per cut / FVS vertex (same for both solvers)

# Immutable per-match context shared by the forest solver and both conditioning strategies.
_Ctx = namedtuple("_Ctx", "ax ay bx by fwd_mask")


def _forward_reach_mask(gb):
    """Boolean ``(NB, NB)`` matrix: ``mask[u, v]`` iff B-vertex ``v`` is forward-reachable from ``u``
    (``u`` itself included). Rows = Reach(u); columns = RevReach(v). Drives the BP constraints."""
    NB = gb.n_vertices
    mask = np.zeros((NB, NB), dtype=bool)
    for v in range(NB):
        s = {v}; st = [v]
        while st:
            u = st.pop()
            for w in gb.succ_arcs[u]:
                if w not in s:
                    s.add(w); st.append(w)
        mask[v, list(s)] = True
    return mask


# --------------------------------------------------------------------------------------
# Undirected structure helpers (cycles, components, cyclomatic number)
# --------------------------------------------------------------------------------------
def _induced(ga, verts):
    """A-graph adjacency restricted to ``verts`` (arcs with both ends inside)."""
    vs = set(verts)
    pred = {a: [p for p in ga.pred_arcs[a] if p in vs] for a in vs}
    succ = {a: [s for s in ga.succ_arcs[a] if s in vs] for a in vs}
    return pred, succ


def _undirected(pred, succ, verts):
    adj = {a: set() for a in verts}
    for a in verts:
        for b in pred[a]:
            adj[a].add(b); adj[b].add(a)
        for b in succ[a]:
            adj[a].add(b); adj[b].add(a)
    return adj


def _components(verts, adj):
    """Connected components of ``verts`` under undirected adjacency ``adj``."""
    vs = set(verts); seen: set = set(); out = []
    for s in verts:
        if s in seen:
            continue
        comp = []; q = deque([s]); seen.add(s)
        while q:
            u = q.popleft(); comp.append(u)
            for w in adj[u]:
                if w in vs and w not in seen:
                    seen.add(w); q.append(w)
        out.append(comp)
    return out


def _cyclomatic(verts, adj):
    """Undirected cyclomatic number ``E - V + C`` (0 ⇔ forest ⇔ nothing left to condition on)."""
    E = sum(len([w for w in adj[a] if w in set(verts)]) for a in verts) // 2
    return E - len(verts) + len(_components(verts, adj))


def _junctions(ga):
    """Vertices that can lie on an undirected cycle: branches (out>1) or merges (in>1). Every
    reconvergence in a DAG passes through such a junction, so a minimum FVS can be sought here."""
    return [v for v in range(ga.n_vertices)
            if len(ga.succ_arcs[v]) > 1 or len(ga.pred_arcs[v]) > 1]


def min_feedback_vertex_set(ga) -> List[int]:
    """A **minimum** feedback vertex set of ``GA``'s undirected skeleton: the smallest vertex set
    whose removal leaves a forest. Searched over junctions by increasing size (exact for the small
    local graphs here; every undirected cycle in a DAG contains a branch/merge junction)."""
    verts = list(range(ga.n_vertices))
    pred, succ = _induced(ga, verts)
    adj = _undirected(pred, succ, verts)
    if _cyclomatic(verts, adj) == 0:
        return []
    juncs = _junctions(ga)
    for k in range(1, len(juncs) + 1):
        for subset in combinations(juncs, k):
            sub = set(subset)
            rest = [v for v in verts if v not in sub]
            radj = {a: {w for w in adj[a] if w not in sub} for a in rest}
            if _cyclomatic(rest, radj) == 0:
                return list(subset)
    return juncs   # fallback (never reached for a DAG: junctions are a FVS)


# --------------------------------------------------------------------------------------
# The forest base solver: EXACT min-sum belief propagation on the polytree
# --------------------------------------------------------------------------------------
def _forest_solve(ga, gb, free_verts, pinned, ctx):
    """Label a FOREST ``free_verts`` given ``pinned`` boundary labels, **exactly**, by min-sum belief
    propagation. Objective: minimise ``Σ_a drift(a, φ(a))`` subject to the reachability constraint
    ``φ(head) ∈ Reach(φ(tail))`` on every A-arc, plus the pinned boundary. BP is exact on a tree, so
    each connected tree component is solved to the global optimum -- which is what makes the two
    conditioning strategies (recursive cut, one-shot FVS) provably agree. Returns
    ``(phi over free_verts, realized cost over free_verts)``."""
    ax, ay, bx, by, fwd_mask, INF = ctx.ax, ctx.ay, ctx.bx, ctx.by, ctx.fwd_mask, float("inf")
    NB = len(bx)
    fset = set(free_verts)

    # Unary potentials: drift + folded PINNED-boundary reachability constraints (pinned neighbours
    # are boundary conditions on a free vertex, NOT BP nodes -- re-adding them would re-form a cycle).
    unary: Dict[int, np.ndarray] = {}
    for a in free_verts:
        u = np.hypot(bx - ax[a], by - ay[a]).astype(float)
        for p in ga.pred_arcs[a]:                          # arc p->a: φ(a) ∈ Reach(pinned p)
            if p in pinned:
                u = np.where(fwd_mask[pinned[p]], u, INF)
        for s in ga.succ_arcs[a]:                          # arc a->s: pinned s ∈ Reach(φ(a))
            if s in pinned:
                u = np.where(fwd_mask[:, pinned[s]], u, INF)
        unary[a] = u

    # Undirected forest adjacency among free vertices, remembering each arc's direction.
    nbrs: Dict[int, List[Tuple[int, str]]] = {a: [] for a in free_verts}
    for a in free_verts:
        for p in ga.pred_arcs[a]:
            if p in fset:
                nbrs[a].append((p, "in"))                  # arc p->a
        for s in ga.succ_arcs[a]:
            if s in fset:
                nbrs[a].append((s, "out"))                 # arc a->s

    phi: Dict[int, int] = {}
    visited: set = set()
    total_cost = 0.0
    for root in free_verts:
        if root in visited:
            continue
        bfs: List[int] = []; parent: Dict[int, int] = {root: -1}
        q = deque([root]); visited.add(root)
        while q:
            x = q.popleft(); bfs.append(x)
            for (nb, _d) in nbrs[x]:
                if nb not in parent:
                    parent[nb] = x; visited.add(nb); q.append(nb)

        def edge_msg(x, p, belief_x):
            """min-sum message x->p over p's labels, given the x--p arc direction."""
            arc_out = any(nb == p and d == "out" for (nb, d) in nbrs[x])  # arc x->p
            if arc_out:                                    # φ(p) ∈ Reach(φ(x)) : mask[v_x, v_p]
                return np.where(fwd_mask, belief_x[:, None], INF).min(axis=0)
            else:                                          # arc p->x : φ(x) ∈ Reach(φ(p)) : mask[v_p, v_x]
                return np.where(fwd_mask, belief_x[None, :], INF).min(axis=1)

        msg: Dict[Tuple[int, int], np.ndarray] = {}        # (child, parent) -> array over parent labels
        for x in reversed(bfs):                            # leaves -> root
            p = parent[x]
            if p == -1:
                continue
            belief = unary[x].copy()
            for (nb, _d) in nbrs[x]:
                if nb != p:
                    belief = belief + msg[(nb, x)]
            msg[(x, p)] = edge_msg(x, p, belief)

        broot = unary[root].copy()
        for (nb, _d) in nbrs[root]:
            broot = broot + msg[(nb, root)]
        bmin = float(broot.min())
        total_cost += bmin                             # min-sum BP value = the EXACT optimum for this
        phi[root] = int(np.argmin(broot))              # component (INF if the pinned boundary is
        if not np.isfinite(bmin):                      # infeasible -- must propagate, not silently
            continue                                   # accept argmin=0 with a finite drift, the bug)
        for x in bfs:                                      # root -> leaves, choose consistent labels
            if parent[x] == -1:
                continue
            p = parent[x]; vp = phi[p]
            belief = unary[x].copy()
            for (nb, _d) in nbrs[x]:
                if nb != p:
                    belief = belief + msg[(nb, x)]
            arc_out = any(nb == p and d == "out" for (nb, d) in nbrs[x])  # arc x->p
            allowed = fwd_mask[:, vp] if arc_out else fwd_mask[vp]        # constraint given φ(p)
            phi[x] = int(np.argmin(np.where(allowed, belief, INF)))

    # Cost is the summed BP optimum (+inf if ANY component was infeasible), NOT a re-summed drift --
    # a re-sum would charge a finite cost for the argmin=0 fallback of an all-INF belief.
    phi = {a: phi.get(a, 0) for a in free_verts}
    return phi, total_cost


# --------------------------------------------------------------------------------------
# Candidate B-vertices near an A-vertex (same policy for both solvers)
# --------------------------------------------------------------------------------------
def _candidates(c, ax, ay, bx, by, k):
    return [int(v) for v in np.argsort(np.hypot(bx - ax[c], by - ay[c]))[:k]]


def _pins_feasible(ga, pinned, fwd_mask):
    """No A-arc between two PINNED vertices may run backward in B. The forest solver only folds
    pinned-to-FREE constraints, so an arc whose *both* endpoints are pinned is invisible to it and
    must be checked here (else an infeasible pinning is scored as if valid)."""
    for h, vh in pinned.items():
        for t in ga.pred_arcs[h]:
            if t in pinned and not fwd_mask[pinned[t], vh]:
                return False
    return True


def _choose_cut(verts, adj, ga, prefer=frozenset()):
    """Minimum vertex cut heuristic: the free vertex whose removal most reduces the cyclomatic
    number. Prefers a vertex in ``prefer`` (the global min-FVS) so the recursion conditions on the
    SAME vertices as :func:`_solve_fvs` -- otherwise the two solvers pick different (both-valid) cut
    sets and, under candidate restriction, can disagree. Then junctions, then higher degree."""
    best = best_key = None
    for c in verts:
        rest = [v for v in verts if v != c]
        radj = {a: {w for w in adj[a] if w != c} for a in rest}
        is_junc = len(ga.succ_arcs[c]) > 1 or len(ga.pred_arcs[c]) > 1
        key = (_cyclomatic(rest, radj), 0 if c in prefer else 1, 0 if is_junc else 1, -len(adj[c]))
        if best_key is None or key < best_key:
            best_key, best = key, c
    return best


# --------------------------------------------------------------------------------------
# Solver 1 -- recursive minimum vertex cut
# --------------------------------------------------------------------------------------
def _solve_recursive(ga, gb, verts_free, pinned, ctx, cand_k, fvs_set=frozenset()):
    ax, ay, bx, by = ctx.ax, ctx.ay, ctx.bx, ctx.by
    fpred, fsucc = _induced(ga, verts_free)
    fadj = _undirected(fpred, fsucc, verts_free)
    if _cyclomatic(verts_free, fadj) == 0:                 # forest -> exact base solve
        return _forest_solve(ga, gb, verts_free, pinned, ctx)

    c = _choose_cut(verts_free, fadj, ga, prefer=fvs_set)  # cut at a min-FVS vertex if one is here
    rest = [v for v in verts_free if v != c]
    radj = {a: {w for w in fadj[a] if w != c} for a in rest}
    comps = _components(rest, radj)
    NB = len(bx)
    wide = min(NB, max(4 * cand_k, 32))                    # bounded widening (avoid NB**|F| blow-up)
    for cands in (_candidates(c, ax, ay, bx, by, cand_k),
                  _candidates(c, ax, ay, bx, by, wide)):   # widen if none of the near ones is feasible
        best_phi, best_cost = None, float("inf")
        for vc in cands:                                   # enumerate the cut, recurse per component
            p2 = dict(pinned); p2[c] = int(vc)
            if not _pins_feasible(ga, p2, ctx.fwd_mask):   # arc between two pinned vertices runs backward
                continue
            phi_all = {c: int(vc)}
            cost = float(np.hypot(ax[c] - bx[vc], ay[c] - by[vc]))
            feasible = True
            for comp in comps:
                phic, costc = _solve_recursive(ga, gb, comp, p2, ctx, cand_k, fvs_set)
                if phic is None or not np.isfinite(costc):  # component infeasible under this pinning
                    feasible = False; break
                phi_all.update(phic); cost += costc
            if feasible and cost < best_cost:
                best_cost, best_phi = cost, phi_all
        if best_phi is not None and np.isfinite(best_cost):
            break                                          # found a feasible pinning; no need to widen
    if best_phi is None:                                   # no feasible labelling within the candidates
        return {}, float("inf")
    return best_phi, best_cost


# --------------------------------------------------------------------------------------
# Solver 2 -- one-shot minimum feedback vertex set
# --------------------------------------------------------------------------------------
def _solve_fvs(ga, gb, ctx, cand_k):
    ax, ay, bx, by = ctx.ax, ctx.ay, ctx.bx, ctx.by
    F = min_feedback_vertex_set(ga)
    all_verts = list(range(ga.n_vertices))
    if not F:
        return _forest_solve(ga, gb, all_verts, {}, ctx)
    free = [v for v in all_verts if v not in set(F)]
    NB = len(bx)
    wide = min(NB, max(4 * cand_k, 32))                    # bounded widening (avoid NB**|F| blow-up)
    knn = [_candidates(f, ax, ay, bx, by, cand_k) for f in F]
    wider = [_candidates(f, ax, ay, bx, by, wide) for f in F]
    for cand_lists in (knn, wider):                        # widen if none of the near ones is feasible
        best_phi, best_cost = None, float("inf")
        for combo in product(*cand_lists):                 # enumerate the WHOLE F jointly
            pinned = dict(zip(F, combo))
            if not _pins_feasible(ga, pinned, ctx.fwd_mask):   # arc INTERNAL to F runs backward in B
                continue
            phi, cost = _forest_solve(ga, gb, free, pinned, ctx)
            cost += sum(float(np.hypot(ax[f] - bx[pinned[f]], ay[f] - by[pinned[f]])) for f in F)
            phi = dict(phi); phi.update(pinned)
            if cost < best_cost:
                best_cost, best_phi = cost, phi
        if best_phi is not None and np.isfinite(best_cost):
            break
    if best_phi is None:                                   # no feasible labelling within the candidates
        return {}, float("inf")
    return best_phi, best_cost


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def conditioned_labels(a_edges, b_edges, method="recursive", *, snap_tolerance_m=0.5,
                       step_meters=2.0, cand_k=CAND_K) -> Tuple[Any, Any, Dict[int, int], float]:
    """Label a source DAG onto ``b_edges`` by exact conditioning. ``method`` is ``"recursive"``
    (minimum vertex cut) or ``"fvs"`` (minimum feedback vertex set). Returns
    ``(ga, gb, phi, realized_cost)``. The two methods must return equal-cost labellings."""
    a_pts = [(float(x), float(y)) for _id, g in a_edges for (x, y) in g.coords]
    b_pts = [(float(x), float(y)) for _id, g in b_edges for (x, y) in g.coords]
    ga = build_local_digraph(a_edges, b_pts, snap_tolerance_m, step_meters)
    gb = build_local_digraph(b_edges, a_pts, snap_tolerance_m, step_meters)
    topological_order(ga)                              # acyclicity guard (raises NotADAG)
    ctx = _Ctx(ga.vx, ga.vy, gb.vx, gb.vy, _forward_reach_mask(gb))
    if method == "recursive":
        F = frozenset(min_feedback_vertex_set(ga))     # cut at these, matching _solve_fvs's set
        phi, cost = _solve_recursive(ga, gb, list(range(ga.n_vertices)), {}, ctx, cand_k, F)
    elif method == "fvs":
        phi, cost = _solve_fvs(ga, gb, ctx, cand_k)
    else:
        raise ValueError(f"unknown method {method!r}; use 'recursive' or 'fvs'")
    return ga, gb, phi, cost
