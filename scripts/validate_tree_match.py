"""Validate a Tree-DTW match at the **anchor** level -- i.e. the decision the forward `D` / backward
`B` tables produce (φ: each source point → one target point), *before* the gap-fill and the `M`
relation. This is the honest check: the gap-fill is built so V1-V4 hold by construction, so validating
the gap-filled `M` is nearly vacuous; validating the anchors tells you whether the tables are right.

The four rules are checked **coverage-aware** (by forward B-reachability, not immediate adjacency), so
a legitimate junction gap (B sampled finer than A) passes, but a backward jump, a disconnected jump,
or two independent A-branches landing on the same B-edge fails. Output is in **edge names +
coordinates**, never bare vertex indices.

Run:  python scripts/validate_tree_match.py
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------------------
# forward reachability in the target graph GB
# --------------------------------------------------------------------------------------
def _reach(gb, s: int) -> set:
    seen = {s}
    st = [s]
    while st:
        u = st.pop()
        for w in gb.succ_arcs[u]:
            if w not in seen:
                seen.add(w)
                st.append(w)
    return seen


def _subtree(ga, root: int) -> set:
    """All A-vertices downstream of (and including) `root` in the source tree."""
    seen = {root}
    st = [root]
    while st:
        u = st.pop()
        for w in ga.succ_arcs[u]:
            if w not in seen:
                seen.add(w)
                st.append(w)
    return seen


# --------------------------------------------------------------------------------------
# the validator
# --------------------------------------------------------------------------------------
def validate_anchor(res) -> Dict[str, Any]:
    """Check the anchor matching φ = res["anchor"] against V1-V4 (coverage-aware). Returns
    ``{ok, v1_backward, v2_merge, v3_split, v4_uncovered, overlap}`` where each list holds
    human-readable violation strings."""
    ga, gb, phi = res["GA"], res["GB"], res["anchor"]
    aeid = lambda a: ga.edge_ids[ga.vert_edge[a]]
    beid = lambda v: gb.edge_ids[gb.vert_edge[v]]
    apt = lambda a: f"({ga.vx[a]:.2f},{ga.vy[a]:.2f})"
    bpt = lambda v: f"({gb.vx[v]:.2f},{gb.vy[v]:.2f})"

    reach_cache: Dict[int, set] = {}

    def reach(v):
        if v not in reach_cache:
            reach_cache[v] = _reach(gb, v)
        return reach_cache[v]

    v4 = [f"A '{aeid(a)}' {apt(a)} has no anchor" for a in range(ga.n_vertices) if a not in phi]

    # V1 -- monotone: every source arc (a→c) must land φ(c) forward-reachable from φ(a).
    v1: List[str] = []
    for a in range(ga.n_vertices):
        if a not in phi:
            continue
        for c in ga.succ_arcs[a]:
            if c in phi and phi[c] not in reach(phi[a]):
                v1.append(f"A '{aeid(a)}'{apt(a)}→B '{beid(phi[a])}'{bpt(phi[a])}  then  "
                          f"A '{aeid(c)}'{apt(c)}→B '{beid(phi[c])}'{bpt(phi[c])}  "
                          f"-- child not forward-reachable (backward/disconnected)")

    # V2 -- merge: every predecessor of a merge must forward-reach the merge's point.
    v2: List[str] = []
    for m in range(ga.n_vertices):
        if len(ga.pred_arcs[m]) > 1 and m in phi:
            for p in ga.pred_arcs[m]:
                if p in phi and phi[m] not in reach(phi[p]):
                    v2.append(f"merge A '{aeid(m)}'{apt(m)}→B{bpt(phi[m])}: predecessor "
                              f"A '{aeid(p)}'{apt(p)}→B{bpt(phi[p])} cannot reach it")

    # V3 -- split: every successor of a split must be forward-reachable from the split's point.
    v3: List[str] = []
    for s in range(ga.n_vertices):
        if len(ga.succ_arcs[s]) > 1 and s in phi:
            for c in ga.succ_arcs[s]:
                if c in phi and phi[c] not in reach(phi[s]):
                    v3.append(f"split A '{aeid(s)}'{apt(s)}→B{bpt(phi[s])}: exit "
                              f"A '{aeid(c)}'{apt(c)}→B{bpt(phi[c])} not reachable from it")

    # DISJOINTNESS -- distinct A-edges must map to distinct B-edges. Two A-edges routed through the
    # SAME B-edge is a mis-route/overlap (e.g. `down` landing on `stem`, which then also carries
    # `stem`). This is the tree-independence property (§2) at the edge level, and it's exactly what a
    # bare V1-V4 check misses. [Assumes A and B are comparably sampled -- the conflation case.]
    overlap: List[str] = []
    b_to_a: Dict[Any, set] = {}
    for a_edge, b_edges in res["routes"].items():
        for be in b_edges:
            b_to_a.setdefault(be, set()).add(a_edge)
    for be, a_set in b_to_a.items():
        if len(a_set) > 1:
            overlap.append(f"B-edge '{be}' is routed by A-edges {sorted(map(str, a_set))} "
                           f"-- distinct A-edges must map to distinct B-edges (mis-route/overlap)")

    ok = not (v1 or v2 or v3 or v4 or overlap)
    return {"ok": ok, "v1_backward": v1, "v2_merge": v2, "v3_split": v3,
            "v4_uncovered": v4, "overlap": overlap}


def report(res, title: str = "") -> bool:
    """Print the structure, the anchor matching, and the validation verdict -- all in edge names +
    coordinates. Returns the ok bool."""
    ga, gb, phi = res["GA"], res["GB"], res["anchor"]
    aeid = lambda a: ga.edge_ids[ga.vert_edge[a]]
    beid = lambda v: gb.edge_ids[gb.vert_edge[v]]
    print("=" * 78)
    if title:
        print(title)
    print("routes:", {k: v for k, v in res["routes"].items()})

    # anchor matching per A-edge, readable
    print("\nanchor matching  (A-edge point  ->  B-edge point   [drift]):")
    for a in sorted(range(ga.n_vertices), key=lambda a: (str(aeid(a)), a)):
        v = phi[a]
        d = ((ga.vx[a] - gb.vx[v]) ** 2 + (ga.vy[a] - gb.vy[v]) ** 2) ** 0.5
        print(f"  A '{aeid(a):>5}' ({ga.vx[a]:6.2f},{ga.vy[a]:6.2f})  ->  "
              f"B '{beid(v):>5}' ({gb.vx[v]:6.2f},{gb.vy[v]:6.2f})   [{d:4.2f} m]")

    r = validate_anchor(res)
    print(f"\nANCHOR VALID: {r['ok']}")
    for key, label in [("v4_uncovered", "V4 uncovered"), ("v1_backward", "V1 backward/disconnected"),
                       ("v2_merge", "V2 merge"), ("v3_split", "V3 split"),
                       ("overlap", "BRANCH-OVERLAP (mis-route)")]:
        for msg in r[key]:
            print(f"  [{label}] {msg}")
    return r["ok"]


# --------------------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from network_matching.tree_dtw import match_tree_to_bgraph
    from network_matching.dag_playground import perturb_dag

    KW = dict(snap_tolerance_m=0.6, step_meters=100.0)
    a_edges = [("stem", [(-2, 0), (0, 0)]), ("up", [(0, 0), (2, 1)]), ("down", [(0, 0), (2, -1)])]
    b_edges = [("stem", [(-2, 0.1), (0, 0.1)]), ("up", [(0, 0.1), (2, 1.1)]), ("down", [(0, 0.1), (2, -0.9)])]

    # (1) correct match -- should be VALID
    ok1 = report(match_tree_to_bgraph(a_edges, b_edges, emission="segment", bearing_weight=0, **KW),
                 "(1) aligned Y, no bearing  -- EXPECT valid")

    # (2) the mis-route we found -- should be INVALID (down lands on stem)
    A = perturb_dag(a_edges, shift=0, rotate=20, noise=0.0, seed=1)
    ok2 = report(match_tree_to_bgraph(A, b_edges, emission="segment", bearing_weight=2, **KW),
                 "(2) rotate=20, bearing=2  -- EXPECT invalid (down->stem overlap)")

    print("\n" + "=" * 78)
    print(f"SUMMARY:  case (1) valid={ok1} (want True)   case (2) valid={ok2} (want False)")
