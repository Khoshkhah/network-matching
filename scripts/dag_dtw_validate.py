#!/usr/bin/env python
"""Validate & test point-to-point DAG-DTW on the synthetic DAG scenarios.

Runs each built-in DAG (chain / y_split / merge / diamond) and checks the invariants of
``docs/dag_dtw_matching.md``:

  1. every A-edge routes to its intended B-edge (allowing a trailing junction spill),
  2. drift is near the built-in B offset,
  3. junction A-vertices land in one junction REGION (v1 region-consistency),
  4. the topological order respects the arcs, and a cyclic source raises ``NotADAG``.

Then it runs a **perturbation sweep** (rigidly shifting the whole DAG) and reports where the route
first changes -- documenting the point-to-point model's behaviour at junctions (a junction spills
onto the nearest B-edge once the shift pulls it there; there is no bearing term in point mode).

    python scripts/dag_dtw_validate.py                 # full report, exits non-zero on failure
    python scripts/dag_dtw_validate.py --case y_split  # one scenario
    python scripts/dag_dtw_validate.py --sweep         # only the perturbation sweep

Exit code 0 = all core invariants held, 1 = a failure.
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching.dag_dtw import NotADAG, match_dag_to_bgraph, topological_order  # noqa: E402
from network_matching.dag_dtw import build_local_digraph  # noqa: E402
from network_matching.dag_synthetic import DAG_SCENARIOS, get_dag, list_dags  # noqa: E402
from network_matching.dag_playground import perturb_dag  # noqa: E402

_OFF = 0.4  # the offset the scenarios put B off A

# Each A-edge's intended primary B-edge.
EXPECT = {
    "chain": {"A1": "B1", "A2": "B2"},
    "y_split": {"A_main": "B_main", "A_left": "B_left", "A_right": "B_right"},
    "merge": {"A_top": "B_top", "A_bot": "B_bot", "A_out": "B_out"},
    "diamond": {"A_in": "B_in", "A_up": "B_up", "A_dn": "B_dn",
                "A_up2": "B_up2", "A_dn2": "B_dn2", "A_out": "B_out"},
}

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"


def _ok(cond):
    return PASS if cond else FAIL


def _fwd_reachable(gb, src, dst):
    """Is B-vertex ``dst`` reachable from ``src`` by walking GB forward arcs (or src == dst)?"""
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


def sequence_rules(res):
    """The matched sequence must obey the map-matching rules. Returns (arc_violations,
    route_violations):

      - **monotone forward B-walk**: every GA arc ``a -> a'`` maps to a forward B-step
        ``φ(a) -> φ(a')`` (reachable along GB arcs, or equal) -- as A advances B never goes
        backward or jumps to a disconnected vertex;
      - **connected route**: consecutive B-edges in a route are graph-connected in GB.
    """
    ga, gb, phi = res["GA"], res["GB"], res["phi"]
    arc_viol = []
    for a in range(ga.n_vertices):
        if a not in phi:
            continue
        for a2 in ga.succ_arcs[a]:
            if a2 in phi and not _fwd_reachable(gb, phi[a], phi[a2]):
                arc_viol.append((ga.edge_ids[ga.vert_edge[a]], ga.edge_ids[ga.vert_edge[a2]]))
    econ = set()                                       # B-edge -> B-edge connectivity in GB
    for u in range(gb.n_vertices):
        for w in gb.succ_arcs[u]:
            eu, ew = gb.edge_ids[gb.vert_edge[u]], gb.edge_ids[gb.vert_edge[w]]
            if eu != ew:
                econ.add((eu, ew))
    route_viol = []
    for aeid, route in res["routes"].items():
        for i in range(1, len(route)):
            if (route[i - 1], route[i]) not in econ:
                route_viol.append((aeid, route[i - 1], route[i]))
    return arc_viol, route_viol


def validate_case(name):
    """Return (n_checks, n_failed) after printing a per-scenario validation block."""
    sc = get_dag(name)
    res = match_dag_to_bgraph(sc["a_edges"], sc["b_edges"], debug=True, **sc["defaults"])
    routes, drift = res["routes"], res["avg_drift"]
    ga, gb, phi = res["GA"], res["GB"], res["phi"]
    fails = 0

    print(f"\n=== {name} ===  ({DAG_SCENARIOS[name]['description'].splitlines()[0]})")
    print(f"    avg drift {drift:.2f} m   sources {len(res['sources'])}   sinks {len(res['sinks'])}")

    # 1. routing
    for aeid, want in EXPECT[name].items():
        got = routes.get(aeid, [])
        ok = bool(got) and got[0] == want
        fails += not ok
        spill = f"  (+spill {got[1:]})" if len(got) > 1 else ""
        print(f"    [{_ok(ok)}] {aeid:8s} -> {got[0] if got else 'NONE':8s} expect {want}{spill}")

    # 2. drift near the offset
    dok = abs(drift - _OFF) < 0.25
    fails += not dok
    print(f"    [{_ok(dok)}] drift {drift:.2f} m  ~ offset {_OFF} m")

    # 3. junction region-consistency -- a junction = an A-LOCATION where >= 2 edge-endpoint
    # vertices coincide (a diamond has two); check ALL coincident vertices there agree in B.
    groups = {}
    for a in range(ga.n_vertices):
        if ga.is_endpoint[a]:
            key = (round(float(ga.vx[a]), 3), round(float(ga.vy[a]), 3))
            groups.setdefault(key, []).append(a)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    n_junc_checks = 0
    for key, verts in sorted(groups.items()):
        locs = np.array([(gb.vx[phi[a]], gb.vy[phi[a]]) for a in verts if a in phi])
        spread = float(np.max(np.ptp(locs, axis=0))) if len(locs) else 0.0
        jok = spread < 0.6
        fails += not jok
        n_junc_checks += 1
        print(f"    [{_ok(jok)}] junction @({key[0]:.0f},{key[1]:.0f}): {len(verts)} A-vertices "
              f"span {spread:.2f} m in B (< 0.6)")

    # 4. SEQUENCE RULES: monotone forward B-walk + connected route
    arc_v, route_v = sequence_rules(res)
    aok, rok = not arc_v, not route_v
    fails += (not aok) + (not rok)
    print(f"    [{_ok(aok)}] monotone forward B-walk (every A arc -> a forward B step)"
          + (f"  {len(arc_v)} violation(s): {arc_v[:3]}" if arc_v else ""))
    print(f"    [{_ok(rok)}] connected route (consecutive B-edges graph-connected)"
          + (f"  {len(route_v)} violation(s): {route_v[:3]}" if route_v else ""))

    return len(EXPECT[name]) + 1 + n_junc_checks + 2, fails


def validate_structure():
    """Topological-order validity + acyclicity guard."""
    fails = 0
    sc = get_dag("diamond")
    b_pts = [p for _e, g in sc["b_edges"] for p in g.coords]
    ga = build_local_digraph(sc["a_edges"], b_pts, 0.5, 2.0)
    order = topological_order(ga)
    pos = {v: i for i, v in enumerate(order)}
    topo_ok = all(pos[u] < pos[w] for u in range(ga.n_vertices) for w in ga.succ_arcs[u])
    fails += not topo_ok
    print(f"\n=== structure ===")
    print(f"    [{_ok(topo_ok)}] topological order respects all arcs")

    try:
        match_dag_to_bgraph([("e1", LineString([(0, 0), (10, 0)])),
                             ("e2", LineString([(10, 0), (0, 0)]))],
                            [("B", LineString([(0, 0.4), (10, 0.4)]))],
                            snap_tolerance_m=0.5, step_meters=2.0)
        cyc_ok = False
    except NotADAG:
        cyc_ok = True
    fails += not cyc_ok
    print(f"    [{_ok(cyc_ok)}] cyclic source raises NotADAG")
    return 2, fails


def perturbation_sweep(name="y_split"):
    """Rigidly shift the whole DAG and report where the route first changes -- point-to-point has
    no bearing, so a junction eventually spills onto the nearest B-edge."""
    sc = get_dag(name)
    print(f"\n=== perturbation sweep: {name} (rigid shift, point-to-point) ===")
    print(f"    {'shift m':>8} | {'avg drift':>9} | routes")
    print(f"    {'shift m':>8} | {'drift':>6} | {'seq-rule':>9} | routes")
    base = None
    for s in [0, 1, 2, 3, 4, 6, 8]:
        a = perturb_dag(sc["a_edges"], shift=float(s))
        res = match_dag_to_bgraph(a, sc["b_edges"], debug=True, **sc["defaults"])
        r = {k: v for k, v in res["routes"].items()}
        base = base or r
        arc_v, route_v = sequence_rules(res)
        seq = "OK" if not arc_v and not route_v else f"VIOLATED({len(arc_v)+len(route_v)})"
        changed = "  <- route changed" if r != base else ""
        print(f"    {s:>8} | {res['avg_drift']:>4.2f} m | {seq:>9} | {r}{changed}")
    print("    A route change AND a monotone-walk VIOLATION appear together once the shift pulls the "
          "junction\n    onto a cross road: point-to-point spills the junction, and the coincident "
          "junction vertices then\n    disagree in B (backtrack resolves them independently). This "
          "is the vertex-exact junction gap.")


def main():
    ap = argparse.ArgumentParser(description="Validate point-to-point DAG-DTW on synthetic DAGs.")
    ap.add_argument("--case", choices=list_dags(), help="validate only this scenario")
    ap.add_argument("--sweep", action="store_true", help="only run the perturbation sweep")
    args = ap.parse_args()

    if args.sweep:
        perturbation_sweep(args.case or "y_split")
        return

    total = failed = 0
    names = [args.case] if args.case else list_dags()
    for name in names:
        n, f = validate_case(name)
        total += n
        failed += f
    if not args.case:
        n, f = validate_structure()
        total += n
        failed += f
        perturbation_sweep("y_split")

    print(f"\n{'='*50}\nVALIDATION: {total - failed}/{total} checks passed"
          f"{'' if not failed else f'  ({failed} FAILED)'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
