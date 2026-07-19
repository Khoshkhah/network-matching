"""Real-world batch: `auto` vs `extract_cell` over a random sample of vancouver_city edges.

Everything else in report/ measures hand-picked hard edges or synthetic families. This measures the
population the library actually runs on.

LADDER SEMANTICS MATTER. `mapconflation.match.direction` (`:201`) escalates `hp.rladder` when a
radius produces no result -- including when EXTRACTION fails, not only `forward`. An earlier version
of this probe broke out of the ladder as soon as `forward` succeeded, tried extraction once, and gave
up; that inflated the refusal count 4x and was mistakenly reported as a defect in `match_task`. This
probe mirrors the real loop.

Run:
  PYTHONPATH=/home/kaveh/projects/map-conflation/src \
  N=150 /home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/probe_batch.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_profiled_hourglass import build_graphs, NET, REF
from mapconflation.clean import load_network
from mapconflation.match import load_reference, Reference, local_dag as ld
from mapconflation.config import load_hyperparams
from network_matching.dag_dtw import (prepare, forward, extract_by_engine, extract_cell,
                                      check_rules, _cost_of)
from network_matching.profiled import profiled_width, merge_pressure


def solve(LA, LB, hp, rladder, engine):
    """(cost, r_used) by the real ladder: advance whenever a radius yields no valid matching."""
    for r in rladder:
        prepare(LA, LB, r=r, k_min=hp.k_min, bearing_weight=hp.bearing_weight)
        try:
            forward(LA, LB, alpha=hp.alpha, beta=hp.beta)
            M = (extract_by_engine(LA, LB, hp.alpha, hp.beta, "auto")[0] if engine == "auto"
                 else extract_cell(LA, LB, hp.alpha, hp.beta)[0])
        except Exception:
            continue
        v1, v2, v3 = check_rules(M, LA, LB)
        if v1 or v2 or v3 or {a for a, _ in M} != set(LA.nodes):
            continue
        return _cost_of(LA, LB, M, hp.alpha, hp.beta), r
    return None, None


if __name__ == "__main__":
    N = int(os.environ.get("N", "150"))
    hp = load_hyperparams("vancouver_city").hp
    G = load_network(NET)
    geoms, adj = ld.from_graph(G, snap_m=hp.snap_m)
    ref = Reference(load_reference(REF))
    ids = sorted(geoms)
    random.Random(0).shuffle(ids)
    ids = ids[:N]
    print(f"{len(ids)} edges, rladder={tuple(hp.rladder)}\n", flush=True)

    solved = refused = 0
    engines, shapes, radii = Counter(), Counter(), Counter()
    t_auto = t_cell = 0.0
    disagree, auto_only, cell_only, unmatched = [], 0, 0, []

    for i, lid in enumerate(ids):
        try:
            LA, LB = build_graphs(lid, geoms, adj, hp, ref)
        except Exception:
            continue
        shapes[(profiled_width(LA), merge_pressure(LA))] += 1
        engines[("profiled" if profiled_width(LA) <= 2
                 else "rebase" if merge_pressure(LA) >= profiled_width(LA) else "cell")] += 1

        t = time.perf_counter()
        a_cost, r_used = solve(LA, LB, hp, hp.rladder, "auto")
        t_auto += time.perf_counter() - t
        LA2, LB2 = build_graphs(lid, geoms, adj, hp, ref)
        t = time.perf_counter()
        c_cost, _ = solve(LA2, LB2, hp, hp.rladder, "cell")
        t_cell += time.perf_counter() - t

        if a_cost is not None:
            solved += 1
            radii[r_used] += 1
        else:
            refused += 1
            unmatched.append(lid)
        if a_cost is not None and c_cost is None:
            auto_only += 1
        elif c_cost is not None and a_cost is None:
            cell_only += 1
        elif a_cost is not None and c_cost is not None and abs(a_cost - c_cost) > 1e-6:
            disagree.append((lid, c_cost, a_cost))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(ids)}  solved={solved} refused={refused}", flush=True)

    print(f"\nsolved {solved} / refused {refused}  of {len(ids)}")
    print(f"engine chosen   {dict(engines)}")
    print(f"(W,Mo) shapes   {dict(sorted(shapes.items()))}")
    print(f"radius used     {dict(sorted(radii.items()))}")
    print(f"total time      auto {t_auto:7.1f}s     extract_cell {t_cell:7.1f}s"
          f"     ({t_cell / max(t_auto, 1e-9):.0f}x)")
    print(f"auto solved where cell could not : {auto_only}")
    print(f"cell solved where auto could not : {cell_only}")
    print(f"cost disagreements               : {len(disagree)}")
    if unmatched:
        print(f"unmatched at every radius        : {unmatched}")
