"""The `match_dag(engine="auto")` arm of the §7 gate: dispatch vs `extract_cell` on cyclic B.

gate_profiled_cyclicB.py exercises `extract_profiled` DIRECTLY. This one goes through the public
entry point, so it also covers the dispatch itself: `predict_work`'s refusal gate, the `W`/`Mo`
routing (docs §9), and the resource ceilings. A source routed to `"cell"` should match `extract_cell`
exactly; one routed to `"profiled"`/`"rebase"` should match it or answer where it refuses.

Reuses the same generator as gate_profiled_cyclicB.py so the population is identical and the two
gates' counts are directly comparable.

Run:  python3 -u report/gate_auto_dispatch.py [n_seeds]
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "scripts", "repro_contraction_eviction"))

from network_matching.dag_dtw import (prepare, forward, extract_cell, check_rules, _cost_of,
                                      match_dag)
from network_matching.profiled import profiled_width, merge_pressure
from regress_hunt import rand_case                              # the repro's own generator


def valid_cost(M, A, B, alpha, beta):
    """Cost when M is a VALID complete matching, else None + why."""
    v1, v2, v3 = check_rules(M, A, B)
    v4 = [a for a in A.nodes if not any(x == a for (x, _w) in M)]
    if v1 or v2 or v3 or v4:
        return None, f"INVALID V1={len(v1)} V2={len(v2)} V3={len(v3)} V4={len(v4)}"
    return _cost_of(A, B, M, alpha, beta), ""


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tot = both = same = auto_only = cell_only = 0
    invalid, costlier = [], []
    widths, pressures = Counter(), Counter()

    for seed in range(500, 500 + n_seeds):
        try:
            A, B = rand_case(seed)
        except Exception:                                       # noqa: BLE001
            continue
        for alpha, beta in ((1.0, 1.0), (0.5, 1.0), (0.5, 1.5)):
            prepare(A, B, r=40.0)
            try:
                forward(A, B, alpha, beta)
            except ValueError:
                continue
            tot += 1
            widths[profiled_width(A)] += 1
            pressures[merge_pressure(A)] += 1

            try:
                c_cost, c_why = valid_cost(extract_cell(A, B, alpha, beta)[0], A, B, alpha, beta)
            except Exception:                                   # noqa: BLE001
                c_cost = None
            # match_dag re-runs prepare/forward itself; r matches the loop above
            try:
                a_cost, a_why = valid_cost(match_dag(A, B, r=40.0, alpha=alpha, beta=beta)[0],
                                           A, B, alpha, beta)
            except Exception as e:                              # noqa: BLE001
                a_cost, a_why = None, f"{type(e).__name__}: {str(e)[:36]}"
            if a_why.startswith("INVALID"):
                invalid.append((seed, alpha, beta, a_why))

            if c_cost is not None and a_cost is not None:
                both += 1
                if abs(c_cost - a_cost) < 1e-6:
                    same += 1
                else:
                    costlier.append((seed, alpha, beta, c_cost, a_cost))
            elif c_cost is None and a_cost is not None:
                auto_only += 1
            elif c_cost is not None and a_cost is None:
                cell_only += 1
                invalid.append((seed, alpha, beta, "cell ok, auto " + a_why))

    print(f"\nmatch_dag(auto) on cyclic B: {tot} cases  (seeds 500-{500+n_seeds-1} x 3 weightings)")
    print(f"  both answered          {both}")
    print(f"  COST PARITY            {same}/{both}   <- the gate")
    print(f"  auto answered where cell RAISED       {auto_only}")
    print(f"  cell answered where auto failed       {cell_only}")
    print(f"  auto returned INVALID                 "
          f"{len([i for i in invalid if 'INVALID' in i[3]])}")
    print(f"  width histogram        {dict(sorted(widths.items()))}")
    print(f"  merge-pressure hist    {dict(sorted(pressures.items()))}")
    for s, al, be, c, a in costlier[:10]:
        print(f"  COST DIFF seed={s} a={al} b={be}: cell={c:.6f} auto={a:.6f}")
    for row in invalid[:10]:
        print(f"  FAIL seed={row[0]} a={row[1]} b={row[2]}: {row[3]}")
    print("\nVERDICT:", "GREEN" if (same == both and not invalid and cell_only == 0) else "RED")
