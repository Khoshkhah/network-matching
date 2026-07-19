"""The cyclic-B arm of the §7 gate: extract_profiled vs extract_cell on random trees over CYCLIC B.

The point-mode envelope (gate_profiled.py) uses acyclic targets, so it never reaches V1 -- the rule
that bites when the matching CROSSES -- V1 forbids a predecessor of `a` sitting on a successor of
`v`, and a cyclic B lets Bsucc(v) wrap so that becomes reachable. Real road networks always have
B-cycles, and the known
open defect lives exactly here: `extract_cell` keeps only the CHEAPEST row per pending signature, so
a cheap-but-invalid row can evict the valid-but-costlier one sharing its signature
(scripts/repro_contraction_eviction/README.md -- 18 spurious raises + 1 displaced optimum per 900).

extract_profiled contracts the SAME way (cheapest per profile) and has NO terminal judge at all, so
the question is whether it inherits the defect, avoids it, or is worse. Reuses that repro's own case
generator so the population is identical.

Run:  python3 -u report/gate_profiled_cyclicB.py [n_seeds]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "scripts", "repro_contraction_eviction"))

from network_matching.dag_dtw import (prepare, forward, extract_cell, check_rules, _cost_of)
from network_matching.profiled import forward_profiled, extract_profiled
from regress_hunt import rand_case                              # the repro's own generator


def arm(fn, A, B, alpha, beta, profiled=False):
    """(cost, why): cost when a VALID complete matching came out, else None + reason."""
    try:
        if profiled:
            forward_profiled(A, B, alpha, beta)
            M = extract_profiled(A, B, alpha, beta)[0]
        else:
            M = fn(A, B, alpha, beta)[0]
    except ValueError as e:
        return None, f"raise: {str(e)[:36]}"
    except Exception as e:                                      # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:36]}"
    v1, v2, v3 = check_rules(M, A, B)
    v4 = [a for a in A.nodes if not any(x == a for (x, _w) in M)]
    if v1 or v2 or v3 or v4:
        return None, f"INVALID V1={len(v1)} V2={len(v2)} V3={len(v3)} V4={len(v4)}"
    return _cost_of(A, B, M, alpha, beta), ""


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tot = both = same = 0
    prof_only = cell_only = 0
    costlier = []
    invalid = []
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
            c_cost, c_why = arm(extract_cell, A, B, alpha, beta)
            p_cost, p_why = arm(None, A, B, alpha, beta, profiled=True)
            if p_why.startswith("INVALID"):
                invalid.append((seed, alpha, beta, p_why))
            if c_cost is not None and p_cost is not None:
                both += 1
                if abs(c_cost - p_cost) < 1e-6:
                    same += 1
                elif p_cost > c_cost + 1e-6:
                    costlier.append((seed, alpha, beta, c_cost, p_cost))
                else:
                    costlier.append((seed, alpha, beta, c_cost, p_cost))   # profiled CHEAPER
            elif c_cost is None and p_cost is not None:
                prof_only += 1
            elif c_cost is not None and p_cost is None:
                cell_only += 1
                invalid.append((seed, alpha, beta, "cell ok, prof " + p_why))

    print(f"\ncyclic-B cases: {tot}  (seeds 500-{500+n_seeds-1} x 3 weightings)")
    print(f"  both answered          {both}")
    print(f"  COST PARITY            {same}/{both}   <- the gate")
    print(f"  profiled answered where cell RAISED   {prof_only}")
    print(f"  cell answered where profiled failed   {cell_only}")
    print(f"  profiled returned INVALID             "
          f"{len([i for i in invalid if 'INVALID' in i[3]])}")
    for s, al, be, c, p in costlier[:10]:
        rel = "profiled COSTLIER" if p > c else "profiled cheaper"
        print(f"  COST DIFF seed={s} a={al} b={be}: cell={c:.6f} prof={p:.6f}  {rel}")
    for row in invalid[:10]:
        print(f"  FAIL seed={row[0]} a={row[1]} b={row[2]}: {row[3]}")
    print("\nVERDICT:", "GREEN" if (same == both and not invalid and cell_only == 0) else "RED")
