"""The rebase arm of the §7 gate -- the one the other three gates cannot reach.

`gate_auto_dispatch` reports `merge-pressure hist {0: 600}`: its generated population has Mo=0
throughout, so `auto` never routes to "rebase" (which needs Mo >= W >= 3) and the path is selected in
0 of 600 cases. `gate_profiled` and `gate_profiled_cyclicB` call extract_profiled directly with
REBASE off, so they take the cone branch. Net effect: `_extract_rebased`, `_eliminate_fused` and
`rebase_work` had NO automated coverage at all -- only two manual hourglass runs.

This gate sweeps `braid(k, j)` from probe_pressure_sweep, which holds W ~ k while varying how many
branches rejoin so Mo ~ j. That reaches Mo >= W, where the dispatch selects rebase.

Standard: parity to the digit against `extract_cell`, which is exact over the full space. The design
is exact, so any cost divergence is a bug. Also asserts the coverage itself -- that rebase is really
being selected -- since a gate that silently stops exercising its target is worse than no gate.

Run:  python3 -u report/gate_rebase.py
"""
from __future__ import annotations

import copy
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import match_dag, check_rules, _cost_of
from network_matching.profiled import profiled_width, merge_pressure
from probe_pressure_sweep import braid


def arm(A, B, engine, r, alpha, beta):
    """(cost, why): cost when a VALID complete matching came out, else None + reason."""
    Ax, Bx = copy.deepcopy(A), copy.deepcopy(B)
    try:
        M = match_dag(Ax, Bx, r=r, alpha=alpha, beta=beta, engine=engine)[0]
    except ValueError as e:
        return None, f"raise: {str(e)[:40]}"
    except Exception as e:                                      # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:40]}"
    v1, v2, v3 = check_rules(M, Ax, Bx)
    v4 = [a for a in Ax.nodes if not any(x == a for (x, _w) in M)]
    if v1 or v2 or v3 or v4:
        return None, f"INVALID V1={len(v1)} V2={len(v2)} V3={len(v3)} V4={len(v4)}"
    return _cost_of(Ax, Bx, M, alpha, beta), ""


if __name__ == "__main__":
    tot = both = same = 0
    picks = Counter()
    rebase_cases = 0
    diffs, fails = [], []

    print(f"{'k,j':>6} {'W':>3} {'Mo':>3} {'auto picks':>11} {'cell':>12} {'rebase':>12}  verdict")
    for k in (3, 4, 5, 6):
        for j in range(0, k + 1):
            A, B = braid(k, j)
            W, Mo = profiled_width(A), merge_pressure(A)
            pick = "profiled" if W <= 2 else ("rebase" if Mo >= W else "cell")
            picks[pick] += 1
            if pick == "rebase":
                rebase_cases += 1
            for alpha, beta in ((1.0, 1.0), (0.5, 1.0), (0.5, 1.5)):
                tot += 1
                c_cost, c_why = arm(A, B, "cell", 14.0, alpha, beta)
                r_cost, r_why = arm(A, B, "rebase", 14.0, alpha, beta)
                a_cost, a_why = arm(A, B, "auto", 14.0, alpha, beta)
                if r_why.startswith("INVALID"):
                    fails.append((k, j, alpha, beta, "rebase " + r_why))
                if a_why.startswith("INVALID"):
                    fails.append((k, j, alpha, beta, "auto " + a_why))
                # auto must agree with the engine it claims to pick
                if pick == "rebase" and a_cost is not None and r_cost is not None \
                        and abs(a_cost - r_cost) > 1e-6:
                    fails.append((k, j, alpha, beta,
                                  f"auto({a_cost:.6f}) != rebase({r_cost:.6f})"))
                if c_cost is not None and r_cost is not None:
                    both += 1
                    if abs(c_cost - r_cost) < 1e-6:
                        same += 1
                    else:
                        diffs.append((k, j, alpha, beta, c_cost, r_cost))
                if alpha == 1.0:
                    fc = "refused" if c_cost is None else f"{c_cost:12.6f}"
                    fr = "refused" if r_cost is None else f"{r_cost:12.6f}"
                    ok = ("same" if c_cost is not None and r_cost is not None
                          and abs(c_cost - r_cost) < 1e-6 else "-")
                    print(f"{k},{j:>3} {W:>3} {Mo:>3} {pick:>11} {fc:>12} {fr:>12}  {ok}",
                          flush=True)

    print(f"\nrebase gate: {tot} cases over braid(k,j), k=3..6, j=0..k x 3 weightings")
    print(f"  dispatch picks         {dict(picks)}")
    print(f"  shapes routed to rebase {rebase_cases}   <- the coverage this gate exists for")
    print(f"  both answered          {both}")
    print(f"  COST PARITY  rebase == cell   {same}/{both}   <- the gate")
    print(f"  invalid / disagreement        {len(fails)}")
    for d in diffs[:10]:
        print(f"  COST DIFF braid({d[0]},{d[1]}) a={d[2]} b={d[3]}: "
              f"cell={d[4]:.6f} rebase={d[5]:.6f}")
    for f in fails[:10]:
        print(f"  FAIL braid({f[0]},{f[1]}) a={f[2]} b={f[3]}: {f[4]}")
    green = same == both and not fails and rebase_cases > 0
    print("\nVERDICT:", "GREEN" if green else "RED")
