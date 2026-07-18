"""The §7 gate for the profiled engine: the structured envelope of scripts/test_dag_point.py, with
extract_profiled added as a third column.

Reuses that script's case construction verbatim (STRUCTURES, build) so the sweep is the SAME 384
combinations the existing engines are measured on -- structure x density x shift x noise x (alpha,
beta). The alpha/beta axis is the point: beta-stalls and alpha-coverage runs are exactly the corners
where docs §6.2's fraction-exactness argument could break, and the acceptance probes barely touched
them (mostly alpha=beta=1).

Standard: parity to the digit. The design is exact, so any cost divergence is a bug.

Run:  python3 -u report/gate_profiled.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_matching.dag_dtw import (prepare, forward, extract_cell, extract_join, check_rules,
                                      _cost_of)
from network_matching.profiled import forward_profiled, extract_profiled
from scripts.test_dag_point import STRUCTURES, build


def engines(A, B, alpha, beta):
    """(name -> (M, cost, why)) for each engine on one prepared case."""
    out = {}

    def validate(name, M):
        v1, v2, v3 = check_rules(M, A, B)
        v4 = [a for a in A.nodes if not any(x == a for (x, _w) in M)]
        if v1 or v2 or v3 or v4:
            out[name] = (None, None, f"V1={len(v1)} V2={len(v2)} V3={len(v3)} V4={len(v4)}")
        else:
            out[name] = (M, _cost_of(A, B, M, alpha, beta), "")

    for name, fn in (("join", extract_join), ("cell", extract_cell)):
        try:
            validate(name, fn(A, B, alpha, beta)[0])
        except ValueError as e:
            out[name] = (None, None, f"infeasible: {str(e)[:40]}")

    try:
        forward_profiled(A, B, alpha, beta)
        M, _committed, join_cost = extract_profiled(A, B, alpha, beta)
        validate("prof", M)
        if out["prof"][0] is not None:                       # docs §6.2: sink-sum == recomputed cost
            out["prof"] = out["prof"] + (join_cost,)
    except ValueError as e:
        out["prof"] = (None, None, f"infeasible: {str(e)[:40]}")
    return out


if __name__ == "__main__":
    n_tot = 0
    ok = {"join": 0, "cell": 0, "prof": 0}
    cross = same = joineq = 0
    both = 0
    fails, cost_fails, join_fails = [], [], []

    for struct in STRUCTURES:
        for a_step, b_step in [(2.0, 2.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.5)]:
            for shift in (0.0, 0.5, 2.0, 5.0):
                for noise in (0.0, 0.3):
                    for alpha, beta in ((1.0, 1.0), (0.7, 1.0), (0.5, 1.5)):
                        A = build(STRUCTURES[struct], a_step, shift=shift, noise=noise, seed=7)
                        B = build(STRUCTURES[struct], b_step, shift=0.0, noise=0.0, seed=0)
                        prepare(A, B, r=20.0)
                        try:
                            forward(A, B, alpha, beta)
                        except ValueError:
                            continue
                        n_tot += 1
                        tag = (struct, a_step, b_step, shift, noise, alpha, beta)
                        try:
                            e = engines(A, B, alpha, beta)
                        except Exception as ex:              # noqa: BLE001
                            fails.append((tag, f"ERROR {type(ex).__name__}: {str(ex)[:40]}"))
                            continue
                        for k in ok:
                            ok[k] += e[k][0] is not None
                        Mc, cc, _ = e["cell"][:3]
                        Mj, cj, _ = e["join"][:3]
                        Mp, cp = e["prof"][0], e["prof"][1]
                        if Mc is not None and Mj is not None and cc <= cj + 1e-6:
                            cross += 1
                        if Mc is not None and Mp is not None:
                            both += 1
                            if abs(cc - cp) < 1e-6:
                                same += 1
                            else:
                                cost_fails.append((tag, f"cell={cc:.6f} prof={cp:.6f} "
                                                        f"({(cp-cc)/max(cc,1e-9)*100:+.2f}%)"))
                        if Mp is not None and len(e["prof"]) > 3:
                            if abs(e["prof"][3] - cp) < 1e-6:
                                joineq += 1
                            else:
                                join_fails.append((tag, f"join={e['prof'][3]:.6f} cost={cp:.6f}"))
                        if e["prof"][0] is None:
                            fails.append((tag, "prof: " + e["prof"][2]))

    print(f"\ncases: {n_tot}")
    print(f"  valid   join {ok['join']}/{n_tot}   cell {ok['cell']}/{n_tot}   "
          f"PROFILED {ok['prof']}/{n_tot}")
    print(f"  invariant cell<=join            {cross}/{n_tot}")
    print(f"  COST PARITY  prof == cell       {same}/{both}   <- the gate")
    print(f"  docs §6.2    join == _cost_of   {joineq}/{ok['prof']}")
    for t, w in cost_fails[:10]:
        print("  COST FAIL", t, w)
    for t, w in join_fails[:6]:
        print("  JOIN FAIL", t, w)
    for t, w in fails[:10]:
        print("  FAIL", t, w)
    print("\nVERDICT:", "GREEN" if (same == both and not fails and not join_fails
                                    and ok['prof'] == ok['cell']) else "RED")
