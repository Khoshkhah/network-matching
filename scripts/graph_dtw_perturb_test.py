#!/usr/bin/env python
"""Perturbation-robustness sweep for graph-DTW: one case x many distortions of the A-edge.

Generalizes ``graph_dtw_shift_test.py`` from lateral shift only to every perturbation family in
``network_matching.synthetic`` (lateral / longitudinal shift, Gaussian noise, rotation, end
cropping / stretching). The A-edge is distorted at each magnitude of each family and re-matched
against the UNCHANGED B-network; the report shows, per family, how drift grows and at which
magnitude the route first changes (or the match dies) -- i.e. the algorithm's stability envelope.

Usage:
    python scripts/graph_dtw_perturb_test.py --case split
    python scripts/graph_dtw_perturb_test.py --case parallel_trap --families shift --negate
    python scripts/graph_dtw_perturb_test.py --edge-id 1377 --families shift,noise,rotate
    python scripts/graph_dtw_perturb_test.py --case curve --emission segment --seed 7

Writes output/graph_dtw_perturb_<case|edge>_<emission>.png; deep-dive one failing cell with
scripts/graph_dtw_debug_viz.py using the same case/family/magnitude/seed.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_dtw_debug_viz import A_COLOR, B_COLOR, COST_CMAP, load_case  # noqa: E402
from network_matching import match_edge_to_bgraph, setup_logging  # noqa: E402
from network_matching.synthetic import PERTURBATIONS, apply_perturbation  # noqa: E402


def run_family(case, kwargs, fam, grid, seed, fam_kwargs=None):
    """Match the case's A-edge at every magnitude of one family; returns list of row dicts."""
    rows = []
    for mag in grid:
        P = apply_perturbation(case["coords_a"], fam, mag, seed=seed, **(fam_kwargs or {}))
        res = match_edge_to_bgraph(P.tolist(), case["b_edges"], **kwargs)
        M = res["metrics"]
        rows.append(dict(
            family=fam, magnitude=mag, A=P, res=res,
            route=[re["dest_id"] for re in M["route_edges"]],
            drift=res["avg_distance"], overlap=M["overlap_pct"],
            bearing=M["bearing_diff"], ok=np.isfinite(res["avg_distance"])))
    base = rows[0]["route"]
    for r in rows:
        r["route_changed"] = r["route"] != base
    return rows


def draw_snapshot(ax, case, row, norm):
    """Small-multiple: B network + route (colored by per-edge drift) + the perturbed A."""
    gb = row["res"]["graph"]
    per_edge = {re["dest_id"]: re["match_dist_avg"]
                for re in row["res"]["metrics"]["route_edges"]}
    for e, eid in enumerate(gb.edge_ids):
        vids = np.where(gb.vert_edge == e)[0]
        if eid in per_edge:
            ax.plot(gb.vx[vids], gb.vy[vids], color=COST_CMAP(norm(per_edge[eid])), lw=4,
                    zorder=2, solid_capstyle="round")
        ax.plot(gb.vx[vids], gb.vy[vids], color=B_COLOR, lw=1.2, zorder=3)
    A = row["A"]
    ax.plot(A[:, 0], A[:, 1], color=A_COLOR, lw=1.8, zorder=4)
    ax.plot(A[0, 0], A[0, 1], marker="^", ms=7, color="#111827", zorder=5)
    flags = []
    if not row["ok"]:
        flags.append("NO MATCH")
    elif row["route_changed"]:
        flags.append("ROUTE CHANGED")
    ax.set_title(f"{row['magnitude']:g} {PERTURBATIONS[row['family']]['unit']} -> "
                 f"drift {row['drift']:.1f} m, overlap {row['overlap']}%"
                 + (f"  [{' ,'.join(flags)}]" if flags else ""),
                 fontsize=7.5, color="#dc2626" if flags else "#111827")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_curve(ax, rows, reject):
    """Drift vs magnitude, with route changes (red star) and NO_MATCH (open circle) marked."""
    mags = [r["magnitude"] for r in rows]
    drifts = [r["drift"] if r["ok"] else np.nan for r in rows]
    ax.plot(mags, drifts, "-o", color="#0e7490", ms=4, lw=1.4)
    for r in rows:
        if not r["ok"]:
            ax.plot(r["magnitude"], 0, marker="o", ms=9, mfc="none", mec="#dc2626", mew=1.5)
        elif r["route_changed"]:
            ax.plot(r["magnitude"], r["drift"], marker="*", ms=13, color="#dc2626", zorder=5)
    if reject:
        ax.axhline(reject, color="#15803d", ls=":", lw=1)
    spec = PERTURBATIONS[rows[0]["family"]]
    ax.set_xlabel(f"{rows[0]['family']} ({spec['unit']})", fontsize=8)
    ax.set_ylabel("avg drift (m)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)


def stability_summary(rows):
    """First magnitude at which the route changed / the match died (None = never)."""
    first_change = next((r["magnitude"] for r in rows if r["route_changed"]), None)
    first_dead = next((r["magnitude"] for r in rows if not r["ok"]), None)
    return first_change, first_dead


def main():
    ap = argparse.ArgumentParser(description="Graph-DTW perturbation-robustness sweep.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="synthetic scenario (see graph_dtw_debug_viz --list-cases)")
    src.add_argument("--edge-id", type=int, help="real OSM A-edge")
    ap.add_argument("--families", default=",".join(sorted(PERTURBATIONS)),
                    help="comma list of perturbation families (default: all)")
    ap.add_argument("--negate", action="store_true",
                    help="flip magnitude signs (e.g. shift toward the other side)")
    ap.add_argument("--translate-bearing", type=float, default=90.0,
                    help="compass direction for the translate family (0=north, 90=east, "
                         "180=south, 270=west)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emission", choices=["point", "segment"], default="point")
    ap.add_argument("--bearing-weight", type=float, default=0.0)
    ap.add_argument("--snap", type=float, default=None)
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--reject", type=float, default=5.0,
                    help="drift guide line on the curves (~resolve threshold)")
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_logging(console=False)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in PERTURBATIONS]
    if unknown:
        ap.error(f"unknown families {unknown}; available: {', '.join(sorted(PERTURBATIONS))}")

    case = load_case(args)
    kwargs = dict(case["kwargs"])
    if args.snap is not None:
        kwargs["snap_tolerance_m"] = args.snap
    if args.step is not None:
        kwargs["step_meters"] = args.step
    kwargs.update(emission=args.emission, bearing_weight=args.bearing_weight)

    results = {}
    print(f"[{case['name']}] emission={args.emission} seed={args.seed}\n")
    print(f"{'family':<13} {'mag':>7} | {'drift':>6} {'overlap':>7} {'bearD':>6}  route")
    for fam in families:
        grid = [(-m if args.negate else m) for m in PERTURBATIONS[fam]["grid"]]
        fam_kwargs = {"bearing": args.translate_bearing} if fam == "translate" else None
        rows = run_family(case, kwargs, fam, grid, args.seed, fam_kwargs)
        results[fam] = rows
        for r in rows:
            mark = ("  <-- NO MATCH" if not r["ok"]
                    else "  <-- route changed" if r["route_changed"] else "")
            drift = f"{r['drift']:6.2f}" if r["ok"] else "   inf"
            print(f"{fam:<13} {r['magnitude']:>7g} | {drift} {r['overlap']:>6}% "
                  f"{r['bearing']:>6.1f}  {r['route']}{mark}")
        print()

    print("stability envelope (baseline route = magnitude 0):")
    for fam, rows in results.items():
        first_change, first_dead = stability_summary(rows)
        unit = PERTURBATIONS[fam]["unit"]
        last = rows[-1]["magnitude"]
        msg = (f"route stable through {last:g} {unit}" if first_change is None
               else f"route first changes at {first_change:g} {unit}")
        if first_dead is not None:
            msg += f"; NO_MATCH from {first_dead:g} {unit}"
        print(f"  {fam:<13} {msg}")

    # figure: one row per family = drift curve + three spatial snapshots (low / mid / max)
    nf = len(families)
    fig = plt.figure(figsize=(15, 3.1 * nf))
    gs = fig.add_gridspec(nf, 4, width_ratios=[1.6, 1, 1, 1])
    dmax = max([r["drift"] for rows in results.values() for r in rows if r["ok"]] + [1.0])
    norm = Normalize(0.0, dmax)
    for fi, fam in enumerate(families):
        rows = results[fam]
        draw_curve(fig.add_subplot(gs[fi, 0]), rows, args.reject)
        picks = sorted({1, len(rows) // 2, len(rows) - 1} - {0})[:3]
        for ci, ri in enumerate(picks):
            draw_snapshot(fig.add_subplot(gs[fi, 1 + ci]), case, rows[ri], norm)
    fig.suptitle(f"graph-DTW robustness -- {case['name']} -- emission={args.emission} "
                 f"(star = route changed, open circle = NO_MATCH; "
                 f"snapshots: blue = perturbed A, ramp = route drift)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out or f"output/graph_dtw_perturb_{case['name']}_{args.emission}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
