#!/usr/bin/env python
"""Full-detail ALGORITHM debug view of one graph-DTW match (synthetic case or real edge).

Where ``graph_dtw_edge_detail.py`` visualizes the *result*, this renders the *algorithm*: the
local directed B-graph the DP ran on, the accumulated-cost table D and the emission table E with
the backtracked state path overlaid (every move classified START / V=A-advance / H=B-advance /
D=both), the overhang trim window, and the per-step drift profile -- plus an optional stdout
trace of every DP state. Runs on a named synthetic scenario (``--case``, see
``network_matching/synthetic.py``) or a real A-edge (``--edge-id``), with optional perturbations
of the A-edge (noise / shift / rotation / crop / ... ) so a failure can be reproduced and
inspected in one command.

Usage:
    python scripts/graph_dtw_debug_viz.py --list-cases
    python scripts/graph_dtw_debug_viz.py --case split
    python scripts/graph_dtw_debug_viz.py --case parallel_trap --shift -6 --trace
    python scripts/graph_dtw_debug_viz.py --case curve --noise 2 --seed 7 --emission segment
    python scripts/graph_dtw_debug_viz.py --edge-id 1377 --rotate 8

Writes output/graph_dtw_debug_<case|edge>_<emission>.png.
"""
import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import match_edge_to_bgraph, setup_logging  # noqa: E402
from network_matching.synthetic import (SCENARIOS, apply_perturbation,  # noqa: E402
                                        as_array, get_scenario, reverse)

# One sequential ramp (viridis, CVD-safe) carries every magnitude: cost tables, drift, route.
COST_CMAP = plt.cm.viridis
A_COLOR = "#2563eb"          # identity: the A-edge is always this blue
B_COLOR = "#b0b6bf"          # identity: candidate B network
STITCH_COLOR = "#7c3aed"     # junction stitches (end->start connectivity)
# Move types: color AND marker shape, so identity is never color-alone.
MOVES = {"START": ("#111827", "*", "start (free entry)"),
         "V": ("#0ea5e9", "^", "V: A advances"),
         "H": ("#f59e0b", ">", "H: B advances (Dijkstra)"),
         "D": ("#10b981", "d", "D: both advance")}

# Perturbations compose in this order (structural cuts first, then rigid moves, then jitter).
PERTURB_ORDER = ["crop", "stretch", "rotate", "translate", "shift", "longitudinal", "noise"]


# --------------------------------------------------------------------------------------
# Case loading (synthetic scenario or real edge) + perturbation
# --------------------------------------------------------------------------------------
def load_case(args):
    """Return dict(name, coords_a, b_edges, kwargs) in a meter CRS."""
    if args.case:
        sc = get_scenario(args.case)
        kw = dict(sc["defaults"])
        return dict(name=args.case, coords_a=list(sc["coords_a"]),
                    b_edges=list(sc["b_edges"]), kwargs=kw,
                    description=sc["description"])
    from shapely.wkt import loads as load_wkt
    from network_matching import DuckDBMapMatcher
    m = DuckDBMapMatcher.from_wkt_csv(
        args.osm, args.sweden, id_a="edge_id", id_b="directed_id",
        utm_srid=args.utm_srid, max_distance=args.max_distance)
    grp = m.generate_candidate_pairs().query("id_a == @args.edge_id")
    if grp.empty:
        sys.exit(f"No B candidates within {args.max_distance} m of edge {args.edge_id}.")
    coords_a = list(load_wkt(grp.wkt_a.iloc[0]).coords)
    b_edges = [(r.id_b, load_wkt(r.wkt_b)) for r in grp.itertuples()
               if load_wkt(r.wkt_b).geom_type == "LineString"]
    return dict(name=f"edge_{args.edge_id}", coords_a=coords_a, b_edges=b_edges,
                kwargs=dict(snap_tolerance_m=0.5, step_meters=10.0),
                description=f"real A-edge {args.edge_id} + {len(b_edges)} candidate B-edges")


def perturb(coords_a, args):
    """Apply the requested perturbations in PERTURB_ORDER; return (coords, description list)."""
    P = as_array(coords_a)
    applied = []
    for fam in PERTURB_ORDER:
        mag = getattr(args, fam)
        if mag:
            kw = {"bearing": args.translate_bearing} if fam == "translate" else {}
            P = apply_perturbation(P, fam, mag, seed=args.seed, **kw)
            applied.append(f"{fam}={mag:g}"
                           + (f"@{args.translate_bearing:g}deg" if fam == "translate" else ""))
    if args.reverse:
        P = reverse(P)
        applied.append("reverse")
    return P, applied


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------
def _route_colors(route_edges, norm):
    return {re["dest_id"]: COST_CMAP(norm(re["match_dist_avg"])) for re in route_edges}


def draw_spatial(ax, res, coords_a0, coords_a, norm):
    """True-position view: B-graph structure, route, warp links, trim, perturbed vs original A."""
    gb = res["graph"]
    M = res["metrics"]
    dbg = res["debug"]
    per_edge = _route_colors(M["route_edges"], norm)

    # B-edges (by vertex chain, so the drawn geometry is exactly what the DP saw)
    for e, eid in enumerate(gb.edge_ids):
        vids = np.where(gb.vert_edge == e)[0]
        X, Y = gb.vx[vids], gb.vy[vids]
        if eid in per_edge:
            ax.plot(X, Y, color=per_edge[eid], lw=5, zorder=2, solid_capstyle="round")
        ax.plot(X, Y, color=B_COLOR, lw=1.4, zorder=3)
        k = len(X) // 2                                    # direction arrow at the midpoint
        if len(X) >= 2:
            ax.annotate("", xy=(X[k], Y[k]), xytext=(X[k - 1], Y[k - 1]), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.0))
        lx, ly = X[len(X) // 2], Y[len(Y) // 2]
        ax.annotate(str(eid), (lx, ly), textcoords="offset points", xytext=(3, -9),
                    fontsize=7, color="#374151",
                    fontweight="bold" if eid in per_edge else "normal")

    # junction stitches: inter-edge arcs (end -> start of a different edge)
    for u in range(gb.n_vertices):
        for w in gb.succ_arcs[u]:
            if gb.vert_edge[u] != gb.vert_edge[w]:
                ax.plot([gb.vx[u], gb.vx[w]], [gb.vy[u], gb.vy[w]], color=STITCH_COLOR,
                        lw=1.2, ls=":", zorder=4)
                ax.plot(gb.vx[u], gb.vy[u], marker="o", ms=7, mfc="none", mec=STITCH_COLOR,
                        mew=1.2, zorder=4)

    # B vertex types: endpoint / real node / projection
    ep, nd = gb.is_endpoint, gb.is_node
    ax.scatter(gb.vx[ep], gb.vy[ep], marker="D", s=22, c=STITCH_COLOR, zorder=5,
               label="B endpoint (junction)")
    ax.scatter(gb.vx[nd & ~ep], gb.vy[nd & ~ep], marker="o", s=12, facecolors="none",
               edgecolors="#111827", zorder=5, label="B real node")
    ax.scatter(gb.vx[~nd], gb.vy[~nd], marker=".", s=6, c="#9ca3af", zorder=5,
               label="B projection/fill pt")

    # warp links: kept span colored by drift, trimmed overhang gray dashed
    if dbg and "pairs_all" in dbg:
        lo, hi = dbg.get("kept_span", (0, len(dbg["pairs_all"]) - 1))
        a_pool = dbg["a_pool"]
        for t, (i, v) in enumerate(dbg["pairs_all"]):
            pa = a_pool[i][:2]
            pb = (gb.vx[v], gb.vy[v])
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            kept = lo <= t <= hi
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color=COST_CMAP(norm(d)) if kept else "#9ca3af",
                    lw=1.1 if kept else 0.8, ls="-" if kept else "--",
                    alpha=0.9 if kept else 0.6, zorder=6)

    # A-edge: original (if perturbed) dashed, matched copy solid + its DP point pool
    A0, A = as_array(coords_a0), as_array(coords_a)
    if not (A0.shape == A.shape and np.allclose(A0, A)):
        ax.plot(A0[:, 0], A0[:, 1], color=A_COLOR, lw=1.2, ls="--", alpha=0.45, zorder=6,
                label="A original")
    ax.plot(A[:, 0], A[:, 1], color=A_COLOR, lw=2.4, zorder=7, label="A (as matched)")
    ax.plot(A[0, 0], A[0, 1], marker="^", ms=9, color="#111827", zorder=8)
    if dbg and "a_pool" in dbg:
        ap = np.asarray([(p[0], p[1]) for p in dbg["a_pool"]])
        isn = np.asarray([p[2] for p in dbg["a_pool"]], bool)
        ax.scatter(ap[isn, 0], ap[isn, 1], marker="^", s=20, c=A_COLOR, zorder=8,
                   label="A node")
        ax.scatter(ap[~isn, 0], ap[~isn, 1], marker=".", s=10, c=A_COLOR, zorder=8,
                   label="A projection/fill pt")

    ax.set_aspect("equal")
    # pad the thin dimension so a near-collinear case doesn't collapse to a sliver
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(x1 - x0, y1 - y0)
    if (y1 - y0) < 0.3 * span:
        pad = (0.3 * span - (y1 - y0)) / 2
        ax.set_ylim(y0 - pad, y1 + pad)
    if (x1 - x0) < 0.3 * span:
        pad = (0.3 * span - (x1 - x0)) / 2
        ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_title("spatial view -- local B-graph, matched route (ramp = per-edge drift), "
                 "warp links (gray dashed = trimmed overhang)", fontsize=9)
    ax.legend(fontsize=6.5, loc="best", ncol=2, framealpha=0.9)
    ax.tick_params(labelsize=7)


def _state_bands(res):
    """Group the DP state axis by B-edge: returns (band starts, band labels, tail-edge of each
    state, stitch mask). Point mode states are vertices; segment mode states are arcs."""
    gb = res["graph"]
    dbg = res["debug"]
    if dbg["params"]["emission"] == "segment":
        arcs = dbg["arcs"]
        state_edge = np.asarray([gb.vert_edge[u] for (u, _w) in arcs])
        stitch = np.asarray([gb.vert_edge[u] != gb.vert_edge[w] for (u, w) in arcs])
    else:
        state_edge = np.asarray(gb.vert_edge)
        stitch = np.zeros(len(state_edge), bool)
    starts, labels = [], []
    for k in range(len(state_edge)):
        if k == 0 or state_edge[k] != state_edge[k - 1]:
            starts.append(k)
            labels.append(str(gb.edge_ids[state_edge[k]]))
    return starts, labels, state_edge, stitch


def draw_table(ax, res, which, norm_dummy):
    """Heatmap of the DP table (``which``: 'D' accumulated / 'E' emission), states on y grouped
    by B-edge, A on x, with the backtracked path overlaid and moves classified."""
    dbg = res["debug"]
    M = np.array(dbg[which], float)
    seg_mode = dbg["params"]["emission"] == "segment"
    disp = M.copy()
    if which == "D":
        # subtract each row's finite minimum: shows which states are COMPETITIVE at each A step
        rmin = np.nanmin(np.where(np.isfinite(disp), disp, np.nan), axis=1, keepdims=True)
        disp = disp - np.where(np.isfinite(rmin), rmin, 0.0)
    disp = np.ma.masked_invalid(disp)
    vmax = np.percentile(disp.compressed(), 95) if disp.count() else 1.0
    cmap = COST_CMAP.copy()
    cmap.set_bad("#e5e7eb")
    im = ax.imshow(disp.T, origin="lower", aspect="auto", cmap=cmap,
                   norm=Normalize(0, max(vmax, 1e-9)), interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=6)
    lims = (ax.get_xlim(), ax.get_ylim())      # overlays (markers at x<0) must not widen the view

    starts, labels, state_edge, stitch = _state_bands(res)
    route_lbls = {str(re["dest_id"]) for re in res["metrics"]["route_edges"]}
    bounds = starts + [len(state_edge)]
    for s in starts[1:]:
        ax.axhline(s - 0.5, color="white", lw=0.8)
    # y-ticks at band centers = the B-edge each state belongs to (bold if in the route);
    # with many edges label only the route bands + a subsample
    keep_every = max(1, int(np.ceil(len(starts) / 30)))
    ticks, tlabels, tbold = [], [], []
    for bi, lab in enumerate(labels):
        if len(starts) <= 30 or lab in route_lbls or bi % keep_every == 0:
            ticks.append((bounds[bi] + bounds[bi + 1] - 1) / 2)
            tlabels.append(lab)
            tbold.append(lab in route_lbls)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tlabels, fontsize=6.5)
    for t, bold in zip(ax.get_yticklabels(), tbold):
        if bold:
            t.set_fontweight("bold")
    for k in np.where(stitch)[0]:                       # segment mode: mark stitch (pass-only) rows
        ax.plot(-0.7, k, marker="x", ms=4, color=STITCH_COLOR, clip_on=False)

    path = dbg.get("arc_path" if seg_mode else "path")
    if path:
        xs = [i for (i, _s, _m) in path]
        ys = [s for (_i, s, _m) in path]
        ax.plot(xs, ys, color="white", lw=2.6, alpha=0.85, zorder=3)
        for mv, (col, mk, _lab) in MOVES.items():
            pts = [(i, s) for (i, s, m) in path if m == mv]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=26,
                           c=col, edgecolors="white", linewidths=0.4, zorder=4)
        if "kept_span" in dbg and not seg_mode:          # trim window on the A axis
            lo, hi = dbg["kept_span"]
            pairs = dbg["pairs_all"]
            ax.axvspan(-0.5, pairs[lo][0] - 0.5, color="0.3", alpha=0.12)
            ax.axvspan(pairs[hi][0] + 0.5, M.shape[0] - 0.5, color="0.3", alpha=0.12)
    if "terminal_state" in dbg:
        ax.plot(M.shape[0] - 1, dbg["terminal_state"], marker="*", ms=13, color="#dc2626",
                mec="white", zorder=5)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])

    xl = "A segment index" if seg_mode else "A pool-point index"
    yl = ("B arc states, banded by owning edge" if seg_mode
          else "B vertex states, banded by owning edge")
    ttl = {"D": "accumulated cost D (per A-step, above that step's best -- 0 = the frontier)",
           "E": "emission E (local cost of pairing each state with each A-step)"}[which]
    ax.set_xlabel(xl, fontsize=7)
    ax.set_ylabel(yl, fontsize=7)
    ax.set_title(ttl + "  -- white path = backtracked alignment, red star = terminal argmin",
                 fontsize=9)
    ax.tick_params(labelsize=7)


def draw_drift(ax, res, norm):
    """Per-step drift along the alignment, colored by the B-edge each step matched."""
    dbg = res["debug"]
    M = res["metrics"]
    if not dbg or "drift_all" not in dbg:
        ax.text(0.5, 0.5, "no drift profile (match failed before backtrack)",
                ha="center", va="center", fontsize=9, color="#6b7280")
        ax.set_axis_off()
        return
    gb = res["graph"]
    drift = dbg["drift_all"]
    pairs = dbg["pairs_all"]
    lo, hi = dbg.get("kept_span", (0, len(pairs) - 1))
    per_edge = _route_colors(M["route_edges"], norm)
    xs = np.arange(len(drift))
    ax.plot(xs, drift, color="#9ca3af", lw=0.8, zorder=1)
    for t in range(len(drift)):
        eid = gb.edge_ids[gb.vert_edge[pairs[t][1]]]
        kept = lo <= t <= hi
        ax.plot(t, drift[t], marker="o", ms=4 if kept else 3,
                color=per_edge.get(eid, "#6b7280") if kept else "#d1d5db", zorder=2)
    ytop = max(max(drift), 0.1) * 1.05
    for t in range(1, len(pairs)):                       # edge-transition boundaries
        e0, e1 = gb.vert_edge[pairs[t - 1][1]], gb.vert_edge[pairs[t][1]]
        if e0 != e1:
            ax.axvline(t - 0.5, color="#6b7280", lw=0.7, ls=":")
            ax.annotate(str(gb.edge_ids[e1]), (t - 0.4, ytop), fontsize=6,
                        va="top", color="#374151", rotation=90)
    if np.isfinite(M["average"]):
        ax.axhline(M["average"], color="#111827", lw=1.0, ls="--")
        ax.annotate(f"avg {M['average']:.2f} m", (0, M["average"]), fontsize=7,
                    textcoords="offset points", xytext=(2, 3))
    if lo > 0:
        ax.axvspan(-0.5, lo - 0.5, color="0.3", alpha=0.12)
    if hi < len(drift) - 1:
        ax.axvspan(hi + 0.5, len(drift) - 0.5, color="0.3", alpha=0.12)
    ax.set_xlabel("alignment step", fontsize=7)
    ax.set_ylabel("drift (m)", fontsize=7)
    ax.set_title("per-step drift along the warping path (dot color = matched B-edge; "
                 "shaded = trimmed overhang)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)


def draw_info(ax, case, applied, res, kwargs):
    """Monospaced metrics + DP statistics panel."""
    M = res["metrics"]
    dbg = res["debug"]
    gb = res["graph"]
    lines = [f"case: {case['name']}", f"perturbation: {', '.join(applied) or 'none'}",
             f"params: emission={kwargs.get('emission', 'point')} "
             f"snap={kwargs.get('snap_tolerance_m')} step={kwargs.get('step_meters')} "
             f"bearing_w={kwargs.get('bearing_weight', 0)}", ""]
    if np.isfinite(M["average"]):
        lines += [f"route   : {[re['dest_id'] for re in M['route_edges']]}",
                  f"drift   : avg {M['average']:.2f}  max {M['max']:.2f}  min {M['min']:.2f} m",
                  f"overlap : {M['overlap_pct']}%   matched_len {M['matched_len']:.1f} m   "
                  f"bearing_diff {M['bearing_diff']:.1f} deg", "",
                  "seq  edge          avg    max  coverA  usesB  bearD  pts"]
        for re in M["route_edges"]:
            lines.append(f"{re['seq']:>3}  {str(re['dest_id']):<12} {re['match_dist_avg']:>5.2f}"
                         f"  {re['match_dist_max']:>5.2f}  {re['cover_pct']:>5.1f}%"
                         f"  {re['b_cover_pct']:>4.1f}%  {re['bearing_diff']:>5.1f}"
                         f"  {re['n_points']:>3}")
    else:
        lines += ["NO_MATCH", f"reason: {dbg.get('reason', '?') if dbg else '?'}"]
    lines.append("")
    if dbg and "D" in dbg:
        D = np.array(dbg["D"])
        path = dbg.get("arc_path") or dbg.get("path") or []
        counts = Counter(m for (_i, _s, m) in path)
        lines += ["-- DP statistics --",
                  f"table   : {D.shape[0]} A-steps x {D.shape[1]} states "
                  f"({'arcs' if dbg['params']['emission'] == 'segment' else 'vertices'}; "
                  f"{gb.n_vertices} vertices, {len(gb.edge_ids)} B-edges)",
                  f"final cost: {dbg.get('final_cost', float('nan')):.2f}   "
                  f"terminal state: {dbg.get('terminal_state', '-')}",
                  f"path    : {len(path)} states  "
                  f"(V={counts.get('V', 0)} H={counts.get('H', 0)} D={counts.get('D', 0)})"]
        if "kept_span" in dbg:
            lo, hi = dbg["kept_span"]
            lines.append(f"trim    : kept alignment steps {lo}..{hi} "
                         f"of 0..{len(dbg['pairs_all']) - 1}")
    lines += ["", "moves:"] + [f"  {mk} {lab}" for (_c, mk, lab) in MOVES.values()]
    ax.text(0.01, 0.99, "\n".join(lines), family="monospace", fontsize=7.6,
            va="top", ha="left", transform=ax.transAxes)
    ax.set_axis_off()


def print_trace(res):
    """Stdout table: one row per backtracked DP state, with emission/accumulated cost."""
    dbg = res["debug"]
    gb = res["graph"]
    seg = dbg["params"]["emission"] == "segment"
    path = dbg.get("arc_path" if seg else "path") or []
    D, E = np.array(dbg["D"]), np.array(dbg["E"])
    print(f"\n{'t':>4} {'move':>5} {'A':>4} {'state':>6}  {'on edge':<14} "
          f"{'emission':>8} {'accum D':>9}  detail")
    for t, (i, s, mv) in enumerate(path):
        if seg:
            u, w = dbg["arcs"][s]
            eid = gb.edge_ids[gb.vert_edge[u]]
            detail = (f"arc v{u}->v{w}" + ("  [stitch]" if not dbg["ridable"][s] else ""))
        else:
            eid = gb.edge_ids[gb.vert_edge[s]]
            kind = ("endpoint" if gb.is_endpoint[s] else
                    "node" if gb.is_node[s] else "projection")
            detail = f"v{s} ({kind})"
        print(f"{t:>4} {mv:>5} {i:>4} {s:>6}  {str(eid):<14} "
              f"{E[i][s]:>8.2f} {D[i][s]:>9.2f}  {detail}")
    if "kept_span" in dbg:
        lo, hi = dbg["kept_span"]
        print(f"kept alignment steps {lo}..{hi} of 0..{len(dbg['pairs_all']) - 1} "
              f"(outside = trimmed overhang)")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Graph-DTW algorithm debug visualization.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--case", choices=sorted(SCENARIOS), help="synthetic scenario")
    src.add_argument("--edge-id", type=int, help="real OSM A-edge (uses --osm/--sweden data)")
    ap.add_argument("--list-cases", action="store_true", help="list scenarios and exit")
    # perturbations (composable; applied crop -> stretch -> rotate -> shift -> long. -> noise)
    ap.add_argument("--shift", type=float, default=0,
                    help="lateral shift m along the road's local normal (+ = left of travel)")
    ap.add_argument("--longitudinal", type=float, default=0,
                    help="lengthwise slide m along the road's local tangent")
    ap.add_argument("--translate", type=float, default=0,
                    help="rigid translation m in an ABSOLUTE compass direction "
                         "(see --translate-bearing)")
    ap.add_argument("--translate-bearing", type=float, default=90.0,
                    help="translate direction deg: 0=north, 90=east (default), 180=south, "
                         "270=west")
    ap.add_argument("--noise", type=float, default=0, help="Gaussian jitter sigma m")
    ap.add_argument("--rotate", type=float, default=0, help="rotation deg about centroid")
    ap.add_argument("--crop", type=float, default=0, help="remove this %% of length (ends)")
    ap.add_argument("--stretch", type=float, default=0, help="extend both ends m")
    ap.add_argument("--reverse", action="store_true", help="reverse A's digitized direction")
    ap.add_argument("--seed", type=int, default=0)
    # algorithm parameters
    ap.add_argument("--emission", choices=["point", "segment"], default="point")
    ap.add_argument("--bearing-weight", type=float, default=0.0)
    ap.add_argument("--snap", type=float, default=None, help="snap tolerance m")
    ap.add_argument("--step", type=float, default=None, help="gap-fill step m")
    ap.add_argument("--trim-ends", type=float, default=0.0)
    # real-data source
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--trace", action="store_true", help="print the per-state DP trace")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.list_cases:
        for name in sorted(SCENARIOS):
            print(f"{name:<15} {SCENARIOS[name]['description']}")
        return
    if not args.case and args.edge_id is None:
        ap.error("pick a --case or an --edge-id (--list-cases shows the scenarios)")
    setup_logging(console=False)

    case = load_case(args)
    kwargs = dict(case["kwargs"])
    if args.snap is not None:
        kwargs["snap_tolerance_m"] = args.snap
    if args.step is not None:
        kwargs["step_meters"] = args.step
    kwargs.update(emission=args.emission, bearing_weight=args.bearing_weight,
                  trim_ends_m=args.trim_ends)

    coords_a, applied = perturb(case["coords_a"], args)
    res = match_edge_to_bgraph(coords_a.tolist(), case["b_edges"], debug=True, **kwargs)
    M = res["metrics"]
    print(f"[{case['name']}] perturbation: {', '.join(applied) or 'none'}")
    print(f"route={[re['dest_id'] for re in M['route_edges']]} "
          f"drift={res['avg_distance']:.2f} m overlap={M['overlap_pct']}% "
          f"bearing_diff={M['bearing_diff']:.1f}")
    if args.trace and res["debug"] and "D" in res["debug"]:
        print_trace(res)

    # shared drift normalization: route drift + warp drift on one ramp
    dmax = max([re["match_dist_avg"] for re in M["route_edges"]]
               + ([np.percentile(res['debug']['drift_all'], 95)]
                  if res["debug"] and "drift_all" in res["debug"] else []) + [1.0])
    norm = Normalize(0.0, float(dmax))

    have_tables = res["debug"] is not None and "D" in res["debug"]
    fig = plt.figure(figsize=(15, 15 if have_tables else 8))
    gs = fig.add_gridspec(4 if have_tables else 2, 5,
                          height_ratios=[2.1, 1.5, 1.5, 0.9] if have_tables else [2.1, 0.9])
    draw_spatial(fig.add_subplot(gs[0, :3]), res, case["coords_a"], coords_a, norm)
    draw_info(fig.add_subplot(gs[0, 3:]), case, applied, res, kwargs)
    if have_tables:
        draw_table(fig.add_subplot(gs[1, :]), res, "D", norm)
        draw_table(fig.add_subplot(gs[2, :]), res, "E", norm)
    draw_drift(fig.add_subplot(gs[-1, :]), res, norm)
    fig.suptitle(f"graph-DTW algorithm debug -- {case['name']} "
                 f"({', '.join(applied) or 'unperturbed'}) -- emission={args.emission}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (f"output/graph_dtw_debug_{case['name']}_{args.emission}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
