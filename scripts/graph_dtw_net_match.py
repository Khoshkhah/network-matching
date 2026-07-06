#!/usr/bin/env python
"""Visualize how ONE OSM A-edge map-matches onto the LOCAL B-NETWORK — point vs segment emission.

For a single A-edge it draws, in true position, side by side for each emission mode:
  - the **local B-network** — every candidate B-edge within `max_distance` (grey), i.e. the roads
    the matcher chose *from*;
  - the **matched route** — the connected B-edges the A-edge actually took, thick, coloured
    green→red by per-edge drift;
  - the **A-edge** itself (blue) with its densified points and start(▲)/end(■);
  - **correspondence rungs** A→B coloured by drift.
So you see which A-edge maps to which *path through the network*, how good the match is, and how
the point vs segment cost changes that path.

Usage:
    python scripts/graph_dtw_net_match.py --edge-id 685
    python scripts/graph_dtw_net_match.py            # auto-pick a rich multi-B-edge route
Writes output/graph_dtw_net_match_<edge>.png.
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.wkt import loads as load_wkt
from network_matching import DuckDBMapMatcher, match_edge_to_bgraph, setup_logging

DMAX = 8.0  # drift colour scale saturates here (m); a good match is < ~5 m

def _arrows(ax, P, color, n=3, zorder=6, scale=13):
    """Draw n arrowheads along polyline P showing its travel direction."""
    L = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if L[-1] <= 0:
        return
    for f in np.linspace(0.25, 0.85, n):
        k = int(np.clip(np.searchsorted(L, f * L[-1]), 1, len(P) - 1))
        ax.annotate("", xy=P[k], xytext=P[k - 1],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.5, mutation_scale=scale),
                    zorder=zorder)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-id", type=int, default=None, help="OSM edge_id; omit to auto-pick a rich route")
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--shift-left", type=float, default=0.0,
                    help="translate the A-edge this many metres to the LEFT (-x/west) before matching; "
                         "negative = to the RIGHT (east)")
    ap.add_argument("--bearing-weight", type=float, default=0.0,
                    help="heading-penalty lambda; > 0 adds a third panel segment+bearing")
    ap.add_argument("--snap", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_logging(console=False)

    m = DuckDBMapMatcher.from_wkt_csv(args.osm, args.sweden, id_a="edge_id", id_b="directed_id",
        utm_srid=args.utm_srid, max_distance=args.max_distance, keep_cols_b=["name"])
    eid = args.edge_id
    if eid is None:
        _, rs = m.match_routes(n_jobs=1)
        rs = rs[rs.match_type != "NO_MATCH"].copy()
        rs["nb"] = rs.dest_ids.apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 1)
        eid = int(rs.sort_values("nb", ascending=False).iloc[0].source_id)

    grp = m.generate_candidate_pairs().query("id_a == @eid")
    coords_a = np.array(load_wkt(grp.wkt_a.iloc[0]).coords)
    if args.shift_left:
        coords_a = coords_a - np.array([args.shift_left, 0.0])   # move the A-edge left (-x/west)
    cand = {r.id_b: np.array(load_wkt(r.wkt_b).coords) for r in grp.itertuples()}   # the local network
    be_list = [(r.id_b, load_wkt(r.wkt_b)) for r in grp.itertuples()]

    modes = [("point", 0.0), ("segment", 0.0)]
    if args.bearing_weight > 0:
        modes.append(("segment", args.bearing_weight))
    fig, axes = plt.subplots(1, len(modes), figsize=(9 * len(modes), 9))
    for col, (mode, bw) in enumerate(modes):
        res = match_edge_to_bgraph(coords_a.tolist(), be_list, snap_tolerance_m=args.snap,
                                   step_meters=args.step, emission=mode, bearing_weight=bw)
        wp, M = res["warping_path"], res["metrics"]
        ax = axes[col]
        for P in cand.values():                       # local B-network (candidates; directed twins)
            ax.plot(P[:, 0], P[:, 1], color="0.82", lw=2, alpha=0.95, zorder=1, solid_capstyle="round")
            _arrows(ax, P, "0.65", n=1, zorder=1, scale=9)
        route = [rid for rid, _, _ in M["route"]]
        if wp:
            A = np.array([a for a, _ in wp]); B = np.array([b for _, b in wp])
            drift = np.hypot(A[:, 0] - B[:, 0], A[:, 1] - B[:, 1])
            for re in M["route_edges"]:               # matched route, coloured by per-edge drift
                if re["dest_id"] in cand:
                    P = cand[re["dest_id"]]
                    ax.plot(P[:, 0], P[:, 1], color=plt.cm.RdYlGn_r(min(re["match_dist_avg"] / DMAX, 1.0)),
                            lw=7, alpha=0.95, zorder=2, solid_capstyle="round")
                    _arrows(ax, P, "black", n=2, zorder=6)
            ax.add_collection(LineCollection(np.stack([A, B], 1), array=drift, cmap="RdYlGn_r",
                                             norm=plt.Normalize(0, DMAX), lw=1.0, alpha=0.6, zorder=3))
            ax.plot(coords_a[:, 0], coords_a[:, 1], color="#123a8f", lw=3.5, zorder=4, solid_capstyle="round")
            _arrows(ax, coords_a, "#123a8f", n=4, zorder=6)
            # solid dot = B advances here (covered); hollow = stalled (this A point maps to the
            # same B point as the previous one -> overhang, excluded from coverage)
            stall = np.zeros(len(A), dtype=bool)
            stall[1:] = np.hypot(*(B[1:] - B[:-1]).T) < 0.01   # B did not advance (metres)
            ax.scatter(A[~stall, 0], A[~stall, 1], s=14, color="#123a8f", zorder=5)
            ax.scatter(A[stall, 0], A[stall, 1], s=18, facecolors="white", edgecolors="#123a8f",
                       linewidths=1.2, zorder=5)
            ax.plot(A[0, 0], A[0, 1], "k^", ms=12, zorder=6); ax.plot(A[-1, 0], A[-1, 1], "ks", ms=11, zorder=6)
            label = mode.upper() + (f" + BEARING λ={bw:g}" if bw > 0 else "")
            ax.set_title(f"{label}   ·   A-edge {eid} matched {len(route)} of {len(cand)} local B-edges\n"
                         f"route = {route}   ·   avg drift {M['average']:.2f} m   ·   "
                         f"coverage {M['overlap_pct']}%", fontsize=11)
        else:
            label = mode.upper() + (f" + BEARING λ={bw:g}" if bw > 0 else "")
            ax.set_title(f"{label}: NO MATCH", fontsize=11)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.autoscale()

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, DMAX))
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01).set_label("match drift (m)", fontsize=10)
    sdir = "LEFT" if args.shift_left > 0 else "RIGHT"
    shift_txt = f"   ·   A-edge shifted {sdir} {abs(args.shift_left):.0f} m" if args.shift_left else ""
    fig.suptitle(f"Map-matching A-edge {eid} onto the local B-network{shift_txt}   ·   "
                 f"grey = candidate B-edges (the network) · thick = matched route (green→red by drift) · blue = A-edge · ▲start ■end\n"
                 f"solid A dot = B advances (covered) · hollow A dot = stalled on one B point (overhang, not covered)\n"
                 f"arrows = travel direction (B is a DIRECTED network: every grey road is two overlapping twins, one arrow each way; "
                 f"the route uses the twin agreeing with A)",
                 fontsize=12)
    tag = f"_{sdir.lower()}{abs(args.shift_left):.0f}" if args.shift_left else ""
    if args.bearing_weight > 0:
        tag += f"_bw{args.bearing_weight:g}"
    out = args.out or f"output/graph_dtw_net_match_{eid}{tag}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight"); print("wrote", out)

if __name__ == "__main__":
    main()
