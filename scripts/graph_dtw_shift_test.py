#!/usr/bin/env python
"""Shift-robustness experiment for graph-DTW, comparing POINT vs SEGMENT emission.

Takes one A-edge, moves it perpendicular (laterally) off its road in steps, and re-matches each
shifted copy against the SAME fixed local B-network (candidates within --pool metres of the
original edge) under BOTH emission costs. Reports route + drift per shift for each, renders a
2-row filmstrip (point / segment) of the shifted edge over the network, and a drift-vs-shift curve
overlaying both emissions.

Usage:
    python scripts/graph_dtw_shift_test.py --edge-id 1377
    python scripts/graph_dtw_shift_test.py                     # auto-pick a clean, long, 1:1 edge
    python scripts/graph_dtw_shift_test.py --max-shift 40 --step 8 --pool 60
Writes output/graph_dtw_shift_test_<edge>.png.
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.wkt import loads as load_wkt
from network_matching import DuckDBMapMatcher, match_edge_to_bgraph, setup_logging

MODES = ["point", "segment"]

def shift_poly(P, s):                         # move polyline sideways by s along its local normal
    T = np.zeros_like(P); T[1:-1] = P[2:] - P[:-2]; T[0] = P[1] - P[0]; T[-1] = P[-1] - P[-2]
    L = np.hypot(T[:, 0], T[:, 1]); L[L == 0] = 1
    return P + np.stack([-T[:, 1] / L, T[:, 0] / L], 1) * s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-id", type=int, default=None)
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--pool", type=float, default=60.0, help="fixed local-network radius (m)")
    ap.add_argument("--max-shift", type=float, default=40.0)
    ap.add_argument("--step", type=float, default=8.0)
    ap.add_argument("--reject", type=float, default=5.0, help="drift line drawn as ~reject threshold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_logging(console=False)

    m = DuckDBMapMatcher.from_wkt_csv(args.osm, args.sweden, id_a="edge_id", id_b="directed_id",
        utm_srid=args.utm_srid, max_distance=args.pool, keep_cols_b=["name"])
    cand_df = m.generate_candidate_pairs()
    eid = args.edge_id
    if eid is None:                           # auto-pick a clean, long, single-road match
        _, rs = m.match_routes(n_jobs=1)
        lens = cand_df.drop_duplicates("id_a").set_index("id_a").wkt_a.apply(lambda w: load_wkt(w).length)
        rs = rs[rs.match_type != "NO_MATCH"].copy()
        rs["len"] = rs.source_id.map(lens); rs["nb"] = rs.dest_ids.apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 1)
        ok = rs[(rs.dtw_distance < 2.0) & (rs.nb == 1) & (rs.len > 45)]
        eid = int((ok if len(ok) else rs).sort_values("len" if len(ok) else "dtw_distance", ascending=len(ok) == 0).iloc[0].source_id)
    print("edge", eid)

    grp = cand_df.query("id_a == @eid")
    coords_a = np.array(load_wkt(grp.wkt_a.iloc[0]).coords)
    b_edges = [(r.id_b, load_wkt(r.wkt_b)) for r in grp.itertuples()]
    b_geom = {r.id_b: np.array(load_wkt(r.wkt_b).coords) for r in grp.itertuples()}
    shifts = list(np.arange(0, args.max_shift + 1e-9, args.step))

    data = {}                                 # (mode, shift) -> result
    for mode in MODES:
        for s in shifts:
            M = match_edge_to_bgraph(shift_poly(coords_a, s).tolist(), b_edges,
                                     snap_tolerance_m=0.5, step_meters=10.0, emission=mode)["metrics"]
            data[(mode, s)] = dict(A=shift_poly(coords_a, s), route=[rid for rid, _, _ in M["route"]],
                                   drift=M["average"], re=M["route_edges"])
    print(f"{'shift':>6} | {'POINT route / drift':40} | SEGMENT route / drift")
    for s in shifts:
        p, g = data[("point", s)], data[("segment", s)]
        tag = "   <-- differ" if p["route"] != g["route"] else ""
        print(f"{s:4.0f} m | {str(p['route'])+' / '+format(p['drift'],'.1f'):40} | {g['route']} / {g['drift']:.1f}{tag}")

    rids = set().union(*[set(v["route"]) for v in data.values()])
    fx = np.concatenate([b_geom[i][:, 0] for i in rids if i in b_geom] + [v["A"][:, 0] for v in data.values()])
    fy = np.concatenate([b_geom[i][:, 1] for i in rids if i in b_geom] + [v["A"][:, 1] for v in data.values()])
    mg = 0.12 * max(np.ptp(fx), np.ptp(fy), 10)
    xlim = (fx.min() - mg, fx.max() + mg); ylim = (fy.min() - mg, fy.max() + mg)
    DMAX = max(args.max_shift, 1.0); nc = len(shifts)

    fig = plt.figure(figsize=(3.5 * nc, 3.2 * 3)); gs = fig.add_gridspec(3, nc)
    for i, mode in enumerate(MODES):
        base = data[(mode, 0)]["route"]
        for j, s in enumerate(shifts):
            ax = fig.add_subplot(gs[i, j]); r = data[(mode, s)]
            for P in b_geom.values():
                ax.plot(P[:, 0], P[:, 1], color="0.83", lw=1.5, zorder=1)
            for re in r["re"]:
                if re["dest_id"] in b_geom:
                    P = b_geom[re["dest_id"]]
                    ax.plot(P[:, 0], P[:, 1], color=plt.cm.RdYlGn_r(min(re["match_dist_avg"] / DMAX, 1)), lw=5, zorder=2)
            ax.plot(r["A"][:, 0], r["A"][:, 1], color="#123a8f", lw=2.5, zorder=3)
            ax.plot(r["A"][0, 0], r["A"][0, 1], "k^", ms=9, zorder=4)
            j2 = "  JUMPED" if r["route"] != base else ""
            ax.set_title(f"{mode.upper()} · shift {s:.0f} m · drift {r['drift']:.1f} m{j2}", fontsize=8.5,
                         color="crimson" if j2 else "black")
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    axc = fig.add_subplot(gs[2, :])
    for mode, style in zip(MODES, ["-o", "--s"]):
        axc.plot(shifts, [data[(mode, s)]["drift"] for s in shifts], style, label=f"{mode} drift")
        base = data[(mode, 0)]["route"]
        for s in shifts:
            if data[(mode, s)]["route"] != base:
                axc.plot(s, data[(mode, s)]["drift"], "*", color="crimson", ms=12)
    axc.plot([0, max(shifts)], [0, max(shifts)], ":", color="0.6", lw=1, label="drift = shift (ideal lateral)")
    axc.axhline(args.reject, color="green", ls=":", lw=1)
    axc.annotate(f"~reject threshold ({args.reject:.0f} m)", (0, args.reject + 0.4), color="green", fontsize=8)
    axc.set_xlabel("perpendicular shift (m)"); axc.set_ylabel("avg match drift (m)")
    axc.legend(fontsize=8, ncol=2); axc.grid(alpha=0.3)
    axc.set_title("match drift vs shift — point vs segment (★ = route jumped)", fontsize=10)

    fig.suptitle(f"Shift-robustness of graph-DTW — A-edge {eid} moved perpendicular off its road · POINT vs SEGMENT\n"
                 f"grey = local B-network · thick = matched route (green→red by drift) · blue = shifted A-edge", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or f"output/graph_dtw_shift_test_{eid}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=105); print("wrote", out)

if __name__ == "__main__":
    main()
