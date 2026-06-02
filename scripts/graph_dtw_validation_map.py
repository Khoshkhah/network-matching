"""
Standalone interactive HTML **validation** map for graph-DTW route matching.

Both networks on one map (Sweden B shifted beside OSM A), with toggleable colour-coded layers so
you can filter matched vs unmatched per network, and inspect the abnormal cases:

  OSM A (true position)
    - A matched            blue
    - A NO_MATCH           gray dashed
    - A under-covered      red          (overlap_pct < --a-cover, off by default)
  Sweden B (shifted)
    - B used (matched)     orange
    - B unused             red dashed
    - B under-used         gold         (aggregated usage < --b-under, off by default)
    - B over-used          purple       (aggregated usage > --b-over -> contention, off by default)

Run:
    python scripts/graph_dtw_validation_map.py
    python scripts/graph_dtw_validation_map.py --offset 0.0006 --a-cover 95 --b-under 50 --b-over 110
"""

import argparse
import os
import sys

from shapely.affinity import translate
from shapely.wkt import loads as load_wkt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import DuckDBMapMatcher, get_logger, setup_logging  # noqa: E402

log = get_logger("scripts.graph_dtw_validation_map")


def _latlon(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[(y, x) for x, y in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[(y, x) for x, y in line.coords] for line in geom.geoms]
    return []


def _fetch(m):
    da = m.conn.execute("SELECT edge_id AS id, ST_AsText(geometry) AS wkt FROM driving_edges").df()
    db = m.conn.execute(
        "SELECT directed_id AS id, name, ST_AsText(geometry) AS wkt FROM vehicle_edges_directed").df()
    a = {r.id: load_wkt(r.wkt) for r in da.itertuples(index=False)}
    b = {r.id: (load_wkt(r.wkt), r.name) for r in db.itertuples(index=False)}
    return a, b


def build_map(m, routes_long, routes_summary, *, offset, a_cover, b_under, b_over, boundary):
    import folium

    a_geom, b_geom = _fetch(m)
    matched = routes_summary[routes_summary["match_type"] != "NO_MATCH"]
    matched_a = set(matched["source_id"])
    a_meta = matched.set_index("source_id")

    used_b = set(routes_long["dest_id"])
    bagg = (routes_long.groupby("dest_id")
            .agg(used=("edge_matched_len", "sum"), blen=("edge_b_len", "first"),
                 n_a=("source_id", "nunique")))
    bagg["pct"] = 100.0 * bagg["used"] / bagg["blen"]
    b_pct, b_na = bagg["pct"].to_dict(), bagg["n_a"].to_dict()

    # Shift network B north-east so it sits beside A.
    b_shift = {bid: (translate(g, xoff=offset, yoff=offset), nm) for bid, (g, nm) in b_geom.items()}

    xs, ys = [], []
    for g in a_geom.values():
        x0, y0, x1, y1 = g.bounds; xs += [x0, x1]; ys += [y0, y1]
    for g, _ in b_shift.values():
        x0, y0, x1, y1 = g.bounds; xs += [x0, x1]; ys += [y0, y1]

    fmap = folium.Map(location=[(min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2],
                      zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(fmap)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite").add_to(fmap)

    fg_a_m = folium.FeatureGroup(name="A matched (blue)", show=True)
    fg_a_nm = folium.FeatureGroup(name="A NO_MATCH (gray)", show=True)
    fg_a_under = folium.FeatureGroup(name=f"A under-covered < {a_cover}% (red)", show=False)
    fg_b_used = folium.FeatureGroup(name="B used (orange)", show=True)
    fg_b_unused = folium.FeatureGroup(name="B unused (red)", show=True)
    fg_b_under = folium.FeatureGroup(name=f"B under-used < {b_under}% (gold)", show=False)
    fg_b_over = folium.FeatureGroup(name=f"B over-used > {b_over}% (purple)", show=False)

    def tip(html):
        return folium.Tooltip(
            f"<div style='font-family:Arial;font-size:12px;white-space:nowrap'>{html}</div>",
            sticky=True)

    def line(paths, fg, color, weight, dash=None, tooltip=None):
        for p in paths:
            folium.PolyLine(p, color=color, weight=weight, opacity=0.9, dash_array=dash,
                            tooltip=tooltip).add_to(fg)

    # --- OSM A (true position) ---
    for eid, g in a_geom.items():
        paths = _latlon(g)
        if eid in matched_a:
            r = a_meta.loc[eid]
            cov = float(r["overlap_pct"]) if r["overlap_pct"] == r["overlap_pct"] else None
            html = (f"<b>OSM A:</b> {eid}<br><b>coverage:</b> {cov:.0f}%<br>"
                    f"<b>route:</b> {r['dest_ids']}<br><b>drift:</b> {r['dtw_distance']:.2f} m")
            line(paths, fg_a_m, "#2563eb", 3, tooltip=tip(html))
            if cov is not None and cov < a_cover:
                line(paths, fg_a_under, "#dc2626", 5,
                     tooltip=tip(f"<b>OSM A {eid} UNDER-COVERED</b><br>coverage {cov:.0f}%"))
        else:
            line(paths, fg_a_nm, "#94a3b8", 2, dash="4,6",
                 tooltip=tip(f"<b>OSM A:</b> {eid}<br><b>NO_MATCH</b>"))

    # --- Sweden B (shifted) ---
    for bid, (g, nm) in b_shift.items():
        paths = _latlon(g)
        if bid in used_b:
            pct, na = b_pct.get(bid, 0.0), b_na.get(bid, 0)
            html = (f"<b>NVDB B:</b> {bid} ({nm})<br><b>used:</b> {pct:.0f}% of edge<br>"
                    f"<b>by:</b> {na} OSM edge(s)")
            line(paths, fg_b_used, "#f59e0b", 3, tooltip=tip(html))
            if pct < b_under:
                line(paths, fg_b_under, "#eab308", 5,
                     tooltip=tip(f"<b>NVDB B {bid} UNDER-USED</b><br>{pct:.0f}% used"))
            if pct > b_over:
                line(paths, fg_b_over, "#7c3aed", 5,
                     tooltip=tip(f"<b>NVDB B {bid} OVER-USED</b><br>{pct:.0f}% used by {na} OSM edges "
                                 f"(contention)"))
        else:
            line(paths, fg_b_unused, "#ef4444", 2, dash="4,6",
                 tooltip=tip(f"<b>NVDB B:</b> {bid} ({nm})<br><b>unused</b>"))

    for fg in (fg_a_m, fg_a_nm, fg_a_under, fg_b_used, fg_b_unused, fg_b_under, fg_b_over):
        fg.add_to(fmap)

    if boundary and os.path.exists(boundary):
        folium.GeoJson(boundary, name="Area boundary",
                       style_function=lambda x: {"color": "#fbbf24", "weight": 2.5,
                                                 "fill": False, "dashArray": "6,4"}).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    n_au = sum(1 for e in matched_a
               if (a_meta.loc[e, "overlap_pct"] == a_meta.loc[e, "overlap_pct"])
               and float(a_meta.loc[e, "overlap_pct"]) < a_cover)
    n_bu = int((bagg["pct"] < b_under).sum())
    n_bo = int((bagg["pct"] > b_over).sum())
    title = (f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:white;"
             f"padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);"
             f"font-family:Arial;font-size:13px'>"
             f"<b>Graph-DTW validation</b> &nbsp; OSM A &harr; NVDB B (shifted)<br>"
             f"A matched {len(matched_a)} / NO_MATCH {(routes_summary['match_type']=='NO_MATCH').sum()}"
             f" &nbsp;|&nbsp; B used {len(used_b)} / unused {len(b_geom)-len(used_b)}<br>"
             f"<span style='color:#dc2626'>A under-covered {n_au}</span> &nbsp; "
             f"<span style='color:#a16207'>B under-used {n_bu}</span> &nbsp; "
             f"<span style='color:#7c3aed'>B over-used {n_bo}</span> "
             f"<span style='color:#888'>(toggle layers, top-right)</span></div>")
    fmap.get_root().html.add_child(folium.Element(title))
    fmap.fit_bounds([[min(ys), min(xs)], [max(ys), max(xs)]])
    return fmap


def main():
    ap = argparse.ArgumentParser(description="Graph-DTW validation HTML map.")
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--boundary", default="data/sundbyberg_boundary.geojson")
    ap.add_argument("--out", default="output/graph_dtw_validation_map.html")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--offset", type=float, default=0.0006,
                    help="degrees to shift network B north-east")
    ap.add_argument("--a-cover", type=float, default=95.0, help="A under-covered threshold (%)")
    ap.add_argument("--b-under", type=float, default=50.0, help="B under-used threshold (%)")
    ap.add_argument("--b-over", type=float, default=110.0, help="B over-used threshold (%)")
    args = ap.parse_args()

    setup_logging()
    m = DuckDBMapMatcher.from_wkt_csv(args.osm, args.sweden, id_a="edge_id", id_b="directed_id",
                                      utm_srid=args.utm_srid, max_distance=args.max_distance,
                                      keep_cols_b=["name"], table_a="driving_edges",
                                      table_b="vehicle_edges_directed")
    log.info("running match_routes...")
    routes_long, routes_summary = m.match_routes(n_jobs=args.n_jobs)
    log.info("building validation map...")
    fmap = build_map(m, routes_long, routes_summary, offset=args.offset, a_cover=args.a_cover,
                     b_under=args.b_under, b_over=args.b_over, boundary=args.boundary)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fmap.save(args.out)
    log.info("saved -> %s", args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
