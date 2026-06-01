"""
Standalone interactive HTML map for graph-DTW (route-based) matching: each OSM A-edge and the
connected Sweden-NVDB B-route it maps to, coloured by alignment drift.

Run:
    python scripts/graph_dtw_map.py
    python scripts/graph_dtw_map.py --osm data/osm_edges.csv --sweden data/sweden_edges.csv \
        --out output/graph_dtw_map.html --max-distance 30 --snap 0.5 --n-jobs -1

Produces a self-contained ``.html`` (open in any browser). Layers (toggle top-right):
  - OSM A (matched)        thin blue
  - OSM A (NO_MATCH)       gray dashed
  - B route (by drift)     thick, green->orange->red by drift  [main layer + legend]
  - A -> B links           thin orange connectors (off by default)
"""

import argparse
import os
import sys

import pandas as pd
from shapely.wkt import loads as load_wkt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import DuckDBMapMatcher, setup_logging, get_logger  # noqa: E402

log = get_logger("scripts.graph_dtw_map")


# ----------------------------------------------------------------------------------------
def load_matcher(osm_csv, sweden_csv, utm_srid, max_distance):
    # One-call initializer; carry `name` through for the B-edge tooltips.
    return DuckDBMapMatcher.from_wkt_csv(
        osm_csv, sweden_csv, id_a="edge_id", id_b="directed_id",
        utm_srid=utm_srid, max_distance=max_distance, keep_cols_b=["name"],
        table_a="driving_edges", table_b="vehicle_edges_directed")


def _fetch_geoms(matcher):
    """Return {edge_id: geom4326} for A and {directed_id: (geom4326, name)} for B."""
    da = matcher.conn.execute(
        "SELECT edge_id AS id, ST_AsText(geometry) AS wkt FROM driving_edges"
    ).df()
    db = matcher.conn.execute(
        "SELECT directed_id AS id, name, ST_AsText(geometry) AS wkt FROM vehicle_edges_directed"
    ).df()
    a_geom = {r.id: load_wkt(r.wkt) for r in da.itertuples(index=False)}
    b_geom = {r.id: (load_wkt(r.wkt), r.name) for r in db.itertuples(index=False)}
    return a_geom, b_geom


def _latlon_paths(geom):
    """Shapely (lon, lat) geometry -> list of [(lat, lon), ...] paths for folium."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[(y, x) for x, y in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[(y, x) for x, y in line.coords] for line in geom.geoms]
    return []


def build_map(matcher, routes_long, routes_summary, boundary_path=None, offset_deg=0.00035):
    """Both networks on one map: OSM A in its true position, Sweden B **shifted** by
    ``offset_deg`` (north-east), and a **match link** per A->B route-edge carrying the matching
    info (drift, route, seq), coloured by alignment drift."""
    import os as _os

    import folium
    from branca.colormap import LinearColormap
    from shapely.affinity import translate

    a_geom, b_geom = _fetch_geoms(matcher)            # {id: geom} ; {id: (geom, name)}
    summ = routes_summary.set_index("source_id")
    matched = routes_summary[routes_summary["match_type"] != "NO_MATCH"]

    # Drift colour scale (clip at p95 so outliers don't wash it out).
    drifts = matched["dtw_distance"].dropna()
    vmax = max(float(drifts.quantile(0.95)) if len(drifts) else 10.0, 1.0)
    cmap = LinearColormap(["#2ca25f", "#fec44f", "#de2d26"], vmin=0.0, vmax=vmax)
    cmap.caption = ("Match-link colour = average match distance A↔route (m), i.e. how far apart "
                    "the OSM edge and NVDB route lie:  GREEN = close/good  →  RED = far/poor (review)")

    # Shift network B so it sits beside A instead of overlapping it.
    b_shift = {bid: (translate(g, xoff=offset_deg, yoff=offset_deg), nm)
               for bid, (g, nm) in b_geom.items()}

    # Bounds across BOTH networks (so the shifted B is in view).
    xs, ys = [], []
    for g in a_geom.values():
        minx, miny, maxx, maxy = g.bounds
        xs += [minx, maxx]; ys += [miny, maxy]
    for g, _nm in b_shift.values():
        minx, miny, maxx, maxy = g.bounds
        xs += [minx, maxx]; ys += [miny, maxy]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    # Multiple selectable base layers.
    m = folium.Map(location=[cy, cx], zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light (CartoDB)").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark (CartoDB)").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (Esri)").add_to(m)

    fg_a = folium.FeatureGroup(name="🟦 OSM network A (blue=matched)", show=True)
    fg_b = folium.FeatureGroup(name="🟧 Sweden network B, shifted (orange=matched)", show=True)
    fg_links = folium.FeatureGroup(name="🔗 Match links A→B (by drift)", show=True)

    def tip(html):
        return folium.Tooltip(
            f"<div style='font-family:Arial;font-size:12px;white-space:nowrap'>{html}</div>",
            sticky=True)

    route_dests = (routes_long.sort_values(["source_id", "seq"])
                   .groupby("source_id")["dest_id"].apply(list).to_dict())
    matched_ids = set(matched["source_id"])
    matched_b = set(routes_long["dest_id"])

    # --- Network A (OSM), true position ---
    for eid, g in a_geom.items():
        on = eid in matched_ids
        if on:
            row = summ.loc[eid]
            color, dash, w = "#2563eb", None, 3
            html = (f"<b>OSM A:</b> {eid}<br>"
                    f"<b>→ B route:</b> {route_dests.get(eid, [])}<br>"
                    f"<b>edges:</b> {int(row['n_edges'])} ({row['match_type']})<br>"
                    f"<b>match dist avg:</b> {row['dtw_distance']:.2f} m &nbsp; "
                    f"<b>max:</b> {row['max_dtw_distance']:.2f} &nbsp; "
                    f"<b>min:</b> {row['min_dtw_distance']:.2f} m<br>"
                    f"<b>bearing &Delta;:</b> {row['bearing_diff']:.1f}&deg; &nbsp; "
                    f"<b>overlap:</b> {int(row['overlap_pct'])}% &nbsp; "
                    f"<b>matched_len:</b> {row['matched_len']:.0f} m")
        else:
            color, dash, w = "#64748b", "4,6", 2
            html = f"<b>OSM A:</b> {eid}<br><b>NO_MATCH</b>"
        for p in _latlon_paths(g):
            folium.PolyLine(p, color=color, weight=w, opacity=0.85, dash_array=dash,
                            tooltip=tip(html)).add_to(fg_a)

    # --- Network B (Sweden), shifted ---
    for bid, (g, nm) in b_shift.items():
        on = bid in matched_b
        color, dash, w = ("#f59e0b", None, 3) if on else ("#ef4444", "4,6", 2)
        html = (f"<b>Sweden B:</b> {bid}<br><b>Name:</b> {nm}<br>"
                f"<b>Status:</b> {'matched' if on else 'NO_MATCH'}")
        for p in _latlon_paths(g):
            folium.PolyLine(p, color=color, weight=w, opacity=0.85, dash_array=dash,
                            tooltip=tip(html)).add_to(fg_b)

    # --- Match links: A -> each B-edge of its route (shifted), with matching info ---
    for eid in matched_ids:
        ga = a_geom.get(eid)
        if ga is None:
            continue
        row = summ.loc[eid]
        color = cmap(float(row["dtw_distance"])) if pd.notna(row["dtw_distance"]) else "#888"
        pa = ga.interpolate(0.5, normalized=True)
        dests = route_dests.get(eid, [])
        for k, did in enumerate(dests):
            gb_nm = b_shift.get(did)
            if gb_nm is None:
                continue
            gb, nm = gb_nm
            pb = gb.interpolate(0.5, normalized=True)
            html = (f"<b>OSM {eid}</b> &harr; <b>B {did}</b> "
                    f"(seq {k + 1}/{len(dests)})<br>"
                    f"<b>name:</b> {nm}<br>"
                    f"<b>route:</b> {dests} ({row['match_type']})<br>"
                    f"<b>match dist avg:</b> {row['dtw_distance']:.2f} m &nbsp; "
                    f"<b>max:</b> {row['max_dtw_distance']:.2f} &nbsp; "
                    f"<b>min:</b> {row['min_dtw_distance']:.2f} m<br>"
                    f"<b>bearing &Delta;:</b> {row['bearing_diff']:.1f}&deg; &nbsp; "
                    f"<b>overlap:</b> {int(row['overlap_pct'])}% &nbsp; "
                    f"<b>matched_len:</b> {row['matched_len']:.0f} m")
            folium.PolyLine([(pa.y, pa.x), (pb.y, pb.x)], color=color, weight=2.5,
                            opacity=0.8, tooltip=tip(html)).add_to(fg_links)

    for fg in (fg_a, fg_b, fg_links):
        fg.add_to(m)

    # Area boundary outline.
    if boundary_path and _os.path.exists(boundary_path):
        folium.GeoJson(
            boundary_path, name="🟨 Area boundary",
            style_function=lambda x: {"color": "#fbbf24", "weight": 2.5,
                                      "fill": False, "dashArray": "6, 4"},
        ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    n_m = len(matched_ids)
    n_multi = int((matched["match_type"] == "1:N_ROUTE").sum())
    title = (f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:white;"
             f"padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);"
             f"font-family:Arial;font-size:13px'>"
             f"<b>Graph-DTW route matching</b> &nbsp; OSM → Sweden NVDB (B shifted)<br>"
             f"{n_m} matched A-edges ({n_multi} multi-edge routes), "
             f"{(routes_summary['match_type']=='NO_MATCH').sum()} NO_MATCH<br>"
             f"<span style='color:#555'>Link colour = match distance (avg gap A↔route): "
             f"<b style='color:#2ca25f'>green = close/good</b> → "
             f"<b style='color:#de2d26'>red = far/poor (review)</b></span></div>")
    m.get_root().html.add_child(folium.Element(title))

    m.fit_bounds([[min(ys), min(xs)], [max(ys), max(xs)]])
    return m


def main():
    ap = argparse.ArgumentParser(description="Standalone graph-DTW route-matching HTML map.")
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--boundary", default="data/sundbyberg_boundary.geojson")
    ap.add_argument("--out", default="output/graph_dtw_map.html")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--snap", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--offset", type=float, default=0.00035,
                    help="degrees to shift network B (north-east) so it sits beside A")
    args = ap.parse_args()

    setup_logging()
    log.info("loading data: %s, %s", args.osm, args.sweden)
    m = load_matcher(args.osm, args.sweden, args.utm_srid, args.max_distance)

    log.info("running graph-DTW match_routes...")
    routes_long, routes_summary = m.match_routes(
        snap_tolerance_m=args.snap, step_meters=args.step, n_jobs=args.n_jobs)

    log.info("building folium map...")
    fmap = build_map(m, routes_long, routes_summary,
                     boundary_path=args.boundary, offset_deg=args.offset)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fmap.save(args.out)
    log.info("saved interactive map -> %s", args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
