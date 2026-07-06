"""
Detailed standalone HTML for ONE OSM A-edge: extract its local B-subgraph, run graph-DTW, and
visualize the full matching result on an interactive map.

Run:
    python scripts/graph_dtw_edge_detail.py --edge-id 3
    python scripts/graph_dtw_edge_detail.py --edge-id 3 --max-distance 30 --snap 0.5 \
        --out output/graph_dtw_edge_3.html

Shows (all in true position; matching runs in UTM, drawn in lat/lon):
  - the A-edge (blue) and its densified points,
  - the whole local B-subgraph (gray = candidate but unused; coloured = in the route),
  - the matched route coloured by **per-edge** match distance (green->red),
  - **every point match** link: crimson = matched to a B node (point-to-point),
    orange = matched to a projection point (point-to-projection),
  - an info panel with the overall metrics and the **per-route-edge breakdown** table.
"""

import argparse
import os
import sys

from shapely.wkt import loads as load_wkt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import (DuckDBMapMatcher, get_logger,  # noqa: E402
                              match_edge_to_bgraph, setup_logging)

log = get_logger("scripts.graph_dtw_edge_detail")


def load_matcher(osm_csv, sweden_csv, utm_srid, max_distance):
    # One-call initializer; carry `name` through for the B-edge popups.
    return DuckDBMapMatcher.from_wkt_csv(
        osm_csv, sweden_csv, id_a="edge_id", id_b="directed_id",
        utm_srid=utm_srid, max_distance=max_distance, keep_cols_b=["name"],
        table_a="driving_edges", table_b="vehicle_edges_directed")


def _latlon(geom):
    """Shapely lon/lat geometry -> list of [(lat, lon), ...] paths."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[(y, x) for x, y in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[(y, x) for x, y in line.coords] for line in geom.geoms]
    return []


def _utm_to_latlon(conn, pts, srid):
    """Transform UTM ``(x, y)`` points back to ``(lat, lon)`` using the SAME engine (DuckDB
    ST_Transform) that produced the UTM coordinates, so the round-trip is exact (pyproj's
    axis-order convention does not match DuckDB's, which puts points in the wrong place)."""
    import pandas as pd
    if not pts:
        return []
    df = pd.DataFrame(pts, columns=["x", "y"])
    df["_i"] = range(len(df))
    conn.register("pts_tmp", df)
    rows = conn.execute(
        f"SELECT _i, ST_Y(g) AS lat, ST_X(g) AS lon FROM ("
        f"  SELECT _i, ST_Transform(ST_Point(x, y), 'EPSG:{srid}', 'EPSG:4326') AS g FROM pts_tmp"
        f") ORDER BY _i"
    ).fetchall()
    conn.unregister("pts_tmp")
    return [(r[1], r[2]) for r in rows]


def build_detail_map(res, a4326, b4326, edge_id, conn, srid, a_shift=0.00012):
    import math

    import folium
    from branca.colormap import LinearColormap

    gb = res["graph"]
    M = res["metrics"]
    route_edges = M["route_edges"]
    route_ids = [re["dest_id"] for re in route_edges]
    per_edge = {re["dest_id"]: re for re in route_edges}
    # re["cover_pct"] is the % of the WHOLE A-edge this B-edge covers (these sum to overlap_pct,
    # NOT to 100%, because A's overhang past B's corridor is uncovered).

    # Pre-transform UTM -> (lat, lon) with DuckDB (consistent with how the UTM was produced).
    vtx_ll = _utm_to_latlon(conn, [(float(gb.vx[i]), float(gb.vy[i]))
                                   for i in range(gb.n_vertices)], srid)
    wp = res["warping_path"]
    a_ll = _utm_to_latlon(conn, [a for (a, _b) in wp], srid)
    b_ll = _utm_to_latlon(conn, [b for (_a, b) in wp], srid)

    vals = [re["match_dist_avg"] for re in route_edges] or [1.0]
    vmax = max(max(vals), 1.0)
    cmap = LinearColormap(["#2ca25f", "#fec44f", "#de2d26"], vmin=0.0, vmax=vmax)
    cmap.caption = "per-edge match distance A↔B-edge (m)"

    cx, cy = a4326.centroid.x, a4326.centroid.y
    m = folium.Map(location=[cy, cx], zoom_start=17, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite").add_to(m)

    fg_sub = folium.FeatureGroup(name="Subgraph B-edges (unused)", show=True)
    fg_route = folium.FeatureGroup(name="Matched route (per-edge dist)", show=True)
    fg_a = folium.FeatureGroup(name="A-edge", show=True)
    fg_links = folium.FeatureGroup(name="Point matches", show=True)

    def tip(html):
        return folium.Tooltip(
            f"<div style='font-family:Arial;font-size:12px;white-space:nowrap'>{html}</div>",
            sticky=True)

    all_latlon = []

    # --- subgraph B-edges (route vs unused) ---
    for did, (g, nm) in b4326.items():
        paths = _latlon(g)
        for p in paths:
            all_latlon += p
        if did in route_ids:
            re = per_edge[did]
            color, w, tgt = cmap(re["match_dist_avg"]), 6, fg_route
            html = (f"<b>B {did}</b> ({nm}) &nbsp; seq {re['seq']} / dir {re['direction']}<br>"
                    f"<b>match dist:</b> avg {re['match_dist_avg']:.2f} "
                    f"(max {re['match_dist_max']:.2f}, min {re['match_dist_min']:.2f}) m<br>"
                    f"<b>covers A:</b> {re['a_len']:.1f} m = "
                    f"<b>{re['cover_pct']:.1f}%</b> of edge A<br>"
                    f"<b>uses {re['b_cover_pct']:.1f}% of this B-edge</b> "
                    f"({re['matched_len']:.1f} / {re['b_edge_len']:.1f} m)<br>"
                    f"<b>bearing Δ:</b> {re['bearing_diff']:.1f}° &nbsp; "
                    f"<b>A pts matched:</b> {re['n_points']}")
        else:
            color, w, tgt = "#c0c4cc", 2, fg_sub
            html = f"<b>B {did}</b> ({nm}) — in subgraph, not in route"
        for p in paths:
            folium.PolyLine(p, color=color, weight=w, opacity=0.9, tooltip=tip(html)).add_to(tgt)

    # --- A-edge (drawn shifted north-east by a_shift so it separates from the B subgraph) ---
    for p in _latlon(a4326):
        sp = [(lat + a_shift, lon + a_shift) for (lat, lon) in p]
        all_latlon += sp
        folium.PolyLine(sp, color="#2563eb", weight=3, opacity=0.95,
                        tooltip=tip(f"<b>OSM A:</b> {edge_id} (shifted {a_shift}° for clarity)")
                        ).add_to(fg_a)

    # --- B vertices: THREE distinct types ---
    #   edge end point (junction/dead-end) : purple diamond
    #   real interior node                  : dark hollow circle
    #   projection point                    : small gray dot
    fg_end = folium.FeatureGroup(name="◆ B edge-endpoints (junctions)", show=True)
    fg_node = folium.FeatureGroup(name="○ B real nodes", show=False)
    fg_proj = folium.FeatureGroup(name="· B projection points", show=False)
    for vi in range(gb.n_vertices):
        lat, lon = vtx_ll[vi]
        if bool(gb.is_endpoint[vi]):
            folium.RegularPolygonMarker(
                location=[lat, lon], number_of_sides=4, rotation=45, radius=7,
                color="#6d28d9", fill_color="#7c3aed", fill_opacity=0.9, weight=1,
                tooltip=tip("B edge end point (junction / dead-end)")).add_to(fg_end)
        elif bool(gb.is_node[vi]):
            folium.CircleMarker([lat, lon], radius=4, color="#111827", weight=2,
                                fill=False, tooltip=tip("B real node (interior vertex)")
                                ).add_to(fg_node)
        else:
            folium.CircleMarker([lat, lon], radius=2, color="#9ca3af", weight=0,
                                fill=True, fill_opacity=1.0,
                                tooltip=tip("B projection point")).add_to(fg_proj)

    # --- every point match link (shifted A point -> B vertex), with per-point match info ---
    wb = M["warp_is_node"]
    wa = M["warp_a_is_node"]
    we = M["warp_edge"]
    for k, (a_utm, b_utm) in enumerate(wp):
        alat, alon = a_ll[k]
        blat, blon = b_ll[k]
        alat += a_shift; alon += a_shift          # follow the shifted A-edge
        dist_m = math.hypot(a_utm[0] - b_utm[0], a_utm[1] - b_utm[1])
        isnode = wb[k]
        col = "#dc2626" if isnode else "#ea580c"
        b_kind = "B node (point-to-point)" if isnode else "B projection (point-to-projection)"
        a_kind = "A node" if wa[k] else "A projection"
        html = (f"<b>match #{k}</b> &nbsp; distance <b>{dist_m:.2f} m</b><br>"
                f"<b>on B-edge:</b> {we[k]}<br>"
                f"<b>B side:</b> {b_kind}<br><b>A side:</b> {a_kind}")
        folium.PolyLine([(alat, alon), (blat, blon)], color=col, weight=1.6,
                        opacity=0.85, tooltip=tip(html)).add_to(fg_links)
        folium.CircleMarker([blat, blon], radius=2.8, color=col, fill=True, fill_opacity=1.0,
                            weight=0, tooltip=tip(html)).add_to(fg_links)
        folium.CircleMarker([alat, alon], radius=2, color="#2563eb", fill=True,
                            fill_opacity=1.0, weight=0, tooltip=tip(html)).add_to(fg_links)
        all_latlon += [(alat, alon), (blat, blon)]

    # --- A points: real nodes vs projection points (shifted, like the A-edge) ---
    fg_a_node = folium.FeatureGroup(name="▲ A real nodes", show=False)
    fg_a_proj = folium.FeatureGroup(name="· A projection points", show=False)
    seen = set()
    for k in range(len(wp)):
        alat, alon = a_ll[k]
        alat += a_shift; alon += a_shift
        key = (round(alat, 7), round(alon, 7))
        if key in seen:
            continue
        seen.add(key)
        if wa[k]:
            folium.RegularPolygonMarker(
                location=[alat, alon], number_of_sides=3, radius=5,
                color="#1d4ed8", fill_color="#3b82f6", fill_opacity=0.9, weight=1,
                tooltip=tip("A real node")).add_to(fg_a_node)
        else:
            folium.CircleMarker([alat, alon], radius=2.5, color="#1d4ed8", weight=1,
                                fill=False, tooltip=tip("A projection point")).add_to(fg_a_proj)

    for fg in (fg_sub, fg_route, fg_a, fg_end, fg_node, fg_proj,
               fg_a_node, fg_a_proj, fg_links):
        fg.add_to(m)
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # --- info panel: overall metrics + per-edge breakdown ---
    rows = "".join(
        f"<tr><td>{re['seq']}</td><td>{re['dest_id']}</td><td>{re['direction']}</td>"
        f"<td>{re['match_dist_avg']:.2f}</td><td>{re['match_dist_max']:.2f}</td>"
        f"<td>{re['match_dist_min']:.2f}</td><td>{re['a_len']:.1f}</td>"
        f"<td>{re['cover_pct']:.1f}%</td>"
        f"<td>{re['bearing_diff']:.1f}°</td>"
        f"<td>{re['matched_len']:.1f}</td><td>{re['b_cover_pct']:.1f}%</td>"
        f"<td>{re['n_points']}</td></tr>"
        for re in route_edges)
    panel = (
        f"<div style='position:fixed;top:10px;left:10px;z-index:9999;background:white;"
        f"padding:10px 12px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.3);"
        f"font-family:Arial;font-size:12px;max-width:600px;max-height:85vh;overflow:auto'>"
        f"<b style='font-size:14px'>Graph-DTW match detail — OSM edge {edge_id}</b><br>"
        f"route = {[re['dest_id'] for re in route_edges]} &nbsp; "
        f"(<b>{M['n_edges']}</b> B-edges)<br>"
        f"<b>match distance:</b> avg <b>{M['average']:.2f}</b> m "
        f"(max {M['max']:.2f}, min {M['min']:.2f})<br>"
        f"<b>bearing Δ:</b> <b>{M['bearing_diff']:.1f}°</b> &nbsp;&nbsp; "
        f"<b>overlap (A covered):</b> <b>{M['overlap_pct']}%</b> &nbsp;&nbsp; "
        f"<b>matched_len:</b> {M['matched_len']:.1f} m"
        f"<hr style='margin:6px 0'>"
        f"<b>per route-edge breakdown</b> (the result divided by the B-edges traversed):"
        f"<table style='border-collapse:collapse;font-size:11px;margin-top:4px' border='1' "
        f"cellpadding='3'>"
        f"<tr style='background:#f1f5f9'><th>seq</th><th>B&nbsp;id</th><th>dir</th>"
        f"<th>avg</th><th>max</th><th>min</th><th>A&nbsp;len</th><th>cover</th>"
        f"<th>bearΔ</th><th>B&nbsp;len</th><th>B&nbsp;used</th><th>pts</th></tr>"
        f"{rows}</table>"
        f"<div style='color:#666;margin-top:6px'>"
        f"<b>cover</b> = % of edge A this B-edge covers; <b>B&nbsp;used</b> = % of this B-edge's "
        f"own length the match traverses; <b>bearΔ</b> = bearing diff (B-edge span vs the A part "
        f"matched to it); <b>pts</b> = A sample points matched onto it."
        f"<br>A-edge drawn <b>shifted</b> for clarity; each link is one point match "
        f"(hover for distance): <b style='color:#dc2626'>red</b>=to B node (point-to-point), "
        f"<b style='color:#ea580c'>orange</b>=to projection (point-to-projection).<br>"
        f"B vertices: <b style='color:#7c3aed'>◆</b> edge end point (junction), "
        f"<b>○</b> real interior node, <b style='color:#9ca3af'>·</b> projection point. &nbsp; "
        f"A vertices: <b style='color:#3b82f6'>▲</b> real node, "
        f"<b style='color:#1d4ed8'>·</b> projection point (toggle layers top-right).</div>"
        f"</div>")
    m.get_root().html.add_child(folium.Element(panel))

    if all_latlon:
        lats = [p[0] for p in all_latlon]; lons = [p[1] for p in all_latlon]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


def main():
    ap = argparse.ArgumentParser(description="Detailed single-edge graph-DTW HTML.")
    ap.add_argument("--edge-id", type=int, required=True, help="OSM edge_id (Source A) to inspect")
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--snap", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--emission", choices=["point", "segment"], default="point",
                    help="local cost: point-to-point (default) or segment (endpoint-average)")
    ap.add_argument("--bearing-weight", type=float, default=0.0,
                    help="optional heading penalty lambda (segment mode only)")
    ap.add_argument("--a-shift", type=float, default=0.00012,
                    help="degrees to shift edge A (north-east) so its point matches are visible")
    args = ap.parse_args()

    setup_logging()
    m = load_matcher(args.osm, args.sweden, args.utm_srid, args.max_distance)

    cand = m.generate_candidate_pairs()
    grp = cand[cand["id_a"] == args.edge_id]
    if grp.empty:
        print(f"No B candidates within {args.max_distance} m of OSM edge {args.edge_id}.")
        return

    coords_a = list(load_wkt(grp["wkt_a"].iloc[0]).coords)  # UTM
    b_edges = []
    for r in grp.itertuples(index=False):
        g = load_wkt(r.wkt_b)
        if g.geom_type == "LineString":
            b_edges.append((r.id_b, g))
    log.info("edge %s: %d candidate B-edges", args.edge_id, len(b_edges))

    res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=args.snap, step_meters=args.step,
                               emission=args.emission, bearing_weight=args.bearing_weight)

    # 4326 geometries for drawing
    a4326 = load_wkt(m.conn.execute(
        f"SELECT ST_AsText(geometry) FROM driving_edges WHERE edge_id={args.edge_id}").fetchone()[0])
    bids = ",".join(str(r.id_b) for r in grp.itertuples(index=False))
    b4326 = {row[0]: (load_wkt(row[1]), row[2]) for row in m.conn.execute(
        f"SELECT directed_id, ST_AsText(geometry), name FROM vehicle_edges_directed "
        f"WHERE directed_id IN ({bids})").fetchall()}

    fmap = build_detail_map(res, a4326, b4326, args.edge_id, m.conn, args.utm_srid,
                            a_shift=args.a_shift)

    out = args.out or f"output/graph_dtw_edge_{args.edge_id}_{args.emission}.html"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fmap.save(out)
    log.info("saved -> %s", out)
    print(f"Saved {out}  (route={[re['dest_id'] for re in res['metrics']['route_edges']]}, "
          f"avg match dist={res['avg_distance']:.2f} m)")


if __name__ == "__main__":
    main()
