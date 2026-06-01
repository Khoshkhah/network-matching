#!/usr/bin/env python
"""
diagnose_match.py — Explain why a specific OSM edge is (or isn't) matched to a
Sweden directed edge in the network-matching pipeline.

USAGE
    python scripts/diagnose_match.py <osm_edge_id> <sweden_directed_id> [options]

EXAMPLES
    python scripts/diagnose_match.py 1914 472
    python scripts/diagnose_match.py 2142 422 --mode all
    python scripts/diagnose_match.py 1914 472 --keep-overlap 30

WHAT IT REPORTS
    1. Attributes + directionality of each edge (oneway / junction; whether each
       lands in the matching set for the chosen --mode).
    2. The raw pair geometry: distance, DTW drift, both overlaps, bearing.
    3. A PASS/FAIL verdict for every gate (distance, dtw, bearing, containment).
    4. Whether the pair survives match_symmetric, and — if not — what each side
       matched to instead.

MODES
    oneway (default) : OSM oneway roads vs Sweden one-way roads (the "directed" run)
    all              : full OSM network vs full Sweden directed network
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from network_matching import DuckDBMapMatcher
from network_matching.dtw import dtw_align
from network_matching.matcher import bearing_between
from shapely.wkt import loads as load_wkt

DATA = PROJECT / "data"
UTM = 3006  # SWEREF99 TM


def _one(con, sql):
    r = con.execute(sql).fetchone()
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("osm", type=int, help="OSM edge_id")
    ap.add_argument("sweden", type=int, help="Sweden directed_id")
    ap.add_argument("--mode", choices=["oneway", "all"], default="oneway")
    ap.add_argument("--osm-csv", default=str(DATA / "osm_edges.csv"))
    ap.add_argument("--sweden-csv", default=str(DATA / "sweden_edges.csv"))
    ap.add_argument("--max-distance", type=float, default=25.0)
    ap.add_argument("--max-dtw", type=float, default=25.0)
    ap.add_argument("--max-angle", type=float, default=45.0)
    ap.add_argument("--min-overlap-m", type=float, default=5.0,
                    help="min shared length in meters (absolute matched-length floor)")
    ap.add_argument("--sym-overlap", type=int, default=70)
    args = ap.parse_args()

    OSM, SWE = args.osm, args.sweden
    osm_csv, swe_csv = args.osm_csv, args.sweden_csv
    m = DuckDBMapMatcher()
    c = m.conn
    c.execute("INSTALL spatial; LOAD spatial;")

    def tick(ok):
        return "PASS" if ok else "FAIL"

    print("=" * 64)
    print(f"  MATCH DIAGNOSIS   OSM {OSM}  ->  Sweden directed {SWE}   (mode={args.mode})")
    print("=" * 64)

    # ── 1. Attributes / directionality ──────────────────────────────────────
    # junction may not exist in older CSVs; probe columns.
    osm_cols = [r[0] for r in c.execute(f"DESCRIBE SELECT * FROM '{osm_csv}'").fetchall()]
    jsel = "CAST(junction AS VARCHAR)" if "junction" in osm_cols else "NULL"
    osm_row = _one(c, f"""SELECT osm_id, highway, CAST(oneway AS VARCHAR), {jsel}
                          FROM '{osm_csv}' WHERE edge_id={OSM}""")
    if not osm_row:
        sys.exit(f"OSM edge_id {OSM} not found in {osm_csv}")
    osm_id, hw, oneway, junction = osm_row
    osm_isoneway = (str(oneway).lower() in ("yes", "true", "1", "-1")) or \
                   (str(junction).lower() in ("roundabout", "circular"))
    print(f"\n[OSM {OSM}]  osm_id={osm_id}  highway={hw}  oneway={oneway}  junction={junction}")
    print(f"   -> one-way? {osm_isoneway}")

    swe_row = _one(c, f"""SELECT original_edge_id, name, CAST(is_reverse AS VARCHAR)
                          FROM '{swe_csv}' WHERE directed_id={SWE}""")
    if not swe_row:
        sys.exit(f"Sweden directed_id {SWE} not found in {swe_csv}")
    orig, name, is_rev = swe_row
    ndir = _one(c, f"SELECT COUNT(*) FROM '{swe_csv}' WHERE original_edge_id={orig}")[0]
    swe_isoneway = ndir == 1
    print(f"[SWE {SWE}]  original_edge_id={orig}  name={name}  is_reverse={is_rev}")
    print(f"   -> {ndir} directed row(s) => {'ONE-WAY' if swe_isoneway else 'TWO-WAY'}")

    if args.mode == "oneway":
        print(f"\n   In one-way matching set?  OSM: {tick(osm_isoneway)}   Sweden: {tick(swe_isoneway)}")
        if not (osm_isoneway and swe_isoneway):
            print("   >>> At least one edge is excluded from the one-way set — that alone "
                  "prevents the match in --mode oneway (try --mode all).")

    # ── 2. Raw pair geometry + metrics ──────────────────────────────────────
    c.execute(f"CREATE TABLE _a AS SELECT ST_GeomFromText(geometry) g FROM '{osm_csv}' WHERE edge_id={OSM}")
    c.execute(f"CREATE TABLE _b AS SELECT ST_GeomFromText(geometry) g FROM '{swe_csv}' WHERE directed_id={SWE}")
    dist, within = _one(c, f"""
        SELECT ST_Distance(ST_Transform(_a.g,'EPSG:4326','EPSG:{UTM}'),
                           ST_Transform(_b.g,'EPSG:4326','EPSG:{UTM}')),
               ST_DWithin(ST_Transform(_a.g,'EPSG:4326','EPSG:{UTM}'),
                          ST_Transform(_b.g,'EPSG:4326','EPSG:{UTM}'), {args.max_distance})
        FROM _a, _b""")
    wa = _one(c, f"SELECT ST_AsText(ST_Transform(g,'EPSG:4326','EPSG:{UTM}')) FROM _a")[0]
    wb = _one(c, f"SELECT ST_AsText(ST_Transform(g,'EPSG:4326','EPSG:{UTM}')) FROM _b")[0]
    ga, gb = load_wkt(wa), load_wkt(wb)
    # undirected=True matches what match_symmetric does (orientation-robust).
    dab, wp, mab = dtw_align(list(ga.coords), list(gb.coords), undirected=True)
    dba, _, mba = dtw_align(list(gb.coords), list(ga.coords), undirected=True)
    a0, b0 = wp[0]; a1, b1 = wp[-1]
    bd = abs(bearing_between(a0, a1) - bearing_between(b0, b1)); bd = min(bd, 360 - bd)
    dtw = min(dab, dba)
    ov_ab, ov_ba = mab["overlap_pct"], mba["overlap_pct"]
    matched_len_m = max(mab["matched_len"], mba["matched_len"])

    print(f"\n[GEOMETRY]  len_OSM={ga.length:.0f} m   len_SWE={gb.length:.0f} m")
    print(f"   distance       = {dist:6.2f} m   (<= max_distance {args.max_distance})   {tick(within)}")
    print(f"   dtw drift      = {dtw:6.2f} m   (<= max_dtw {args.max_dtw})           {tick(dtw <= args.max_dtw)}")
    print(f"   bearing diff   = {bd:6.1f}°   (<= max_angle {args.max_angle})         {tick(bd <= args.max_angle)}")
    print(f"   overlap a->b={ov_ab}%  b->a={ov_ba}%   (classification only, not a gate)")
    print(f"   matched length = {matched_len_m:6.1f} m  (>= min_overlap_m {args.min_overlap_m})   {tick(matched_len_m >= args.min_overlap_m)}")

    # ── 3. Run the actual symmetric matcher for this mode ────────────────────
    if args.mode == "oneway":
        c.execute(f"""CREATE TABLE a AS SELECT edge_id::BIGINT AS edge_id, ST_GeomFromText(geometry) AS geometry
                      FROM '{osm_csv}'
                      WHERE lower(CAST(oneway AS VARCHAR)) IN ('yes','true','1','-1')""")
        c.execute(f"""CREATE TABLE b AS
                      WITH cnt AS (SELECT original_edge_id FROM '{swe_csv}'
                                   GROUP BY original_edge_id HAVING COUNT(*)=1)
                      SELECT s.directed_id::BIGINT AS directed_id, ST_GeomFromText(s.geometry) AS geometry
                      FROM '{swe_csv}' s JOIN cnt USING (original_edge_id)""")
    else:
        c.execute(f"CREATE TABLE a AS SELECT edge_id::BIGINT AS edge_id, ST_GeomFromText(geometry) AS geometry FROM '{osm_csv}'")
        c.execute(f"CREATE TABLE b AS SELECT directed_id::BIGINT AS directed_id, ST_GeomFromText(geometry) AS geometry FROM '{swe_csv}'")

    m.configure_sources(source_a="a", id_col_a="edge_id", geom_col_a="geometry",
                        source_b="b", id_col_b="directed_id", geom_col_b="geometry", utm_srid=UTM)
    m.set_parameters(max_distance=args.max_distance)
    sym = m.match_symmetric(max_dtw=args.max_dtw, max_angle=args.max_angle,
                            min_overlap_m=args.min_overlap_m, sym_overlap=args.sym_overlap)

    pair = sym[(sym.a_id == OSM) & (sym.b_id == SWE)]
    print("\n[RESULT]")
    if not pair.empty:
        r = pair.iloc[0]
        print(f"   ✅ MATCHED — relation={r['relation']}  cardinality={r['cardinality']}  "
              f"ov_ab={int(r['ov_ab'])}  ov_ba={int(r['ov_ba'])}  dtw={r['dtw']:.1f}")
    else:
        print(f"   ❌ NOT MATCHED in --mode {args.mode}.")
        cols = ["a_id", "b_id", "relation", "cardinality", "ov_ab", "ov_ba", "dtw"]
        a_m = sym[sym.a_id == OSM]
        b_m = sym[sym.b_id == SWE]
        print(f"\n   What OSM {OSM} matched instead:")
        print("   " + (a_m[cols].to_string(index=False).replace("\n", "\n   ") if not a_m.empty else "(nothing)"))
        print(f"\n   What matched to Sweden {SWE} instead:")
        print("   " + (b_m[cols].to_string(index=False).replace("\n", "\n   ") if not b_m.empty else "(nothing)"))


if __name__ == "__main__":
    main()
