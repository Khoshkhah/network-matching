"""Mode 4 -- point-to-edge matching (``match_points``, docs/point_matching.md).

Same input system as the other modes: lon/lat WKT CSVs, ``utm_srid`` projection inside DuckDB.
Local layout (meters around LON0/LAT0): two parallel roads 8 m apart running opposite ways, one
far road, and four sensors:

    e_north  (0,0)   -> (100,0)     west->east
    e_south  (100,-8)-> (0,-8)      east->west
    e_far    (0,500) -> (100,500)

    s1 (50,2)    2 m north of e_north (mid-edge); e_south also in radius
    s2 (25,-6)   2 m north of e_south (at 75% along its east->west run)
    s3 (50,200)  no edge within max_distance -> NO_MATCH
    s4 (120,1)   past e_north's east end -> snaps to the endpoint (position 100%)
"""
import csv
import math

import pytest
from shapely.wkt import loads as load_wkt

from network_matching import DuckDBMapMatcher

LON0, LAT0 = 18.06, 59.33
MX = 1.0 / (111320 * math.cos(math.radians(LAT0)))
MY = 1.0 / 111320.0


def _pt(x, y):
    return f"POINT ({LON0 + x * MX:.8f} {LAT0 + y * MY:.8f})"


def _ls(pts):
    return "LINESTRING (" + ", ".join(f"{LON0 + x * MX:.8f} {LAT0 + y * MY:.8f}"
                                      for x, y in pts) + ")"


@pytest.fixture()
def matcher(tmp_path):
    points = [("s1", _pt(50, 2)), ("s2", _pt(25, -6)), ("s3", _pt(50, 200)),
              ("s4", _pt(120, 1))]
    edges = [("e_north", _ls([(0, 0), (100, 0)])),
             ("e_south", _ls([(100, -8), (0, -8)])),
             ("e_far", _ls([(0, 500), (100, 500)]))]
    for name, rows in (("points.csv", points), ("edges.csv", edges)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fid", "geometry"])
            w.writerows(rows)
    return DuckDBMapMatcher.from_wkt_csv(str(tmp_path / "points.csv"), str(tmp_path / "edges.csv"),
                                         id_a="fid", id_b="fid", utm_srid=3006,
                                         max_distance=25, id_cast=None)


def test_match_points_schema_and_ranking(matcher):
    res = matcher.match_points()
    assert list(res.columns) == DuckDBMapMatcher.POINT_COLUMNS
    assert set(res.source_id) == {"s1", "s2", "s3", "s4"}       # every point accounted for

    best = {r.source_id: r for r in res[res["rank"] == 1].itertuples()}
    assert best["s1"].dest_id == "e_north"
    assert best["s2"].dest_id == "e_south"
    assert best["s4"].dest_id == "e_north"

    # s1: 2 m lateral, mid-edge, west->east bearing (~90 deg + grid convergence)
    assert 1.5 < best["s1"].distance_m < 2.5
    assert 45 < best["s1"].position_pct < 55
    assert 75 < best["s1"].edge_bearing_deg < 105
    assert best["s1"].match_type == "1:N_CANDIDATES"            # e_south also inside 25 m
    assert (res.source_id == "s1").sum() == 2                   # ranked, both kept

    # s2: e_south runs east->west, so bearing ~270 and x=25 sits 75% along it
    assert 255 < best["s2"].edge_bearing_deg < 285
    assert 70 < best["s2"].position_pct < 80

    # s4: past the east end -> snaps to the endpoint
    assert best["s4"].position_pct > 99
    assert 18 < best["s4"].distance_m < 22

    # the snapped point sits on the matched edge (distance_m away from the sensor)
    assert best["s1"].snap_wkt.startswith("POINT")
    snap = load_wkt(best["s1"].snap_wkt)
    assert abs(snap.y - load_wkt(best["s4"].snap_wkt).y) < 3    # both on e_north (same UTM y +- curve)

    # s3: nothing within 25 m
    s3 = res[res.source_id == "s3"].iloc[0]
    assert s3["match_type"] == "NO_MATCH" and pd_isna(s3["dest_id"]) and pd_isna(s3["rank"])


def pd_isna(v):
    import pandas as pd
    return v is None or pd.isna(v)


def test_resolve_on_point_table(matcher):
    res = matcher.match_points()

    # best_per_source: one row per point, ranked by distance_m (auto-detected score column)
    dec = matcher.resolve(res, strategy="best_per_source")
    assert len(dec) == 4 and set(dec.source_id) == {"s1", "s2", "s3", "s4"}
    by = {r.source_id: r for r in dec.itertuples()}
    assert by["s1"].dest_id == "e_north" and by["s2"].dest_id == "e_south"
    assert by["s3"].match_type == "NO_MATCH"

    # one_to_one: s1 and s2 win their edges (~2 m each); s4's candidates are both taken
    one = matcher.resolve(res, strategy="one_to_one")
    by = {r.source_id: r for r in one.itertuples()}
    assert by["s1"].dest_id == "e_north" and by["s2"].dest_id == "e_south"
    assert by["s4"].match_type == "NO_MATCH" and by["s3"].match_type == "NO_MATCH"


def test_match_points_all_unmatched(tmp_path):
    """No candidates at all: the output is pure NO_MATCH rows with the Mode-4 schema."""
    with open(tmp_path / "p.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fid", "geometry"])
        w.writerow(["lonely", _pt(0, 0)])
    with open(tmp_path / "e.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fid", "geometry"])
        w.writerow(["far", _ls([(0, 900), (100, 900)])])
    m = DuckDBMapMatcher.from_wkt_csv(str(tmp_path / "p.csv"), str(tmp_path / "e.csv"),
                                      id_a="fid", id_b="fid", utm_srid=3006,
                                      max_distance=25, id_cast=None)
    res = m.match_points()
    assert list(res.columns) == DuckDBMapMatcher.POINT_COLUMNS
    assert len(res) == 1 and res.iloc[0].match_type == "NO_MATCH"
