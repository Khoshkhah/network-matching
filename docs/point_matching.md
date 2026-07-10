# Point-to-Edge Matching (Mode 4)

Assigns **points** (sensors, measurement stations, stops, incidents) to the road **edges** they
belong to. The simple sibling of Mode 2: the same input system and the same three-tier flow, but
Source A holds POINT geometry, the score is plain geometry (no DTW), and the assignment decision
is the same `resolve()` cardinality step.

## Pipeline

| Tier | What happens | Controlled by |
|------|--------------|---------------|
| 1. Candidate search | DuckDB spatial join: every B-edge within `max_distance` of each A-point (both transformed to `utm_srid`, meters). | `max_distance` |
| 2. Scoring | Per candidate, in Python (shapely): `distance_m` = lateral point→edge distance; `position_pct` = where the point projects along the edge; `snap_wkt` = the projected (snapped) point; `edge_bearing_deg` = the edge's local direction at the snap point (tangent over ±1 m). | — |
| 3. Ranking | Per point, candidates ranked by `distance_m` (`rank` = 1 nearest); points with no candidate come back as `NO_MATCH`. | — |

## API

```python
from network_matching import DuckDBMapMatcher

m = DuckDBMapMatcher.from_wkt_csv(
    "data/sensors.csv", "data/osm_edges.csv",              # A = POINT WKT, B = LINESTRING WKT
    id_a="sensor_id", id_b="edge_id", utm_srid=3006, max_distance=25)

points     = m.match_points()                              # all ranked candidates (+ NO_MATCH rows)
assignment = m.resolve(points, strategy="best_per_source")  # each sensor -> its nearest road
```

Same input system as every other mode — `from_wkt_csv` / `from_geofiles` / manual
`configure_sources(...)`; geometry is lon/lat (EPSG:4326) and transformed to `utm_srid` (meters)
internally.

## Output schema

One row per (point, candidate edge), ranked; one `NO_MATCH` row per unmatched point.

| Column | Meaning |
|--------|---------|
| `source_id` | the A-point id |
| `dest_id` | the candidate B-edge id (`None` on `NO_MATCH`) |
| `distance_m` | lateral point→edge distance, meters |
| `position_pct` | snap position along the edge: 0 (start) … 100 (end) |
| `edge_bearing_deg` | the edge's travel direction at the snap point, degrees clockwise from north |
| `snap_wkt` | the snapped point on the edge (POINT WKT in `utm_srid` meters — same CRS as Mode 1's `route_geom_wkt`) |
| `rank` | 1 = nearest edge for this point |
| `match_type` | `1:1` (single candidate) / `1:N_CANDIDATES` (several qualify) / `NO_MATCH` |

## The decision (`resolve`)

`match_points()` keeps **every** qualifying edge per point. `resolve()` commits to an assignment
exactly as in Mode 2 — it ranks by `distance_m` here (`dtw_distance` in Mode 2), auto-detected
from the columns:

| `strategy` | Cardinality | Use when |
|------------|-------------|----------|
| `"best_per_source"` *(default)* | many-to-one | each point → its nearest edge (sensors → roads) |
| `"best_per_dest"` | one-to-many | one representative point per edge |
| `"one_to_one"` | global unique | each point and each edge used at most once |
| `"all"` | no decision | keep every ranked candidate |

## Notes

* **Directed edge pairs.** A divided road — or a two-way road stored as two directed edges — puts
  two near-parallel candidates inside the radius, and `rank` alone cannot tell direction. Filter on
  `edge_bearing_deg` against the point's known travel direction (e.g. a sensor's facing) before
  `resolve`.
* **Endpoints.** A point past the end of an edge snaps to the endpoint (`position_pct` 0 or 100);
  `distance_m` is then the distance to that endpoint, not a perpendicular offset.
* **No shape thresholds.** `max_angle` / `min_overlap` are Mode-2 DTW-pair filters and are not
  applied here; the only gate is `max_distance`.
