# Graph-DTW Route Matching — End-to-End Pipeline

How to run the whole route-based conflation pipeline: initialize → generate candidates →
graph-DTW per A-edge → the two output tables → save / visualize. For the *algorithm* itself
(the DTW-on-a-graph dynamic program), see [`graph_dtw_matching.md`](graph_dtw_matching.md).

Implementation: [`network_matching/matcher.py`](../network_matching/matcher.py),
[`network_matching/graph_dtw.py`](../network_matching/graph_dtw.py).

---

## 1. What it does

Matches each edge of **network A** (OSM) to a connected **route of edges** in **network B**
(Sweden NVDB), so a single A-edge that spans several differently-split B-edges is stitched into
one clean match — and a topologically-disconnected parallel road can never be picked. Output is
two tables: one row per A-edge (`routes_summary`) and one row per matched B-edge in each route
(`routes_long`).

---

## 2. Initialize

One call loads both networks into in-memory DuckDB spatial tables, configures the column
mapping, and sets the search radius. Geometry is assumed lon/lat (EPSG:4326) and is transformed
to the local projected `utm_srid` (meters) during candidate generation.

```python
from network_matching import DuckDBMapMatcher, setup_logging
setup_logging()                                  # optional: logs/network_matching_*.log

# (a) WKT CSVs — geometry column holds WKT:
m = DuckDBMapMatcher.from_wkt_csv(
    "data/osm_edges.csv", "data/sweden_edges.csv",
    id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=30,
    keep_cols_b=["name"])                         # carry extra columns through

# (b) GIS files (GeoPackage / GeoJSON / Shapefile) via DuckDB ST_Read:
m = DuckDBMapMatcher.from_geofiles(
    "osm.gpkg", "nvdb.gpkg",
    id_a="edge_id", id_b="directed_id", utm_srid=3006, src_srid=3006)

# (c) manual (what the initializers wrap):
m = DuckDBMapMatcher()
m.conn.execute("CREATE TABLE a AS SELECT edge_id, ST_GeomFromText(geometry) AS geometry FROM '...';")
m.conn.execute("CREATE TABLE b AS SELECT directed_id, ST_GeomFromText(geometry) AS geometry FROM '...';")
m.configure_sources(source_a="a", id_col_a="edge_id", geom_col_a="geometry",
                    source_b="b", id_col_b="directed_id", geom_col_b="geometry", utm_srid=3006)
m.set_parameters(max_distance=30)
```

Inputs live in [`data/`](../data/); see §7 for the folder layout.

---

## 3. Run

```python
routes_long, routes_summary = m.match_routes(
    snap_tolerance_m=0.5, step_meters=10, n_jobs=-1)
```

### Steps (what `match_routes` does)

1. **Candidates** — `generate_candidate_pairs()` runs a DuckDB `ST_DWithin` spatial join once,
   returning `id_a, wkt_a, id_b, wkt_b` (projected to UTM meters). Each A-edge gets the B-edges
   within `max_distance`.
2. **Group** — candidates are grouped by `id_a`; each group `(coords_a, [B-edges])` is an
   independent unit.
3. **Graph-DTW per A-edge** — `match_edge_to_bgraph` builds the local directed B-graph and aligns
   A to it (the algorithm in [`graph_dtw_matching.md`](graph_dtw_matching.md)). Runs in parallel
   over A-edges with **joblib** (`n_jobs=-1` = all cores).
4. **Assemble** — results become the two tables; every A-edge appears (unmatched as `NO_MATCH`).
   (`trim_ends_m > 0` would optionally remove a junk end edge here; it is **off by default**.)

### Parameters

| parameter           | meaning |
|---------------------|---------|
| `max_distance`      | candidate search radius (m) for `ST_DWithin` (set in the initializer / `set_parameters`). |
| `snap_tolerance_m`  | one B-edge's **end** is joined to another's **start** when within this distance — the head-to-tail junction crossing (B is a directed table; no reverse arcs are synthesized). |
| `step_meters`       | gap-fill density: a vertex every ~N m on top of node+projection pools (default 10). Smaller = denser/slower; `0` = projection-only (fastest). |
| `trim_ends_m`       | **default `0` (off).** optional: *remove* a leading/trailing route edge covering `<` this many m of A. Not a gap-filler (use `snap_tolerance_m`); off by default as it can delete real corridor edges. |
| `n_jobs`            | parallel workers over A-edges: `-1` = all cores, `1` = serial. |

### Filter by quality (`resolve_routes`)

`match_routes` returns the best route for **every** A-edge regardless of quality. To keep only
confident matches, filter by thresholds — the route-mode analogue of the edge-to-edge quality
filters:

```python
routes_summary, routes_long = m.resolve_routes(
    routes_summary, routes_long,
    max_match_dist=10,      # drop routes with avg match distance (dtw_distance) > 10 m
    max_bearing_diff=30,    # ...or whole-route bearing difference > 30°
    min_overlap_pct=95)     # ...or covering < 95% of the A-edge
```

Any threshold left `None` is not applied. A route that fails is reset to a `NO_MATCH` row (route
cleared, metrics `NaN`) and its rows are removed from `routes_long`; every A-edge still appears.
(`min_overlap_pct` drops edges whose ends **overhang** too far past B's corridor — the overlap
part shrinks below 100% of A where A's first/last records pile onto the route's first/last
B-arc/vertex; common on differently-segmented networks. See `graph_dtw_matching.md` §4.1.)

#### Estimating the thresholds (`suggest_thresholds`)

Rather than pick the cuts by eye, estimate them from the data. Each quality metric forms a tight
**good cluster** plus a **tail** of wrong/poor matches; `suggest_thresholds` runs several
outlier / two-population estimators (Tukey IQR & MAD fences, a high percentile, a 2-component
Gaussian-mixture EM, a KDE valley, Otsu, Kneedle, and an IsolationForest cut) and recommends the
cut between cluster and tail (a bimodal separator when a real bad cluster exists, else a robust
fence). The numpy estimators need no extra deps; IsolationForest needs `scikit-learn` (`[ml]`) and
the diagnostic plot needs `matplotlib` (`[viz]`) — both guarded and skipped if absent.

```python
from network_matching import suggest_thresholds
sugg = suggest_thresholds(routes_summary, report=True,
                          plot_path="output/threshold_suggestions.png")
routes_summary, routes_long = m.resolve_routes(routes_summary, routes_long, **sugg["recommended"])
```

`sugg["recommended"]` is a dict of `resolve_routes` kwargs (`max_match_dist`,
`max_bearing_diff`, `min_overlap_pct`); per-metric breakdowns (every method's value + the chosen
cut + a rationale) are under `sugg["metrics"]`. CLI: `python scripts/suggest_thresholds.py`;
walk-through: [`notebooks/threshold_estimation.ipynb`](../notebooks/threshold_estimation.ipynb).
Full reference (every estimator, the recommendation rule, the multivariate IsolationForest):
[`threshold_estimation.md`](threshold_estimation.md).

**Multivariate review (`isolation_forest_flags`).** Per-metric cuts treat each axis independently.
`isolation_forest_flags(routes_summary)` fits one IsolationForest on all quality signals **jointly**
(z-scored; `overlap_pct` flipped to a deficit) and flags matches anomalous *in combination* — a
route can look acceptable on each axis yet sit in a sparse region of the joint space. It returns
the frame with `if_outlier` (bool) and `if_score` (lower = more anomalous) columns, a complement to
the thresholds for surfacing jointly-weird matches to inspect. Needs `scikit-learn` (`[ml]`).

---

## 4. Output tables

### `routes_summary` — one row per A-edge

| column                          | meaning |
|---------------------------------|---------|
| `source_id`                     | A-edge id |
| `n_edges`                       | number of B-edges in the route (0 for `NO_MATCH`) |
| `dest_ids`                      | ordered list of B-edge ids in the route |
| `dtw_distance`                  | average match distance (m) over the whole edge — main quality signal |
| `max_dtw_distance` / `min_dtw_distance` | max / min match distance |
| `bearing_diff`                  | whole-route bearing difference (degrees) |
| `part_drift`                    | mean match distance (m) over the end-trimmed **overlap part** (`graph_dtw_matching.md` §4.1) |
| `part_bearing_diff`             | mean per-segment heading diff (°) over the overlap part — segment emission only; equals `bearing_diff` in point mode |
| `overlap_pct`                   | **A-length share of the overlap part** (%); < 100 where A's ends pile up on the route's first/last B-arc/vertex |
| `matched_len`                   | total B-length (m) traversed |
| `route_geom_wkt`                | matched corridor geometry, WKT in **UTM (`utm_srid`)** |
| `match_type`                    | `1:1` (single edge) · `1:N_ROUTE` (multi-edge) · `NO_MATCH` |

### `routes_long` — one row per (A-edge, B-edge in its route)

The result **divided per B-edge** (`seq` = order of matching along the route).

| column                          | meaning |
|---------------------------------|---------|
| `source_id`                     | A-edge id |
| `dest_id`                       | B-edge id |
| `seq`                           | **order of matching** (0,1,2,…) |
| `direction`                     | always `forward` (B is a directed table; the reverse direction is a separate `directed_id`) |
| `edge_match_dist_avg/max/min`   | match distance over just this edge's matched points |
| `edge_a_len`                    | metres of A **covered** by this edge (where its B vertex advances) |
| `edge_cover_pct`                | **% of the whole A-edge** this edge covers (these sum to `overlap_pct`) |
| `edge_matched_len`              | metres of this B-edge traversed |
| `edge_b_len`                    | this B-edge's total length (m) |
| `edge_b_used_pct`               | **% of this B-edge** used (`edge_matched_len` ÷ `edge_b_len`) |
| `edge_bearing_diff`             | bearing of this B-edge's span vs the A part matched to it (degrees) |
| `n_points`                      | A sample points matched onto this edge |
| `route_match_dist`              | whole-route average match distance (repeated) |
| `n_edges`                       | whole-route edge count (repeated) |

### Column types

Both tables are returned with a fixed dtype schema (`DuckDBMapMatcher.ROUTES_LONG_DTYPES` /
`ROUTES_SUMMARY_DTYPES`). Ids and counts are plain **`int64`** — always-present integers (an edge
id is never null and never a float, so `dest_id` is `580`, not `580.0`); distances/percentages are
`float64`; `direction`/`match_type` are `string`; `dest_ids` (a list) and `route_geom_wkt` stay
`object`. The one genuinely nullable column is `routes_summary.overlap_pct` (nullable `Int64`,
`<NA>` for `NO_MATCH` rows).

---

## 5. Save

`match_routes` returns DataFrames **in memory** — nothing is written automatically. Persist to
[`output/`](../output/):

```python
routes_summary.to_csv("output/routes_summary.csv", index=False)
routes_long.to_csv("output/routes_long.csv", index=False)
```

(The `route_geom_wkt` column is UTM `utm_srid`, e.g. EPSG:3006; set that CRS in QGIS.)

---

## 6. Inspect & visualize

- **Pre-flight:** `validate_b_geometry(b_edges)` reports the endpoint-gap distribution to choose
  `snap_tolerance_m`.
- **Whole-network map** → [`scripts/graph_dtw_map.py`](../scripts/graph_dtw_map.py): both networks
  (B shifted), match links coloured by match distance. `python scripts/graph_dtw_map.py`.
- **Validation map** → [`scripts/graph_dtw_validation_map.py`](../scripts/graph_dtw_validation_map.py):
  both networks (B shifted) with toggleable layers — A matched / NO_MATCH, B used / unused, and the
  abnormal overlays (A under-covered, B under-used, B over-used). Plus the
  [validation notebook](../notebooks/graph_dtw_validation.ipynb) (raw vs resolved coverage report).
- **Single-edge deep dive** → [`scripts/graph_dtw_edge_detail.py`](../scripts/graph_dtw_edge_detail.py):
  the local subgraph, every point match, and the per-edge breakdown table.
  `python scripts/graph_dtw_edge_detail.py --edge-id 3597`.
- **Notebooks:** [`graph_dtw_results_table.ipynb`](../notebooks/graph_dtw_results_table.ipynb)
  (tables), [`graph_dtw_real_data.ipynb`](../notebooks/graph_dtw_real_data.ipynb) (sampled plots),
  [`graph_dtw_visualization.ipynb`](../notebooks/graph_dtw_visualization.ipynb) (synthetic cases).

Both scripts write their `.html` to `output/`.

---

## 7. Folder layout & environment

```
data/                INPUT only — osm_edges.csv, sweden_edges.csv, sundbyberg_boundary.geojson
output/              generated maps + result CSVs (git-ignored)
conflation_issues/   conflation_issues.schema.yaml + <area>_conflation_issues.yaml
logs/                run logs (git-ignored; from setup_logging())
network_matching/    the library
scripts/ notebooks/ docs/ tests/
```

Run in the dedicated conda env: `conda activate network-matching` (Python 3.11 + deps +
the package installed editable + a registered Jupyter kernel "Python (network-matching)").

---

## 8. Scope

Directed A→B; deterministic (candidate edges sorted by id). `overlap_pct` (the overlap-part share
of A) is < 100 where A overhangs past the route's first/last B-arc/vertex (end-edge trimming via
`trim_ends_m` off by default).
The cost is count-weighted (route choice depends on `step_meters` density); a density-independent
objective and symmetric (B→A) reconciliation are future work — see
[`graph_dtw_matching.md`](graph_dtw_matching.md) §7.
