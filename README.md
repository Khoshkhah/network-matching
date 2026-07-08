# network-matching

A reusable, high-performance library for **directed road map-to-map matching (conflation)** —
aligning two road networks (for example OpenStreetMap against Sweden's NVDB), even when they are
segmented differently. It combines **Dynamic Time Warping (DTW)** for shape alignment with
**DuckDB Spatial** for fast candidate search, and works in a local projected CRS so all distances
are in meters.

The library offers **two matching modes**, both built on the same `DuckDBMapMatcher` class,
inputs, and CRS handling:

| Mode | Entry point | Produces | Best for |
|------|-------------|----------|----------|
| **Route-based (graph-DTW)** | `match_routes()` | each A-edge → a connected **route** of B-edges | conflating networks split differently; the recommended default |
| **Edge-to-edge** | `match()` + `resolve()` | ranked A↔B candidate **pairs** + a cardinality decision | assigning points/edges to a single nearest segment; fine-grained control |

---

## Installation

```bash
pip install -e .                 # core library
pip install -r requirements.txt  # core + visualization + notebook tooling
# extras:  pip install -e ".[viz]"   (maps)     pip install -e ".[dev]"   (tests, notebooks)
```

**Dependencies.** Core: `duckdb` (spatial extension auto-loaded), `numpy`, `pandas`, `shapely`,
`geopandas`, `joblib` (parallel route matching). Visualization (scripts/notebooks): `folium`,
`branca`, `matplotlib`. Optional: `scipy` (faster endpoint validation; pure-numpy fallback).

---

## Mode 1 — Route-based matching (graph-DTW)

Instead of aligning an A-edge to a single B-edge, graph-DTW aligns it to the **whole local directed
graph** of nearby B-edges and returns the connected **route** of B-edges it maps to. This stitches
a road that is split differently in the two networks into one clean match, and can never select a
topologically-disconnected parallel road.

```python
from network_matching import DuckDBMapMatcher

# one-call initializer (WKT CSVs; or .from_geofiles(...) for GeoPackage / GeoJSON / Shapefile)
m = DuckDBMapMatcher.from_wkt_csv(
    "data/osm_edges.csv", "data/sweden_edges.csv",
    id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=30)

routes_long, routes_summary = m.match_routes(n_jobs=-1)   # parallel over A-edges
```

**Output — two tables:**

- **`routes_summary`** — one row per A-edge: the chosen route (`dest_ids`), average match distance,
  coverage, `route_geom_wkt`, and `match_type` (`1:1` / `1:N_ROUTE` / `NO_MATCH`).
- **`routes_long`** — one row per B-edge in each route (the result *divided per edge*): `seq` (order
  of matching), `edge_cover_pct` (% of A covered), `edge_b_used_pct` (% of the B-edge used), and the
  per-edge match distance and bearing.

**Key parameters:** `snap_tolerance_m` (junction snapping / connectivity), `step_meters` (sampling
density), `n_jobs` (parallel cores). (`trim_ends_m`, an optional end-edge remover, is off by default.)

**Local cost (`emission`).** Two modes. Defaults to `"point"` (point-to-point drift). Pass
`emission="segment"` for the **segment-to-segment** cost: one distance between the two segment
middles per matched (A-segment, B-arc) pair, with sliver-free pools and free junction crossings,
and the reported distances *are* those middle-to-middle distances. Because a middle-to-middle
distance is blind to a segment rotating about its own middle, pair `"segment"` with a
`bearing_weight` heading term (λ ≈ 1–5) — see [docs/weighted_emission.md](docs/weighted_emission.md).
`"point"` is unchanged, so existing results are unaffected.

`match_routes` returns a route for every A-edge; keep only confident matches by filtering on
thresholds (failures become `NO_MATCH`):

```python
routes_summary, routes_long = m.resolve_routes(
    routes_summary, routes_long, max_match_dist=10, max_bearing_diff=30, min_overlap_pct=95)
```

**Visualize** (standalone HTML written to `output/`):

```bash
python scripts/graph_dtw_map.py                          # whole-network map
python scripts/graph_dtw_edge_detail.py --edge-id 3597   # single-edge deep dive
```

**Debug the algorithm** (DP cost tables + backtracked path on synthetic cases or real edges,
and robustness sweeps under noise / shift / rotation / crop —
see [docs/graph_dtw_debugging.md](docs/graph_dtw_debugging.md)):

```bash
python scripts/graph_dtw_debug_viz.py --case parallel_trap --shift -6 --trace
python scripts/graph_dtw_perturb_test.py --case split      # or --edge-id 1377
```

**Play with the matching interactively** — the playground notebook shows *which A points
matched which B-edge* (colored correspondence links, no analysis panels), with sliders for
shift / rotate / noise / translate / crop and a build-your-own-network section (edges are plain
coordinate lists). Also runs as a standalone dashboard:

```bash
jupyter lab notebooks/graph_dtw_playground.ipynb   # notebook
voila notebooks/graph_dtw_playground.ipynb         # same file as a dashboard
```

Full pipeline reference (init, steps, output schemas, parameters):
[docs/graph_dtw_pipeline.md](docs/graph_dtw_pipeline.md). The algorithm itself (DTW on a directed
graph): [docs/graph_dtw_matching.md](docs/graph_dtw_matching.md).

---

## Mode 2 — Edge-to-edge matching

Pairwise matching: a three-tier pipeline behind `match()` that scores every nearby B-segment
against each A-segment, then ranks the candidates.

| Tier | What happens | Controlled by |
|------|--------------|---------------|
| 1. Candidate search | DuckDB spatial join finds every B-segment within `max_distance` of each A-segment. | `max_distance` |
| 2. Shape scoring | DTW measures average drift (`dtw_distance`), direction difference (`bearing_diff`), and coverage (`overlap_pct`). | — |
| 3. Reconciliation | Drop pairs failing `max_angle` / `min_overlap`; rank the survivors per source by `dtw_distance`. | `max_angle`, `min_overlap` |

```python
from network_matching import DuckDBMapMatcher

m = DuckDBMapMatcher.from_wkt_csv(
    "data/osm_edges.csv", "data/sweden_edges.csv",
    id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=25)
m.set_parameters(max_angle=45, min_overlap=50)        # optional quality filters (off by default)

results    = m.match()                                # all ranked candidates (+ NO_MATCH rows)
assignment = m.resolve(results, strategy="best_per_source")
print(assignment["match_type"].value_counts())
```

### Deciding the assignment (`resolve`)

`match()` is a candidate *generator* — for every source it keeps all qualifying destinations,
ranked by `dtw_distance`. `resolve()` commits to an assignment by **cardinality**:

| `strategy` | Cardinality | Use when |
|------------|-------------|----------|
| `"all"` | no decision | you'll apply your own logic |
| `"best_per_source"` *(default)* | many-to-one | assign each A to its closest B (e.g. sensors → roads) |
| `"best_per_dest"` | one-to-many | best representative A per B |
| `"one_to_one"` | global unique | unique segment-to-segment conflation |

Every source appears exactly once; unassigned ones come back as `NO_MATCH`. Read the decision from
the returned rows, not from `match_type`. For a **symmetric** (A→B and B→A) split/merge-aware
reconciliation, see [docs/symmetric_matching.md](docs/symmetric_matching.md).

### Result columns

`source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance, bearing_diff, overlap_pct,
rank, match_type`. `rank == 1` is the best match per source; `dtw_distance` (average drift in
meters) is the primary quality score. `match_type` is `1:1_SYMMETRIC`, `1:N_SPLIT`,
`UNIDIRECTIONAL_PARTIAL`, or `NO_MATCH`. Drop unmatched rows with
`results[results["match_type"] != "NO_MATCH"]`.

---

## Inputs

Geometry is assumed lon/lat (EPSG:4326) and transformed to your local projected `utm_srid`
(meters) during matching. The one-call initializers cover the common formats:

```python
DuckDBMapMatcher.from_wkt_csv(csv_a, csv_b, id_a=..., id_b=..., utm_srid=...)      # WKT CSVs
DuckDBMapMatcher.from_geofiles(file_a, file_b, id_a=..., id_b=..., utm_srid=...,   # GeoPackage /
                               src_srid=...)                                       # GeoJSON / Shapefile
```

Both accept `keep_cols_a` / `keep_cols_b` to carry extra attributes (e.g. `["name"]`) through.

For full control, build the matcher manually and call `configure_sources(...)`. The only part that
changes with your data is how the two tables are made available to `m.conn`:

- **WKT CSV** — `CREATE TABLE t AS SELECT id, ST_GeomFromText(geom) AS geom FROM 'file.csv';`
- **DuckDB databases** — `DuckDBMapMatcher(db_path_a=..., db_path_b=...)` (B attached read-only as `db_b`).
- **GeoPandas** — `m.conn.register("t", gdf)`.
- **GIS files in place** — point a source at `ST_Read('file.gpkg')`.

```python
m = DuckDBMapMatcher()
m.conn.execute("CREATE TABLE a AS SELECT id, ST_GeomFromText(geom) AS geom FROM 'data/a.csv';")
m.conn.execute("CREATE TABLE b AS SELECT id, ST_GeomFromText(geom) AS geom FROM 'data/b.csv';")
m.configure_sources(source_a="a", id_col_a="id", geom_col_a="geom",
                    source_b="b", id_col_b="id", geom_col_b="geom", utm_srid=3006)
m.set_parameters(max_distance=25)
```

---

## Documentation

| Document | Covers |
|----------|--------|
| [docs/graph_dtw_pipeline.md](docs/graph_dtw_pipeline.md) | Route-based pipeline — init, steps, output tables, parameters (start here for Mode 1). |
| [docs/graph_dtw_matching.md](docs/graph_dtw_matching.md) | Graph-DTW algorithm — DTW generalized to a directed graph. |
| [docs/dag_dtw_matching.md](docs/dag_dtw_matching.md) | DAG-DTW — DTW generalized so the *source* is a topologically-ordered DAG, matched jointly with consistent junctions (spec + point-to-point v1; demo `notebooks/dag_dtw_playground.ipynb`). |
| [docs/tree_dtw_matching.md](docs/tree_dtw_matching.md) | Tree-DTW — exact DTW matching when the *source* is a directed **tree** (branches and merges, no loops); single forward + backward pass, one number per cell. |
| [docs/weighted_emission.md](docs/weighted_emission.md) | Emission cost — point-to-point vs segment-to-segment (endpoint-average + optional bearing). |
| [docs/graph_dtw_debugging.md](docs/graph_dtw_debugging.md) | Algorithm debugging — `debug=True` internals, synthetic cases, perturbation-robustness tests. |
| [docs/dtw_matching.md](docs/dtw_matching.md) | DTW shape-alignment deep dive (Mode 2). |
| [docs/algorithm.md](docs/algorithm.md) | The three-tier edge-to-edge architecture. |
| [docs/symmetric_matching.md](docs/symmetric_matching.md) | Symmetric (two-way) split/merge reconciliation. |
| [docs/framework.md](docs/framework.md) | Software design. |

---

## Project layout

```
network_matching/   library — matcher, graph_dtw, synthetic (test cases), bgraph_prep, dtw
scripts/            CLI tools — graph_dtw_map.py, graph_dtw_debug_viz.py, graph_dtw_perturb_test.py, ...
notebooks/          demos — route tables, real-data plots, synthetic cases
docs/               documentation
tests/              pytest suite
data/               INPUT data only (osm_edges.csv, sweden_edges.csv, boundary)
output/             generated maps + result CSVs (git-ignored)
conflation_issues/  issue schema + area-specific curated issues
logs/               run logs (git-ignored)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

A dedicated conda env (`network-matching`, Python 3.11, with a registered Jupyter kernel) is
recommended for the notebooks and tests.
