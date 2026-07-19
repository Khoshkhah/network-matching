# network-matching

A reusable, high-performance library for **directed road map-to-map matching (conflation)** —
aligning two road networks (for example OpenStreetMap against Sweden's NVDB), even when they are
segmented differently. It combines **Dynamic Time Warping (DTW)** for shape alignment with
**DuckDB Spatial** for fast candidate search, and works in a local projected CRS so all distances
are in meters.

The library offers **four matchers**, all behind the same `DuckDBMapMatcher` class — same
inputs (WKT CSV / geofiles / DuckDB tables), same CRS handling, same output conventions
(ranked rows, `match_type`, `NO_MATCH`). Mode 3 (DAG-DTW) additionally has a standalone
`networkx` API:

| Mode | Entry point | Produces | Best for |
|------|-------------|----------|----------|
| **Route-based (graph-DTW)** | `m.match_routes()` | each A-edge → a connected **route** of B-edges | conflating networks split differently; the recommended default |
| **Edge-to-edge** | `m.match()` + `m.resolve()` | ranked A↔B candidate **pairs** + a cardinality decision | assigning edges to a single nearest segment; fine-grained control |
| **DAG-to-network (DAG-DTW)** | `m.match_dag()` | an **exact**, validity-checked matching of the whole source DAG onto the target, as Mode-1-style tables | matching a DAG-shaped subnetwork (a route tree, a sensor cone, a divided road that rejoins) in one globally consistent pass |
| **Point-to-edge** | `m.match_points()` + `m.resolve()` | each A-**point** → ranked nearby B-edges, with snap position and local road bearing | assigning sensors / stations / stops to the road they sit on |

---

## Installation

```bash
pip install -e .                 # core library
pip install -r requirements.txt  # core + visualization + notebook tooling
# extras:  pip install -e ".[viz]"   (maps)     pip install -e ".[dev]"   (tests, notebooks)
```

**Dependencies.** Core: `duckdb` (spatial extension auto-loaded), `numpy`, `pandas`, `shapely`,
`geopandas`, `networkx` (Mode 3), `joblib` (parallel route matching). Visualization
(scripts/notebooks): `folium`, `branca`, `matplotlib`, `plotly` (Mode 3 playground). Optional:
`scipy` (faster candidate gating and endpoint validation; pure-numpy fallback).

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

## Mode 3 — DAG-to-network matching (DAG-DTW)

An **exact** matcher for a directed source **DAG** — branches, merges, reconvergences (diamonds,
divided roads) all legal; only a directed cycle is rejected (`NotADAG`) — against any directed
target network (cycles allowed). Where Modes 1–2 match each A-edge independently, Mode 3 matches
the **whole source structure in one globally consistent pass**: the result is a matching relation
validated by four warping rules (V1–V4) and minimized by direct cost. Weights: `alpha ∈ (0, 1]`
discounts 1:N coverage, `beta ∈ [1, ∞)` penalizes N:1 stalls.

Same inputs and output shape as the other modes:

```python
from network_matching import DuckDBMapMatcher

m = DuckDBMapMatcher.from_wkt_csv(
    "data/osm_edges.csv", "data/sweden_edges.csv",
    id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=30)

dag_long, dag_summary = m.match_dag(alpha=0.5, beta=1.5)   # engine="cell" (exact) by default
```

Geometry is transformed to `utm_srid` and converted to `networkx` graphs internally (each
polyline densified at `step_meters` — this supplies the required subdivision; shared endpoints
become junctions). Matching runs at **segment resolution** (arc states with a `bearing_weight`
heading term, default 2.0). Output is the Mode-1-style pair:

- **`dag_summary`** — one row per A-edge: `dest_ids` (ordered `;`-join), `n_dest`, `n_parts`,
  `n_pairs`, `avg_dist_m`, `avg_bearing_diff`, `match_type` (`1:1` / `1:N_ROUTE`).
- **`dag_long`** — one row per matched (A-edge, B-edge): `seq` (order along the A-edge),
  `n_pairs` (matched arc pairs), `avg_dist_m` (mean midpoint drift, meters), `avg_bearing_diff`.

Pass `parts=True` for the third table, **`dag_parts`** — the per-edge decomposition into
contiguous **parts** (one row per stretch of the A-edge matched to one B-edge, in order, re-entry
kept separate), plus a dedicated row for the route's begin and end **non-overlap** when the
A-edge extends past the B coverage (`part_type` = `head` / `match` / `tail`, also summarized as
`a_head_m`/`a_tail_m` on `dag_summary`): `a_from_m/a_to_m` span along A, per-part `n_pairs`,
`drift_m` and `bearing_diff_deg` scores, and the used B span `b_from_m/b_to_m` with the
non-overlapping `b_head_m`/`b_tail_m` leftovers — everything needed to compose a whole-edge
score your own way (see [docs/dag_dtw_matching.md](docs/dag_dtw_matching.md) §11):

```python
dag_long, dag_summary, dag_parts = m.match_dag(alpha=0.5, beta=1.5, parts=True)
```

**Standalone `networkx` API** — plain `DiGraph` inputs whose nodes carry projected `x, y` in
meters, no DuckDB involved; the source must be **subdivided** (≥ 1 interior point per real edge):

```python
from network_matching import match_dag

# A: source DAG (nodes carry x, y; subdivided; diamonds OK). B: target network (may cycle).
M, committed = match_dag(A, B, r=20.0, alpha=0.5, beta=1.5)         # point mode, M ⊆ V(A)×V(B)
M_seg, _     = match_dag(A, B, r=20.0, mode="segment",              # arc mode: nodes are (u, v)
                          bearing_weight=2.0, engine="all")          # edge tuples of the originals
```

Feasibility failures never return a broken matching — they raise `ValueError` telling you to
increase the radius (`max_distance` / `r`), and every returned matching has passed the V1–V4
validity judge.

Three **cross-validating extraction engines** share one validity judge (rules V1–V4) and one cost:
`engine="cell"` (the cell-level join — exact over the full space; default), `"branch"` (branching
exploration), `"join"` (vertex-level junction join), or `"all"` (run all three, return the cheapest
valid matching). On the structured 384-case envelope the cell engine is valid **384/384** and never
costlier than either other engine; on reconvergent DAG sources it is verified exact 195/195 vs
full-space brute force. Spec (algorithm + all three engines):
[docs/dag_dtw_matching.md](docs/dag_dtw_matching.md).

**Play with it interactively** — scenarios, the historical failure demos and their fixes:

```bash
jupyter lab notebooks/dag_dtw_playground.ipynb          # interactive Plotly playground
python scripts/dag_dtw_debug_viz.py --case diamond      # debug view: cell states (alive /
                                                         # forbidden / removed / D=∞) + engine
                                                         # comparison with a dropdown (HTML to output/)
python scripts/test_dag_point.py                        # three-engine cross-validation sweep
                                                         # (structure × density × shift × noise × weights)
```

---

### The profiled forward table — `engine="auto"` (default)

`network_matching/profiled.py`. Since `engine="auto"` became the default, `match_dag` **dispatches to
it automatically** on sources where it wins. It exists because
the Mode-3 forward table is *optimistic at splits*: where several exit cells are usable by every
child, each child's row picks its own cheapest and the trace can place one split on two cells at
once. Measured on a real conflation corpus, that happens on the two slowest edges of four.

Carrying a cost **per profile** — per placement of the upstream splits — prices a split's children
jointly and blocks it at construction. On those four edges, against `extract_cell`:

| edge | current | profiled | V3 violations |
|---|---|---|---|
| 100042 | 4.8 s · 63 MB | 0.03 s · 3 MB | 0 → 0 |
| 102752 | 30.0 s · 248 MB | 0.50 s · 16 MB | **2 → 0** |
| 100341 | 33.4 s · 215 MB | 0.03 s · 4 MB | 0 → 0 |
| 100350 | **687.7 s · 783 MB** | **0.21 s · 14 MB** | **3 → 0** |

It also answers **168 of 900** random cyclic-target cases that `extract_cell` refuses with a
spurious *"no valid root row"* — the open contraction-eviction defect in
`scripts/repro_contraction_eviction/`, resolved as a side effect of keying rows on profiles rather
than on merge signatures.

**How `auto` chooses.** Per call, from the source's shape, before any matching work:

```python
engine = "profiled" if profiled_width(A) <= 2 else "cell"
```

`profiled_width` is the **maximum live profile keys at any vertex** — one `O(V+E)` topological
sweep. Width ≤ 2 means splits are disjoint or discharge quickly at a merge, and the profiled engine
wins; width ≥ 3 means splits nested with no merge between them, where its key grows with depth and
`extract_cell` is faster. It predicts the engine's actual stored width exactly on every test family.
Every real conflation edge measured is width 2.

```python
M, committed = match_dag(A, B, r=20.0, alpha=0.5, beta=1.5)                   # auto (default)
M, committed = match_dag(A, B, r=20.0, alpha=0.5, beta=1.5, engine="cell")    # force the classic
```

**Where it is worse, and the `rebase` variant.** On nested splits the key grows with depth
(usable to depth 4), which is why `auto` routes those to `"cell"`. `engine="rebase"` re-bases costs
to *"since the last split"*, holding width 1 to depth 7+ — but it is slower than `"cell"` there too,
so it is never chosen automatically and exists for shapes that would otherwise fail.

Gates: unit suite 198; structured envelope and cyclic-B both 384/384 and 487/487 cost parity
**through `match_dag(engine="auto")`**, 0 invalid, 109 cases answered where the classic engine
refuses; all four hourglass edges exact end-to-end. Reproduce with the probes in `report/`.

---

## Mode 4 — Point-to-edge matching

The simple sibling of Mode 2 for **POINT** sources: assign each point (a sensor, a measurement
station, a stop) to the road edges near it. Same three-tier flow — DuckDB candidate search within
`max_distance`, plain-geometry scoring (no DTW), ranking — and the same `resolve()` decision step
(it ranks by `distance_m` here):

```python
from network_matching import DuckDBMapMatcher

m = DuckDBMapMatcher.from_wkt_csv(
    "data/sensors.csv", "data/osm_edges.csv",              # A = POINT WKT, B = LINESTRING WKT
    id_a="sensor_id", id_b="edge_id", utm_srid=3006, max_distance=25)

points     = m.match_points()                              # all ranked candidates (+ NO_MATCH rows)
assignment = m.resolve(points, strategy="best_per_source")  # each sensor -> its nearest road
```

One row per (point, candidate edge): `distance_m` (lateral drift), `position_pct` (where along
the edge the point snaps, 0–100), `edge_bearing_deg` (the road's direction at the snap point —
use it to pick the right edge of a divided road), `snap_wkt` (the snapped point, `utm_srid`
meters), `rank` (1 = nearest), `match_type` (`1:1` / `1:N_CANDIDATES` / `NO_MATCH`). Reference:
[docs/point_matching.md](docs/point_matching.md).

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
| [docs/dag_dtw_matching.md](docs/dag_dtw_matching.md) | DAG-DTW (Mode 3) — the complete spec: forward table with the split (V3) coupling, the three extraction engines (§5 branching, §10 vertex & **cell-level joins** — exact over the full space), validity rules V1–V4, point & segment modes, DAG sources. |
| [docs/profiled_forward_table.md](docs/profiled_forward_table.md) | The profiled forward table — a cost *per profile* (where the upstream splits are placed), so a split's children are priced jointly and the V3 phantom is blocked at construction. `match_dag`'s default engine on width ≤ 2 sources; §8 covers the `rebase` variant. |
| [docs/weighted_emission.md](docs/weighted_emission.md) | Emission cost — point-to-point vs segment-to-segment (endpoint-average + optional bearing). |
| [docs/graph_dtw_debugging.md](docs/graph_dtw_debugging.md) | Algorithm debugging — `debug=True` internals, synthetic cases, perturbation-robustness tests. |
| [docs/dtw_matching.md](docs/dtw_matching.md) | DTW shape-alignment deep dive (Mode 2). |
| [docs/point_matching.md](docs/point_matching.md) | Point-to-edge matching (Mode 4) — pipeline, output schema, `resolve`. |
| [docs/algorithm.md](docs/algorithm.md) | The three-tier edge-to-edge architecture. |
| [docs/symmetric_matching.md](docs/symmetric_matching.md) | Symmetric (two-way) split/merge reconciliation. |
| [docs/threshold_estimation.md](docs/threshold_estimation.md) | Data-driven match-quality thresholds (`suggest_thresholds`). |
| [docs/framework.md](docs/framework.md) | Software design. |

---

## Project layout

```
network_matching/   library — matcher, graph_dtw, dag_dtw (Mode 3), profiled (Mode 3 default engine), synthetic, bgraph_prep, dtw
scripts/            CLI tools — graph_dtw_map.py, graph_dtw_debug_viz.py, dag_dtw_debug_viz.py + test_dag_point.py (Mode 3), ...
notebooks/          playgrounds — graph_dtw_playground.ipynb (Mode 1), dag_dtw_playground.ipynb (Mode 3)
docs/               documentation
report/             measured results + the probes that reproduce them
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
