# 🗺️ General-Purpose Directed Map-to-Map Matching Library

Welcome to the **`network-matching`** Python library — a high-performance, reusable map conflation engine that aligns and merges directed road networks (for example, matching an OpenStreetMap network against Sweden's NVDB network).

The engine combines **Dynamic Time Warping (DTW)** for shape alignment with **DuckDB Spatial** for fast candidate search, producing direction-aware, high-fidelity geometry matches — even when the two networks are segmented differently (e.g. one road in network A split into several pieces in network B, a `1:N` split).

---

## 🚀 Features

- **Direction-Aware (Directed) Matching** — uses travel bearings and coordinate sequences, so a road and its reverse-direction twin are not confused.
- **Progressive Overlap-Based DTW Alignment** — finds the natural overlap start and end dynamically, ignoring the non-overlapping tails of each segment without length assumptions or segment swapping.
- **Configurable Quality Thresholds** — candidate search radius (`max_distance`), maximum bearing difference (`max_angle`), and minimum overlap (`min_overlap`) are all tunable. By default the two quality filters are wide open, so every spatial candidate is evaluated; tighten them to discard weak matches.
- **Ranked Match Selection** — every qualifying candidate is kept and ranked by alignment quality. The best match for a given source is the row with `rank == 1`; split roads (`1:N`) cleanly preserve all their pieces.
- **Complete Output** — every source segment appears in the result, even those with no match (flagged `match_type == 'NO_MATCH'`), so nothing is silently dropped.
- **Flexible Input Sources** — CSV files, DuckDB tables, Pandas/GeoPandas DataFrames, and GIS files (GeoPackage, Shapefile, GeoJSON).

---

## 📦 Installation

From the root of the project:

```bash
pip install -e .
```

### Dependencies
- `duckdb` (with the spatial extension — loaded automatically)
- `numpy`
- `pandas`
- `shapely`
- `geopandas` (optional, only for the GeoDataFrame workflow)

---

## ⚙️ How It Works

Matching runs as a three-tier pipeline, all wrapped behind a single `match()` call:

| Tier | What happens | Controlled by |
|------|--------------|---------------|
| **1. Candidate search** | A spatial R-Tree join finds every Source-B segment within `max_distance` meters of each Source-A segment. | `max_distance` |
| **2. Shape scoring** | For each candidate pair, DTW measures the average geometric drift (`dtw_distance`), the direction difference (`bearing_diff`), and how much of the source is covered (`overlap_pct`). | — |
| **3. Reconciliation** | Candidates failing the `max_angle` / `min_overlap` thresholds are dropped; the survivors are ranked per source by `dtw_distance`. Sources with no surviving match get a `NO_MATCH` row. | `max_angle`, `min_overlap` |

---

## 🛠️ Quick Start

The core class is **`DuckDBMapMatcher`**. The general workflow is the same regardless of where your data lives:

> **1. Initialize → 2. Load / point to your two networks → 3. Configure column mappings → 4. (optional) Set thresholds → 5. `match()` (score candidates) → 6. `resolve()` (decide the assignment) → 7. Inspect or save.**

The column and table names below (`network_a`, `id`, `geom`, …) are **placeholders** — substitute whatever your own data uses. The matcher only needs to know, for each network: the table/source name, its **ID column**, and its **geometry column**.

```python
from network_matching import DuckDBMapMatcher

# 1. Initialize the matcher (clean in-memory DuckDB by default)
matcher = DuckDBMapMatcher()

# 2. Make your two networks available to the matcher's connection.
#    (Here we load them from CSV; see the other input cases below.)
matcher.conn.execute("""
    CREATE TABLE network_a AS
    SELECT id::BIGINT AS id, ST_GeomFromText(geom) AS geom
    FROM 'data/network_a.csv';
""")
matcher.conn.execute("""
    CREATE TABLE network_b AS
    SELECT id::BIGINT AS id, ST_GeomFromText(geom) AS geom
    FROM 'data/network_b.csv';
""")

# 3. Tell the matcher which columns hold the ID and geometry of each network.
#    `utm_srid` is a LOCAL projected (metre-based) CRS so distances are in meters.
matcher.configure_sources(
    source_a="network_a", id_col_a="id", geom_col_a="geom",
    source_b="network_b", id_col_b="id", geom_col_b="geom",
    utm_srid=3006,  # e.g. SWEREF99 TM for Sweden; use the right EPSG for your area
)

# 4. (Optional) Tune the matching thresholds. Any argument you omit keeps its default.
matcher.set_parameters(
    max_distance=25.0,   # candidate search radius, meters          (default 25.0)
    max_angle=180.0,     # max bearing difference, degrees 0–180    (default 180.0 = off)
    min_overlap=0.0,     # min overlap percentage, 0–100            (default 0.0 = off)
)

# 5. Run the full pipeline (directed: A → B). Source-A segments with no match
#    come back as NO_MATCH rows.
results = matcher.match()

# 6. Decide the final assignment. match() returns ALL ranked candidates; resolve()
#    commits to one according to your problem (see "Making the Decision" below).
assignment = matcher.resolve(results, strategy="best_per_source")  # many-to-one

# 7. Inspect …
print(assignment["match_type"].value_counts())

# … or save to CSV.
matcher.conn.register("final_matches", assignment)
matcher.conn.execute(
    "COPY final_matches TO 'data/conflation_results.csv' (HEADER, DELIMITER ',');"
)
```

### Tuning the thresholds

| Parameter | Unit | Default | Effect |
|-----------|------|---------|--------|
| `max_distance` | meters | `25.0` | Tier 1 search radius — how far apart two segments can be and still be considered candidates. |
| `max_angle` | degrees, `0–180` | `180.0` | Tier 3 filter — drop pairs whose travel directions differ by more than this. `180` disables it. |
| `min_overlap` | percent, `0–100` | `0.0` | Tier 3 filter — drop pairs covering less than this fraction of the source's length. `0` disables it. |

With the defaults, the two quality filters are off and **every** candidate within `max_distance` is matched and ranked. Tighten `max_angle` (e.g. `45`) and `min_overlap` (e.g. `50`) to keep only confident matches.

---

## 🎯 Making the Decision (`resolve`)

`match()` is a **candidate generator**, not a decision maker. For every source it keeps *all*
qualifying destinations, ranked by `dtw_distance` — it deliberately does **not** commit to "this
A goes to that B," because the right answer depends on your problem's **cardinality**:

> *Example:* if A is a list of sensor locations and B is road segments, two sensors in different
> lanes should both map to the **same** road — a **many-to-one** assignment is correct. But for
> unique segment-to-segment network conflation, you want each segment used **once** — a
> **one-to-one** assignment.

`resolve()` applies that decision to the table from `match()`:

```python
assignment = matcher.resolve(results, strategy="best_per_source")
```

| `strategy` | Cardinality | Each source → | Each dest → | Use when |
|------------|-------------|---------------|-------------|----------|
| `"all"` | none (no decision) | all candidates | reused | you'll apply your own logic |
| `"best_per_source"` *(default)* | **many-to-one** | its 1 closest dest | may be shared | **sensors → roads**; assign each A to the road it's on |
| `"best_per_dest"` | **one-to-many** | may be shared | its 1 closest source | "best representative A per B" |
| `"one_to_one"` | **global unique** | its 1 unique dest | its 1 unique source | unique segment-to-segment conflation |

`"one_to_one"` accepts pairs greedily from the smallest `dtw_distance` upward, so the globally
closest pairs win and conflicting weaker pairs are dropped (a fast approximation of optimal
assignment). Whatever the strategy, **every source still appears exactly once** — any source left
unassigned comes back as a `NO_MATCH` row.

> ⚠️ After resolving, read the decision from the **returned rows themselves**, not from
> `match_type` — that column still describes the *original* candidate fan-out, not the resolved
> cardinality.

---

## 📥 Input Source Variations

Step 2 above is the only part that changes with your data format; everything else (`configure_sources` → `match`) is identical.

### Case 1 — Local CSV files with WKT geometries
For flat CSVs whose geometry column is Well-Known Text (e.g. `LINESTRING (312345 6123456, ...)`):

```python
matcher = DuckDBMapMatcher()
matcher.conn.execute("""
    CREATE TABLE network_a AS
    SELECT id::BIGINT AS id, ST_GeomFromText(geom) AS geom FROM 'data/network_a.csv';
""")
matcher.conn.execute("""
    CREATE TABLE network_b AS
    SELECT id::BIGINT AS id, ST_GeomFromText(geom) AS geom FROM 'data/network_b.csv';
""")
matcher.configure_sources(
    source_a="network_a", id_col_a="id", geom_col_a="geom",
    source_b="network_b", id_col_b="id", geom_col_b="geom",
    utm_srid=3006,
)
results = matcher.match()
```

### Case 2 — Physical DuckDB databases (ATTACH)
Connect to database A and attach database B read-only to avoid file locking:

```python
matcher = DuckDBMapMatcher(
    db_path_a="path/to/network_a.duckdb",
    db_path_b="path/to/network_b.duckdb",   # attached under the alias 'db_b'
)
matcher.configure_sources(
    source_a="schema_a.edges",      id_col_a="id", geom_col_a="geom",
    source_b="db_b.main.edges",     id_col_b="id", geom_col_b="geom",
    utm_srid=3006,
)
results = matcher.match()
```

### Case 3 — GeoPandas / GeoDataFrames
Register in-memory GeoDataFrames directly into the matcher's connection:

```python
import geopandas as gpd

gdf_a = gpd.read_file("network_a.gpkg", layer="edges")
gdf_b = gpd.read_file("network_b.shp")

matcher = DuckDBMapMatcher()
matcher.conn.register("network_a", gdf_a)
matcher.conn.register("network_b", gdf_b)
matcher.configure_sources(
    source_a="network_a", id_col_a="id", geom_col_a="geometry",
    source_b="network_b", id_col_b="id", geom_col_b="geometry",
    utm_srid=3006,
)
results_df = matcher.match()
```

### Case 4 — GIS files via DuckDB `ST_Read` (no Python load)
Point the sources straight at on-disk spatial files:

```python
matcher = DuckDBMapMatcher()
matcher.configure_sources(
    source_a="ST_Read('data/network_a.gpkg')",    id_col_a="fid", geom_col_a="geom",
    source_b="ST_Read('data/network_b.geojson')", id_col_b="id",  geom_col_b="geom",
    utm_srid=3006,
)
results = matcher.match()
```

---

## 📈 Result Columns Explained

`match()` returns a Pandas DataFrame with one row per matched source→destination pair (plus one `NO_MATCH` row for each source that matched nothing):

| Column | Meaning |
|--------|---------|
| **`source_id`** | ID of the source segment (network A). |
| **`dest_id`** | ID of the matched destination segment (network B). `None` for `NO_MATCH` rows. |
| **`dtw_distance`** | Average geometric drift in meters along the DTW warping path — *the* primary match-quality score (lower = better). |
| **`max_dtw_distance`** | Largest point-to-point offset along the warping path (meters). |
| **`min_dtw_distance`** | Smallest point-to-point offset along the warping path (meters). |
| **`bearing_diff`** | Absolute travel-direction difference, `0–180°` (0 = same direction). |
| **`overlap_pct`** | Integer percentage of the source segment's length covered by the aligned section, `0–100` (nullable `Int64`; `<NA>` for `NO_MATCH` rows). |
| **`rank`** | Rank of this destination among the source's candidates, by `dtw_distance`. **`rank == 1` is the best match.** `NaN`/`<NA>` for `NO_MATCH` rows. |
| **`match_type`** | Category of the match (see below). |

### `match_type` values
- **`1:1_SYMMETRIC`** — the source matched exactly one destination (a clean one-to-one pairing).
- **`1:N_SPLIT`** — the source matched several destinations (one road split across multiple pieces in the other network). *All* rows of such a source carry this label; use `rank == 1` to pick the best piece.
- **`UNIDIRECTIONAL_PARTIAL`** — fallback label for a lower-confidence single candidate.
- **`NO_MATCH`** — the source had no candidate survive (no nearby segment, or none passing the thresholds). `dest_id` is `None` and all metric columns are `NaN`.

> **Committing to a final assignment:** use [`resolve()`](#-making-the-decision-resolve) rather than filtering by hand — e.g. `matcher.resolve(results, strategy="best_per_source")`. **Dropping unmatched rows:** filter to `results[results["match_type"] != "NO_MATCH"]`.
