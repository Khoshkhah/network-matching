# 🗺️ General-Purpose Directed Map-to-Map Matching Library

Welcome to the **`network-matching`** Python library! This is a high-performance, reusable, and general-purpose map conflation engine designed to align and merge directed road networks (such as OpenStreetMap and Sweden's NVDB network).

The engine leverages **Dynamic Time Warping (DTW)** and **DuckDB Spatial** to perform direction-aware, high-fidelity geometry matching—even when road networks have different segmentations (e.g., $1:N$ split roads).

---

## 🚀 Features

- **Direction-Aware (Directed) Matching**: Uses travel bearings and coordinate sequences to prevent matching against opposite lanes or incorrect travel directions.
- **Progressive Overlap-Based DTW Shape Alignment**: Identifies natural overlap starts and boundaries dynamically, ignoring pre-overlap and post-overlap tails without relying on arbitrary length classification or segment swapping.
- **Ranked Match Selection**: Evaluates all qualifying candidate matches, sorting and ranking them by alignment quality. Handles split parallel roads cleanly, flagging the absolute best matching target (`is_best = True`) while preserving all qualifying alternatives.
- **Optional Bidirectional Reconstruction**: Run matching independently in both directions ($A \to B$ and $B \to A$) and obtain a unified, reciprocal network-to-network conflation table.
- **Extremely Flexible Input Sources**: Connects seamlessly to CSV files, DuckDB tables, Pandas/GeoPandas DataFrames, and standard GIS files (GeoPackage, Shapefile, GeoJSON).

---

## 📦 Installation

To make the library installable in your active environment, run the following from the root directory of the project:

```bash
pip install -e .
```

### Dependencies
The library requires standard, lightweight Python numerical and spatial packages:
- `duckdb` (with spatial extension)
- `numpy`
- `pandas`
- `shapely`
- `geopandas` (optional, for GeoPandas usage)

---

## 🛠️ Usage Guides for Different Input Types

The core class is `DuckDBMapMatcher`. It unifies Tier 1 spatial R-Tree filtering, Tier 2 Python DTW alignment, and Tier 3 SQL window-function reconciliation under a single method call: `match()`.

### Case 1: Matching Local CSV Files (WKT Geometries)
If your datasets are stored as flat CSV files with coordinate sequences represented as Well-Known Text (WKT) (e.g. `LINESTRING (312345 6123456, ...)`), you can load them directly into an in-memory database:

```python
from network_matching import DuckDBMapMatcher

# 1. Initialize map matcher (uses clean, fast in-memory DuckDB by default)
matcher = DuckDBMapMatcher()

# 2. Parse WKT geometry columns from CSV into active in-memory spatial tables
matcher.conn.execute("""
    CREATE TABLE driving_edges AS
    SELECT 
        edge_id::BIGINT AS edge_id,
        ST_GeomFromText(geometry) AS geometry
    FROM 'data/osm_edges.csv';
""")

matcher.conn.execute("""
    CREATE TABLE vehicle_edges_directed AS
    SELECT 
        directed_id::BIGINT AS directed_id,
        ST_GeomFromText(geometry) AS geometry
    FROM 'data/sweden_edges.csv';
""")

# 3. Configure matcher sources
matcher.configure_sources(
    source_a="driving_edges", id_col_a="edge_id", geom_col_a="geometry",
    source_b="vehicle_edges_directed", id_col_b="directed_id", geom_col_b="geometry",
    utm_srid=3006  # Local projected metric system (e.g., SWEREF99 TM for Sweden)
)

# 4. Run Map Matching (directed match A -> B)
results = matcher.match(bidirectional=False)

# 5. Export matching results directly to a local CSV file!
matcher.conn.register("final_matches", results)
matcher.conn.execute("COPY final_matches TO 'data/conflation_results.csv' (HEADER, DELIMITER ',');")
```

---

### Case 2: Matching Direct DuckDB Tables (Multi-Database Connection)
If your datasets are stored inside physical `.duckdb` files, you can connect to one and ATTACH the other read-only to avoid file locking:

```python
from network_matching import DuckDBMapMatcher

# Connect to Database A, and attach Database B under alias 'db_b'
matcher = DuckDBMapMatcher(
    db_path_a="path/to/osm.duckdb",
    db_path_b="path/to/sweden.duckdb"
)

# Configure sources using database table paths
matcher.configure_sources(
    source_a="driving.edges", id_col_a="edge_id", geom_col_a="geometry",
    source_b="db_b.main.vehicle_edges", id_col_b="edge_id", geom_col_b="geometry",
    utm_srid=3006
)

# Run full bidirectional map conflation
results = matcher.match(bidirectional=True)

# Save results directly to a permanent table in the attached database B
matcher.conn.register("final_matches", results)
matcher.conn.execute("CREATE OR REPLACE TABLE db_b.main.conflation_results AS SELECT * FROM final_matches;")
```

---

### Case 3: Using GeoPandas / GeoDataFrames
If you are already working with GeoDataFrames in your Python code, you can easily register them as temporary tables inside the map matcher's connection:

```python
import geopandas as gpd
from network_matching import DuckDBMapMatcher

# Load standard GIS files (GeoPackage, Shapefile, or GeoJSON)
gdf_osm = gpd.read_file("osm_network.gpkg", layer="edges")
gdf_nvdb = gpd.read_file("nvdb_network.shp")

# Initialize matcher
matcher = DuckDBMapMatcher()

# Register GeoDataFrames directly inside DuckDB
matcher.conn.register("osm_layer", gdf_osm)
matcher.conn.register("nvdb_layer", gdf_nvdb)

# Configure matcher
matcher.configure_sources(
    source_a="osm_layer", id_col_a="edge_id", geom_col_a="geometry",
    source_b="nvdb_layer", id_col_b="edge_id", geom_col_b="geometry",
    utm_srid=3006
)

# Execute matching and retrieve a standard Pandas DataFrame
results_df = matcher.match(bidirectional=True)
```

---

### Case 4: Reading Direct GIS Files via DuckDB Spatial (GPKG, Shapefiles, GeoJSON)
DuckDB Spatial's `ST_Read` is incredibly fast. You can use it to query spatial files on disk without even loading them in Python memory first!

```python
from network_matching import DuckDBMapMatcher

matcher = DuckDBMapMatcher()

# Configure sources directly pointing to the ST_Read commands
matcher.configure_sources(
    source_a="ST_Read('data/osm_roads.gpkg')", id_col_a="fid", geom_col_a="geom",
    source_b="ST_Read('data/sweden_roads.geojson')", id_col_b="edge_id", geom_col_b="geom",
    utm_srid=3006
)

# Run map matching pipeline
results = matcher.match(bidirectional=False)
```

---

## 📈 Match Result Fields Explained

The output table contains the following columns:
- **`source_id`**: Identifier of the Source segment (A if directed, A or B if bidirectional).
- **`dest_id`**: Identifier of the matched Destination segment (B if directed, B or A if bidirectional).
- **`dtw_distance`**: Average physical offset (drift) in meters calculated along the progressive DTW warping path.
- **`max_dtw_distance` / `min_dtw_distance`**: Maximum and minimum alignment offsets (meters).
- **`bearing_diff`**: Absolute difference in travel direction (0 - 180 degrees).
- **`overlap_pct`**: The coverage percentage representing what proportion of the Source segment's length was aligned.
- **`rank`**: The rank of this destination for the respective source (rank 1 is the closest).
- **`is_best`**: Boolean flag indicating whether this is the absolute closest qualifying destination (rank == 1).
- **`direction`** *(only when `bidirectional=True`)*: Indicates matching direction (`A_to_B` or `B_to_A`).
