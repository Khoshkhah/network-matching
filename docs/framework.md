# Software Architecture & Reusable Framework

This document outlines the software design, class specifications, and execution lifecycle of the **`network-matching`** framework. It describes how the project combines **In-Memory DuckDB Spatial SQL** and **Python DTW** into a universal, reusable map-matching package.

---

## 1. Architectural Philosophy

To build a general and reusable tool, we decouple the **storage format** from the **computation engine**:

1. **In-Memory DuckDB Engine**: Rather than binding the software to static database files, the framework uses an in-memory DuckDB connection (`duckdb.connect(':memory:')`) loaded with the `spatial` extension as its core computation engine.
2. **File-Format Agnostic**: DuckDB's ability to query raw files directly allows the framework to support **DuckDB tables, CSV files (WKT geometries), Parquet files, and Shapefiles** as inputs without altering the SQL logic.
3. **Hybrid Processing (Tier 1 SQL + Tier 2 Python + Tier 3 SQL)**:
   * **SQL** is used for spatial indices and joins (Tier 1) because it is highly parallel and optimized in C++.
   * **Python** is used for progressive DTW shape-alignment (Tier 2) because recursive time-series matrix algorithms are best expressed in Python/NumPy.
   * **SQL** is used again for directional ranking and post-processing (Tier 3).

---

## 2. System Data Flow

```
                      +-----------------------------+
                      | Input Sources (DB/CSV/Parq) |
                      +-----------------------------+
                                     |
                                     v (Configured SQL URIs)
  +-------------------------------------------------------------------+
  |                  IN-MEMORY DUCKDB ENGINE                          |
  |                                                                   |
  |  1. TIER 1: Bounding-box spatial join (ST_DWithin)                |
  |     Result: Unique Candidate Pairs                                |
  |                                                                   |
  +-------------------------------------------------------------------+
                                     |
                                     v (Coordinate lists fetched to Python)
  +-------------------------------------------------------------------+
  |                       PYTHON MATH ENGINE                          |
  |                                                                   |
  |  2. TIER 2: Progressive DTW Shape Alignment                       |
  |     - Bounded Projection Dynamic Programming                      |
  |     - Aligned Overlap Percentage                                  |
  |     - Direction Alignment & Heading Checks                        |
  |                                                                   |
  +-------------------------------------------------------------------+
                                     |
                                     v (Pushed back to DuckDB table)
  +-------------------------------------------------------------------+
  |                  IN-MEMORY DUCKDB ENGINE                          |
  |                                                                   |
  |  3. TIER 3: Directional Ranking & Reconciliation                  |
  |     - Filters & ranks candidates; emits NO_MATCH for unmatched    |
  |                                                                   |
  +-------------------------------------------------------------------+
                                     |
                                     v
                        +---------------------------+
                        | Output Matches (CSV/GPKG) |
                        +---------------------------+
```

---

**Beside this DuckDB flow** sits one standalone matcher: **DAG-DTW** (`match_dag`,
`network_matching/dag_dtw.py`) — pure `networkx`, no database; a directed source *tree or subdivided DAG* matched
exactly onto a directed target network, with three cross-validating extraction engines. See
`docs/dag_dtw_matching.md`.

## 3. Class API Reference: `DuckDBMapMatcher`

Below is the complete, reusable interface for the map matching module.

```python
import duckdb
import pandas as pd

class DuckDBMapMatcher:
    def __init__(self, db_path_a: Optional[str] = None, db_path_b: Optional[str] = None):
        """
        Initializes an in-memory DuckDB instance and installs/loads 
        the spatial extension.
        """
        self.conn = duckdb.connect(database=':memory:')
        self.conn.execute("INSTALL spatial; LOAD spatial;")
        
        # Source configurations
        self.source_a = None
        self.source_b = None
        self.columns_a = {}
        self.columns_b = {}
        self.utm_srid = None
        
        # Default thresholds
        self.max_distance = 25.0       # search radius in meters to find candidate segments
        self.max_angle = 180.0         # max bearing difference (deg); 180 = no angle filter
        self.min_overlap = 0.0         # min overlap percent (0-100); 0 = no overlap filter
        
    def configure_sources(self, 
                          source_a: str, id_col_a: str, geom_col_a: str,
                          source_b: str, id_col_b: str, geom_col_b: str,
                          utm_srid: int):
        """
        Sets the tables or files to match and configures column mappings.
        """
        self.source_a = source_a
        self.source_b = source_b
        self.columns_a = {"id": id_col_a, "geom": geom_col_a}
        self.columns_b = {"id": id_col_b, "geom": geom_col_b}
        self.utm_srid = utm_srid

    def set_parameters(self, max_distance=None, max_angle=None, min_overlap=None):
        """
        Override matching thresholds. Any argument left as None keeps its default.
        - max_distance: Tier 1 candidate search radius (meters).
        - max_angle:    Tier 3 max bearing difference (degrees, 0-180).
        - min_overlap:  Tier 3 min aligned overlap (percent, 0-100).
        """
        if max_distance is not None:
            self.max_distance = max_distance
        if max_angle is not None:
            self.max_angle = max_angle
        if min_overlap is not None:
            self.min_overlap = min_overlap
```

---

## 4. The 4-Stage Execution Lifecycle

### Stage 1: Candidate Generation (Tier 1)
Using DuckDB's spatial index, the engine queries the unique candidates. This query does not double-calculate:

```python
    def generate_candidate_pairs(self) -> pd.DataFrame:
        """
        Runs an in-memory spatial join to retrieve unique candidate pairs.
        Returns a Pandas DataFrame of [id_a, wkt_a, id_b, wkt_b] pairs.
        """
        query = f"""
            SELECT DISTINCT
                A.{self.columns_a['id']} AS id_a,
                ST_AsText(ST_Transform(A.{self.columns_a['geom']}, 'EPSG:4326', 'EPSG:{self.utm_srid}')) AS wkt_a,
                B.{self.columns_b['id']} AS id_b,
                ST_AsText(ST_Transform(B.{self.columns_b['geom']}, 'EPSG:4326', 'EPSG:{self.utm_srid}')) AS wkt_b
            FROM {self.source_a} AS A, {self.source_b} AS B
            WHERE ST_DWithin(
                ST_Transform(A.{self.columns_a['geom']}, 'EPSG:4326', 'EPSG:{self.utm_srid}'),
                ST_Transform(B.{self.columns_b['geom']}, 'EPSG:4326', 'EPSG:{self.utm_srid}'),
                {self.max_distance}
            );
        """
        return self.conn.execute(query).df()
```

### Stage 2: DTW Distance Calculation (Tier 2)
The coordinate lists for each candidate pair are loaded into Python. The DTW algorithm computes the progressive DTW alignment and coverage percentage:

```python
    def compute_dtw_metrics(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches geometries for the candidate pairs, runs the 2D progressive DTW algorithm 
        in Python, and returns a DataFrame of evaluated metrics:
        [id_a, id_b, dtw_distance, bearing_diff, overlap_pct]
        """
        # 1. Parse WKT to Shapely LineString
        # 2. Call dtw_align(coords_a, coords_b) to retrieve alignment distance and overlap_pct
        # 3. Compute travel bearing mismatch
        # 4. Return the populated metrics DataFrame
        pass
```

### Stage 3: Reconciliation & Classification (Tier 3)
The evaluated metrics are pushed back to the in-memory DuckDB instance to rank candidate targets using SQL window functions:

```python
    def reconcile_matches(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pushes evaluated_df into DuckDB and runs ranking queries to 
        identify qualifying matches.
        """
        self.conn.register("evaluated_pairs", evaluated_df)
        
        query = f"""
            WITH Qualified AS (
                -- Keep only pairs that pass the optional bearing / overlap thresholds
                SELECT *
                FROM evaluated_pairs
                WHERE bearing_diff <= {self.max_angle}
                  AND overlap_pct >= {self.min_overlap}
            ),
            Ranked AS (
                -- Rank each source's qualifying destinations by alignment (closest first)
                SELECT
                    id_a AS source_id,
                    id_b AS dest_id,
                    dtw_distance, max_dtw_distance, min_dtw_distance,
                    bearing_diff, overlap_pct,
                    ROW_NUMBER() OVER (PARTITION BY id_a ORDER BY dtw_distance ASC) AS rnk
                FROM Qualified
            )
            SELECT
                source_id, dest_id,
                dtw_distance, max_dtw_distance, min_dtw_distance,
                bearing_diff, overlap_pct,
                rnk AS rank,
                CASE
                    WHEN COUNT(dest_id) OVER (PARTITION BY source_id) > 1 THEN '1:N_SPLIT'
                    WHEN rnk = 1 THEN '1:1_SYMMETRIC'
                    ELSE 'UNIDIRECTIONAL_PARTIAL'
                END AS match_type
            FROM Ranked
            ORDER BY source_id, rnk;
        """
        results_df = self.conn.execute(query).df()
        self.conn.unregister("evaluated_pairs")
        return results_df
```

### Stage 4: High-Level Directed Matching
The class provides a high-level `match` method to run the full directed Source A -> Destination B pipeline. Source segments with no candidate are appended as `NO_MATCH` rows:

```python
    def match(self) -> pd.DataFrame:
        """
        Runs the full 3-tier directed map-matching pipeline (Source A -> Destination B).
        Returns every ranked candidate plus a NO_MATCH row for each unmatched source.
        """
        # 1. Generate spatial candidate pairs (Tier 1)
        # 2. Compute DTW metrics (Tier 2)
        # 3. Reconcile & rank, then append NO_MATCH rows (Tier 3)
        pass
```

### Stage 5: The Assignment Decision
`match()` returns *all* ranked candidates. To commit to a final assignment whose cardinality
fits the problem, pass the result through `resolve`:

```python
    def resolve(self, results: pd.DataFrame, strategy: str = "best_per_source") -> pd.DataFrame:
        """
        Apply a cardinality decision to the ranked candidates from match():
        - "best_per_source" (many-to-one, default): each source -> its closest dest.
        - "best_per_dest"   (one-to-many):          each dest   -> its closest source.
        - "one_to_one"      (global unique):        each source and dest used at most once.
        - "all":                                    keep every candidate (no decision).
        Unassigned sources are returned as NO_MATCH rows.
        """
        pass
```

---

## 5. Concrete Reusable Workflows

### Case A: Conflating Two Separate DuckDB Database Tables
```python
matcher = DuckDBMapMatcher()

# Connect to db_a and attach db_b internally
matcher.conn.execute("ATTACH 'projects/network-matching/data/tomtom.db' AS db_b;")

matcher.configure_sources(
    source_a="main.osm_roads", id_col_a="id", geom_col_a="geom",
    source_b="db_b.main.tomtom_roads", id_col_b="id", geom_col_b="geom",
    utm_srid=32639
)

# Run directed pipeline, then decide a unique segment-to-segment assignment
results = matcher.match()
assignment = matcher.resolve(results, strategy="one_to_one")

# Save result back to primary database
matcher.conn.register("final_results", assignment)
matcher.conn.execute("CREATE TABLE main.conflation_results AS SELECT * FROM final_results")
```

### Case B: Conflating Two Flat CSV Files (Zero-Database Footprint)
```python
matcher = DuckDBMapMatcher()

matcher.configure_sources(
    source_a="'projects/network-matching/data/osm.csv'", 
    id_col_a="osm_id", 
    geom_col_a="ST_GeomFromText(wkt_geom)",
    
    source_b="'projects/network-matching/data/here.csv'", 
    id_col_b="here_id", 
    geom_col_b="ST_GeomFromText(wkt_geom)",
    
    utm_srid=32639
)

# Run directed pipeline in-memory, then decide the assignment (many-to-one here)
results = matcher.match()
assignment = matcher.resolve(results, strategy="best_per_source")

# Export directly to a new CSV file
matcher.conn.register("final_results", assignment)
matcher.conn.execute("COPY final_results TO 'projects/network-matching/data/matches.csv' (HEADER, DELIMITER ',')")
```
