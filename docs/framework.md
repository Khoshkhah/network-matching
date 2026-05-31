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
  |     - Ranks qualifying destination edges for each source edge     |
  |     - Flags best match (is_best) and supports bidirectional union  |
  |                                                                   |
  +-------------------------------------------------------------------+
                                     |
                                     v
                        +---------------------------+
                        | Output Matches (CSV/GPKG) |
                        +---------------------------+
```

---

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

    def set_parameters(self, max_distance: float):
        """Overrides the default candidate search radius."""
        self.max_distance = max_distance
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
            WITH Ranked AS (
                -- Rank each source's qualifying destinations by alignment (closest first)
                SELECT
                    id_a AS source_id,
                    id_b AS dest_id,
                    dtw_distance, max_dtw_distance, min_dtw_distance,
                    bearing_diff, overlap_pct,
                    ROW_NUMBER() OVER (PARTITION BY id_a ORDER BY dtw_distance ASC) AS rnk
                FROM evaluated_pairs
            )
            SELECT
                source_id, dest_id,
                dtw_distance, max_dtw_distance, min_dtw_distance,
                bearing_diff, overlap_pct,
                rnk AS rank,
                (rnk = 1) AS is_best,
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

### Stage 4: High-Level Directed/Bidirectional Matching
The class provides a high-level `match` method to execute either a directed Source A -> Destination B match or run both directions independently and return their union:

```python
    def match(self, bidirectional: bool = False) -> pd.DataFrame:
        """
        Runs the full 3-tier map-matching pipeline in either directed or bidirectional mode.
        """
        # 1. Generate spatial candidate pairs
        # 2. If bidirectional=False, run directional matching A -> B
        # 3. If bidirectional=True, run A -> B and B -> A independently and return their union.
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

# Run bidirectional pipeline
results = matcher.match(bidirectional=True)

# Save result back to primary database
matcher.conn.register("final_results", results)
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

# Run directional pipeline in-memory
results = matcher.match(bidirectional=False)

# Export directly to a new CSV file
matcher.conn.register("final_results", results)
matcher.conn.execute("COPY final_results TO 'projects/network-matching/data/matches.csv' (HEADER, DELIMITER ',')")
```
