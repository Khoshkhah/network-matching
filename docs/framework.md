# Software Architecture & Reusable Framework

This document outlines the software design, class specifications, and execution lifecycle of the **`network-matching`** framework. It describes how the project combines **In-Memory DuckDB Spatial SQL** and **Python DTW** into a universal, reusable map-matching package.

---

## 1. Architectural Philosophy

To build a general and reusable tool, we decouple the **storage format** from the **computation engine**:

1. **In-Memory DuckDB Engine**: Rather than binding the software to static database files, the framework uses an in-memory DuckDB connection (`duckdb.connect(':memory:')`) loaded with the `spatial` extension as its core computation engine.
2. **File-Format Agnostic**: DuckDB's ability to query raw files directly allows the framework to support **DuckDB tables, CSV files (WKT geometries), Parquet files, and Shapefiles** as inputs without altering the SQL logic.
3. **Hybrid Processing (Tier 1 SQL + Tier 2 Python)**:
   * **SQL** is used for spatial indices and joins (Tier 1) because it is highly parallel and optimized in C++.
   * **Python** is used for DTW shape-alignment (Tier 2) because recursive time-series matrix algorithms are best expressed in Python/NumPy.
   * **SQL** is used again for reconciliation and groupings (post-processing).

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
  |  2. TIER 2: De-duplicated DTW Distance calculation                |
  |     - Normalized DTW Proximity                                    |
  |     - Direction Alignment & Heading Checks                        |
  |                                                                   |
  +-------------------------------------------------------------------+
                                     |
                                     v (Pushed back to DuckDB table)
  +-------------------------------------------------------------------+
  |                  IN-MEMORY DUCKDB ENGINE                          |
  |                                                                   |
  |  3. RECONCILIATION: Grouping & Argmin SQL queries                 |
  |     - Identifies Symmetric 1:1, 1:N Splits, Conflicts             |
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
    def __init__(self):
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
        
        # Thresholds
        self.max_distance = 20.0
        self.max_angle = 30.0
        self.min_overlap = 0.50
        
    def configure_sources(self, 
                          source_a: str, id_col_a: str, geom_col_a: str,
                          source_b: str, id_col_b: str, geom_col_b: str,
                          utm_srid: int):
        """
        Sets the data sources and maps their columns.
        
        Parameters:
        - source_a: SQL representation of Source A (e.g. 'table_name' or "'file.csv'")
        - id_col_a: ID column name in Source A
        - geom_col_a: Geometry column representation (e.g. 'geom' or 'ST_GeomFromText(wkt_col)')
        - utm_srid: Local projected coordinate system SRID in meters (e.g. 32639)
        """
        self.source_a = source_a
        self.source_b = source_b
        self.columns_a = {"id": id_col_a, "geom": geom_col_a}
        self.columns_b = {"id": id_col_b, "geom": geom_col_b}
        self.utm_srid = utm_srid

    def set_parameters(self, max_distance: float, max_angle: float, min_overlap: float):
        """Overrides the default spatial matching thresholds."""
        self.max_distance = max_distance
        self.max_angle = max_angle
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
        Returns a Pandas DataFrame of [id_a, id_b] pairs.
        """
        query = f"""
            SELECT DISTINCT
                A.{self.columns_a['id']} AS id_a,
                B.{self.columns_b['id']} AS id_b
            FROM {self.source_a} AS A, {self.source_b} AS B
            WHERE ST_DWithin(
                ST_Transform(A.{self.columns_a['geom']}, 4326, {self.utm_srid}),
                ST_Transform(B.{self.columns_b['geom']}, 4326, {self.utm_srid}),
                {self.max_distance}
            );
        """
        return self.conn.execute(query).df()
```

### Stage 2: DTW Distance Calculation (Tier 2)
The coordinate lists for each candidate pair are loaded into Python. The DTW algorithm computes the Normalized DTW distance exactly once per pair:

```python
    def compute_dtw_metrics(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches geometries for the candidate pairs, runs the 2D DTW algorithm 
        in Python, and returns a DataFrame of evaluated metrics:
        [id_a, id_b, dtw_distance, bearing_diff, overlap_pct]
        """
        # 1. Fetch exact geometries in UTM meters from DuckDB
        # 2. Iterate over pairs and calculate DTW distance (Python/NumPy)
        # 3. Return the populated metrics DataFrame
        pass
```

### Stage 3: Reconciliation & Classification (Post-Processing)
The evaluated metrics are pushed back to the in-memory DuckDB instance to classify matches using simple SQL aggregation:

```python
    def reconcile_matches(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pushes evaluated_df into DuckDB and runs reconciliation queries to 
        identify Symmetric 1:1, 1:N Splits, and Conflicts.
        """
        # Register the DataFrame as a virtual table in DuckDB
        self.conn.register("evaluated_pairs", evaluated_df)
        
        reconciliation_query = f"""
            WITH 
            -- 1. Apply baseline cutoff threshold
            FilteredPairs AS (
                SELECT * FROM evaluated_pairs 
                WHERE dtw_distance <= {self.max_distance}
                  AND bearing_diff <= {self.max_angle}
            ),
            -- 2. Find the minimum distance (argmin) from A to B
            BestAB AS (
                SELECT id_a, id_b, dtw_distance,
                       ROW_NUMBER() OVER(PARTITION BY id_a ORDER BY dtw_distance ASC) as rank
                FROM FilteredPairs
            ),
            -- 3. Find the minimum distance (argmin) from B to A
            BestBA AS (
                SELECT id_b, id_a, dtw_distance,
                       ROW_NUMBER() OVER(PARTITION BY id_b ORDER BY dtw_distance ASC) as rank
                FROM FilteredPairs
            )
            -- 4. Reconcile and Classify Matches
            SELECT 
                F.id_a, 
                F.id_b,
                F.dtw_distance,
                CASE 
                    WHEN AB.rank = 1 AND BA.rank = 1 THEN '1:1_SYMMETRIC'
                    WHEN BA.rank = 1 AND COUNT(F.id_b) OVER(PARTITION BY F.id_a) > 1 THEN '1:N_SPLIT'
                    WHEN AB.rank = 1 AND BA.rank != 1 THEN 'CONFLICT'
                    ELSE 'UNIDIRECTIONAL_PARTIAL'
                END AS match_type
            FROM FilteredPairs F
            LEFT JOIN BestAB AB ON F.id_a = AB.id_a AND F.id_b = AB.id_b AND AB.rank = 1
            LEFT JOIN BestBA BA ON F.id_b = BA.id_b AND F.id_a = BA.id_a AND BA.rank = 1;
        """
        return self.conn.execute(reconciliation_query).df()
```

### Stage 4: Export Result
Write the final table back to your desired destination (e.g. database table or a flat CSV file).

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

# Run pipeline
candidates = matcher.generate_candidate_pairs()
evaluated = matcher.compute_dtw_metrics(candidates)
results = matcher.reconcile_matches(evaluated)

# Save result back to primary database
matcher.conn.execute("CREATE TABLE main.conflation_results AS SELECT * FROM results")
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

# Run pipeline in-memory
candidates = matcher.generate_candidate_pairs()
evaluated = matcher.compute_dtw_metrics(candidates)
results = matcher.reconcile_matches(evaluated)

# Export directly to a new CSV file
matcher.conn.register("final_results", results)
matcher.conn.execute("COPY final_results TO 'projects/network-matching/data/matches.csv' (HEADER, DELIMITER ',')")
```
