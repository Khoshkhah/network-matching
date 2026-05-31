import duckdb
import pandas as pd
import numpy as np
from shapely.wkt import loads as load_wkt
from shapely.geometry import LineString
from typing import Optional

from .dtw import dtw_align

def calculate_bearing(line: LineString) -> float:
    """Calculate absolute bearing (0-360 degrees) of a LineString from start to end."""
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    start, end = coords[0], coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360

class DuckDBMapMatcher:
    def __init__(self, db_path_a: Optional[str] = None, db_path_b: Optional[str] = None):
        """
        Initializes an in-memory DuckDB connection, installs/loads the spatial extension.
        If db_path_a is provided, it connects to that database file.
        If db_path_b is provided, it ATTACHES that database file under the alias 'db_b'.
        """
        if db_path_a:
            self.conn = duckdb.connect(database=db_path_a)
        else:
            self.conn = duckdb.connect(database=':memory:')
            
        # Ensure spatial extension is loaded
        self.conn.execute("INSTALL spatial; LOAD spatial;")
        
        if db_path_b:
            self.conn.execute(f"ATTACH '{db_path_b}' AS db_b;")
            
        self.source_a = None
        self.source_b = None
        self.columns_a = {}
        self.columns_b = {}
        self.utm_srid = None
        
        # Default thresholds
        self.max_distance = 25.0       # meters
        self.max_angle = 30.0          # degrees
        self.min_overlap = 0.50        # ratio (50%)
        
    def configure_sources(self, 
                          source_a: str, id_col_a: str, geom_col_a: str,
                          source_b: str, id_col_b: str, geom_col_b: str,
                          utm_srid: int):
        """
        Sets the tables or files to match and configures column mappings.
        
        Parameters:
        - source_a: SQL representation of A (e.g. 'table_a' or "'file_a.csv'")
        - id_col_a: Primary key/ID column of table A
        - geom_col_a: Geometry column of table A (WKT/WKB or geometry type)
        - source_b: SQL representation of B (e.g. 'table_b' or "'file_b.csv'")
        - id_col_b: Primary key/ID column of table B
        - geom_col_b: Geometry column of table B (WKT/WKB or geometry type)
        - utm_srid: The local projected coordinate system EPSG code in meters (e.g. 32639)
        """
        self.source_a = source_a
        self.source_b = source_b
        self.columns_a = {"id": id_col_a, "geom": geom_col_a}
        self.columns_b = {"id": id_col_b, "geom": geom_col_b}
        self.utm_srid = utm_srid
        
    def set_parameters(self, 
                       max_distance: Optional[float] = None, 
                       max_angle: Optional[float] = None, 
                       min_overlap: Optional[float] = None):
        """Allows overriding standard matching thresholds."""
        if max_distance is not None:
            self.max_distance = max_distance
        if max_angle is not None:
            self.max_angle = max_angle
        if min_overlap is not None:
            self.min_overlap = min_overlap
            
    def generate_candidate_pairs(self) -> pd.DataFrame:
        """
        TIER 1: Performs an in-memory spatial index overlap join in DuckDB.
        Returns projected WKT geometries in meters to prevent duplicate projection steps.
        
        Returns:
        - DataFrame containing [id_a, wkt_a, id_b, wkt_b]
        """
        if not self.source_a or not self.source_b or not self.utm_srid:
            raise ValueError("Matcher sources and UTM SRID must be configured first.")
            
        # Spatial query uses ST_DWithin to leverage DuckDB's internal R-Tree index
        # We transform both geometries to the local UTM meter-based projection
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

    def compute_dtw_metrics(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        TIER 2: Performs shape-matching for candidate pairs in Python.
        Calculates DTW distance, bearing difference, and corridor overlap percentage.
        
        Returns:
        - DataFrame containing [id_a, id_b, dtw_distance, bearing_diff, overlap_pct]
        """
        evaluated_records = []
        
        for _, row in candidates_df.iterrows():
            id_a = row["id_a"]
            id_b = row["id_b"]
            
            # 1. Parse WKT geometry strings to Shapely LineStrings
            try:
                geom_a = load_wkt(row["wkt_a"])
                geom_b = load_wkt(row["wkt_b"])
            except Exception:
                # Fallback in case of parsing issues with corrupted geometries
                continue
                
            if not isinstance(geom_a, LineString) or not isinstance(geom_b, LineString):
                continue
                
            coords_a = list(geom_a.coords)
            coords_b = list(geom_b.coords)
            
            # 2. Compute Normalized 2D DTW Distance (returns average, path, metrics)
            dtw_dist, _, dtw_metrics = dtw_align(coords_a, coords_b)
            
            # 3. Compute Direction / Bearing Difference
            bearing_a = calculate_bearing(geom_a)
            bearing_b = calculate_bearing(geom_b)
            bearing_diff = abs(bearing_a - bearing_b)
            bearing_diff = min(bearing_diff, 360 - bearing_diff)
            
            # 4. Retrieve Parallel Alignment Overlap Percentage
            overlap_pct = dtw_metrics.get("overlap_pct", 0.0)
                
            evaluated_records.append({
                "id_a": id_a,
                "id_b": id_b,
                "dtw_distance": dtw_dist,
                "max_dtw_distance": dtw_metrics.get("max", float('inf')),
                "min_dtw_distance": dtw_metrics.get("min", float('inf')),
                "bearing_diff": bearing_diff,
                "overlap_pct": overlap_pct
            })
            
        return pd.DataFrame(evaluated_records)

    def reconcile_matches(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        TIER 3: Resolve evaluated candidate pairs into a **directional** source->destination
        match table.

        Each surviving row is a match FROM a source edge (A) TO a destination edge (B): candidate
        pairs are filtered by the distance / bearing / overlap thresholds, then for every source
        its qualifying destinations are ranked by alignment quality (``rank`` = 1 is the closest;
        ``is_best`` flags it). **All** qualifying destinations are kept, so a source that is split
        across several destinations yields several rows.

        This is intentionally one-directional, mapping Source A to Destination B.
        Swapping the sources gives a different table since all projections and overlap
        metrics are evaluated relative to Source A (e.g., ``overlap_pct`` is the aligned
        coverage of Source A's length). To obtain a symmetric result, run the matcher
        both ways and UNION the two outputs -- no reciprocal A<->B reconciliation is
        performed here.

        Returns columns: ``source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance,
        bearing_diff, overlap_pct, rank, is_best, match_type``.
        """
        cols = ["source_id", "dest_id", "dtw_distance", "max_dtw_distance",
                "min_dtw_distance", "bearing_diff", "overlap_pct", "rank", "is_best", "match_type"]
        if evaluated_df.empty:
            return pd.DataFrame(columns=cols)

        self.conn.register("evaluated_pairs", evaluated_df)
        query = f"""
            WITH FilteredPairs AS (
                -- Absolute cutoffs: drop poor candidate pairs (also "missing road" detection)
                SELECT * FROM evaluated_pairs
                WHERE dtw_distance <= {self.max_distance}
                  AND bearing_diff <= {self.max_angle}
                  AND overlap_pct  >= {self.min_overlap * 100.0}
            ),
            Ranked AS (
                -- Rank each source's qualifying destinations by alignment (closest first)
                SELECT
                    id_a AS source_id,
                    id_b AS dest_id,
                    dtw_distance, max_dtw_distance, min_dtw_distance,
                    bearing_diff, overlap_pct,
                    ROW_NUMBER() OVER (PARTITION BY id_a ORDER BY dtw_distance ASC) AS rnk
                FROM FilteredPairs
            )
            SELECT
                source_id, dest_id,
                dtw_distance, max_dtw_distance, min_dtw_distance,
                bearing_diff, overlap_pct,
                rnk AS rank,
                (rnk = 1) AS is_best,
                -- Backwards-compatible match_type classification for legacy scripts/visualizers
                CASE
                    WHEN COUNT(dest_id) OVER (PARTITION BY source_id) > 1 THEN '1:N_SPLIT'
                    WHEN rnk = 1 AND overlap_pct >= 80.0 AND bearing_diff <= 15.0 THEN '1:1_SYMMETRIC'
                    WHEN bearing_diff > 25.0 OR overlap_pct < 60.0 THEN 'CONFLICT'
                    ELSE 'UNIDIRECTIONAL_PARTIAL'
                END AS match_type
            FROM Ranked
            ORDER BY source_id, rnk;
        """
        results_df = self.conn.execute(query).df()
        self.conn.unregister("evaluated_pairs")
        return results_df

    def match(self, bidirectional: bool = False) -> pd.DataFrame:
        """
        Runs the full 3-tier map-matching pipeline.
        
        If bidirectional=False (default):
            Performs directed matching from Source A to Destination B.
            Returns a DataFrame with columns: 
            [source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance,
             bearing_diff, overlap_pct, rank, is_best, match_type]
             
        If bidirectional=True:
            Runs matching in both directions (Source A -> Destination B AND Destination B -> Source A)
            and returns the UNION of the two reconciled directional matching tables.
            Returns a DataFrame with columns:
            [source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance,
             bearing_diff, overlap_pct, rank, is_best, match_type, direction]
             where `direction` is either 'A_to_B' or 'B_to_A'.
        """
        candidates_df = self.generate_candidate_pairs()
        
        if candidates_df.empty:
            cols = ["source_id", "dest_id", "dtw_distance", "max_dtw_distance",
                    "min_dtw_distance", "bearing_diff", "overlap_pct", "rank", "is_best", "match_type"]
            if bidirectional:
                cols.append("direction")
            return pd.DataFrame(columns=cols)
            
        if not bidirectional:
            evaluated = self.compute_dtw_metrics(candidates_df)
            return self.reconcile_matches(evaluated)
            
        # Direction 1: A -> B
        evaluated_a_to_b = self.compute_dtw_metrics(candidates_df)
        reconciled_a_to_b = self.reconcile_matches(evaluated_a_to_b)
        reconciled_a_to_b["direction"] = "A_to_B"
        
        # Direction 2: B -> A
        candidates_b_to_a = candidates_df.rename(columns={
            "id_a": "id_b",
            "wkt_a": "wkt_b",
            "id_b": "id_a",
            "wkt_b": "wkt_a"
        })[["id_a", "wkt_a", "id_b", "wkt_b"]]
        
        evaluated_b_to_a = self.compute_dtw_metrics(candidates_b_to_a)
        reconciled_b_to_a = self.reconcile_matches(evaluated_b_to_a)
        reconciled_b_to_a["direction"] = "B_to_A"
        
        return pd.concat([reconciled_a_to_b, reconciled_b_to_a], ignore_index=True)


