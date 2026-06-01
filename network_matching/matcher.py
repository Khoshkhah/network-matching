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
        self.max_distance = 25.0       # search radius in meters to find candidate segments
        self.max_angle = 180.0         # max allowed bearing difference in degrees (180 = no angle filter)
        self.min_overlap = 0.0         # min required corridor overlap fraction (0-1; 0 = no overlap filter)

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
        
    def set_parameters(self, max_distance: Optional[float] = None,
                       max_angle: Optional[float] = None,
                       min_overlap: Optional[float] = None):
        """Override matching thresholds.

        Parameters
        ----------
        max_distance:
            Candidate search radius in meters (Tier 1 spatial join).
        max_angle:
            Maximum allowed bearing difference in degrees for a pair to qualify
            as a match (Tier 3 reconciliation).
        min_overlap:
            Minimum required corridor overlap fraction (0-1) for a pair to
            qualify as a match (Tier 3 reconciliation).
        """
        if max_distance is not None:
            self.max_distance = max_distance
        if max_angle is not None:
            self.max_angle = max_angle
        if min_overlap is not None:
            self.min_overlap = min_overlap
            
    def _get_all_ids_a(self) -> pd.DataFrame:
        """
        Returns a single-column DataFrame of every ID present in Source Network A.
        Used to identify segments that had no candidate in Network B.
        """
        query = f"SELECT DISTINCT {self.columns_a['id']} AS id_a FROM {self.source_a};"
        return self.conn.execute(query).df()

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
        its qualifying destinations are ranked by alignment quality (``rank`` = 1 is the closest).
        **All** qualifying destinations are kept, so a source that is split across several
        destinations yields several rows.

        This is intentionally one-directional, mapping Source A to Destination B.
        Swapping the sources gives a different table since all projections and overlap
        metrics are evaluated relative to Source A (e.g., ``overlap_pct`` is the aligned
        coverage of Source A's length). To obtain a symmetric result, run the matcher
        both ways and UNION the two outputs -- no reciprocal A<->B reconciliation is
        performed here.

        Returns columns: ``source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance,
        bearing_diff, overlap_pct, rank, match_type``.
        """
        cols = ["source_id", "dest_id", "dtw_distance", "max_dtw_distance",
                "min_dtw_distance", "bearing_diff", "overlap_pct", "rank", "match_type"]
        if evaluated_df.empty:
            return pd.DataFrame(columns=cols)

        self.conn.register("evaluated_pairs", evaluated_df)
        query = f"""
            WITH Qualified AS (
                -- Keep only pairs that pass the bearing / overlap thresholds
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
                -- Backwards-compatible match_type classification for legacy scripts/visualizers
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

    def _append_unmatched(self, results: pd.DataFrame, all_ids_a: pd.DataFrame) -> pd.DataFrame:
        """
        Appends a NO_MATCH row for every Source A segment that does not appear in *results*.

        Unmatched rows have NULL (NaN) for all metric columns and ``rank`` is left as
        ``pd.NA``.  ``match_type`` is set to ``'NO_MATCH'`` so callers can filter them out
        with a simple boolean mask::

            matched = results[results['match_type'] != 'NO_MATCH']

        Parameters
        ----------
        results:
            Reconciled match table (output of ``reconcile_matches``).
        all_ids_a:
            Single-column DataFrame with column ``id_a`` containing every Source A ID.
        """
        matched_ids = set(results["source_id"].dropna().unique()) if not results.empty else set()
        unmatched_ids = set(all_ids_a["id_a"].tolist()) - matched_ids

        if not unmatched_ids:
            return results

        nan = float("nan")
        unmatched_rows = pd.DataFrame({
            "source_id": list(unmatched_ids),
            "dest_id": None,
            "dtw_distance": nan,
            "max_dtw_distance": nan,
            "min_dtw_distance": nan,
            "bearing_diff": nan,
            "overlap_pct": nan,
            "rank": pd.array([pd.NA] * len(unmatched_ids), dtype="Int64"),
            "match_type": "NO_MATCH",
        })

        return pd.concat([results, unmatched_rows], ignore_index=True)

    def match(self) -> pd.DataFrame:
        """
        Runs the full 3-tier map-matching pipeline (directed: Source A -> Destination B).

        Source A segments that have **no candidate** in Network B within ``max_distance``
        are included in the output with ``dest_id = None``, all metric columns set to
        ``NaN``, and ``match_type = 'NO_MATCH'``.  To drop unmatched rows::

            matched = results[results['match_type'] != 'NO_MATCH']

        Returns a DataFrame with columns:
            [source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance,
             bearing_diff, overlap_pct, rank, match_type]
        """
        # Fetch all A IDs upfront so we can detect unmatched segments later
        all_ids_a = self._get_all_ids_a()
        candidates_df = self.generate_candidate_pairs()

        if candidates_df.empty:
            evaluated = pd.DataFrame()
        else:
            evaluated = self.compute_dtw_metrics(candidates_df)
        results = self.reconcile_matches(evaluated)
        return self._append_unmatched(results, all_ids_a)

    # Strategies understood by ``resolve()`` (see that method for the semantics).
    RESOLVE_STRATEGIES = ("all", "best_per_source", "best_per_dest", "one_to_one")

    def resolve(self, results: pd.DataFrame, strategy: str = "best_per_source") -> pd.DataFrame:
        """
        Apply a cardinality DECISION to the ranked candidate table produced by ``match()``.

        ``match()`` only generates and scores candidates -- for every source it keeps
        *every* qualifying destination, ranked by ``dtw_distance``. It does not decide which
        single pairing is "the" match, because that decision depends on the problem. This
        method makes that decision according to ``strategy``:

        - ``"all"`` (no decision):
              Return the candidates unchanged (every ranked pair). Use when you want to apply
              your own downstream logic.

        - ``"best_per_source"`` (MANY-TO-ONE, the default):
              Each source is assigned its single closest destination (``rank == 1``). A
              destination may be chosen by many sources. This is the right choice when several
              A's legitimately map to the same B -- e.g. assigning sensor locations to road
              segments, where sensors in different lanes all belong to the same road.

        - ``"best_per_dest"`` (ONE-TO-MANY):
              The mirror image: each destination keeps only its single closest source. A source
              may be chosen by many destinations.

        - ``"one_to_one"`` (GLOBAL UNIQUE):
              Each source and each destination is used at most once. Pairs are accepted greedily
              in ascending ``dtw_distance`` order (a fast approximation of optimal assignment),
              so the globally closest pairs win and conflicting weaker pairs are dropped. Use for
              unique segment-to-segment network conflation.

        Regardless of strategy, every source from ``results`` appears exactly once in the output:
        sources left without an assignment come back as ``NO_MATCH`` rows (``dest_id = None``,
        metrics ``NaN``). Note that ``match_type`` still reflects the *original* candidate
        fan-out, not the resolved cardinality -- filter on the returned rows, not on
        ``match_type``, to read the decision.

        Parameters
        ----------
        results:
            Output of ``match()`` (or ``reconcile_matches``).
        strategy:
            One of ``RESOLVE_STRATEGIES``.

        Returns
        -------
        The decided assignment table, same columns as ``results``.
        """
        if strategy not in self.RESOLVE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected one of {self.RESOLVE_STRATEGIES}."
            )

        if results.empty:
            return results

        # All sources we must account for (includes sources that were already NO_MATCH).
        all_src = pd.DataFrame({"id_a": results["source_id"].dropna().unique()})

        # Work only with real candidate pairs; NO_MATCH rows are re-derived at the end.
        matched = results[results["match_type"] != "NO_MATCH"].copy()

        if strategy == "all" or matched.empty:
            decided = matched
        elif strategy == "best_per_source":
            # Each source keeps its closest destination.
            decided = (matched.sort_values("dtw_distance")
                              .drop_duplicates(subset="source_id", keep="first"))
        elif strategy == "best_per_dest":
            # Each destination keeps its closest source.
            decided = (matched.sort_values("dtw_distance")
                              .drop_duplicates(subset="dest_id", keep="first"))
        else:  # "one_to_one" -- greedy global assignment
            used_src, used_dst, keep_idx = set(), set(), []
            for idx, src, dst in matched.sort_values("dtw_distance")[
                ["source_id", "dest_id"]
            ].itertuples():
                if src in used_src or dst in used_dst:
                    continue
                used_src.add(src)
                used_dst.add(dst)
                keep_idx.append(idx)
            decided = matched.loc[keep_idx]

        # Re-append a NO_MATCH row for every source that ended up unassigned.
        decided = decided.sort_values(["source_id", "dtw_distance"]).reset_index(drop=True)
        return self._append_unmatched(decided, all_src)


