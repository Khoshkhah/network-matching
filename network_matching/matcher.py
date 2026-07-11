import logging
import time

import duckdb
import pandas as pd
import numpy as np
from shapely.wkt import loads as load_wkt
from shapely.geometry import LineString, Point
from typing import Optional

from .dtw import dtw_align
from .graph_dtw import match_edge_to_bgraph

logger = logging.getLogger("network_matching.matcher")


def _graph_dtw_group(task, snap_tolerance_m, step_meters, trim_ends_m,
                     emission="point", bearing_weight=0.0, alpha=1.0, beta=1.0):
    """Pure, picklable worker: run graph-DTW for one A-edge group.

    ``task`` is ``(id_a, coords_a, b_edges)`` where ``b_edges`` is a list of
    ``(id_b, shapely LineString in UTM)``. Returns ``(id_a, result_or_None)``.
    """
    id_a, coords_a, b_edges = task
    if len(coords_a) < 2 or not b_edges:
        return id_a, None
    res = match_edge_to_bgraph(
        coords_a, b_edges,
        snap_tolerance_m=snap_tolerance_m, step_meters=step_meters,
        trim_ends_m=trim_ends_m, emission=emission, bearing_weight=bearing_weight,
        alpha=alpha, beta=beta,
    )
    if not res["route"]:
        return id_a, None
    return id_a, res


def bearing_between(start, end) -> float:
    """Absolute bearing (0-360 degrees) of the vector from point ``start`` to point ``end``."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def calculate_bearing(line: LineString) -> float:
    """Calculate absolute bearing (0-360 degrees) of a LineString from start to end."""
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    return bearing_between(coords[0], coords[-1])

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
        # ST_Transform axis handling. True treats EPSG:4326 coords as (lon, lat) = (x, y) — correct
        # for lon/lat WKT and required on DuckDB/PROJ builds that otherwise return POINT(inf inf)
        # for a 4326->UTM transform. Set False for the pre-`always_xy` behaviour.
        self.always_xy = True

        # Default thresholds
        self.max_distance = 25.0       # search radius in meters to find candidate segments
        self.max_angle = 180.0         # max allowed bearing difference in degrees (180 = no angle filter)
        self.min_overlap = 0.0         # min required corridor overlap fraction (0-1; 0 = no overlap filter)

    @classmethod
    def from_wkt_csv(cls, csv_a: str, csv_b: str, *, id_a: str, id_b: str, utm_srid: int,
                     geom_col: str = "geometry", id_cast: Optional[str] = "BIGINT",
                     max_distance: float = 25.0, keep_cols_a=None, keep_cols_b=None,
                     table_a: str = "network_a", table_b: str = "network_b",
                     always_xy: bool = True) -> "DuckDBMapMatcher":
        """Build and fully configure a matcher from two CSVs whose ``geom_col`` holds WKT
        geometry in EPSG:4326. One-call replacement for the usual
        ``CREATE TABLE ... ST_GeomFromText`` + :meth:`configure_sources` +
        :meth:`set_parameters` boilerplate::

            m = DuckDBMapMatcher.from_wkt_csv(
                "data/osm_edges.csv", "data/sweden_edges.csv",
                id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=30)
            routes_long, routes_summary = m.match_routes(n_jobs=-1)

        Each source is loaded into an in-memory spatial table; geometries are assumed lon/lat
        (EPSG:4326) and transformed to ``utm_srid`` (meters) during candidate generation.

        Parameters
        ----------
        csv_a, csv_b: paths to the Source-A and Source-B CSVs.
        id_a, id_b:   id column names in each CSV.
        utm_srid:     local projected CRS (EPSG, meters), e.g. 3006 for Sweden.
        geom_col:     WKT geometry column name (same in both, default ``"geometry"``).
        id_cast:      optional DuckDB cast for the id (default ``"BIGINT"``; ``None`` keeps text).
        max_distance: candidate search radius (m).
        keep_cols_a, keep_cols_b: extra columns to carry into each table (e.g. ``["name"]``).
        table_a, table_b: in-memory table names to create.
        """
        m = cls()
        srcs = ((csv_a, table_a, id_a, keep_cols_a or []),
                (csv_b, table_b, id_b, keep_cols_b or []))
        for path, table, idc, keep in srcs:
            cast = f"::{id_cast}" if id_cast else ""
            extra = "".join(f"{c}, " for c in keep)
            m.conn.execute(
                f"CREATE OR REPLACE TABLE {table} AS "
                f"SELECT {idc}{cast} AS {idc}, {extra}ST_GeomFromText({geom_col}) AS geometry "
                f"FROM '{path}';"
            )
        m.configure_sources(source_a=table_a, id_col_a=id_a, geom_col_a="geometry",
                            source_b=table_b, id_col_b=id_b, geom_col_b="geometry",
                            utm_srid=utm_srid)
        m.set_parameters(max_distance=max_distance, always_xy=always_xy)
        return m

    @classmethod
    def from_geofiles(cls, path_a: str, path_b: str, *, id_a: str, id_b: str, utm_srid: int,
                      src_srid: int = 4326, max_distance: float = 25.0,
                      keep_cols_a=None, keep_cols_b=None,
                      table_a: str = "network_a", table_b: str = "network_b",
                      always_xy: bool = True) -> "DuckDBMapMatcher":
        """Same as :meth:`from_wkt_csv` but reads GIS files (GeoPackage / GeoJSON / Shapefile /
        FlatGeobuf, anything DuckDB ``ST_Read`` supports) instead of WKT CSVs::

            m = DuckDBMapMatcher.from_geofiles(
                "osm.gpkg", "nvdb.gpkg", id_a="edge_id", id_b="directed_id",
                utm_srid=3006, src_srid=3006)

        ``src_srid`` is the CRS the files are stored in; geometry is reprojected to EPSG:4326 on
        load (the rest of the pipeline transforms 4326 -> ``utm_srid``).
        """
        m = cls()
        axy = f", always_xy := {'true' if always_xy else 'false'}"
        srcs = ((path_a, table_a, id_a, keep_cols_a or []),
                (path_b, table_b, id_b, keep_cols_b or []))
        for path, table, idc, keep in srcs:
            geom = ("geom" if src_srid == 4326
                    else f"ST_Transform(geom, 'EPSG:{src_srid}', 'EPSG:4326'{axy})")
            extra = "".join(f"{c}, " for c in keep)
            m.conn.execute(
                f"CREATE OR REPLACE TABLE {table} AS "
                f"SELECT {idc} AS {idc}, {extra}{geom} AS geometry FROM ST_Read('{path}');"
            )
        m.configure_sources(source_a=table_a, id_col_a=id_a, geom_col_a="geometry",
                            source_b=table_b, id_col_b=id_b, geom_col_b="geometry",
                            utm_srid=utm_srid)
        m.set_parameters(max_distance=max_distance, always_xy=always_xy)
        return m

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
                       min_overlap: Optional[float] = None,
                       always_xy: Optional[bool] = None):
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
        always_xy:
            ST_Transform axis handling for the 4326->UTM candidate projection.
            ``True`` (default) treats 4326 coords as (lon, lat); set ``False``
            for the legacy behaviour.
        """
        if max_distance is not None:
            self.max_distance = max_distance
        if max_angle is not None:
            self.max_angle = max_angle
        if min_overlap is not None:
            self.min_overlap = min_overlap
        if always_xy is not None:
            self.always_xy = always_xy
            
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
            
        # Spatial query uses ST_DWithin to leverage DuckDB's internal R-Tree index.
        # We transform both geometries to the local UTM meter-based projection; `always_xy` keeps
        # lon/lat WKT in (x, y) order (see __init__ / set_parameters).
        axy = f", always_xy := {'true' if self.always_xy else 'false'}"
        tA = f"ST_Transform(A.{self.columns_a['geom']}, 'EPSG:4326', 'EPSG:{self.utm_srid}'{axy})"
        tB = f"ST_Transform(B.{self.columns_b['geom']}, 'EPSG:4326', 'EPSG:{self.utm_srid}'{axy})"
        query = f"""
            SELECT DISTINCT
                A.{self.columns_a['id']} AS id_a,
                ST_AsText({tA}) AS wkt_a,
                B.{self.columns_b['id']} AS id_b,
                ST_AsText({tB}) AS wkt_b
            FROM {self.source_a} AS A, {self.source_b} AS B
            WHERE ST_DWithin({tA}, {tB}, {self.max_distance});
        """
        return self.conn.execute(query).df()

    def compute_dtw_metrics(self, candidates_df: pd.DataFrame, undirected: bool = False) -> pd.DataFrame:
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
            
            # 2. Compute Normalized 2D DTW Distance (returns average, warping path, metrics).
            #    undirected=True aligns against the better orientation of B, so a road
            #    digitized the opposite way still aligns instead of collapsing to 0 overlap.
            dtw_dist, warping_path, dtw_metrics = dtw_align(coords_a, coords_b, undirected=undirected)

            # 3. Compute Direction / Bearing Difference over the MATCHED span only.
            #    Use the first and last matched points of the DTW warping path (the start
            #    and end of the overlapping region) rather than each full segment's endpoints,
            #    so partial overlaps and splits compare the direction of the aligned portion.
            if len(warping_path) >= 2:
                a_start, b_start = warping_path[0]
                a_end, b_end = warping_path[-1]
                bearing_a = bearing_between(a_start, a_end)
                bearing_b = bearing_between(b_start, b_end)
            else:
                bearing_a = calculate_bearing(geom_a)
                bearing_b = calculate_bearing(geom_b)
            bearing_diff = abs(bearing_a - bearing_b)
            bearing_diff = min(bearing_diff, 360 - bearing_diff)
            
            # 4. Retrieve Parallel Alignment Overlap Percentage and absolute matched length
            overlap_pct = dtw_metrics.get("overlap_pct", 0.0)
            matched_len = dtw_metrics.get("matched_len", 0.0)

            evaluated_records.append({
                "id_a": id_a,
                "id_b": id_b,
                "dtw_distance": dtw_dist,
                "max_dtw_distance": dtw_metrics.get("max", float('inf')),
                "min_dtw_distance": dtw_metrics.get("min", float('inf')),
                "bearing_diff": bearing_diff,
                "overlap_pct": overlap_pct,
                "matched_len": matched_len
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
        # Metric columns come from the results frame itself, so the same appender serves every
        # per-pair schema (Mode 2's DTW metrics, Mode 4's point metrics, ...).
        metric_cols = [c for c in results.columns if c not in ("source_id", "rank", "match_type")] \
            if len(results.columns) else ["dest_id", "dtw_distance", "max_dtw_distance",
                                          "min_dtw_distance", "bearing_diff", "overlap_pct"]

        def _fill(c):
            if c == "dest_id" or (not results.empty and results[c].dtype == object):
                return None                      # id / WKT columns stay object, not float NaN
            return nan

        unmatched_rows = pd.DataFrame({
            "source_id": list(unmatched_ids),
            **{c: _fill(c) for c in metric_cols},
            "rank": pd.array([pd.NA] * len(unmatched_ids), dtype="Int64"),
            "match_type": "NO_MATCH",
        })

        combined = pd.concat([results, unmatched_rows], ignore_index=True)
        # Keep overlap_pct an integer percentage even though NO_MATCH rows add NaN
        # (use a nullable Int64 so unmatched rows stay NULL rather than forcing floats).
        if "overlap_pct" in combined.columns:
            combined["overlap_pct"] = combined["overlap_pct"].round().astype("Int64")
        return combined

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
        *every* qualifying destination, ranked by ``dtw_distance`` (for a :meth:`match_points`
        table the score column is ``distance_m``; auto-detected). It does not decide which
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

        # Ranking score: Mode 2 tables carry ``dtw_distance``, Mode 4 (point-to-edge) tables
        # carry ``distance_m`` -- auto-detect so the same decision step serves both.
        score = "dtw_distance" if "dtw_distance" in results.columns else "distance_m"
        if score not in results.columns:
            raise ValueError("results has neither 'dtw_distance' nor 'distance_m' to rank by.")

        # All sources we must account for (includes sources that were already NO_MATCH).
        all_src = pd.DataFrame({"id_a": results["source_id"].dropna().unique()})

        # Work only with real candidate pairs; NO_MATCH rows are re-derived at the end.
        matched = results[results["match_type"] != "NO_MATCH"].copy()

        if strategy == "all" or matched.empty:
            decided = matched
        elif strategy == "best_per_source":
            # Each source keeps its closest destination.
            decided = (matched.sort_values(score)
                              .drop_duplicates(subset="source_id", keep="first"))
        elif strategy == "best_per_dest":
            # Each destination keeps its closest source.
            decided = (matched.sort_values(score)
                              .drop_duplicates(subset="dest_id", keep="first"))
        else:  # "one_to_one" -- greedy global assignment
            used_src, used_dst, keep_idx = set(), set(), []
            for idx, src, dst in matched.sort_values(score)[
                ["source_id", "dest_id"]
            ].itertuples():
                if src in used_src or dst in used_dst:
                    continue
                used_src.add(src)
                used_dst.add(dst)
                keep_idx.append(idx)
            decided = matched.loc[keep_idx]

        # Re-append a NO_MATCH row for every source that ended up unassigned.
        decided = decided.sort_values(["source_id", score]).reset_index(drop=True)
        return self._append_unmatched(decided, all_src)

    # Columns returned by match_points().
    POINT_COLUMNS = ["source_id", "dest_id", "distance_m", "position_pct",
                     "edge_bearing_deg", "snap_wkt", "rank", "match_type"]

    def match_points(self) -> pd.DataFrame:
        """Mode 4 -- point-to-edge matching: assign each Source-A **POINT** (a sensor, a
        measurement station, a stop) to the nearby Source-B edges. The simple sibling of
        :meth:`match`: same configured sources and ``max_distance`` candidate search, but the
        score is plain geometry -- no DTW (see ``docs/point_matching.md``).

        Returns one row per (point, candidate edge) with columns ``POINT_COLUMNS``:
        ``distance_m`` (lateral point->edge distance), ``position_pct`` (snap position along the
        edge, 0 = start .. 100 = end), ``edge_bearing_deg`` (the edge's direction at the snap
        point, degrees clockwise from north), ``snap_wkt`` (the snapped point, POINT WKT in
        ``utm_srid`` meters), ``rank`` (1 = nearest edge for this point) and ``match_type``
        (``1:1`` / ``1:N_CANDIDATES`` / ``NO_MATCH``). Points with no edge within
        ``max_distance`` come back as ``NO_MATCH`` rows.

        Commit to an assignment with :meth:`resolve` (it ranks by ``distance_m`` here)::

            points     = m.match_points()
            assignment = m.resolve(points, strategy="best_per_source")
        """
        all_ids_a = self._get_all_ids_a()
        candidates = self.generate_candidate_pairs()

        rows = []
        for _, row in candidates.iterrows():
            try:
                pt = load_wkt(row["wkt_a"])
                edge = load_wkt(row["wkt_b"])
            except Exception:
                continue
            if not isinstance(pt, Point) or not isinstance(edge, LineString):
                continue
            s = edge.project(pt)                                # arc-length of the snap point
            snap = edge.interpolate(s)
            t0 = edge.interpolate(max(0.0, s - 1.0))            # local tangent over +-1 m
            t1 = edge.interpolate(min(edge.length, s + 1.0))
            rows.append({
                "source_id": row["id_a"],
                "dest_id": row["id_b"],
                "distance_m": float(pt.distance(edge)),
                "position_pct": float(100.0 * s / edge.length) if edge.length > 0 else 0.0,
                "edge_bearing_deg": bearing_between((t0.x, t0.y), (t1.x, t1.y)),
                "snap_wkt": snap.wkt,
            })

        if not rows:
            return self._append_unmatched(pd.DataFrame(columns=self.POINT_COLUMNS), all_ids_a)

        df = pd.DataFrame(rows).sort_values(["source_id", "distance_m"]).reset_index(drop=True)
        df["rank"] = (df.groupby("source_id").cumcount() + 1).astype("Int64")
        n_cand = df.groupby("source_id")["dest_id"].transform("size")
        df["match_type"] = np.where(n_cand > 1, "1:N_CANDIDATES", "1:1")
        return self._append_unmatched(df, all_ids_a)

    # Columns returned by reconcile_symmetric()/match_symmetric().
    SYMMETRIC_COLUMNS = ["a_id", "b_id", "dtw", "bearing_diff", "ov_ab", "ov_ba",
                         "matched_len_m", "containment", "symmetry", "relation", "cardinality"]

    def match_symmetric(self, max_dtw: Optional[float] = None, max_angle: float = 45.0,
                        min_overlap_m: float = 5.0, sym_overlap: int = 70) -> pd.DataFrame:
        """
        Run the directed matcher BOTH ways (A->B and B->A) and reconcile the two
        evaluations into a single **symmetric**, split-aware match table.

        Unlike ``match()``/``resolve()`` (which decide cardinality within one direction),
        this preserves split roads (1:N) and merges (N:1) by using *both* overlap values
        per pair. See ``docs/symmetric_matching.md`` for the full algorithm.

        ``max_dtw`` defaults to ``max_distance`` (the candidate search radius): a pair can
        never be a candidate beyond that distance, so the feasibility gate does not impose
        a hidden, tighter distance filter. Pass a smaller value to deliberately tighten it.

        Returns a DataFrame with ``SYMMETRIC_COLUMNS``.
        """
        if max_dtw is None:
            max_dtw = self.max_distance

        candidates = self.generate_candidate_pairs()
        if candidates.empty:
            return pd.DataFrame(columns=self.SYMMETRIC_COLUMNS)

        # Candidate generation is symmetric, so the same pairs are evaluated both ways;
        # the swapped run yields the overlap relative to B (ov_ba). Use undirected alignment
        # so a road digitized in opposite directions by the two networks still aligns.
        eval_ab = self.compute_dtw_metrics(candidates, undirected=True)
        swapped = candidates.rename(columns={
            "id_a": "id_b", "wkt_a": "wkt_b", "id_b": "id_a", "wkt_b": "wkt_a",
        })[["id_a", "wkt_a", "id_b", "wkt_b"]]
        eval_ba = self.compute_dtw_metrics(swapped, undirected=True)

        return self.reconcile_symmetric(
            eval_ab, eval_ba,
            max_dtw=max_dtw, max_angle=max_angle,
            min_overlap_m=min_overlap_m, sym_overlap=sym_overlap,
        )

    @classmethod
    def reconcile_symmetric(cls, eval_ab: pd.DataFrame, eval_ba: pd.DataFrame,
                            max_dtw: float = 25.0, max_angle: float = 45.0,
                            min_overlap_m: float = 5.0, sym_overlap: int = 70) -> pd.DataFrame:
        """
        Combine the two directed Tier-2 evaluation tables (A->B and B->A) into a single
        symmetric, split-aware match table. See ``docs/symmetric_matching.md``.

        Parameters
        ----------
        eval_ab, eval_ba:
            Outputs of ``compute_dtw_metrics`` in each direction. ``eval_ba`` has the A and
            B roles swapped (its ``id_a`` is a B id, its ``overlap_pct`` is overlap of B).
        max_dtw, max_angle:
            Feasibility gate (Step B): drop pairs with larger drift / direction difference.
        min_overlap_m:
            Minimum length (meters) of the shared co-linear stretch to keep an edge
            (Step C). This is the absolute matched length -- derived from overlap but NOT a
            raw overlap-% threshold -- so split/partial matches survive and only trivial
            point-touches/crossings are dropped.
        sym_overlap:
            Min ``symmetry = min(ov_ab, ov_ba)`` overlap-% to *label* an edge ``1:1``
            (Step D, classification only -- never drops a match).

        Returns
        -------
        DataFrame with ``SYMMETRIC_COLUMNS``; ``relation`` in {1:1, split, merge} and
        ``cardinality`` in {1:1, 1:N_SPLIT, N:1_MERGE, N:M_COMPLEX} (per connected component).
        """
        if eval_ab.empty or eval_ba.empty:
            return pd.DataFrame(columns=cls.SYMMETRIC_COLUMNS)

        # Step A: join the two directions on the unordered pair {a, b}.
        ab = eval_ab.rename(columns={
            "id_a": "a_id", "id_b": "b_id", "overlap_pct": "ov_ab", "dtw_distance": "dtw_ab",
            "matched_len": "mlen_ab",
        })
        ba = eval_ba.rename(columns={
            "id_a": "b_id", "id_b": "a_id", "overlap_pct": "ov_ba", "dtw_distance": "dtw_ba",
            "matched_len": "mlen_ba",
        })
        U = ab[["a_id", "b_id", "ov_ab", "dtw_ab", "bearing_diff", "mlen_ab"]].merge(
            ba[["a_id", "b_id", "ov_ba", "dtw_ba", "mlen_ba"]], on=["a_id", "b_id"], how="inner",
        )
        if U.empty:
            return pd.DataFrame(columns=cls.SYMMETRIC_COLUMNS)

        # Use the more favourable direction for drift; overlaps give containment/symmetry.
        U["dtw"] = U[["dtw_ab", "dtw_ba"]].min(axis=1)
        U["containment"] = U[["ov_ab", "ov_ba"]].max(axis=1)
        U["symmetry"] = U[["ov_ab", "ov_ba"]].min(axis=1)
        # Absolute length (m) of the shared co-linear stretch. Derived from overlap but kept
        # as a length, so a partial/split match is judged by how much real geometry it shares
        # -- not by what fraction that is of either (differently-segmented) edge.
        U["matched_len_m"] = U[["mlen_ab", "mlen_ba"]].max(axis=1).round(1)

        # Step B: feasibility gate (geometric "score": close + same direction).
        U = U[(U["dtw"] <= max_dtw) & (U["bearing_diff"] <= max_angle)]
        # Step C: shared-length floor -- drops trivial point-touches/crossings (a few meters)
        # while keeping genuine partial/split matches. NOT a raw overlap-% threshold.
        U = U[U["matched_len_m"] >= min_overlap_m].copy()
        if U.empty:
            return pd.DataFrame(columns=cls.SYMMETRIC_COLUMNS)

        # Step D (relation): symmetric -> 1:1; else the contained side names split/merge.
        U["relation"] = np.where(
            U["symmetry"] >= sym_overlap, "1:1",
            np.where(U["ov_ba"] >= U["ov_ab"], "split", "merge"),
        )

        # Step D (cardinality): label each bipartite connected component by its degrees.
        U["cardinality"] = cls._component_cardinality(U["a_id"].tolist(), U["b_id"].tolist())

        U["bearing_diff"] = U["bearing_diff"].round(1)
        return U[cls.SYMMETRIC_COLUMNS].sort_values(["a_id", "b_id"]).reset_index(drop=True)

    @staticmethod
    def _component_cardinality(a_ids, b_ids):
        """
        Union-find over the bipartite edges (a_ids[i], b_ids[i]); returns a per-edge
        cardinality label based on how many A and B nodes share its connected component.
        """
        from collections import defaultdict

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:        # path compression
                parent[x], x = root, parent[x]
            return root

        def union(x, y):
            parent[find(x)] = find(y)

        edges = [(("A", a), ("B", b)) for a, b in zip(a_ids, b_ids)]
        for na, nb in edges:
            union(na, nb)

        comp_a, comp_b = defaultdict(set), defaultdict(set)
        for (na, nb) in edges:
            r = find(na)
            comp_a[r].add(na)
            comp_b[r].add(nb)

        labels = []
        for (na, nb) in edges:
            r = find(na)
            ca, cb = len(comp_a[r]), len(comp_b[r])
            if ca == 1 and cb == 1:
                labels.append("1:1")
            elif ca == 1 and cb > 1:
                labels.append("1:N_SPLIT")
            elif ca > 1 and cb == 1:
                labels.append("N:1_MERGE")
            else:
                labels.append("N:M_COMPLEX")
        return labels

    # ------------------------------------------------------------------
    # Graph-DTW (route-based) matching  --  see network_matching/graph_dtw.py
    # ------------------------------------------------------------------
    ROUTES_LONG_COLUMNS = ["source_id", "dest_id", "seq", "direction",
                           "edge_match_dist_avg", "edge_match_dist_max", "edge_match_dist_min",
                           "edge_a_len", "edge_cover_pct", "edge_matched_len", "edge_b_len",
                           "edge_b_used_pct", "edge_bearing_diff", "n_points",
                           "route_match_dist", "n_edges"]
    ROUTES_SUMMARY_COLUMNS = ["source_id", "n_edges", "dest_ids", "dtw_distance",
                              "max_dtw_distance", "min_dtw_distance", "bearing_diff",
                              "part_drift", "part_bearing_diff",
                              "overlap_pct", "matched_len", "route_geom_wkt", "match_type"]

    # Column type schema for the two output tables. Ids/counts are plain int64 -- they are always
    # present integers (an edge id is never null and never a float). The ONE genuinely nullable
    # column is routes_summary.overlap_pct (NO_MATCH rows), kept as nullable Int64.
    ROUTES_LONG_DTYPES = {
        "source_id": "int64", "dest_id": "int64", "seq": "int64", "direction": "string",
        "edge_match_dist_avg": "float64", "edge_match_dist_max": "float64",
        "edge_match_dist_min": "float64", "edge_a_len": "float64", "edge_cover_pct": "float64",
        "edge_matched_len": "float64", "edge_b_len": "float64", "edge_b_used_pct": "float64",
        "edge_bearing_diff": "float64", "n_points": "int64", "route_match_dist": "float64",
        "n_edges": "int64",
    }
    ROUTES_SUMMARY_DTYPES = {
        "source_id": "int64", "n_edges": "int64", "dtw_distance": "float64",
        "max_dtw_distance": "float64", "min_dtw_distance": "float64", "bearing_diff": "float64",
        "part_drift": "float64", "part_bearing_diff": "float64",
        "overlap_pct": "Int64", "matched_len": "float64", "match_type": "string",
    }  # dest_ids (list) and route_geom_wkt (WKT/None) stay object

    @staticmethod
    def _apply_dtypes(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
        present = {c: t for c, t in dtypes.items() if c in df.columns}
        return df.astype(present) if not df.empty else df

    def compute_graph_dtw_routes(self, candidates_df: Optional[pd.DataFrame] = None,
                                 snap_tolerance_m: float = 0.75, step_meters: float = 10.0,
                                 trim_ends_m: float = 0.0, n_jobs: int = 1,
                                 emission: str = "point", bearing_weight: float = 0.0,
                                 alpha: float = 1.0, beta: float = 1.0):
        """Route-based matching: align each Source-A edge to the local directed graph of its
        candidate B-edges (graph-DTW), returning one connected B-edge route per A-edge.

        Parameters
        ----------
        candidates_df:
            Output of :meth:`generate_candidate_pairs` (``id_a, wkt_a, id_b, wkt_b`` in UTM
            meters). If ``None`` it is generated.
        snap_tolerance_m, step_meters, trim_ends_m, emission, bearing_weight, alpha, beta:
            Passed through to :func:`network_matching.graph_dtw.match_edge_to_bgraph`
            (``alpha``/``beta`` = the §12 step weights: 1:N coverage / N:1 stall pricing,
            both emission modes; defaults ``1``/``1`` = unweighted).
        n_jobs:
            >1 fans the per-A-edge work out with joblib (each A-edge is an independent unit).

        Returns
        -------
        (routes_long, routes_summary) : two DataFrames.
            ``routes_long`` has one row per (A-edge, B-edge in its route) with ``seq``/``direction``
            and the route-level metrics repeated (columns ``ROUTES_LONG_COLUMNS``).
            ``routes_summary`` has one row per A-edge (columns ``ROUTES_SUMMARY_COLUMNS``);
            every A-edge appears, unmatched ones as a single ``NO_MATCH`` row.
        """
        t_start = time.time()
        if candidates_df is None:
            logger.info("graph-DTW: generating candidate pairs...")
            candidates_df = self.generate_candidate_pairs()
        n_pairs = len(candidates_df)
        n_a = candidates_df["id_a"].nunique() if not candidates_df.empty else 0
        logger.info("graph-DTW: %d candidate pairs over %d A-edges "
                    "(snap=%.2fm, step=%.2fm, n_jobs=%d)",
                    n_pairs, n_a, snap_tolerance_m, step_meters, n_jobs)

        # Build the independent per-A-edge tasks: (id_a, coords_a, [(id_b, LineString), ...]).
        tasks = []
        if not candidates_df.empty:
            for id_a, grp in candidates_df.groupby("id_a"):
                try:
                    geom_a = load_wkt(grp["wkt_a"].iloc[0])
                except Exception:
                    continue
                if not isinstance(geom_a, LineString):
                    continue
                coords_a = list(geom_a.coords)
                b_edges = []
                for r in grp.itertuples(index=False):
                    try:
                        gb = load_wkt(r.wkt_b)
                    except Exception:
                        continue
                    if isinstance(gb, LineString):
                        b_edges.append((r.id_b, gb))
                if b_edges:
                    tasks.append((id_a, coords_a, b_edges))

        logger.info("graph-DTW: %d A-edge tasks to align", len(tasks))

        # Run graph-DTW per A-edge (optionally in parallel).
        t_align = time.time()
        if n_jobs and n_jobs != 1 and tasks:
            from joblib import Parallel, delayed
            logger.info("graph-DTW: aligning in parallel (n_jobs=%d)...", n_jobs)
            outcomes = Parallel(n_jobs=n_jobs)(
                delayed(_graph_dtw_group)(t, snap_tolerance_m, step_meters, trim_ends_m,
                                          emission, bearing_weight, alpha, beta)
                for t in tasks
            )
        else:
            outcomes = []
            n_tasks = len(tasks)
            step = max(1, n_tasks // 10)
            for k, t in enumerate(tasks):
                outcomes.append(_graph_dtw_group(t, snap_tolerance_m, step_meters,
                                                 trim_ends_m, emission, bearing_weight,
                                                 alpha, beta))
                if (k + 1) % step == 0 or (k + 1) == n_tasks:
                    logger.info("graph-DTW: aligned %d/%d A-edges (%.0f%%)",
                                k + 1, n_tasks, 100.0 * (k + 1) / max(1, n_tasks))
        logger.info("graph-DTW: alignment finished in %.1fs", time.time() - t_align)

        long_rows, summary_rows, matched_ids = [], [], set()
        for id_a, res in outcomes:
            if res is None:
                continue
            matched_ids.add(id_a)
            m = res["metrics"]
            route = res["route"]
            # one row per B-edge in the route, each with its OWN sliced metrics
            for re in m["route_edges"]:
                long_rows.append({
                    "source_id": id_a, "dest_id": re["dest_id"], "seq": re["seq"],
                    "direction": re["direction"],
                    "edge_match_dist_avg": re["match_dist_avg"],
                    "edge_match_dist_max": re["match_dist_max"],
                    "edge_match_dist_min": re["match_dist_min"],
                    "edge_a_len": re["a_len"],            # A length COVERED by this edge (m)
                    "edge_cover_pct": re["cover_pct"],    # % of the whole A-edge this edge covers
                    "edge_matched_len": re["matched_len"],
                    "edge_b_len": re["b_edge_len"],
                    "edge_b_used_pct": re["b_cover_pct"],
                    "edge_bearing_diff": re["bearing_diff"],
                    "n_points": re["n_points"],
                    "route_match_dist": m["average"], "n_edges": m["n_edges"],
                })
            # matched path geometry (UTM) from the B-side of the warping path
            bpts = []
            for _pa, pb in res["warping_path"]:
                if not bpts or abs(bpts[-1][0] - pb[0]) > 1e-9 or abs(bpts[-1][1] - pb[1]) > 1e-9:
                    bpts.append(pb)
            route_geom_wkt = LineString(bpts).wkt if len(bpts) >= 2 else None
            summary_rows.append({
                "source_id": id_a, "n_edges": m["n_edges"],
                "dest_ids": [d for (d, _dir, _s) in route],
                "dtw_distance": m["average"], "max_dtw_distance": m["max"],
                "min_dtw_distance": m["min"], "bearing_diff": m["bearing_diff"],
                "part_drift": m["part_drift"], "part_bearing_diff": m["part_bearing_diff"],
                "overlap_pct": m["overlap_pct"], "matched_len": m["matched_len"],
                "route_geom_wkt": route_geom_wkt,
                "match_type": "1:1" if m["n_edges"] == 1 else "1:N_ROUTE",
            })

        routes_long = pd.DataFrame(long_rows, columns=self.ROUTES_LONG_COLUMNS)
        routes_summary = pd.DataFrame(summary_rows, columns=self.ROUTES_SUMMARY_COLUMNS)

        # Append NO_MATCH rows so every Source-A edge is accounted for.
        all_ids = set(self._get_all_ids_a()["id_a"].tolist())
        unmatched = all_ids - matched_ids
        if unmatched:
            nm = pd.DataFrame([{
                "source_id": i, "n_edges": 0, "dest_ids": None, "dtw_distance": float("nan"),
                "max_dtw_distance": float("nan"), "min_dtw_distance": float("nan"),
                "bearing_diff": float("nan"), "part_drift": float("nan"),
                "part_bearing_diff": float("nan"), "overlap_pct": pd.NA, "matched_len": float("nan"),
                "route_geom_wkt": None, "match_type": "NO_MATCH",
            } for i in unmatched], columns=self.ROUTES_SUMMARY_COLUMNS)
            routes_summary = pd.concat([routes_summary, nm], ignore_index=True)

        n_matched = len(matched_ids)
        n_nomatch = int((routes_summary["match_type"] == "NO_MATCH").sum())
        n_multi = int((routes_summary["match_type"] == "1:N_ROUTE").sum())
        logger.info("graph-DTW: %d matched (%d multi-edge routes), %d NO_MATCH; "
                    "%d route-edge rows; total %.1fs",
                    n_matched, n_multi, n_nomatch, len(routes_long), time.time() - t_start)

        # Enforce the column-type schema (stable dtypes; ids stay Int64, never float).
        routes_long = self._apply_dtypes(routes_long, self.ROUTES_LONG_DTYPES)
        routes_summary = self._apply_dtypes(routes_summary, self.ROUTES_SUMMARY_DTYPES)
        return routes_long, routes_summary

    def match_dag(self, alpha: float = 1.0, beta: float = 1.0, engine: str = "cell",
                  bearing_weight: float = 2.0, max_distance: Optional[float] = None,
                  step_meters: float = 5.0, snap_decimals: int = 3):
        """Mode 3 — DAG-to-network matching (DAG-DTW), on the SAME configured sources as the other
        modes: source A must form a directed **acyclic** graph (a route tree, a sensor cone, a
        divided road that rejoins); B may cycle. Geometry is transformed to ``utm_srid`` and both
        edge tables are converted to ``networkx`` graphs internally (each polyline densified at
        ``step_meters`` — this supplies the subdivision; shared endpoints become junctions snapped
        at ``snap_decimals``). Runs the segment-mode pipeline (arc states with a ``bearing_weight``
        heading term) and the chosen extraction engine (``"cell"`` exact / ``"join"``
        cross-validation / ``"all"`` = cheapest valid of the two); ``alpha ∈ (0,1]`` discounts 1:N
        coverage, ``beta ∈ [1,∞)`` penalizes N:1 stalls (docs/dag_dtw_matching.md §3).

        Returns ``(dag_long, dag_summary)`` — the Mode-1-style pair:

        - ``dag_long``: one row per (A-edge, B-edge) the matching connects — ``source_id, dest_id,
          seq`` (order along the A-edge), ``n_pairs`` (matched arc pairs), ``avg_dist_m`` (mean
          midpoint drift, meters).
        - ``dag_summary``: one row per A-edge — ``source_id, dest_ids`` (ordered ``;``-join),
          ``n_dest, n_pairs, avg_dist_m, match_type`` (``1:1`` / ``1:N_ROUTE``).

        Raises ``NotADAG`` if source A has a directed cycle, and ``ValueError`` on infeasibility
        (increase ``max_distance``)."""
        from shapely import wkt as _shapely_wkt
        from .dag_dtw import (edges_to_digraph, line_digraph, prepare, forward,
                              extract_join, extract_cell, _cost_of, check_rules)
        if not self.source_a or not self.source_b or not self.utm_srid:
            raise ValueError("Sources not configured. Call configure_sources() first.")
        r = float(max_distance if max_distance is not None else (self.max_distance or 30.0))
        axy = f", always_xy := {'true' if self.always_xy else 'false'}"

        def fetch(source, cols):
            q = f"""SELECT {cols['id']} AS id,
                           ST_AsText(ST_Transform({cols['geom']}, 'EPSG:4326',
                                                  'EPSG:{self.utm_srid}'{axy})) AS wkt
                    FROM {source}"""
            df = self.conn.execute(q).fetchdf()
            return [(row.id, list(_shapely_wkt.loads(row.wkt).coords)) for row in df.itertuples()]

        A = edges_to_digraph(fetch(self.source_a, self.columns_a), step_meters, snap_decimals)
        B = edges_to_digraph(fetch(self.source_b, self.columns_b), step_meters, snap_decimals)
        LA, LB = line_digraph(A), line_digraph(B)
        for (u, v) in LA.nodes:                                 # carry the input-edge bookkeeping
            LA.nodes[(u, v)]["road_id"] = A[u][v]["road_id"]
            LA.nodes[(u, v)]["seq"] = A[u][v]["seq"]
        for (u, v) in LB.nodes:
            LB.nodes[(u, v)]["road_id"] = B[u][v]["road_id"]
        prepare(LA, LB, r=r, bearing_weight=bearing_weight)
        forward(LA, LB, alpha=alpha, beta=beta)
        engines = {"cell": extract_cell, "join": extract_join}
        if engine in engines:
            M, _ = engines[engine](LA, LB, alpha, beta)
        elif engine == "all":                                   # cheapest valid of the two
            best = None
            for fn in (extract_cell, extract_join):
                try:
                    Mx, _ = fn(LA, LB, alpha, beta)
                except ValueError:
                    continue
                c = _cost_of(LA, LB, Mx, alpha, beta)
                if best is None or c < best[0] - 1e-12:
                    best = (c, Mx)
            if best is None:
                raise ValueError("both extraction engines infeasible -- increase max_distance")
            M = best[1]
        else:
            raise ValueError(f"unknown engine {engine!r} (use 'cell', 'join' or 'all')")

        rows = []
        for (sa, sb) in M:                                      # arc pair -> input-edge pair + drift
            ax, ay = LA.nodes[sa]["x"], LA.nodes[sa]["y"]
            bx, by = LB.nodes[sb]["x"], LB.nodes[sb]["y"]
            rows.append(dict(source_id=LA.nodes[sa]["road_id"], dest_id=LB.nodes[sb]["road_id"],
                             a_seq=LA.nodes[sa]["seq"],
                             dist_m=float(np.hypot(ax - bx, ay - by))))
        pairs = pd.DataFrame(rows)
        dag_long = (pairs.groupby(["source_id", "dest_id"], as_index=False)
                    .agg(seq=("a_seq", "min"), n_pairs=("a_seq", "size"),
                         avg_dist_m=("dist_m", "mean")))
        dag_long["seq"] = dag_long.groupby("source_id")["seq"].rank(method="first").astype(int)
        dag_long = dag_long.sort_values(["source_id", "seq"]).reset_index(drop=True)
        summary = (dag_long.sort_values(["source_id", "seq"])
                   .groupby("source_id", as_index=False)
                   .agg(dest_ids=("dest_id", lambda d: ";".join(map(str, d))),
                        n_dest=("dest_id", "nunique"), n_pairs=("n_pairs", "sum"),
                        avg_dist_m=("avg_dist_m", "mean")))
        summary["match_type"] = np.where(summary["n_dest"] > 1, "1:N_ROUTE", "1:1")
        return dag_long, summary

    def match_routes(self, snap_tolerance_m: float = 0.75, step_meters: float = 10.0,
                     trim_ends_m: float = 0.0, n_jobs: int = 1,
                     emission: str = "point", bearing_weight: float = 0.0,
                     alpha: float = 1.0, beta: float = 1.0):
        """Run the full route-based (graph-DTW) pipeline: generate candidates, then align each
        Source-A edge to the local B-graph. Returns ``(routes_long, routes_summary)`` -- the
        graph-DTW analogue of :meth:`match`. See :meth:`compute_graph_dtw_routes`.

        ``emission="segment"`` uses the middle-to-middle segment cost (optionally with
        ``bearing_weight``); default ``"point"`` is unchanged. ``alpha``/``beta`` weight the
        1:N coverage / N:1 stall steps in BOTH emission modes (same semantics as
        :meth:`match_dag`; defaults ``1``/``1`` = unweighted). See
        ``docs/weighted_emission.md`` §12.

        ``routes_summary`` also carries **``part_drift``** and **``part_bearing_diff``**: the same
        mean drift / heading diff as ``dtw_distance`` / ``bearing_diff`` but restricted to the
        edge's **overlap part** -- the end-trimmed span that also defines ``overlap_pct``. Of the
        leading run of A-units piled on the route's first B-arc/vertex only the last member
        overlaps, of the trailing run on its last unit only the first; the rest of the edge is the
        overlap part (``docs/graph_dtw_matching.md`` §4.1). They enable a *partial-match* tier: an
        A-edge whose full metrics fail but whose ``part_*`` pass has an interior that follows B,
        only its ends diverging. ``part_bearing_diff`` is per-segment and segment-emission only
        (equals ``bearing_diff`` in point mode); both fall back to the full measures when the route
        lives on a single arc/vertex."""
        candidates_df = self.generate_candidate_pairs()
        return self.compute_graph_dtw_routes(
            candidates_df, snap_tolerance_m=snap_tolerance_m, step_meters=step_meters,
            trim_ends_m=trim_ends_m, n_jobs=n_jobs,
            emission=emission, bearing_weight=bearing_weight, alpha=alpha, beta=beta,
        )

    def resolve_routes(self, routes_summary: pd.DataFrame, routes_long: Optional[pd.DataFrame] = None,
                       *, max_match_dist: Optional[float] = None,
                       max_bearing_diff: Optional[float] = None,
                       min_overlap_pct: Optional[float] = None):
        """Filter route matches by quality thresholds (the route-mode analogue of the edge-to-edge
        quality filters).

        :meth:`match_routes` returns the best route for **every** A-edge regardless of quality.
        ``resolve_routes`` keeps only routes that pass the given thresholds; an A-edge whose route
        fails any threshold is reset to a ``NO_MATCH`` row (route cleared, metrics ``NaN``) and its
        rows are removed from ``routes_long``. Any threshold left ``None`` is not applied.

        Parameters
        ----------
        routes_summary, routes_long:
            The two tables from :meth:`match_routes`. ``routes_long`` is optional.
        max_match_dist:
            Drop routes whose average match distance (``dtw_distance``, meters) exceeds this.
        max_bearing_diff:
            Drop routes whose whole-route bearing difference (degrees) exceeds this.
        min_overlap_pct:
            Drop routes covering less than this percent of the A-edge (``overlap_pct``). Coverage
            is < 100 where A **overhangs** past the route's first/last B-edge endpoint (its end
            samples pile onto a single vertex); this drops edges whose ends stick out too far.

        Returns
        -------
        ``routes_summary`` (failing routes turned into ``NO_MATCH``), or the tuple
        ``(routes_summary, routes_long)`` if ``routes_long`` was given.
        """
        rs = routes_summary.copy()
        matched = rs["match_type"] != "NO_MATCH"
        fail = pd.Series(False, index=rs.index)
        if max_match_dist is not None:
            fail |= matched & (rs["dtw_distance"] > max_match_dist)
        if max_bearing_diff is not None:
            fail |= matched & (rs["bearing_diff"] > max_bearing_diff)
        if min_overlap_pct is not None:
            ov = pd.to_numeric(rs["overlap_pct"], errors="coerce")
            fail |= matched & (ov < min_overlap_pct)

        failed_ids = set(rs.loc[fail, "source_id"])
        nan = float("nan")
        for col in ("dtw_distance", "max_dtw_distance", "min_dtw_distance",
                    "bearing_diff", "matched_len"):
            rs.loc[fail, col] = nan
        rs["overlap_pct"] = rs["overlap_pct"].astype("Int64")
        rs.loc[fail, "overlap_pct"] = pd.NA
        rs.loc[fail, "n_edges"] = 0
        rs.loc[fail, "dest_ids"] = None
        rs.loc[fail, "route_geom_wkt"] = None
        rs.loc[fail, "match_type"] = "NO_MATCH"

        n_kept = int((rs["match_type"] != "NO_MATCH").sum())
        logger.info("resolve_routes: %d routes failed thresholds -> NO_MATCH; %d kept",
                    len(failed_ids), n_kept)

        if routes_long is None:
            return rs
        rl = routes_long[~routes_long["source_id"].isin(failed_ids)].copy()
        return rs, rl


