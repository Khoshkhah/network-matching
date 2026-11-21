"""Measure the graph-DTW speed of the Python and the Rust engine, and compare their tables.

    python scripts/bench_rust_engine.py                        # data/ (OSM -> NVDB, Sundbyberg)
    python scripts/bench_rust_engine.py --monaco monaco.duckdb # also duckOSM Monaco driving.edges onto itself

Each case makes the candidate pairs once (DuckDB), then runs ``compute_graph_dtw_routes`` with
``n_jobs=1`` for each engine and each emission. The time is the time of the whole call (the
alignment and the output tables). The script stops with an error if the tables are not equal.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network_matching import DuckDBMapMatcher  # noqa: E402
from network_matching.graph_dtw import rust_available  # noqa: E402

RUNS = {"point": dict(emission="point"),
        "segment": dict(emission="segment", bearing_weight=2.0)}


def data_matcher():
    return DuckDBMapMatcher.from_wkt_csv("data/osm_edges.csv", "data/sweden_edges.csv",
                                         id_a="edge_id", id_b="directed_id", utm_srid=3006,
                                         max_distance=30)


def monaco_matcher(path):
    m = DuckDBMapMatcher()
    m.conn.execute(f"ATTACH '{path}' AS mc (READ_ONLY)")
    m.conn.execute("CREATE TABLE edges AS SELECT edge_id, geometry FROM mc.driving.edges")
    m.configure_sources("edges", "edge_id", "geometry", "edges", "edge_id", "geometry", 32632)
    m.set_parameters(max_distance=30)
    return m


def bench(name, m):
    t = time.perf_counter()
    cand = m.generate_candidate_pairs()
    print(f"\n{name}: {cand['id_a'].nunique()} A-edges, {len(cand)} candidate pairs "
          f"({time.perf_counter() - t:.2f} s in DuckDB)")
    for run, kw in RUNS.items():
        out, secs = {}, {}
        for engine in ("python", "rust"):
            t = time.perf_counter()
            out[engine] = m.compute_graph_dtw_routes(cand, engine=engine, **kw)
            secs[engine] = time.perf_counter() - t
        for k in range(2):
            a, b = (out[e][k].sort_values("source_id", kind="stable").reset_index(drop=True)
                    for e in ("python", "rust"))
            if "seq" in a:
                a, b = (x.sort_values(["source_id", "seq"]).reset_index(drop=True) for x in (a, b))
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        matched = int((out["rust"][1]["match_type"] != "NO_MATCH").sum())
        print(f"  {run:8s} python {secs['python']:7.2f} s   rust {secs['rust']:6.2f} s   "
              f"x{secs['python'] / secs['rust']:5.1f}   {matched} matched, tables equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monaco", help="a duckOSM Monaco .duckdb with the table driving.edges")
    args = ap.parse_args()
    if not rust_available():
        sys.exit("network_matching._rust is not built: pip install -e . (needs Rust)")
    bench("data/ OSM -> NVDB", data_matcher())
    if args.monaco:
        bench("Monaco driving.edges -> itself", monaco_matcher(args.monaco))


if __name__ == "__main__":
    main()
