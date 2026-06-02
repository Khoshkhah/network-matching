"""
Estimate good ``resolve_routes`` quality thresholds from graph-DTW match results.

For each quality signal -- average match distance, max match distance, bearing difference, and
A-coverage -- this runs several outlier / two-population estimators and recommends a cut that
separates the good cluster from the tail of wrong/poor matches. The recommended values map
directly to ``resolve_routes`` keywords.

Run (rebuild matches from the data):
    python scripts/suggest_thresholds.py
    python scripts/suggest_thresholds.py --osm data/osm_edges.csv --sweden data/sweden_edges.csv \
        --out output/threshold_suggestions.png

Or analyze a previously-saved routes_summary CSV:
    python scripts/suggest_thresholds.py --summary-csv output/routes_summary.csv

Writes a 2x2 diagnostic PNG (histogram + KDE + candidate cuts per metric) to ``--out`` and prints
a copy-paste ``resolve_routes(...)`` kwargs line.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import (  # noqa: E402
    DuckDBMapMatcher, suggest_thresholds, setup_logging, get_logger,
)

log = get_logger("scripts.suggest_thresholds")


def _load_summary(args) -> pd.DataFrame:
    """Either read a saved routes_summary CSV, or rebuild it by running match_routes on the data."""
    if args.summary_csv:
        log.info("loading routes_summary from %s", args.summary_csv)
        return pd.read_csv(args.summary_csv)
    log.info("rebuilding matches from %s / %s", args.osm, args.sweden)
    m = DuckDBMapMatcher.from_wkt_csv(
        args.osm, args.sweden, id_a=args.id_a, id_b=args.id_b,
        utm_srid=args.utm_srid, max_distance=args.max_distance)
    _rl, rs = m.match_routes(snap_tolerance_m=args.snap, step_meters=args.step, n_jobs=args.n_jobs)
    return rs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm", default="data/osm_edges.csv")
    ap.add_argument("--sweden", default="data/sweden_edges.csv")
    ap.add_argument("--summary-csv", default=None,
                    help="analyze this saved routes_summary CSV instead of rebuilding")
    ap.add_argument("--out", default="output/threshold_suggestions.png")
    ap.add_argument("--id-a", default="edge_id")
    ap.add_argument("--id-b", default="directed_id")
    ap.add_argument("--utm-srid", type=int, default=3006)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--snap", type=float, default=0.75)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    # estimator tuning
    ap.add_argument("--k-iqr", type=float, default=1.5)
    ap.add_argument("--k-mad", type=float, default=3.0)
    ap.add_argument("--percentile", type=float, default=97.5)
    ap.add_argument("--sep-min", type=float, default=2.0,
                    help="min standardized GMM component separation to count a metric as bimodal")
    args = ap.parse_args()

    setup_logging()
    rs = _load_summary(args)

    result = suggest_thresholds(
        rs, report=True, plot_path=args.out,
        k_iqr=args.k_iqr, k_mad=args.k_mad, percentile=args.percentile, sep_min=args.sep_min)

    kwargs = ", ".join(f"{k}={v}" for k, v in result["recommended"].items() if v is not None)
    print(f"\nApply with:\n  rs2, rl2 = m.resolve_routes(rs, rl, {kwargs})")
    if result.get("plot_path"):
        print(f"Saved {result['plot_path']}")


if __name__ == "__main__":
    main()
