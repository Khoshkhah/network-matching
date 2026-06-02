"""
Tests for the threshold-suggestion tool (network_matching/thresholds.py).

Build a synthetic ``routes_summary`` with a tight "good" cluster plus an outlier tail and check
that the recommended cut lands in the gap between them. Runnable two ways:

    pytest tests/test_thresholds.py        # assertion-based
    python tests/test_thresholds.py        # prints the report

Scenarios:
  1. BIMODAL distance  -- good cluster ~2m + outlier tail 60..300m -> cut in the gap, "bimodal".
  2. OVERLAP (mirror)  -- good ~99% + bad 10..80% -> a sensible min_overlap in the gap.
  3. UNIMODAL          -- all-good column, no tail -> finite robust fence, "unimodal".
  4. EDGE CASES        -- all-NaN, tiny N, constant column: finite/None, no crash.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import suggest_thresholds, isolation_forest_flags  # noqa: E402


def _sklearn_available():
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def _make_summary(rng):
    """A routes_summary-shaped frame: bimodal distances/bearing, left-tailed overlap, NO_MATCH rows."""
    good = np.clip(rng.normal(2.0, 0.8, 1500), 0, None)
    bad = rng.uniform(60.0, 300.0, 120)
    dtw = np.concatenate([good, bad])

    # max distance: a bit higher than avg, same bimodal shape
    maxd = dtw + np.abs(rng.normal(1.0, 0.5, dtw.size))

    # bearing: unimodal good cluster only (no real bad cluster)
    bearing = np.abs(rng.normal(3.0, 1.5, dtw.size))

    # overlap: good near 100, bad tail down to 10..80
    good_ov = np.clip(rng.normal(99.0, 0.6, 1500), 0, 100)
    bad_ov = rng.uniform(10.0, 80.0, 120)
    overlap = np.concatenate([good_ov, bad_ov])

    n = dtw.size
    df = pd.DataFrame({
        "source_id": np.arange(n),
        "n_edges": 1,
        "dest_ids": [[0]] * n,
        "dtw_distance": dtw,
        "max_dtw_distance": maxd,
        "min_dtw_distance": np.clip(dtw - 1.0, 0, None),
        "bearing_diff": bearing,
        "overlap_pct": pd.array(np.round(overlap).astype("int64"), dtype="Int64"),
        "matched_len": rng.uniform(50, 300, n),
        "route_geom_wkt": None,
        "match_type": "1:1",
    })
    # a block of NO_MATCH rows with NaN metrics, to exercise cleaning
    nm = pd.DataFrame({
        "source_id": np.arange(n, n + 40),
        "n_edges": 0,
        "dest_ids": None,
        "dtw_distance": np.nan,
        "max_dtw_distance": np.nan,
        "min_dtw_distance": np.nan,
        "bearing_diff": np.nan,
        "overlap_pct": pd.array([pd.NA] * 40, dtype="Int64"),
        "matched_len": np.nan,
        "route_geom_wkt": None,
        "match_type": "NO_MATCH",
    })
    return pd.concat([df, nm], ignore_index=True)


def test_bimodal_distance_cut_in_gap():
    rng = np.random.default_rng(0)
    rs = _make_summary(rng)
    out = suggest_thresholds(rs, report=False)
    sub = out["metrics"]["dtw_distance"]
    assert sub["n"] == 1620                       # NO_MATCH excluded
    assert 8.0 < sub["recommended"] < 60.0        # in the gap between good (~2) and bad (>=60)
    assert "bimodal" in sub["rationale"]
    # maps to the resolve_routes kwarg
    assert out["recommended"]["max_match_dist"] == sub["recommended"]


def test_overlap_min_in_gap():
    rng = np.random.default_rng(1)
    rs = _make_summary(rng)
    out = suggest_thresholds(rs, report=False)
    sub = out["metrics"]["overlap_pct"]
    assert sub["direction"] == "higher"
    # good overlaps ~99, bad up to 80 -> a min-overlap somewhere above the bad tail, below ~99
    assert 80.0 < sub["recommended"] <= 99.0
    assert out["recommended"]["min_overlap_pct"] == sub["recommended"]


def test_unimodal_bearing_robust_fence():
    rng = np.random.default_rng(2)
    rs = _make_summary(rng)
    out = suggest_thresholds(rs, report=False)
    sub = out["metrics"]["bearing_diff"]
    assert "unimodal" in sub["rationale"]
    # a finite fence above p95 but not absurd
    assert sub["summary"]["p95"] <= sub["recommended"] <= sub["summary"]["max"] + 1e-9


def test_recommended_maps_to_resolve_kwargs():
    rng = np.random.default_rng(3)
    rs = _make_summary(rng)
    out = suggest_thresholds(rs, report=False)
    assert set(out["recommended"]) == {
        "max_match_dist", "max_max_dist", "max_bearing_diff", "min_overlap_pct"}
    # all finite for this data
    assert all(v is not None for v in out["recommended"].values())


def test_edge_cases_no_crash():
    # all-NaN column, tiny N, and a constant column -- must not raise and must be finite/None
    df = pd.DataFrame({
        "source_id": np.arange(5),
        "dtw_distance": [1.0, 1.0, 1.0, 1.0, 1.0],       # constant
        "max_dtw_distance": [np.nan] * 5,                 # all NaN
        "bearing_diff": [2.0, 3.0, 100.0, 4.0, 5.0],      # tiny N
        "overlap_pct": pd.array([90, 95, 99, 100, 100], dtype="Int64"),
        "match_type": ["1:1"] * 5,
    })
    out = suggest_thresholds(df, report=False)
    assert out["metrics"]["max_dtw_distance"]["n"] == 0
    assert out["recommended"]["max_max_dist"] is None
    # constant column -> finite recommendation, no divide-by-zero
    assert out["recommended"]["max_match_dist"] is not None
    # tiny-N column still produces something finite
    assert out["recommended"]["max_bearing_diff"] is not None


@pytest.mark.skipif(not _sklearn_available(), reason="scikit-learn not installed")
def test_iforest_method_in_gap():
    rng = np.random.default_rng(4)
    rs = _make_summary(rng)
    out = suggest_thresholds(rs, report=False)
    sub = out["metrics"]["dtw_distance"]
    iforest = sub["methods"]["iforest"]
    assert iforest is not None
    # 1-D IsolationForest flags the low-density upper tail; the cut sits above the good-cluster
    # centre and below the bad cluster (it may dip into the good tail, unlike GMM/KDE).
    assert sub["summary"]["p50"] < iforest < 60.0


@pytest.mark.skipif(not _sklearn_available(), reason="scikit-learn not installed")
def test_isolation_forest_flags_multivariate():
    rng = np.random.default_rng(5)
    rs = _make_summary(rng)
    out = isolation_forest_flags(rs, report=False)
    assert {"if_outlier", "if_score"} <= set(out.columns)
    # NO_MATCH rows are never flagged
    assert not out.loc[out.match_type == "NO_MATCH", "if_outlier"].any()
    flagged = out[out.if_outlier]
    assert len(flagged) > 0
    # flagged matches skew toward the bad cluster (high distance), so their median dist far exceeds
    # the overall matched median
    matched = out[out.match_type != "NO_MATCH"]
    assert flagged["dtw_distance"].median() > 5 * matched["dtw_distance"].median()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rs = _make_summary(rng)
    suggest_thresholds(rs, report=True)
    isolation_forest_flags(rs, report=True)
    test_bimodal_distance_cut_in_gap()
    test_overlap_min_in_gap()
    test_unimodal_bearing_robust_fence()
    test_recommended_maps_to_resolve_kwargs()
    test_edge_cases_no_crash()
    if _sklearn_available():
        test_iforest_method_in_gap()
        test_isolation_forest_flags_multivariate()
    print("\nALL THRESHOLD TESTS PASSED")
