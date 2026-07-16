"""Mode 3 edge-level parts output (``parts_from_matching`` / ``match_dag(parts=True)``,
docs/dag_dtw_matching.md §11): the per-edge decomposition into contiguous (A-edge, B-edge)
parts with per-part drift/bearing scores, pair counts, and the non-overlapping B head/tail.

Geometry is built with round-number coordinates so ``_densify`` resample points land exactly
on the B-junction vertices (edges_to_digraph snaps shared endpoints into one node)."""
import csv
import math

import pytest

from network_matching.dag_dtw import (edges_to_digraph, line_digraph, prepare, forward,
                                      extract_cell, parts_from_matching)


def _run(edges_a, edges_b, r=10.0, step=5.0, bw=2.0):
    """Mirror the DuckDB pipeline: densified digraphs -> line graphs carrying
    road_id/seq/length -> prepare/forward/extract_cell -> parts."""
    A = edges_to_digraph(edges_a, step_meters=step)
    B = edges_to_digraph(edges_b, step_meters=step)
    LA, LB = line_digraph(A), line_digraph(B)
    for (u, v) in LA.nodes:
        LA.nodes[(u, v)]["road_id"] = A[u][v]["road_id"]
        LA.nodes[(u, v)]["seq"] = A[u][v]["seq"]
    for (u, v) in LB.nodes:
        LB.nodes[(u, v)]["road_id"] = B[u][v]["road_id"]
        LB.nodes[(u, v)]["seq"] = B[u][v]["seq"]
    prepare(LA, LB, r=r, bearing_weight=bw)
    forward(LA, LB)
    M, _ = extract_cell(LA, LB, 1.0, 1.0)
    return M, LA, LB, parts_from_matching(M, LA, LB)


def test_parts_split_two_b_edges():
    """A single 40 m A-edge over two 20 m B-edges -> two ordered contiguous parts with the
    expected spans, per-part drift ~= the 2 m offset, straight bearings, no B leftovers."""
    M, _, _, parts = _run([("a", [(0, 0), (40, 0)])],
                          [("b1", [(0, 2), (20, 2)]), ("b2", [(20, 2), (40, 2)])])
    assert [p["dest_id"] for p in parts] == ["b1", "b2"]
    assert [p["part"] for p in parts] == [1, 2]
    p1, p2 = parts
    assert p1["a_from_m"] == pytest.approx(0.0, abs=1e-6)
    assert p1["a_to_m"] == pytest.approx(p2["a_from_m"], abs=1e-6)      # contiguous along A
    assert p2["a_to_m"] == pytest.approx(40.0, abs=1e-6)
    assert p1["a_pct"] + p2["a_pct"] == pytest.approx(100.0, abs=1e-6)
    for p in parts:
        assert p["n_a_arcs"] == 4 and p["n_pairs"] >= 4
        assert p["drift_m"] == pytest.approx(2.0, abs=0.5)
        assert p["drift_max_m"] >= p["drift_m"]
        assert p["bearing_diff_deg"] == pytest.approx(0.0, abs=1e-6)
        assert p["b_head_m"] == pytest.approx(0.0, abs=1e-6)            # whole B edge used
        assert p["b_tail_m"] == pytest.approx(0.0, abs=1e-6)
        assert p["b_len_m"] == pytest.approx(20.0, abs=1e-6)


def test_parts_b_overhang_saved():
    """A 20 m A-edge matching the middle of a 40 m B-edge: the non-overlapping begin/end of
    the B match are saved as b_head_m/b_tail_m (~10 m each)."""
    M, _, _, parts = _run([("a", [(10, 0), (30, 0)])],
                          [("b", [(0, 2), (40, 2)])])
    assert len(parts) == 1
    p = parts[0]
    assert p["b_len_m"] == pytest.approx(40.0, abs=1e-6)
    assert p["b_from_m"] == pytest.approx(10.0, abs=2.5)                # ~half an arc slack
    assert p["b_to_m"] == pytest.approx(30.0, abs=2.5)
    assert p["b_head_m"] == pytest.approx(10.0, abs=2.5)
    assert p["b_tail_m"] == pytest.approx(10.0, abs=2.5)
    assert p["a_pct"] == pytest.approx(100.0, abs=1e-6)                 # all of A covered (V4)


def test_parts_route_overhang_rows():
    """An A-edge extending ~10 m past each end of B gets explicit rows for the route's
    begin/end non-overlap (part_type head/tail), and the match part's scores exclude them."""
    M, _, _, parts = _run([("a", [(-10, 0), (50, 0)])],
                          [("b", [(0, 2), (40, 2)])])
    assert [p["part_type"] for p in parts] == ["head", "match", "tail"]
    head, match, tail = parts
    assert [p["part"] for p in parts] == [1, 2, 3]
    assert head["a_from_m"] == pytest.approx(0.0, abs=1e-6)
    assert head["a_len_m"] == pytest.approx(10.0, abs=3.0)
    assert tail["a_to_m"] == pytest.approx(60.0, abs=1e-6)
    assert tail["a_len_m"] == pytest.approx(10.0, abs=3.0)
    assert match["a_from_m"] == pytest.approx(head["a_to_m"], abs=1e-6)   # contiguous partition
    assert match["a_to_m"] == pytest.approx(tail["a_from_m"], abs=1e-6)
    assert match["drift_m"] == pytest.approx(2.0, abs=0.5)                # clean, overhang excluded
    assert head["drift_m"] > match["drift_m"]                             # overhang drift grows
    assert tail["drift_m"] > match["drift_m"]


def test_parts_exact_cover_has_no_overhang_rows():
    """A perfectly covered A-edge emits match parts only."""
    M, _, _, parts = _run([("a", [(0, 0), (40, 0)])],
                          [("b1", [(0, 2), (20, 2)]), ("b2", [(20, 2), (40, 2)])])
    assert all(p["part_type"] == "match" for p in parts)


def test_parts_reentry_kept_separate():
    """A route that leaves a B-edge onto a detour and returns to it yields THREE parts
    (b1, b2, b1) -- the (source, dest) grain of dag_long collapses this into two rows."""
    a_coords = [(0, 0), (20, 0), (22, 10), (28, 10), (30, 0), (40, 0)]
    M, _, _, parts = _run([("a", a_coords)],
                          [("b1", [(0, 2), (40, 2)]),
                           ("b2", [(20, 2), (22, 12), (28, 12), (30, 2)])])
    dests = [p["dest_id"] for p in parts]
    assert dests == ["b1", "b2", "b1"]
    assert len({(p["source_id"], p["dest_id"]) for p in parts}) == 2    # the collapsed grain
    first_b1, detour, second_b1 = parts
    assert first_b1["b_tail_m"] > 5.0                                   # b1 continues past the exit
    assert second_b1["b_head_m"] > 5.0                                  # ... and before the return
    assert first_b1["b_to_m"] < second_b1["b_from_m"]                   # forward along b1
    assert detour["bearing_diff_deg"] < 45.0
    # parts are ordered along A and cover it fully (V4)
    assert first_b1["a_from_m"] == pytest.approx(0.0, abs=1e-6)
    assert first_b1["a_to_m"] <= detour["a_to_m"] <= second_b1["a_to_m"] + 1e-9
    total = sum(p["a_len_m"] for p in parts)
    a_len = sum(math.hypot(x2 - x1, y2 - y1)
                for (x1, y1), (x2, y2) in zip(a_coords, a_coords[1:]))
    assert total == pytest.approx(a_len, rel=0.05)


def test_parts_head_when_match_spans_two_b_arcs():
    """Regression: the head/tail scan must not be suppressed when the whole matching uses only
    two B-arcs (the leading stall run abuts the trailing one). A 25 m A-edge over a B-edge that
    starts 15 m in: the first 15 m are a genuine head overhang, not part of a 100% match."""
    M, _, _, parts = _run([("a", [(0, 0), (25, 0)])],
                          [("b", [(15, 3), (25, 3)])], r=25.0)
    assert [p["part_type"] for p in parts] == ["head", "match"]
    head, match = parts
    assert head["a_from_m"] == pytest.approx(0.0, abs=1e-6)
    assert head["a_len_m"] == pytest.approx(15.0, abs=5.0)
    assert match["drift_m"] < head["drift_m"]


def test_parts_partition_the_a_edge_at_unaligned_junction():
    """Regression: an A-arc straddling a B junction (junction NOT on a densify point) must be
    attributed to ONE part -- parts stay a true partition, so spans never overlap and a_pct
    sums to 100%, keeping the docs §11.2 matched_pct recipe sound."""
    M, _, _, parts = _run([("a", [(0, 0), (30, 0)])],
                          [("b0", [(0, 2), (9, 2)]), ("b1", [(9, 2), (11, 2)]),
                           ("b2", [(11, 2), (30, 2)])], r=15.0)
    assert len(parts) >= 2
    for p, q in zip(parts, parts[1:]):                      # contiguous, non-overlapping
        assert q["a_from_m"] == pytest.approx(p["a_to_m"], abs=1e-9)
    assert parts[0]["a_from_m"] == pytest.approx(0.0, abs=1e-9)
    assert parts[-1]["a_to_m"] == pytest.approx(30.0, abs=1e-9)
    assert sum(p["a_len_m"] for p in parts) == pytest.approx(30.0, abs=1e-6)
    assert sum(p["a_pct"] for p in parts) == pytest.approx(100.0, abs=1e-6)


def test_parts_duplicate_edge_id_rows():
    """Regression: a multipart geometry exported as two rows sharing one edge id must not collide
    in (road_id, seq) -- spans stay a partition and the two disjoint stretches match cleanly."""
    M, _, _, parts = _run([("e1", [(0, 0), (20, 0)]), ("e1", [(100, 0), (120, 0)])],
                          [("b1", [(0, 2), (20, 2)]), ("b2", [(100, 2), (120, 2)])], r=15.0)
    assert [p["dest_id"] for p in parts] == ["b1", "b2"]     # no phantom alternation
    assert sum(p["a_len_m"] for p in parts) == pytest.approx(40.0, abs=1e-6)
    assert sum(p["a_pct"] for p in parts) <= 100.0 + 1e-6
    for p, q in zip(parts, parts[1:]):
        assert q["a_from_m"] == pytest.approx(p["a_to_m"], abs=1e-9)


def test_parts_numeric_id_order_is_geometric_not_lexicographic():
    """Regression: within one A-arc, pairs must be ordered by B's chain, not by str(dest_id) --
    with ids 2 and 10, "10" < "2" lexicographically and would invent a re-entry (10, 2, 10)."""
    M, _, _, parts = _run([("a", [(0, 0), (20, 0)])],
                          [(2, [(0, 2), (10, 2)]), (10, [(10, 2), (20, 2)])], r=15.0)
    assert [p["dest_id"] for p in parts] == [2, 10]          # geometric order along A
    assert [p["part"] for p in parts] == [1, 2]


def test_parts_require_pipeline_attributes():
    """Point-mode graphs carry no road_id/seq/length -> a clear ValueError, not a KeyError."""
    from network_matching.dag_dtw import digraph
    A = digraph({0: (0, 0), 1: (10, 0)}, [(0, 1)])
    B = digraph({"u": (0, 1), "v": (10, 1)}, [("u", "v")])
    prepare(A, B, r=20.0)
    forward(A, B)
    M, _ = extract_cell(A, B, 1.0, 1.0)
    with pytest.raises(ValueError, match="road_id"):
        parts_from_matching(M, A, B)


# ---------------------------------------------------------------------------------------
# DuckDB pipeline (DuckDBMapMatcher.match_dag)
# ---------------------------------------------------------------------------------------
LON0, LAT0 = 18.06, 59.33
MX = 1.0 / (111320 * math.cos(math.radians(LAT0)))
MY = 1.0 / 111320.0


def _ls(pts):
    return "LINESTRING (" + ", ".join(f"{LON0 + x * MX:.8f} {LAT0 + y * MY:.8f}"
                                      for x, y in pts) + ")"


@pytest.fixture()
def matcher(tmp_path):
    from network_matching import DuckDBMapMatcher
    edges_a = [(1, _ls([(0, 0), (200, 0)]))]
    edges_b = [(11, _ls([(0, 3), (100, 3)])), (12, _ls([(100, 3), (200, 3)]))]
    for name, rows in (("a.csv", edges_a), ("b.csv", edges_b)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fid", "geometry"])
            w.writerows(rows)
    return DuckDBMapMatcher.from_wkt_csv(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
                                         id_a="fid", id_b="fid", utm_srid=3006,
                                         max_distance=30)


def test_match_dag_parts_pipeline(matcher):
    dag_long, dag_summary, dag_parts = matcher.match_dag(step_meters=10.0, parts=True)
    # additive columns on the existing tables
    assert "avg_bearing_diff" in dag_long.columns
    assert list(dag_summary.columns) == ["source_id", "dest_ids", "n_dest", "n_parts",
                                         "n_pairs", "avg_dist_m", "avg_bearing_diff",
                                         "a_head_m", "a_tail_m", "match_type"]
    # the parts table decomposes the single A edge over both B edges, in order
    assert list(dag_parts["dest_id"]) == [11, 12]
    assert list(dag_parts["part"]) == [1, 2]
    assert (dag_parts["part_type"] == "match").all()            # exact cover, no overhang
    assert dag_summary["a_head_m"].iloc[0] == pytest.approx(0.0)
    assert dag_summary["a_tail_m"].iloc[0] == pytest.approx(0.0)
    assert dag_parts["drift_m"].max() < 10.0
    assert dag_parts["bearing_diff_deg"].max() < 10.0
    # consistency with the summary aggregates
    assert dag_parts["n_pairs"].sum() == dag_summary["n_pairs"].iloc[0]
    assert dag_summary["n_parts"].iloc[0] == len(dag_parts)
    assert dag_summary["match_type"].iloc[0] == "1:N_ROUTE"
    # whole-edge score is composable from the parts (length-weighted drift)
    w = (dag_parts["a_len_m"] * dag_parts["drift_m"]).sum() / dag_parts["a_len_m"].sum()
    assert 0.0 < w < 10.0


def test_match_dag_default_return_unchanged(matcher):
    out = matcher.match_dag(step_meters=10.0)
    assert isinstance(out, tuple) and len(out) == 2
    dag_long, dag_summary = out
    for col in ("source_id", "dest_id", "seq", "n_pairs", "avg_dist_m"):
        assert col in dag_long.columns
    for col in ("source_id", "dest_ids", "n_dest", "n_pairs", "avg_dist_m", "match_type"):
        assert col in dag_summary.columns
