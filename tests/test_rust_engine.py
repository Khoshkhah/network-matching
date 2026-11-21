"""The Rust engine of graph-DTW gives the same result as the Python code.

Both engines run ``match_edge_to_bgraph`` on the synthetic cases (each perturbation family at each
grid magnitude, point and segment emission, with and without the alpha/beta step weights) and on
real OSM -> NVDB candidate groups from ``data/``. Routes, ranks, vertex numbers and flags must be
equal; floats must agree to 1e-9. The test is skipped when ``network_matching._rust`` is not built.
"""
import math
import os

import pytest
from shapely.wkt import loads as load_wkt

from network_matching.graph_dtw import match_edge_to_bgraph, rust_available
from network_matching.synthetic import PERTURBATIONS, SCENARIOS, apply_perturbation

pytestmark = pytest.mark.skipif(not rust_available(), reason="network_matching._rust is not built")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-9

RUNS = [
    dict(emission="point"),
    dict(emission="point", alpha=0.5, beta=1.5),
    dict(emission="segment", bearing_weight=2.0),
    dict(emission="segment", bearing_weight=0.0, alpha=0.7, beta=2.0),
    dict(emission="point", trim_ends_m=5.0, step_meters=4.0),
]


def assert_same(a, b, path="res"):
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb) or math.isnan(fa) or math.isnan(fb):
            assert (fa == fb) or (math.isnan(fa) and math.isnan(fb)), f"{path}: {a!r} != {b!r}"
        else:
            assert abs(fa - fb) <= TOL * max(1.0, abs(fa)), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for k, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{k}]")
    else:
        assert a == b and type(a) is type(b), f"{path}: {a!r} != {b!r}"


def compare(coords_a, b_edges, **kw):
    py = match_edge_to_bgraph(coords_a, b_edges, engine="python", **kw)
    rs = match_edge_to_bgraph(coords_a, b_edges, engine="rust", **kw)
    for key in ("route", "warping_path", "metrics", "avg_distance"):
        assert_same(py[key], rs[key], key)
    # the graph of the Rust result is built on demand and equals the Python graph
    assert rs["graph"].n_vertices == py["graph"].n_vertices
    return py


def synthetic_cases():
    for name, sc in SCENARIOS.items():
        for fam, spec in PERTURBATIONS.items():
            for mag in spec["grid"]:
                yield pytest.param(name, fam, mag, id=f"{name}-{fam}-{mag:g}")


@pytest.mark.parametrize("name,fam,mag", list(synthetic_cases()))
def test_synthetic_equal(name, fam, mag):
    sc = SCENARIOS[name]
    coords = [tuple(p) for p in apply_perturbation(sc["coords_a"], fam, mag)]
    for run in RUNS:
        kw = {**sc.get("defaults", {}), **run}
        compare(coords, sc["b_edges"], **kw)


@pytest.fixture(scope="module")
def real_groups():
    from network_matching import DuckDBMapMatcher
    m = DuckDBMapMatcher.from_wkt_csv(
        os.path.join(ROOT, "data/osm_edges.csv"), os.path.join(ROOT, "data/sweden_edges.csv"),
        id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=30)
    cand = m.generate_candidate_pairs()
    groups = []
    for k, (_id_a, grp) in enumerate(cand.groupby("id_a")):
        if k % 12:                                # every 12th A-edge: ~300 real cases
            continue
        coords_a = list(load_wkt(grp["wkt_a"].iloc[0]).coords)
        b_edges = [(r.id_b, load_wkt(r.wkt_b)) for r in grp.itertuples(index=False)]
        groups.append((coords_a, b_edges))
    return groups


@pytest.mark.parametrize("run", RUNS[:3], ids=["point", "point-weighted", "segment"])
def test_real_data_equal(real_groups, run):
    n_matched = 0
    for coords_a, b_edges in real_groups:
        n_matched += bool(compare(coords_a, b_edges, **run)["route"])
    assert n_matched > len(real_groups) // 2


def test_engine_switch():
    sc = SCENARIOS["split"]
    with pytest.raises(ValueError):
        match_edge_to_bgraph(sc["coords_a"], sc["b_edges"], engine="fast")
    # debug=True runs the Python code (also with engine="rust") and gives the tables
    res = match_edge_to_bgraph(sc["coords_a"], sc["b_edges"], engine="rust", debug=True)
    assert "D" in res["debug"]
