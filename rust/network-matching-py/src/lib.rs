//! Python module `network_matching._rust`: the Rust engine of graph-DTW.
//!
//! `network_matching.graph_dtw.match_edge_to_bgraph` calls [`match_edge`]; do not call it directly.
//! The coordinates come as raw little-endian float64 bytes (`numpy.ndarray.tobytes()`), so that no
//! per-point Python object is made on the way in.

use network_matching_core::{match_edge as core_match, Emission, Params, Pt};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn f64s(b: &[u8]) -> PyResult<Vec<f64>> {
    if b.len() % 8 != 0 {
        return Err(PyValueError::new_err(
            "the byte length is not a multiple of 8",
        ));
    }
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn points(b: &[u8]) -> PyResult<Vec<Pt>> {
    let v = f64s(b)?;
    if v.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "the coordinates are not (x, y) pairs",
        ));
    }
    Ok(v.chunks_exact(2).map(|c| (c[0], c[1])).collect())
}

/// Match one A-edge to the B-edges.
///
/// `a_xy`: the A coordinates; `b_xy`: the coordinates of all B-edges one after the other;
/// `b_counts`: the number of points of each B-edge (int64 bytes); `edge_ids`: the B-edge ids in
/// the same sequence (already sorted by `str(id)`). Returns `None` when there is no match, else the
/// dict `{"route", "warping_path", "metrics", "avg_distance"}` of `match_edge_to_bgraph`.
#[pyfunction]
#[pyo3(signature = (a_xy, b_xy, b_counts, edge_ids, *, snap_tolerance_m, step_meters, trim_ends_m,
                    segment, bearing_weight, alpha, beta, min_gap_m))]
#[allow(clippy::too_many_arguments)]
fn match_edge<'py>(
    py: Python<'py>,
    a_xy: &[u8],
    b_xy: &[u8],
    b_counts: &[u8],
    edge_ids: &Bound<'py, PyList>,
    snap_tolerance_m: f64,
    step_meters: f64,
    trim_ends_m: f64,
    segment: bool,
    bearing_weight: f64,
    alpha: f64,
    beta: f64,
    min_gap_m: f64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let a = points(a_xy)?;
    let b = points(b_xy)?;
    let counts: Vec<usize> = b_counts
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    if counts.iter().sum::<usize>() != b.len() || counts.len() != edge_ids.len() {
        return Err(PyValueError::new_err(
            "b_counts, b_xy and edge_ids do not agree",
        ));
    }
    let p = Params {
        snap_tolerance_m,
        step_meters,
        trim_ends_m,
        emission: if segment {
            Emission::Segment
        } else {
            Emission::Point
        },
        bearing_weight,
        alpha,
        beta,
        min_gap_m,
    };
    let r = py.detach(|| {
        let mut edges: Vec<&[Pt]> = Vec::with_capacity(counts.len());
        let mut at = 0;
        for &c in &counts {
            edges.push(&b[at..at + c]);
            at += c;
        }
        core_match(&a, &edges, &p).1
    });
    let m = match r {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    let route = PyList::empty(py);
    let route_edges = PyList::empty(py);
    for re in &m.route_edges {
        let id = edge_ids.get_item(re.edge)?;
        route.append((id.clone(), "forward", re.seq))?;
        let d = PyDict::new(py);
        d.set_item("dest_id", id)?;
        d.set_item("direction", "forward")?;
        d.set_item("seq", re.seq)?;
        d.set_item("match_dist_avg", re.match_dist_avg)?;
        d.set_item("match_dist_max", re.match_dist_max)?;
        d.set_item("match_dist_min", re.match_dist_min)?;
        d.set_item("a_len", re.a_len)?;
        d.set_item("matched_len", re.matched_len)?;
        d.set_item("b_edge_len", re.b_edge_len)?;
        d.set_item("b_cover_pct", re.b_cover_pct)?;
        d.set_item("bearing_diff", re.bearing_diff)?;
        d.set_item("n_points", re.n_points)?;
        d.set_item("cover_pct", re.cover_pct)?;
        route_edges.append(d)?;
    }
    let warp_edge = PyList::empty(py);
    for &e in &m.warp_edge {
        warp_edge.append(edge_ids.get_item(e)?)?;
    }

    let metrics = PyDict::new(py);
    metrics.set_item("average", m.average)?;
    metrics.set_item("max", m.max)?;
    metrics.set_item("min", m.min)?;
    metrics.set_item("overlap_pct", m.overlap_pct)?;
    metrics.set_item("matched_len", m.matched_len)?;
    metrics.set_item("route", &route)?;
    metrics.set_item("route_edges", route_edges)?;
    metrics.set_item("n_edges", m.route_edges.len())?;
    metrics.set_item("bearing_diff", m.bearing_diff)?;
    metrics.set_item("part_drift", m.part_drift)?;
    metrics.set_item("part_bearing_diff", m.part_bearing_diff)?;
    metrics.set_item("warp_vertices", &m.warp_vertices)?;
    metrics.set_item("warp_is_node", &m.warp_is_node)?;
    metrics.set_item("warp_a_is_node", &m.warp_a_is_node)?;
    metrics.set_item("warp_edge", warp_edge)?;

    let res = PyDict::new(py);
    res.set_item("route", route)?;
    res.set_item("warping_path", &m.warping_path)?;
    res.set_item("metrics", metrics)?;
    res.set_item("avg_distance", m.average)?;
    Ok(Some(res))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(match_edge, m)?)?;
    Ok(())
}
