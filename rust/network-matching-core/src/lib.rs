//! Graph-DTW: align one A-edge to the local directed graph of nearby B-edges.
//!
//! This is the Rust engine of `network_matching.graph_dtw.match_edge_to_bgraph`. It does the same
//! steps in the same sequence, so that the results are the same:
//!
//! 1. The projection pools ([`node_projection_pool`]). The line functions copy the GEOS functions
//!    that shapely calls (`length`, `project`, `interpolate`), so that the pool points are the same.
//! 2. The local directed B-graph ([`build_local_digraph`]): forward arcs only, junction endpoints
//!    are separate vertices, and inter-edge arcs join one edge's end to another edge's start.
//! 3. The dynamic table: point emission (states = A-point, B-vertex) or segment emission (states =
//!    A-segment, B-arc). The horizontal move of each row is a Dijkstra relaxation. The heap order is
//!    (cost, index), the same as Python's `heapq` on tuples, so ties break the same way.
//! 4. The route and its metrics, with the trims of the Python code.
//!
//! Coordinates are in a projected CRS in meters. The algorithm is in `docs/graph_dtw_matching.md`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A point `(x, y)` in meters.
pub type Pt = (f64, f64);

/// The local cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emission {
    /// States are (A-point, B-vertex); each state pays the point-to-point distance.
    Point,
    /// States are (A-segment, B-arc); each state pays the distance between the two segment middles.
    Segment,
}

/// The parameters of [`match_edge`]. The names and the defaults are those of `match_edge_to_bgraph`.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub snap_tolerance_m: f64,
    /// Gap-fill resolution of the pools; `0` disables the gap fill.
    pub step_meters: f64,
    /// `0` disables the trim of short end edges.
    pub trim_ends_m: f64,
    pub emission: Emission,
    pub bearing_weight: f64,
    pub alpha: f64,
    pub beta: f64,
    /// Minimum spacing of the added pool points (Python: `min_pool_gap_m`, already resolved).
    pub min_gap_m: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            snap_tolerance_m: 0.75,
            step_meters: 10.0,
            trim_ends_m: 0.0,
            emission: Emission::Point,
            bearing_weight: 0.0,
            alpha: 1.0,
            beta: 1.0,
            min_gap_m: 0.0,
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Line functions, equal to the GEOS functions that shapely calls
// ------------------------------------------------------------------------------------------------

/// GEOS `Coordinate::distance`.
#[inline]
fn dist(a: Pt, b: Pt) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// numpy `hypot` (the C library `hypot`).
#[inline]
fn hypot(dx: f64, dy: f64) -> f64 {
    dx.hypot(dy)
}

/// GEOS `Length::ofLine`.
fn line_length(c: &[Pt]) -> f64 {
    let mut len = 0.0;
    for k in 1..c.len() {
        len += dist(c[k], c[k - 1]);
    }
    len
}

/// GEOS `Distance::pointToSegment`.
fn point_to_segment(p: Pt, a: Pt, b: Pt) -> f64 {
    if a.0 == b.0 && a.1 == b.1 {
        return dist(p, a);
    }
    let len2 = (b.0 - a.0) * (b.0 - a.0) + (b.1 - a.1) * (b.1 - a.1);
    let r = ((p.0 - a.0) * (b.0 - a.0) + (p.1 - a.1) * (b.1 - a.1)) / len2;
    if r <= 0.0 {
        return dist(p, a);
    }
    if r >= 1.0 {
        return dist(p, b);
    }
    let s = ((a.1 - p.1) * (b.0 - a.0) - (a.0 - p.0) * (b.1 - a.1)) / len2;
    s.abs() * len2.sqrt()
}

/// GEOS `LineSegment::projectionFactor`.
fn projection_factor(p: Pt, a: Pt, b: Pt) -> f64 {
    if p == a {
        return 0.0;
    }
    if p == b {
        return 1.0;
    }
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len2 = dx * dx + dy * dy;
    ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2
}

/// shapely `line.project(point)` = GEOS `LengthIndexOfPoint::indexOfFromStart(pt, -1)`.
fn line_project(c: &[Pt], p: Pt) -> f64 {
    let mut min_distance = f64::INFINITY;
    let mut pt_measure = -1.0;
    let mut seg_start = 0.0;
    for k in 0..c.len().saturating_sub(1) {
        let (a, b) = (c[k], c[k + 1]);
        let seg_distance = point_to_segment(p, a, b);
        let seg_len = dist(a, b);
        let f = projection_factor(p, a, b);
        let measure = if f <= 0.0 {
            seg_start
        } else if f <= 1.0 {
            seg_start + f * seg_len
        } else {
            seg_start + seg_len
        };
        if seg_distance < min_distance && measure > -1.0 {
            pt_measure = measure;
            min_distance = seg_distance;
        }
        seg_start += seg_len;
    }
    pt_measure
}

/// shapely `line.interpolate(s)` = GEOS `LengthLocationMap::getLocationForward` + `LinearLocation::getCoordinate`.
fn line_interpolate(c: &[Pt], s: f64) -> Pt {
    if s <= 0.0 {
        return c[0];
    }
    let mut total = 0.0;
    for k in 0..c.len() - 1 {
        let (p0, p1) = (c[k], c[k + 1]);
        let seg_len = dist(p1, p0);
        if total + seg_len > s {
            let frac = (s - total) / seg_len;
            if frac <= 0.0 {
                return p0;
            }
            if frac >= 1.0 {
                return p1;
            }
            return ((p1.0 - p0.0) * frac + p0.0, (p1.1 - p0.1) * frac + p0.1);
        }
        total += seg_len;
    }
    c[c.len() - 1]
}

/// Absolute bearing (0-360 degrees) from `p0` to `p1`, with numpy's operations.
#[inline]
fn bearing(p0: Pt, p1: Pt) -> f64 {
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    (dx.atan2(dy) * (180.0 / std::f64::consts::PI) + 360.0) % 360.0
}

#[inline]
fn circ_diff(b0: f64, b1: f64) -> f64 {
    let d = (b0 - b1).abs();
    d.min(360.0 - d)
}

/// numpy's pairwise summation (`np.add.reduce` on a contiguous float64 array).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.mean` (the caller makes sure that `a` is not empty).
fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

fn max(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn min(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Python `round(x, 1)`: the decimal formatter rounds the exact binary value, half to even.
fn round1(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.1}").parse().unwrap_or(x)
}

// ------------------------------------------------------------------------------------------------
// Projection pools
// ------------------------------------------------------------------------------------------------

/// A pool point: `(x, y, is_node)`.
pub type PoolPoint = (f64, f64, bool);

/// The projection-enriched point pool along the polyline `coords` (Python `_node_projection_pool`):
/// the vertices of `coords`, the projections of `other_nodes` that fall inside the line, and an even
/// gap fill where two points are more than `step_meters` apart (`0` = no fill). `min_gap_m` > 0 drops
/// added points that are nearer than that to a kept neighbour.
pub fn node_projection_pool(
    coords: &[Pt],
    other_nodes: &[Pt],
    step_meters: f64,
    min_gap_m: f64,
) -> Vec<PoolPoint> {
    let length = line_length(coords);
    if length == 0.0 {
        return vec![(coords[0].0, coords[0].1, true)];
    }
    let mut entries: Vec<(f64, bool)> = Vec::with_capacity(coords.len() + other_nodes.len());
    entries.push((0.0, true));
    let mut cum = 0.0;
    for k in 1..coords.len() {
        cum += dist(coords[k - 1], coords[k]);
        entries.push((cum.min(length), true));
    }
    for &on in other_nodes {
        let s = line_project(coords, on);
        if 1e-9 < s && s < length - 1e-9 {
            entries.push((s, false));
        }
    }
    // a stable sort on the arc length, as Python's `sort(key=...)`
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    if step_meters != 0.0 {
        let mut filled: Vec<(f64, bool)> = Vec::with_capacity(entries.len() * 2);
        for idx in 0..entries.len() {
            let ent = entries[idx];
            filled.push(ent);
            if idx + 1 < entries.len() {
                let (s0, s1) = (ent.0, entries[idx + 1].0);
                let gap = s1 - s0;
                if gap > step_meters {
                    let n = (gap / step_meters).ceil() as i64 - 1;
                    let spacing = gap / (n + 1) as f64;
                    for j in 1..=n {
                        filled.push((s0 + j as f64 * spacing, false));
                    }
                }
            }
        }
        filled.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        entries = filled;
    }

    // de-duplicate by arc length; a node wins over an added point at the same place
    let mut pool: Vec<(f64, bool)> = Vec::with_capacity(entries.len());
    for (s, isn) in entries {
        if let Some(last) = pool.last_mut() {
            if (last.0 - s).abs() <= 1e-9 {
                if isn {
                    last.1 = true;
                }
                continue;
            }
        }
        pool.push((s, isn));
    }

    if min_gap_m > 0.0 && pool.len() > 2 {
        let mut kept: Vec<(f64, bool)> = vec![pool[0]];
        for &(s, isn) in &pool[1..] {
            let last = kept[kept.len() - 1];
            if isn {
                if !last.1 && s - last.0 < min_gap_m && kept.len() > 1 {
                    kept.pop();
                }
                kept.push((s, isn));
            } else if s - last.0 >= min_gap_m {
                kept.push((s, isn));
            }
        }
        pool = kept;
    }

    pool.into_iter()
        .map(|(s, isn)| {
            let p = line_interpolate(coords, s);
            (p.0, p.1, isn)
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Local directed B-graph
// ------------------------------------------------------------------------------------------------

/// The directed graph of the B-edges near one A-edge (Python `LocalBGraph`). Every vertex belongs
/// to exactly one B-edge; junction endpoints are separate coincident vertices.
#[derive(Clone, Debug, Default)]
pub struct LocalBGraph {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub pred_arcs: Vec<Vec<usize>>,
    pub succ_arcs: Vec<Vec<usize>>,
    /// For each graph edge, its index in the input `b_edges` (degenerate edges are skipped).
    pub edge_input: Vec<usize>,
    pub is_node: Vec<bool>,
    pub is_endpoint: Vec<bool>,
    pub vert_edge: Vec<usize>,
    pub b_raw_nodes: Vec<Pt>,
    pub edge_len: Vec<f64>,
}

impl LocalBGraph {
    pub fn n_vertices(&self) -> usize {
        self.vx.len()
    }
}

/// Build the local directed graph from the B-edge polylines. The caller gives the edges in the
/// order that fixes the vertex numbers (Python sorts them by `str(id)`).
pub fn build_local_digraph(
    b_edges: &[&[Pt]],
    a_coords: &[Pt],
    snap_tolerance_m: f64,
    step_meters: f64,
    min_gap_m: f64,
) -> LocalBGraph {
    let mut g = LocalBGraph::default();
    let mut edge_vertex_lists: Vec<(usize, usize)> = Vec::new(); // (first vertex, vertex count)
    for (input_idx, coords) in b_edges.iter().enumerate() {
        if coords.len() < 2 {
            continue;
        }
        let e = g.edge_input.len();
        g.edge_input.push(input_idx);
        let pool = node_projection_pool(coords, a_coords, step_meters, min_gap_m);
        g.b_raw_nodes.extend_from_slice(coords);
        let first = g.vx.len();
        let m = pool.len();
        for (k, &(px, py, isn)) in pool.iter().enumerate() {
            g.vx.push(px);
            g.vy.push(py);
            g.is_node.push(isn);
            g.is_endpoint.push(k == 0 || k == m - 1);
            g.vert_edge.push(e);
        }
        edge_vertex_lists.push((first, m));
        let mut len = 0.0;
        for t in 1..m {
            len += hypot(pool[t].0 - pool[t - 1].0, pool[t].1 - pool[t - 1].1);
        }
        g.edge_len.push(len);
    }
    let nv = g.vx.len();
    g.pred_arcs = vec![Vec::new(); nv];
    g.succ_arcs = vec![Vec::new(); nv];
    for &(first, m) in &edge_vertex_lists {
        for k in 0..m.saturating_sub(1) {
            g.succ_arcs[first + k].push(first + k + 1);
            g.pred_arcs[first + k + 1].push(first + k);
        }
    }
    let tol2 = snap_tolerance_m * snap_tolerance_m;
    for (eu, &(fu, mu)) in edge_vertex_lists.iter().enumerate() {
        let u = fu + mu - 1;
        for (ew, &(w, _)) in edge_vertex_lists.iter().enumerate() {
            if eu == ew {
                continue;
            }
            let dx = g.vx[u] - g.vx[w];
            let dy = g.vy[u] - g.vy[w];
            if dx * dx + dy * dy <= tol2 {
                g.succ_arcs[u].push(w);
                g.pred_arcs[w].push(u);
            }
        }
    }
    g
}

// ------------------------------------------------------------------------------------------------
// The dynamic table
// ------------------------------------------------------------------------------------------------

/// A heap entry; the smallest (cost, index) comes out first, as with `heapq` on tuples.
#[derive(Clone, Copy)]
struct HeapItem(f64, usize);

impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        // reversed: BinaryHeap is a max-heap
        o.0.partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.1.cmp(&self.1))
    }
}

const START: u8 = 0;
const MV_V: u8 = 1;
const MV_H: u8 = 2;
const MV_D: u8 = 3;

/// Dijkstra relaxation of one row: `d` and `back` are the row's slices; `next(k)` lists the states
/// that state `k` feeds; entering state `nb` costs `alpha * e[nb]`.
fn relax_row<'a, F>(
    d: &mut [f64],
    back: &mut [(u8, usize)],
    e: &[f64],
    alpha: f64,
    next: F,
    heap: &mut BinaryHeap<HeapItem>,
) where
    F: Fn(usize) -> &'a [usize],
{
    heap.clear();
    heap.extend(d.iter().enumerate().map(|(k, &c)| HeapItem(c, k)));
    while let Some(HeapItem(c, k)) = heap.pop() {
        if c > d[k] {
            continue;
        }
        for &nb in next(k) {
            let cand = d[k] + alpha * e[nb];
            if cand < d[nb] {
                d[nb] = cand;
                back[nb] = (MV_H, k);
                heap.push(HeapItem(cand, nb));
            }
        }
    }
}

/// Point emission: returns the `(a_index, vertex)` pairs of the best alignment.
fn point_dp(
    g: &LocalBGraph,
    ax: &[f64],
    ay: &[f64],
    alpha: f64,
    beta: f64,
) -> Option<Vec<(usize, usize)>> {
    let n = ax.len();
    let nv = g.n_vertices();
    let mut d = vec![f64::INFINITY; n * nv];
    let mut back = vec![(START, 0usize); n * nv];
    let mut e = vec![0.0; nv];
    let mut heap = BinaryHeap::new();
    let next = |v: usize| g.succ_arcs[v].as_slice();

    for v in 0..nv {
        e[v] = hypot(ax[0] - g.vx[v], ay[0] - g.vy[v]);
        d[v] = e[v];
    }
    relax_row(&mut d[..nv], &mut back[..nv], &e, alpha, next, &mut heap);
    for i in 1..n {
        for v in 0..nv {
            e[v] = hypot(ax[i] - g.vx[v], ay[i] - g.vy[v]);
        }
        let (prev, cur) = d.split_at_mut(i * nv);
        let prev = &prev[(i - 1) * nv..];
        let cur = &mut cur[..nv];
        let bk = &mut back[i * nv..(i + 1) * nv];
        for v in 0..nv {
            let mut cand = beta * e[v] + prev[v];
            let mut mv = (MV_V, v);
            for &u in &g.pred_arcs[v] {
                let c2 = e[v] + prev[u];
                if c2 < cand {
                    cand = c2;
                    mv = (MV_D, u);
                }
            }
            cur[v] = cand;
            bk[v] = mv;
        }
        relax_row(cur, bk, &e, alpha, next, &mut heap);
    }
    let last = &d[(n - 1) * nv..];
    let v_best = argmin(last);
    if !last[v_best].is_finite() {
        return None;
    }
    let mut pairs = Vec::new();
    let (mut i, mut v) = (n - 1, v_best);
    loop {
        let (mv, u) = back[i * nv + v];
        pairs.push((i, v));
        match mv {
            START => break,
            MV_V => i -= 1,
            MV_D => {
                i -= 1;
                v = u;
            }
            _ => v = u,
        }
    }
    pairs.reverse();
    Some(pairs)
}

/// The first index of the smallest value (`np.argmin`).
fn argmin(a: &[f64]) -> usize {
    let mut best = 0;
    for k in 1..a.len() {
        if a[k] < a[best] {
            best = k;
        }
    }
    best
}

/// The arc state axis of the segment emission.
struct SegPath {
    au: Vec<usize>,
    av: Vec<usize>,
    ridable: Vec<bool>,
    /// backtracked states `(a_segment, arc)`
    states: Vec<(usize, usize)>,
}

/// Segment emission (Python `_segment_dp_pairs`): states are (A-segment, B-arc).
fn segment_dp(
    g: &LocalBGraph,
    ax: &[f64],
    ay: &[f64],
    bw: f64,
    alpha: f64,
    beta: f64,
) -> Option<SegPath> {
    let nv = g.n_vertices();
    let n = ax.len();
    if n < 2 {
        return None;
    }
    let mut au = Vec::new();
    let mut av = Vec::new();
    for u in 0..nv {
        for &w in &g.succ_arcs[u] {
            au.push(u);
            av.push(w);
        }
    }
    let na = au.len();
    if na == 0 {
        return None;
    }
    let mut arcs_from: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut arcs_to: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for k in 0..na {
        arcs_from[au[k]].push(k);
        arcs_to[av[k]].push(k);
    }
    // the arcs that a state on arc k feeds within a row
    let feeds: Vec<&[usize]> = (0..na).map(|k| arcs_from[av[k]].as_slice()).collect();
    let (mut ux, mut uy, mut hx, mut hy) =
        (vec![0.0; na], vec![0.0; na], vec![0.0; na], vec![0.0; na]);
    let mut ridable = vec![false; na];
    let mut arc_bear = vec![0.0; na];
    for k in 0..na {
        ux[k] = g.vx[au[k]];
        uy[k] = g.vy[au[k]];
        hx[k] = g.vx[av[k]];
        hy[k] = g.vy[av[k]];
        ridable[k] = hypot(hx[k] - ux[k], hy[k] - uy[k]) >= 0.5;
        arc_bear[k] = bearing((ux[k], uy[k]), (hx[k], hy[k]));
    }
    let ns = n - 1;
    let mut d = vec![f64::INFINITY; ns * na];
    let mut back = vec![(START, 0usize); ns * na];
    let mut e = vec![0.0; na];
    let mut heap = BinaryHeap::new();
    let emit = |i: usize, e: &mut [f64]| {
        let mx = 0.5 * (ax[i] + ax[i + 1]);
        let my = 0.5 * (ay[i] + ay[i + 1]);
        let sb = if bw > 0.0 {
            bearing((ax[i], ay[i]), (ax[i + 1], ay[i + 1]))
        } else {
            0.0
        };
        for k in 0..na {
            let mut v = if ridable[k] {
                hypot(mx - 0.5 * (ux[k] + hx[k]), my - 0.5 * (uy[k] + hy[k]))
            } else {
                0.0
            };
            if bw > 0.0 {
                v += bw
                    * if ridable[k] {
                        circ_diff(sb, arc_bear[k])
                    } else {
                        0.0
                    };
            }
            e[k] = v;
        }
    };
    let next = |k: usize| feeds[k];

    emit(0, &mut e);
    for k in 0..na {
        if ridable[k] {
            d[k] = e[k];
        }
    }
    relax_row(&mut d[..na], &mut back[..na], &e, alpha, next, &mut heap);
    for i in 1..ns {
        emit(i, &mut e);
        let (prev, cur) = d.split_at_mut(i * na);
        let prev = &prev[(i - 1) * na..];
        let cur = &mut cur[..na];
        let bk = &mut back[i * na..(i + 1) * na];
        for k in 0..na {
            if !ridable[k] {
                continue;
            }
            let mut cand = beta * e[k] + prev[k];
            let mut mv = (MV_V, k);
            for &p in &arcs_to[au[k]] {
                let c2 = e[k] + prev[p];
                if c2 < cand {
                    cand = c2;
                    mv = (MV_D, p);
                }
            }
            if cand.is_finite() {
                cur[k] = cand;
                bk[k] = mv;
            }
        }
        relax_row(cur, bk, &e, alpha, next, &mut heap);
    }
    let last = &d[(ns - 1) * na..];
    let k_best = argmin(last);
    if !last[k_best].is_finite() {
        return None;
    }
    let mut states = Vec::new();
    let (mut i, mut k) = (ns - 1, k_best);
    loop {
        let (mv, p) = back[i * na + k];
        states.push((i, k));
        match mv {
            START => break,
            MV_V => i -= 1,
            MV_D => {
                i -= 1;
                k = p;
            }
            _ => k = p,
        }
    }
    states.reverse();
    Some(SegPath {
        au,
        av,
        ridable,
        states,
    })
}

// ------------------------------------------------------------------------------------------------
// The route and its metrics
// ------------------------------------------------------------------------------------------------

/// One B-edge of the route (Python `route_edges` item). `edge` is the index in the input `b_edges`.
#[derive(Clone, Debug)]
pub struct RouteEdge {
    pub edge: usize,
    pub seq: usize,
    pub match_dist_avg: f64,
    pub match_dist_max: f64,
    pub match_dist_min: f64,
    pub a_len: f64,
    pub matched_len: f64,
    pub b_edge_len: f64,
    pub b_cover_pct: f64,
    pub bearing_diff: f64,
    pub n_points: usize,
    pub cover_pct: f64,
}

/// The match of one A-edge (Python `metrics` and `warping_path`). Edge numbers are indices in the
/// input `b_edges`; `warp_vertices` are vertices of [`MatchResult::graph`].
#[derive(Clone, Debug)]
pub struct MatchResult {
    pub average: f64,
    pub max: f64,
    pub min: f64,
    pub overlap_pct: i64,
    pub matched_len: f64,
    pub route_edges: Vec<RouteEdge>,
    pub bearing_diff: f64,
    pub part_drift: f64,
    pub part_bearing_diff: f64,
    pub warping_path: Vec<(Pt, Pt)>,
    pub warp_vertices: Vec<usize>,
    pub warp_is_node: Vec<bool>,
    pub warp_a_is_node: Vec<bool>,
    pub warp_edge: Vec<usize>,
}

/// Why no match was found (Python debug `reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoMatch {
    EmptyInputs,
    NoFiniteAlignment,
    ZeroBTraversal,
}

/// Match one A-edge to the local directed graph of `b_edges` (Python `match_edge_to_bgraph`).
///
/// `b_edges` must be in the order that fixes the vertex numbers: Python sorts them by `str(id)`.
/// Returns the graph too, and the match or the reason for no match.
pub fn match_edge(
    coords_a: &[Pt],
    b_edges: &[&[Pt]],
    p: &Params,
) -> (LocalBGraph, Result<MatchResult, NoMatch>) {
    let g = build_local_digraph(
        b_edges,
        coords_a,
        p.snap_tolerance_m,
        p.step_meters,
        p.min_gap_m,
    );
    let r = align(coords_a, &g, p);
    (g, r)
}

/// Python `graph_dtw_align` (without the debug tables).
pub fn align(coords_a: &[Pt], g: &LocalBGraph, p: &Params) -> Result<MatchResult, NoMatch> {
    if coords_a.is_empty() {
        return Err(NoMatch::EmptyInputs);
    }
    let a_pool = node_projection_pool(coords_a, &g.b_raw_nodes, p.step_meters, p.min_gap_m);
    let n = a_pool.len();
    let nv = g.n_vertices();
    if n < 1 || nv < 1 {
        return Err(NoMatch::EmptyInputs);
    }
    let ax: Vec<f64> = a_pool.iter().map(|q| q.0).collect();
    let ay: Vec<f64> = a_pool.iter().map(|q| q.1).collect();
    let (vx, vy) = (&g.vx, &g.vy);

    let mut seg: Option<SegPath> = None;
    let pairs: Vec<(usize, usize)> = match p.emission {
        Emission::Segment => {
            let sp = segment_dp(g, &ax, &ay, p.bearing_weight, p.alpha, p.beta)
                .ok_or(NoMatch::NoFiniteAlignment)?;
            // the entry arc gives its tail, every state its head
            let mut pr = vec![(sp.states[0].0, sp.au[sp.states[0].1])];
            for &(i, k) in &sp.states {
                pr.push((i + 1, sp.av[k]));
            }
            seg = Some(sp);
            pr
        }
        Emission::Point => {
            point_dp(g, &ax, &ay, p.alpha, p.beta).ok_or(NoMatch::NoFiniteAlignment)?
        }
    };

    let step_e: Vec<usize> = pairs.iter().map(|&(_, v)| g.vert_edge[v]).collect();
    let wa: Vec<Pt> = pairs.iter().map(|&(i, _)| (ax[i], ay[i])).collect();
    let wb: Vec<Pt> = pairs.iter().map(|&(_, v)| (vx[v], vy[v])).collect();

    let k0 = pairs.len();
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut k = 0;
    while k < k0 {
        let e = step_e[k];
        let mut j = k;
        while j + 1 < k0 && step_e[j + 1] == e {
            j += 1;
        }
        groups.push((k, j, e));
        k = j + 1;
    }
    let span_len = |w: &[Pt], gk: usize, gj: usize| {
        let mut l = 0.0;
        for t in gk.max(1)..=gj {
            l += hypot(w[t].0 - w[t - 1].0, w[t].1 - w[t - 1].1);
        }
        l
    };
    let (mut g0, mut g1) = (0usize, groups.len() - 1);
    while g0 < g1 && span_len(&wb, groups[g0].0, groups[g0].1) <= 1e-9 {
        g0 += 1;
    }
    while g1 > g0 && span_len(&wb, groups[g1].0, groups[g1].1) <= 1e-9 {
        g1 -= 1;
    }
    if p.trim_ends_m != 0.0 && g1 > g0 {
        while g0 < g1 && span_len(&wa, groups[g0].0, groups[g0].1) < p.trim_ends_m {
            g0 += 1;
        }
        while g1 > g0 && span_len(&wa, groups[g1].0, groups[g1].1) < p.trim_ends_m {
            g1 -= 1;
        }
    }
    let (lo, hi) = (groups[g0].0, groups[g1].1);
    if span_len(&wb, lo, hi) <= 1e-9 {
        return Err(NoMatch::ZeroBTraversal);
    }
    let drift_all: Vec<f64> = (0..k0)
        .map(|t| hypot(wa[t].0 - wb[t].0, wa[t].1 - wb[t].1))
        .collect();
    let (ve, vt) = (pairs[lo].1, pairs[hi].1);

    let mut route_edges: Vec<RouteEdge> = Vec::new();
    for &(k, j, e) in &groups[g0..=g1] {
        let sg = &drift_all[k..=j];
        let (mut a_len, mut b_len) = (0.0, 0.0);
        for t in k.max(1)..=j {
            b_len += hypot(wb[t].0 - wb[t - 1].0, wb[t].1 - wb[t - 1].1);
            let is_lead = pairs[t].1 == ve;
            let is_trail = pairs[t - 1].1 == vt;
            if !is_lead && !is_trail {
                a_len += hypot(wa[t].0 - wa[t - 1].0, wa[t].1 - wa[t - 1].1);
            }
        }
        let edge_bearing = if j > k {
            circ_diff(bearing(wa[k], wa[j]), bearing(wb[k], wb[j]))
        } else {
            0.0
        };
        let b_edge_len = g.edge_len[e];
        let b_cover = if b_edge_len > 0.0 {
            100.0 * b_len / b_edge_len
        } else {
            0.0
        };
        route_edges.push(RouteEdge {
            edge: g.edge_input[e],
            seq: route_edges.len(),
            match_dist_avg: mean(sg),
            match_dist_max: max(sg),
            match_dist_min: min(sg),
            a_len,
            matched_len: b_len,
            b_edge_len,
            b_cover_pct: round1(b_cover.min(100.0)),
            bearing_diff: edge_bearing,
            n_points: sg.len(),
            cover_pct: 0.0,
        });
    }
    let mut matched_len = 0.0;
    for re in &route_edges {
        matched_len += re.matched_len;
    }
    let a_xy: Vec<Pt> = a_pool.iter().map(|q| (q.0, q.1)).collect();
    let total_a_len = if n > 1 { line_length(&a_xy) } else { 0.0 };
    for re in route_edges.iter_mut() {
        re.cover_pct = if total_a_len > 0.0 {
            round1(100.0 * re.a_len / total_a_len)
        } else {
            0.0
        };
    }

    let drift = &drift_all[lo..=hi];
    let mut average = mean(drift);
    let mut bearing_diff = circ_diff(bearing(wa[lo], wa[hi]), bearing(wb[lo], wb[hi]));
    let (mut dmax, mut dmin) = (max(drift), min(drift));

    // records for the overlap part: (drift, bearing diff, A length, unit)
    let mut recs: Vec<(f64, f64, f64, usize)> = Vec::new();
    if let Some(sp) = &seg {
        // segment mode: the reported distances are the middle-to-middle state costs
        let mut mid_stats: Vec<(usize, Vec<f64>)> = Vec::new(); // (graph edge, distances), first-seen order
        let mut seg_bd: Vec<f64> = Vec::new();
        for (t, &(i, k)) in sp.states.iter().enumerate() {
            if !(lo <= t + 1 && t + 1 <= hi) || !sp.ridable[k] {
                continue;
            }
            let (u, w) = (sp.au[k], sp.av[k]);
            let eid = g.vert_edge[u];
            let d = hypot(
                0.5 * (ax[i] + ax[i + 1]) - 0.5 * (vx[u] + vx[w]),
                0.5 * (ay[i] + ay[i + 1]) - 0.5 * (vy[u] + vy[w]),
            );
            let bd = circ_diff(
                bearing((ax[i], ay[i]), (ax[i + 1], ay[i + 1])),
                bearing((vx[u], vy[u]), (vx[w], vy[w])),
            );
            let al = hypot(ax[i + 1] - ax[i], ay[i + 1] - ay[i]);
            match mid_stats.iter_mut().find(|(e, _)| *e == eid) {
                Some((_, ds)) => ds.push(d),
                None => mid_stats.push((eid, vec![d])),
            }
            seg_bd.push(bd);
            recs.push((d, bd, al, k));
        }
        let all_d: Vec<f64> = mid_stats
            .iter()
            .flat_map(|(_, ds)| ds.iter().copied())
            .collect();
        if !all_d.is_empty() {
            average = mean(&all_d);
            dmax = max(&all_d);
            dmin = min(&all_d);
            for re in route_edges.iter_mut() {
                let ge = g.edge_input.iter().position(|&x| x == re.edge).unwrap();
                if let Some((_, ds)) = mid_stats.iter().find(|(e, _)| *e == ge) {
                    re.match_dist_avg = mean(ds);
                    re.match_dist_max = max(ds);
                    re.match_dist_min = min(ds);
                }
            }
        }
        if !seg_bd.is_empty() {
            bearing_diff = mean(&seg_bd);
        }
    } else {
        for t in lo + 1..=hi {
            recs.push((
                drift_all[t],
                0.0,
                hypot(wa[t].0 - wa[t - 1].0, wa[t].1 - wa[t - 1].1),
                pairs[t].1,
            ));
        }
    }

    let (mut part_drift, mut part_bearing_diff) = (average, bearing_diff);
    let part_a: f64 = match interior_part(&recs) {
        Some(kept) => {
            let ds: Vec<f64> = kept.iter().map(|&i| recs[i].0).collect();
            part_drift = mean(&ds);
            if seg.is_some() {
                let bs: Vec<f64> = kept.iter().map(|&i| recs[i].1).collect();
                part_bearing_diff = mean(&bs);
            }
            let mut s = 0.0;
            for &i in &kept {
                s += recs[i].2;
            }
            s
        }
        None => {
            let mut s = 0.0;
            for r in &recs {
                s += r.2;
            }
            s
        }
    };
    let overlap_pct = if total_a_len > 0.0 {
        ((100.0 * part_a / total_a_len).round_ties_even() as i64).min(100)
    } else {
        0
    };

    let warp_vertices: Vec<usize> = pairs[lo..=hi].iter().map(|&(_, v)| v).collect();
    Ok(MatchResult {
        average,
        max: dmax,
        min: dmin,
        overlap_pct,
        matched_len,
        route_edges,
        bearing_diff,
        part_drift,
        part_bearing_diff,
        warping_path: (lo..=hi).map(|t| (wa[t], wb[t])).collect(),
        warp_is_node: warp_vertices.iter().map(|&v| g.is_node[v]).collect(),
        warp_a_is_node: pairs[lo..=hi].iter().map(|&(i, _)| a_pool[i].2).collect(),
        warp_edge: step_e[lo..=hi].iter().map(|&e| g.edge_input[e]).collect(),
        warp_vertices,
    })
}

/// Python `_interior_part`: drop the leading run on the first unit (keep its last member) and the
/// trailing run on the last unit (keep its first member). `None` = the route is on one unit.
fn interior_part(recs: &[(f64, f64, f64, usize)]) -> Option<Vec<usize>> {
    let (first_u, last_u) = (recs.first()?.3, recs.last()?.3);
    if first_u == last_u {
        return None;
    }
    let lead = recs.iter().take_while(|r| r.3 == first_u).count();
    let trail = recs.iter().rev().take_while(|r| r.3 == last_u).count();
    let n = recs.len();
    // drop indices 0..lead-1 and n-trail+1..n
    let kept: Vec<usize> = (0..n)
        .filter(|&i| !(i + 1 < lead || i > n - trail))
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_chain_is_one_route() {
        // A runs along three chained B-edges, 2 m to the side
        let a = [(0.0, 2.0), (300.0, 2.0)];
        let b1 = [(0.0, 0.0), (100.0, 0.0)];
        let b2 = [(100.0, 0.0), (200.0, 0.0)];
        let b3 = [(200.0, 0.0), (300.0, 0.0)];
        let (_, r) = match_edge(&a, &[&b1, &b2, &b3], &Params::default());
        let r = r.unwrap();
        let route: Vec<usize> = r.route_edges.iter().map(|e| e.edge).collect();
        assert_eq!(route, vec![0, 1, 2]);
        assert!((r.average - 2.0).abs() < 1e-9);
        assert_eq!(r.overlap_pct, 100);
    }

    #[test]
    fn pairwise_sum_matches_sequential_for_small() {
        let a: Vec<f64> = (0..300).map(|k| k as f64 * 0.1).collect();
        assert!((pairwise_sum(&a) - a.iter().sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn interior_part_drops_overhang_runs() {
        let r = |u| (0.0, 0.0, 1.0, u);
        // units 5 5 5 6 7 7 -> drop the first two 5s and the last 7
        let recs = [r(5), r(5), r(5), r(6), r(7), r(7)];
        assert_eq!(interior_part(&recs).unwrap(), vec![2, 3, 4]);
    }
}
