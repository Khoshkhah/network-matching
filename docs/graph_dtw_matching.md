# Graph-DTW: Matching a Road Segment to a Directed Network

This document explains **graph-DTW**, a generalization of the segment-to-segment
[`dtw_align`](dtw_matching.md) used elsewhere in this library. Instead of aligning one A-edge
to **one** B-edge, graph-DTW aligns an A-edge to the whole **local directed graph** of nearby
B-edges, so the alignment can only follow *graph-connected* B-edges. The result is the ordered,
connected **route** of B-edges that the A-edge maps to, plus a single drift metric.

Implementation: [`network_matching/graph_dtw.py`](../network_matching/graph_dtw.py).
Tests: [`tests/test_graph_dtw.py`](../tests/test_graph_dtw.py).
For the **end-to-end pipeline** (initialize → run → the two output tables → save → visualize),
see [`graph_dtw_pipeline.md`](graph_dtw_pipeline.md).

---

## 1. Why generalize DTW?

The pairwise matcher aligns each A-edge to one B-edge at a time. Two failure modes follow from
"one B-edge at a time":

1. **Wrong neighbour.** At parallel roads or junctions, edge-to-edge DTW can latch onto a
   geometrically-close but *topologically unrelated* B-edge (e.g. the opposite carriageway, or
   a service road running alongside).
2. **Segmentation mismatch.** OSM and NVDB split roads at different points, so a single A-edge
   only *partially* overlaps several B-edges, producing fragmented `1:N` matches and noisy
   overlap.

Graph-DTW fixes both at the source: it matches the A-edge to a **route** through the B-network.
The warping path is forced to walk forward along connected B-edges, so it physically cannot hop
onto a disconnected parallel road, and it naturally stitches several B-edges into one clean
match.

---

## 2. From a line to a graph

In ordinary DTW the target is a single polyline `B = [b_0, b_1, …, b_{M-1}]`, and the only
predecessor of point `b_j` is `b_{j-1}`. Graph-DTW replaces that line with a **directed graph**
`GB`:

- Like `dtw_align`, it is **projection-based and continuous** — both sides are enriched with the
  *projections of the other side's nodes* before the table is built:
  - the **A axis** = A's own nodes **+ the projection (foot of perpendicular) of every candidate
    B vertex onto A** (so projection points land on edge `a` too), and
  - each **B-edge** = its own nodes **+ the projections of A's nodes onto that edge**.
  (A small `step_meters` gap-fill keeps long straight stretches from being coarse; set it falsy
  for pure node+projection pools exactly like `dtw_align`.)
- The B network is a **fully directed edge table** — every bidirectional road already appears as
  two opposing directed edges (e.g. the NVDB `is_reverse` twins) — so `GB` uses **forward arcs
  only**: consecutive vertices of an edge are joined by a directed arc in its digitized direction
  (`start → end`). No `backward`/synthesized reverse arcs: if A runs opposite to an edge, it
  simply matches that edge's reverse twin (a different `directed_id`) instead.
- **Every vertex belongs to exactly one edge** (`vert_edge`). B-edges whose endpoints coincide
  (within a snap tolerance) are joined head-to-tail by a directed **inter-edge arc** — one edge's
  **end** → another edge's **start**. Junction endpoints are kept as *separate coincident
  vertices* (one per incident edge), **not** merged into one shared vertex. Keeping the owning
  edge in the vertex — hence in the DP state — is what makes the route unambiguous at a junction:
  a state is never "at the junction", it is "at edge `u`'s end" or "at edge `w`'s start". This
  also makes a **U-turn impossible** — an edge that only *ends* at a junction has no arc leaving
  it, so the path can flow *through* a junction but never dip onto a side edge and return.

So the single predecessor `b_{j-1}` becomes "**any graph-predecessor** `u` of vertex `v`", and a
matched point on either side is either an original **node** (point-to-point) or a **projection**
(point-to-projection) — exposed per warping step as `warp_is_node` / `warp_a_is_node`.

```
            A-edge a  (the "time" axis, swept left -> right)
            a0 ───────────────────────────────────► a_{N-1}

GB (target graph):     B1 ──► B2 ──► B3        (connected chain: a real corridor)
                        \
                         └─► B5                 (a branch leaving the corridor)

                       B4 ─────────────         (isolated parallel road: NO arcs in/out)
```

The warping path must move **monotonically along A** while stepping only along **real arcs of
`GB`**. It can follow `B1→B2→B3` (or take the `B5` branch), but it can never *use* `B4`, because
no arc connects `B4` to anything — any route through it would strand the rest of A.

---

## 3. The dynamic program

Let `a_0 … a_{N-1}` be the densified A-edge and `v` range over the vertices of `GB`.
`D[i][v]` is the minimum cost to align `a_0 … a_i` ending at vertex `v`. Costs are Euclidean
distances in meters (coordinates are pre-projected). The recurrence mirrors DTW's three moves,
with `j-1` replaced by `pred(v)`:

```
D[i][v] = dist(a_i, v) + min(
    D[i-1][v],                          # vertical : advance A, stay at v
    min over u in pred(v) of D[i][u],   # horizontal: advance B along an arc, A stays
    min over u in pred(v) of D[i-1][u]  # diagonal : advance both
)
```

The `u → v` step exists **only when it is a real arc of `GB`** — this single constraint is the
whole wrong-match guard.

- **Init (free entry).** `D[0][v] = dist(a_0, v)` for every `v`: the match may start at the
  beginning of A against *any* B vertex.
- **Termination — finishes exactly when A finishes.** The table has `N` rows, one per A sample
  from `a_0` (A's start) to `a_{N-1}` (A's end); termination is `argmin_v D[N-1][v]`, the **last**
  A row. So the warping path always spans the *entire* A-edge — never early, never beyond.

### 3.1 The one subtlety — cycles

The **vertical** and **diagonal** terms depend only on row `i-1` (already known), so they are
computed directly. The **horizontal** term depends on `D[i][u]` *within the same row* `i`, and
`GB` may contain **cycles** (loops, roundabouts) — there is no topological order to sweep.

For a fixed `i`, this is exactly a **non-negative-weight shortest-path** problem, so it is
solved exactly by **Dijkstra**:

```
base[v] = dist(a_i, v) + min( D[i-1][v], min_{u in pred(v)} D[i-1][u] )   # vertical + diagonal
D[i][·] = base[·];  push all (base[v], v) into a min-heap
pop (cost_u, u); for each arc u -> w:  cand = D[i][u] + dist(a_i, w)
                                       if cand < D[i][w]: D[i][w] = cand; push
```

Re-paying `dist(a_i, w)` on each horizontal step is precisely DTW's `D[i][j-1] + dist(a_i, b_j)`
(one `a_i` may align to a run of B vertices). All arc weights are distances ≥ 0, so Dijkstra is
exact regardless of cycles. Per A-edge the cost is `O(N · E·log V)` with `V, E` tiny (~10 edges
→ a few hundred vertices) — microseconds.

### 3.2 Backtracking → route

Each `D[i][v]` records which move won (`vertical`, `horizontal`, or `diagonal`) and the
predecessor vertex. Tracing back from the best terminal vertex to row 0 yields the warping path
of `(a_i, v)` pairs. The B-edge of each step is read **directly from the vertex** (`vert_edge[v]`)
— no arc inference, no shared-junction ambiguity, because every vertex belongs to exactly one
edge. Consecutive steps on the same B-edge are grouped into one **route entry**, giving the
**route** `[(b_edge_id, direction, seq), …]`. Traversal is always `forward` (the directed table
already encodes orientation; which of a road's two twins matched tells you the physical
direction).

### 3.3 Zero-traversal touches, end trimming + determinism

- **Zero-traversal touches (always dropped).** A leading/trailing route edge that A only *touches*
  at a junction vertex — its end overhangs onto the next edge's start, but it never runs **along**
  that edge (`B-used == 0`) — is removed from the route unconditionally; it is overhang, not a
  match. If **no** edge is actually traversed (A only touches boundary vertices, e.g. a stub that
  ends at but never runs along a B-edge), the result is **`NO_MATCH`**. This is what keeps a route
  to the edges A truly walks (so an A-edge that visibly follows one B-edge is reported as one
  edge, not two).
- **End trimming (`trim_ends_m`, default `0` = OFF).** An optional cleanup that *removes* a
  leading/trailing route edge from the route list when it covers less than `trim_ends_m` of A
  (intended for a free-entry/exit snap onto a crossing B-edge). It is **off by default** because
  on real data A's points often cluster near a junction, so a legitimate corridor edge can have a
  tiny per-edge A-span and be removed by mistake. It does **not** fill gaps — connectivity between
  near-but-unequal endpoints is handled by `snap_tolerance_m`, not this. With the default the full
  DTW route is kept and the warping path spans all of A.
- **Deterministic.** Candidate B-edges are sorted by id before the graph is built, so vertex
  numbering — and therefore every DP tie-break — is stable regardless of the order candidate rows
  arrive in. The same input always yields the same route.

---

### 3.4 Local cost — point-to-point vs segment-to-segment

The recurrence above is the **point-to-point** model (`emission="point"`, the default): states are
(A-point, B-vertex) and each cell adds `dist(a_i, v)`. `emission="segment"` switches to a **true
segment-to-segment DP**: the states become (A-**segment**, B-**arc**), and *every* state pays the
endpoint-average `½(‖aᵢ−u‖ + ‖aᵢ₊₁−v‖)` — plus, with `bearing_weight` λ > 0, a per-state heading
penalty `λ·Δbearing(segment, arc)`. Because both sides of every pairing are segments, no alignment
move can bypass either term. Junction-snap stitches are connectivity, not segments: they are
passed through within a row, never ridden by an A-segment. `"point"` is byte-for-byte unchanged;
see [weighted_emission.md](weighted_emission.md) (§10) for the design, the failure mode of the
earlier transition-cost variant, and validation results.

## 4. Output

`graph_dtw_align(coords_a, gb)` returns `(average_distance, warping_path, metrics)` — the same
shape as `dtw_align`, so existing plotting code works unchanged. The public primitive
`match_edge_to_bgraph(coords_a, b_edges)` wraps build + align and returns
`{route, warping_path, metrics, avg_distance, graph}`.

**Route-level `metrics`:**

| key            | meaning                                                                |
|----------------|------------------------------------------------------------------------|
| `average`      | mean match distance (m) over the warping path — main quality signal     |
| `max` / `min`  | max / min match distance along the path                                |
| `matched_len`  | total length (m) traversed in B along the route                        |
| `overlap_pct`  | **% of A covered** (A-length matched to *advancing* B geometry); < 100 where A overhangs past the route's first/last B-edge endpoint |
| `bearing_diff` | whole-route bearing difference over the matched span (degrees)         |
| `route`        | ordered `[(b_edge_id, 'forward', seq), …]` (`seq` = order of matching)  |
| `n_edges`      | number of B-edges in the route                                         |
| `route_edges`  | **per-B-edge breakdown** (below)                                        |
| `warp_*`       | per warping-step arrays: `warp_vertices`, `warp_edge`, `warp_is_node`, `warp_a_is_node` (node vs projection on each side) |

**`metrics["route_edges"]`** — one dict per B-edge in the route, the result *divided by edge*:

| field            | meaning                                                              |
|------------------|---------------------------------------------------------------------|
| `dest_id`        | the B-edge id                                                       |
| `seq`            | order of this edge along the route (0,1,2,…)                        |
| `direction`      | always `forward` (the directed table encodes orientation; the reverse twin is a separate edge) |
| `match_dist_avg/max/min` | match distance over just this edge's matched points        |
| `a_len`          | metres of A **covered** by this edge (where its B vertex advances)  |
| `cover_pct`      | **% of the whole A-edge** this edge covers (`a_len / A-length`)      |
| `matched_len`    | metres of this B-edge traversed                                    |
| `b_edge_len`     | this B-edge's total length (m)                                     |
| `b_cover_pct`    | **% of *this B-edge* used** = `matched_len / b_edge_len`            |
| `bearing_diff`   | bearing of this B-edge's span vs the A part matched to it (degrees) |
| `n_points`       | A sample points matched onto this edge                             |

The two coverage axes are independent: **`cover_pct`** is how much of *A* this edge covers, while
**`b_cover_pct`** is how much of *this B-edge* A uses. The per-edge `cover_pct` sum to `overlap_pct`.

### 4.1 Coverage and overhang

A-coverage is all of A **except the leading run on the route's entry vertex and the trailing run on
its terminal vertex** — i.e. only where A **overhangs** past the route's first/last B-edge endpoint
(a run of A-points collapsing onto that single end vertex, with no more B to walk). This is a
segmentation/overhang effect (A and B don't start/end at the same place) and happens on any network;
it is *not* a dead-end concept.

Crucially, only the **ends** count: a *mid-corridor* stall — where A is simply denser than B so the
warping advances A while the B vertex momentarily waits (drift stays low, A is on the corridor) — is
**still covered**. Earlier the metric counted every non-advancing step as uncovered and so
under-reported coverage on real data; it now charges overhang only at the two ends.

*Example.* A is 60 m; the route is `B1 → B2` but B only covers A's middle 20–40 m (A overhangs
0–20 m and 40–60 m):

```
A:  0────10────20────30────40────50────60        (samples every 10 m)
              └─ B1 ─┘└─ B2 ─┘                    (B covers only 20..40)
a0,a1,a2  ── all map to B1's start vertex  (overhang 0..20 → uncovered)
a2→a3→a4  ── B advances along B1 then B2   (covered 20..40)
a4,a5,a6  ── all map to B2's end vertex    (overhang 40..60 → uncovered)
```

Result: `overlap_pct = 20/60 = 33%`. Per edge: `B1` and `B2` each **cover 10 m of A** (`cover_pct`
≈ 17% each, summing to 33%), yet each is **100% used** (`b_cover_pct = 100`) because A walks their
full geometry. So a B-edge can be fully used while A is only partly covered.

---

## 5. Worked example

A straight A-edge with a connected three-edge corridor **and** an isolated full-length parallel
road:

```python
from shapely.geometry import LineString
from network_matching import match_edge_to_bgraph

coords_a = [(0.0, 0.0), (30.0, 0.0)]
b_edges = [
    ("B1", LineString([(0, 0.2), (10, 0.2)])),   # ┐
    ("B2", LineString([(10, 0.2), (20, 0.2)])),  # ├ connected corridor, 0.2 m off A
    ("B3", LineString([(20, 0.2), (30, 0.2)])),  # ┘
    ("B4", LineString([(0, -1.0), (30, -1.0)])), # isolated parallel road, 1.0 m off A
]

res = match_edge_to_bgraph(coords_a, b_edges, snap_tolerance_m=0.5)
print(res["route"])
print(round(res["avg_distance"], 3))
```

```
[('B1', 'forward', 0), ('B2', 'forward', 1), ('B3', 'forward', 2)]
0.2
```

Graph-DTW stitches `B1 → B2 → B3` into one route at 0.2 m drift. The isolated `B4` is rejected:
although it is a single full-length parallel edge, it has no arcs in or out, so any route using
it would strand the rest of A at high cost. **Edge-to-edge DTW would instead report `B4` as a
clean full-length 1:1 match** — the exact failure graph-DTW removes.

A picture of this alignment (and the split, parallel, cycle, and no-U-turn cases) is produced by
[`notebooks/graph_dtw_visualization.ipynb`](../notebooks/graph_dtw_visualization.ipynb).

---

## 6. Running it on a whole network

`DuckDBMapMatcher` wires the primitive into the existing candidate pipeline. Use a one-call
initializer to load + configure both networks:

```python
from network_matching import DuckDBMapMatcher, setup_logging
setup_logging()                              # logs/network_matching_*.log

# WKT CSVs (geometry column = WKT in EPSG:4326):
m = DuckDBMapMatcher.from_wkt_csv(
    "data/osm_edges.csv", "data/sweden_edges.csv",
    id_a="edge_id", id_b="directed_id", utm_srid=3006, max_distance=30,
    keep_cols_b=["name"])                    # carry extra columns through

# ...or GIS files (GeoPackage / GeoJSON / Shapefile, via DuckDB ST_Read):
# m = DuckDBMapMatcher.from_geofiles("osm.gpkg", "nvdb.gpkg",
#         id_a="edge_id", id_b="directed_id", utm_srid=3006, src_srid=3006)

routes_long, routes_summary = m.match_routes(snap_tolerance_m=0.5, step_meters=10, n_jobs=-1)
```

(The manual path — `DuckDBMapMatcher()` + `configure_sources(...)` + `set_parameters(...)` — still
works; the initializers just wrap that boilerplate.) `generate_candidate_pairs()` (DuckDB
`ST_DWithin`) runs once and returns `id_a, wkt_a, id_b, wkt_b` (in UTM); the per-A-edge work is
then an independent unit fanned out with **joblib**.

**`match_routes` parameters.**

| parameter           | meaning |
|---------------------|---------|
| `snap_tolerance_m`  | One edge's **end** is joined to another edge's **start** when they fall within this distance — the head-to-tail junction-crossing tolerance. |
| `step_meters`       | **Gap-fill density**: a vertex every ~N m on top of the node+projection pools (default 10). Smaller = denser/slower and shifts which route wins (cost is count-weighted); `0` = projection-only, fastest. |
| `trim_ends_m`       | **Default `0` (off).** Optional: *remove* a leading/trailing route edge covering **< this many meters of A**. Not a gap-filler (that is `snap_tolerance_m`); off by default because it can delete legitimate corridor edges. |
| `n_jobs`            | Parallel workers over A-edges (joblib): `-1` = all cores, `1` = serial. |
| `max_distance`      | (set via the initializer / `set_parameters`) candidate search radius for `ST_DWithin`. |

**Output — two DataFrames.**

- **`routes_long`** — one row per *(A-edge, B-edge in its route)*, carrying the **per-edge** slice
  (the result divided by edge):
  `source_id, dest_id, seq, direction, edge_match_dist_avg/max/min, edge_a_len, edge_cover_pct`
  (% of A this edge covers), `edge_matched_len, edge_b_len, edge_b_used_pct` (% of the B-edge
  used), `edge_bearing_diff, n_points, route_match_dist, n_edges`. `seq` is the **order of
  matching** along the route.
- **`routes_summary`** — one row per A-edge: `source_id, n_edges, dest_ids, dtw_distance` (avg),
  `max/min_dtw_distance, bearing_diff, overlap_pct, matched_len, route_geom_wkt`,
  `match_type` ∈ {`1:1`, `1:N_ROUTE`, `NO_MATCH`}. Every A-edge appears; unmatched ones as a
  single `NO_MATCH` row.
- **`validate_b_geometry(b_edges)`** reports the endpoint-gap distribution to choose
  `snap_tolerance_m`; `setup_logging()` records progress/timing to a log file.

A real-data demo (OSM ↔ Sweden NVDB, Sundbyberg) is in
[`notebooks/graph_dtw_real_data.ipynb`](../notebooks/graph_dtw_real_data.ipynb); single-edge deep
dives via [`scripts/graph_dtw_edge_detail.py`](../scripts/graph_dtw_edge_detail.py) and a whole-
network map via [`scripts/graph_dtw_map.py`](../scripts/graph_dtw_map.py).

## 7. Scope of this version

- **Directed A → B** only (no symmetric reconciliation).
- B is a **directed graph built forward-only** from the directed edge table; connectivity is
  inferred from coincident **end → start** endpoints within a snap tolerance; candidate edges are
  sorted by id so results are **deterministic**.
- A road's two travel directions are **separate directed edges** in B, so no reverse/backward
  arcs are synthesized: A matches whichever twin agrees with its direction, traversed `forward`.
- Each vertex carries its owning edge (`vert_edge`), so the route is unambiguous at junctions and
  a **U-turn onto a side edge is structurally impossible**.
- The warping path spans the entire A-edge, but **coverage** (`overlap_pct`) is the A-length
  matched to *advancing* B geometry — it drops below 100% wherever A **overhangs** past the
  route's first/last B-edge endpoint (see §4.1). Match distance is the other primary quality
  signal. (`trim_ends_m`, an optional end-edge remover, is **off by default**.)
- **Cost is count-weighted** (a sum over discrete vertices), so route choice depends on
  `step_meters` density; a length-weighted / per-step-projection objective (density-independent)
  and symmetric (B→A) reconciliation remain future work.
