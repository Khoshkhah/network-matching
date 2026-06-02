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
- Consecutive vertices of an edge are joined by a directed **arc** in the digitized direction
  (`forward`) and, unless the edge is one-way, also the reverse (`backward`) — geometry-only
  conflation, so a B road digitized opposite to A is still walkable along A.
- B-edges whose **endpoints** coincide (within a snap tolerance) are joined: their shared
  endpoint becomes a single **junction vertex**, which is how the route crosses from one B-edge
  to the next.

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

### 3.2 Backtracking → route + direction

Each `D[i][v]` records which move won (`vertical`, `horizontal`, or `diagonal`) and, for the two
B-advancing moves, the **arc** (its B-edge and traversal direction). Tracing back from the best
terminal vertex to row 0 yields the warping path of `(a_i, v)` pairs; consecutive steps on the
same B-edge are collapsed into one **route entry**, giving the **route**
`[(b_edge_id, direction, seq), …]`.

### 3.3 End trimming + determinism

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
| `overlap_pct`  | % of A covered by the kept route (≈100, lower if ends were trimmed)     |
| `bearing_diff` | whole-route bearing difference over the matched span (degrees)         |
| `route`        | ordered `[(b_edge_id, direction, seq), …]` (`seq` = order of matching)  |
| `n_edges`      | number of B-edges in the route                                         |
| `route_edges`  | **per-B-edge breakdown** (below)                                        |
| `warp_*`       | per warping-step arrays: `warp_vertices`, `warp_edge`, `warp_is_node`, `warp_a_is_node` (node vs projection on each side) |

**`metrics["route_edges"]`** — one dict per B-edge in the route, the result *divided by edge*:

| field            | meaning                                                              |
|------------------|---------------------------------------------------------------------|
| `dest_id`        | the B-edge id                                                       |
| `seq`            | order of this edge along the route (0,1,2,…)                        |
| `direction`      | `forward` / `backward` (vs the B-edge's digitized geometry)         |
| `match_dist_avg/max/min` | match distance over just this edge's matched points        |
| `a_len`          | metres of A matched onto this edge                                  |
| `matched_len`    | metres of this B-edge traversed                                    |
| `b_edge_len`     | this B-edge's total length (m)                                     |
| `b_cover_pct`    | % of *this B-edge* used = `matched_len / b_edge_len`               |
| `bearing_diff`   | bearing of this B-edge's span vs the A part matched to it (degrees) |
| `n_points`       | A sample points matched onto this edge                             |

(The per-edge `a_len` values sum to A's length; "% of A covered by this edge" = `a_len / Σ a_len`.)

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

A picture of this alignment (and the split, parallel, cycle, and one-way cases) is produced by
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
| `snap_tolerance_m`  | B-edge endpoints within this distance are merged into one **junction** vertex — how the route crosses between connected B-edges (connectivity-inference tolerance). |
| `step_meters`       | **Gap-fill density**: a vertex every ~N m on top of the node+projection pools (default 10). Smaller = denser/slower and shifts which route wins (cost is count-weighted); `0` = projection-only, fastest. |
| `trim_ends_m`       | **Default `0` (off).** Optional: *remove* a leading/trailing route edge covering **< this many meters of A**. Not a gap-filler (that is `snap_tolerance_m`); off by default because it can delete legitimate corridor edges. |
| `oneway_ids`        | B-edge ids walkable only in their digitized direction (no `backward` arc). Default: all bidirectional. |
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
- B is treated as a **directed graph**; connectivity is inferred from **endpoints** within a
  snap tolerance; candidate edges are sorted by id so results are **deterministic**.
- B-edges are walkable in **both directions** by default (geometry-only conflation); one-way
  edges can be restricted to their digitized direction via `oneway_ids`.
- A is matched **in full** — the warping path spans the entire A-edge, so `overlap_pct` is 100%
  except for a genuine **dead-end** (A's road extends past the end of B's corridor, where A
  advances while the matched B point is stuck at a terminal vertex). Match distance is the primary
  quality signal. (`trim_ends_m`, an optional end-edge remover, is **off by default**.)
- **Cost is count-weighted** (a sum over discrete vertices), so route choice depends on
  `step_meters` density; a length-weighted / per-step-projection objective (density-independent)
  and symmetric (B→A) reconciliation remain future work.
