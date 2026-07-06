# Debugging Graph-DTW: Algorithm Internals, Synthetic Cases, Perturbation Tests

This document covers the **algorithm-level debug tooling**: how to see *why* graph-DTW chose a
route (not just which route it chose), how to reproduce a behaviour on a hand-built synthetic
case, and how to measure the algorithm's robustness by distorting the A-edge and re-matching.

Three pieces, designed to be used together:

| Piece | What it gives you |
|-------|-------------------|
| `match_edge_to_bgraph(..., debug=True)` | the raw DP internals (cost tables, backtracked path, moves, trim window) |
| [`network_matching/synthetic.py`](../network_matching/synthetic.py) | named synthetic (A, B-network) cases + perturbation families |
| [`scripts/graph_dtw_debug_viz.py`](../scripts/graph_dtw_debug_viz.py) | one-command full-detail debug figure + per-state stdout trace |
| [`scripts/graph_dtw_perturb_test.py`](../scripts/graph_dtw_perturb_test.py) | families × magnitudes robustness sweep with a stability report |

The algorithm itself is described in [`graph_dtw_matching.md`](graph_dtw_matching.md); the
emission variants in [`weighted_emission.md`](weighted_emission.md). Tests:
[`tests/test_graph_dtw_perturbations.py`](../tests/test_graph_dtw_perturbations.py).

---

## 1. The `debug=True` payload

`match_edge_to_bgraph` (and `graph_dtw_align` / `GraphDTWMatcher.match_edge`) accept
`debug=True`. Off by default and free when off; on, the result carries a `debug` dict (also at
`metrics["debug"]`, and **also on failure returns**, with a `reason`):

| Key | Contents |
|-----|----------|
| `params` | `emission`, `bearing_weight`, `step_meters`, `trim_ends_m` as run |
| `a_pool` | the A axis the DP actually used: `(x, y, is_node)` per pool point |
| `D` | accumulated-cost table (`(N_A_points, V)` point mode; `(N_A_segments, N_arcs)` segment mode) |
| `E` | per-state emission (local cost) table, same shape as `D` |
| `path` / `arc_path` | backtracked states `(a_index, state, move)`, moves in `START / V / H / D` |
| `arcs`, `ridable` | segment mode only: the arc state axis and which arcs may host a state |
| `pairs_all` | the full untrimmed vertex-level alignment `(a_index, vertex)` |
| `kept_span` | `(lo, hi)` indices into `pairs_all` kept after overhang trimming |
| `groups` | consecutive alignment runs per B-edge (pre-trim) |
| `drift_all` | per-step drift over the full (untrimmed) alignment |
| `terminal_state`, `final_cost` | the argmin end state and its accumulated cost |
| `reason` | on failure: `empty_inputs`, `no_finite_alignment`, or `zero_b_traversal` |

The tables obey an exact invariant (unit-tested): along the backtracked path every state's
accumulated cost is its predecessor's cost plus its own emission, `D[s] = D[prev] + E[s]`
(START states: `D = E`). If you suspect the DP, check that first.

```python
res = match_edge_to_bgraph(coords_a, b_edges, debug=True)
dbg = res["debug"]
dbg["D"].shape, dbg["path"][:3], dbg["kept_span"]
```

---

## 2. Synthetic test cases (`network_matching.synthetic`)

Hand-built cases in a plain meter CRS, each isolating one behaviour
(`--list-cases` on the debug script prints this table):

| Case | Isolates |
|------|----------|
| `split` | route stitching across a 3-edge chain |
| `parallel_trap` | connected chain vs an isolated parallel road 8 m away |
| `junction_turn` | turning at a junction with a straight competing exit |
| `cycle` | termination and sane routing with a loop in `GB` |
| `stub` | no U-turn onto a perpendicular edge digitized into the junction |
| `overhang` | A longer than the B corridor (overlap < 100%, trimmed ends) |
| `gap` | a mid-corridor connectivity break larger than `snap_tolerance_m` |
| `curve` | non-axis-aligned geometry (quarter circle, 3 B-arcs) |

```python
from network_matching.synthetic import get_scenario
sc = get_scenario("parallel_trap")
res = match_edge_to_bgraph(sc["coords_a"], sc["b_edges"], debug=True, **sc["defaults"])
```

**Perturbation families** distort the A-edge (B never changes): `shift` (lateral along the
road's local normal, + = left of travel), `longitudinal` (along the local tangent), `translate`
(rigid move in an **absolute compass direction** — `bearing` kwarg / `--translate-bearing`,
0 = north, 90 = east, 180 = south, 270 = west), `noise` (Gaussian σ, densified to 5 m first),
`rotate` (about the centroid), `crop` (% removed, half per end), `stretch` (m extrapolated per
end); plus `reverse()`. Each family carries a unit and a default magnitude grid
(`PERTURBATIONS[name]["grid"]`). Note `shift`/`longitudinal` are orientation-relative (on an
east–west road they move the edge north–south / east–west respectively); use `translate` to
move the edge in any fixed direction regardless of its orientation.

```python
from network_matching.synthetic import apply_perturbation
noisy = apply_perturbation(sc["coords_a"], "noise", 2.0, seed=7)
```

---

## 3. The debug figure (`scripts/graph_dtw_debug_viz.py`)

One command renders the whole algorithm state for one match — synthetic case or real edge,
optionally perturbed:

```bash
python scripts/graph_dtw_debug_viz.py --list-cases
python scripts/graph_dtw_debug_viz.py --case split --trace
python scripts/graph_dtw_debug_viz.py --case parallel_trap --shift -6
python scripts/graph_dtw_debug_viz.py --case curve --noise 2 --seed 7 --emission segment
python scripts/graph_dtw_debug_viz.py --edge-id 1377 --rotate 8         # real data
```

Panels (top to bottom):

1. **Spatial view** — the local B-graph exactly as the DP saw it: per-edge vertex chains with
   direction arrows, vertex types (◆ endpoint / ○ real node / · projection), junction stitch
   arcs, the matched route on the drift ramp, every warp link (gray dashed = trimmed overhang),
   and original vs perturbed A.
2. **Accumulated cost `D`** — heatmap with states banded by owning B-edge (bold tick = in the
   route), shown *per A-step above that step's best* so the competitive frontier is visible;
   the backtracked path is overlaid in white with each move classified by marker
   (★ START, ▲ V = A advances, ▶ H = B advances via Dijkstra, ◆ D = both), red star = terminal
   argmin, shaded columns = trimmed overhang.
3. **Emission `E`** — the same layout for the local cost, so you can separate "the DP chose a
   locally-expensive state for global reasons" from "the local cost already preferred it".
4. **Drift profile** — per-step drift along the alignment, dot color = matched B-edge,
   with edge-transition boundaries.

`--trace` prints the same information as text: one row per DP state with move, A index, state,
owning edge, emission, and accumulated cost — the algorithm's decision log.

Reading failures: in the `gap` case the path visibly stalls (a run of `V` moves on one vertex)
because no arc reaches the next band even though `E` shows cheap cells there — connectivity,
not cost, is the blocker. In `parallel_trap --shift -6` the path rides the `B_trap` band from
row 0 — cost, not connectivity.

---

## 4. The robustness sweep (`scripts/graph_dtw_perturb_test.py`)

Distort A at every magnitude of every family, re-match against the unchanged B-network, and
report the **stability envelope** — at which magnitude the route first changes, and when the
match dies:

```bash
python scripts/graph_dtw_perturb_test.py --case split
python scripts/graph_dtw_perturb_test.py --case parallel_trap --families shift --negate
python scripts/graph_dtw_perturb_test.py --edge-id 1377 --families shift,noise,rotate
```

Output: a per-magnitude table (drift / overlap / bearing / route, with `<-- route changed` and
`<-- NO MATCH` flags), a stability summary per family, and a figure with one row per family
(drift-vs-magnitude curve, ★ = route changed, ○ = NO_MATCH, plus three spatial snapshots).

`--negate` flips magnitude signs — e.g. on `parallel_trap`, positive shift moves away from the
trap and stays stable through 16 m, `--negate` moves toward it and the trap captures the route
at −4 m. To dissect any cell of the sweep, re-run the *debug figure* with the same
case/family/magnitude/seed.

Robustness properties that are pinned as unit tests (`tests/test_graph_dtw_perturbations.py`):
small noise keeps the route; lateral drift ≈ lateral shift; the trap captures at −8 m but not
−2 m; small rotations survive on the curve; a hard crop yields a contiguous sub-route; a
reversed A is NO_MATCH on a directed table.

---

## 5. The interactive playground (notebook / dashboard)

[`notebooks/graph_dtw_playground.ipynb`](../notebooks/graph_dtw_playground.ipynb) is the
no-analysis companion to the tools above: it shows only the **correspondence** — every A sample
point painted in the color of the B-edge it matched, with a link to its matched B point — and
re-matches live as you drag sliders (shift / longitudinal / translate+bearing / rotate / noise /
crop / stretch / reverse, plus `snap`, `step`, `emission`). A build-your-own-sample section takes
the network as plain coordinate lists, so a new B-edge is one appended
`("id", [(x, y), ...])` line.

Run it as a notebook (`network-matching` kernel, needs `ipywidgets`) or as a standalone
dashboard: `voila notebooks/graph_dtw_playground.ipynb`.
