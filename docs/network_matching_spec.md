# Directed↔Directed Whole-Network Matching — Specification

**Status:** design spec, ready to implement. Supersedes the "global reconciliation" track sketched in the improvement plan §3/§4 and corrects decisions **D10** and **D11**. *(The improvement plan — the Hootenanny comparison — is product material, not library material; it lives at `~/projects/product/improvement_plan.md`. All `improvement_plan.md` references below point there.)*
**Scope:** conflating two *whole* road networks (OSM ↔ NVDB today; OSM ↔ DRA/NRN next), directed↔directed, with undirected↔undirected handled by reduction.
**Audience:** the one developer who will build it, and the reader of the paper who needs to know exactly what is proved and what is measured.

---

## 0. TL;DR

The winning architecture is **local exactness + global selection**, in three stages:

1. **`graph_dtw` stays the oracle.** Per A-edge, the DP is exact against the whole local *cyclic* B-graph. Nothing in this spec changes the recurrence, the emission, or `alpha`/`beta`.
2. **Candidates come from an entry-masked DP lattice.** Masking DP row 0 to the vertices of one candidate B-edge yields the *exact* optimum over routes **starting** on that edge; the final row already partitions by exit vertex for free. `n_candidates + 1` DP runs per A-edge give a set of genuinely distinct candidate routes, each with an exact, unmodified DTW cost.
3. **The global layer is an ILP over `(A-edge, route)` variables** whose only hard constraint is **sub-edge interval disjointness on B** (measure-injectivity), with **NO_MATCH priced in the objective**. Solved by HiGHS via `scipy.optimize.milp` — no new dependency. Measured: whole of Sundbyberg, 35 343 vars / 10 916 rows, **proven optimal in 0.4–2.2 s**, LP relaxation integral.

**Milestone 1 ships before any of that and beats the status quo on its own:** sub-edge intervals, a stub trim, a real U-turn bug fix, and the DP objective exposed.

**And the gate that decides whether stages 2–3 ever get built is 3 days of work, not 3 weeks** (§6.6). On Sundbyberg the global layer changes ~4–6% of decisions, ~90% of which a drift gate would also catch. That number must be re-measured on a motorway/dual-carriageway tile before the solver is funded.

---

## 1. The object

A cyclic source has **no order**, so V1-monotonicity has nothing to range over: **a global warping of a road network is not a well-defined object.** Do not look for one. The right object is:

> **φ** — a *partial*, *orientation-preserving*, *measure-injective*, piecewise-linear map from **A's carrier** (the 1-complex of A's geometry) into **B's carrier**, commuting with the reverse-twin involution.

- Each A-edge's DTW route is a **chart** of φ.
- "Globally consistent" = **the charts glue**, i.e. no B-metre is claimed twice.
- φ is honestly **partial** (12.2% of Sundbyberg A-edges are NO_MATCH; 34.8% of A-edges are out of NVDB's scope entirely — `service`, `living_street`), **non-surjective** (5.0% of B directed edges are never claimed), and **injective only in measure** — never edge-wise.

Two measured facts fix the shape of every constraint below:

| fact (Sundbyberg, OSM↔NVDB) | consequence |
|---|---|
| **44.8%** of A-junctions land in the *interior* of a B-edge | "adjacent routes must share a B **node**" is simply **false**. Continuity is a statement about **(b_edge, arc-length offset)** positions. |
| **51.2%** of claimed B-edges are legally claimed by **>1** A-edge (A is 1.5× finer; A **tiles** B). True 1:1 = **1.9%** of the network. | B-edge **exclusivity** is catastrophic. The constraint is **interval disjointness**, which permits N:1 tiling and forbids only genuine double-cover. |

Both are only *expressible* once B-side arc-length offsets exist. That is Milestone 1.

---

## 2. Negative results — what does NOT work, and why

This section is normative. Each item below was built or measured and killed. A reader must not re-enter these.

### 2.1 SCC condensation of the source — **NO-GO**

`nx.condensation(A)` *is* a DAG, so "run Mode 3 on it" looks obvious. It is empty and type-incorrect.

- 48 SCCs; the giant SCC holds **3797 of 3948 arcs (96.2%)** and **219.4 of 229.6 km (95.6%)**.
- The condensation has **48 super-nodes, 41 arcs**; only **45 arcs (1.1%)** are inter-SCC — dead-end spurs and one-way stubs, i.e. the *easiest* matches in the map.
- A super-node containing 1734 nodes **has no `(x, y)` and no bearing**. `dag_dtw`'s emission `E(a,v) = ‖pos(a) − pos(v)‖ + λ·circ(bearing)` is **undefined** on it. Contraction destroys precisely the object being warped.
- It **does not recurse**: the inside of the giant super-node *is* a strongly connected road network of 1734 nodes — the original problem, verbatim, on 96% of the input.
- Worst: the giant SCC is largely a **twin artifact**. Drop one arc per twin pair and it collapses from **1734 nodes to 187**. Condensation's first act on a road network is to contract every two-way street into a super-node — i.e. to *destroy the directedness Mode 1 exists to exploit*.

**This closes `improvement_plan.md` "Cyclic-exactness M2 — SCC survey (GO/NO-GO gate)" with a NO-GO, on measured data, citable in the paper.**

### 2.2 DAG corridor decomposition (any carving) — **structurally impossible**

**Theorem (twin barrier).** A DAG contains no directed 2-cycle. A two-way street's twin pair `u→v` / `v→u` **is** a directed 2-cycle. Measured: **3414 of 3948 arcs (86.5%)** are one half of a twin pair (**100%** under the undirected reduction). Therefore **no corridor, under any carving, at any K, from any seed, can ever contain both directions of a two-way street** — so **twin consistency, the constraint the undirected case is *defined* by, is inexpressible inside every corridor, forever.**

And even ignoring twins, the **gluing is circular**:
- A corridor's only product is a **joint** warping (`route(e)` was chosen *because* it agrees with `route(f)` at their shared junction). **Per-arc voting discards exactly that joint object** — you pay 2–3× redundancy for a consistency certificate and throw it away in the last step.
- **Whole-corridor selection** is a strictly *harder* restatement of the original assignment problem (more objects, denser coupling, no exactness).
- **Edge-disjoint corridors** internalize only **47.0%** of junction pairs, and the disagreements land exactly on the corridor boundaries.

There is no fourth option.

### 2.3 Whole-DAG (non-star) `dag_dtw` on a network — **intractable and unsafe**

- **V4 has no null state.** DAG-DTW demands *every* A-vertex match. **15.4%** of densified A-arcs have **zero** B-arc within 30 m (NVDB does not carry `service`/`living_street`). One unmatchable arm empties its split's V3 exits and **kills the whole component**. Measured feasibility on real carved corridors: **50% fail at length 8, 67% at length 16**, all-or-nothing `ValueError`. Raising `r` 30→50 m changed the failure counts by **exactly zero** and cost +60% — the library's error message ("increase match_radius_m") **lies**.
- Worse, `dag_dtw.prepare(k_min=1)` **silently force-matches a candidate-less arc to the nearest B-arc at unbounded distance**, and V1/V2/V3 then *chain that fabrication into its neighbours' routes*. This is the worst failure mode available: a confidently exactly-optimal alignment of a problem whose constraints are known-false.
- **`extract_cell` blows up on reconvergence**, which is what a city block *is*: 0.1 s at 105 A-arcs → **64 s at 278** → **did not finish in 10 min at ~600**. A real θ-carved corridor has **321 open pending keys** against `max_rows=50000`.

**Mode 1's per-A-edge independence is not a limitation — it is a robustness property.** A scope difference stays local. Do not give it up.

### 2.4 Min-cost flow / "conservation at junctions" — **unsatisfiable, not merely wrong**

The phrase "road network + global consistency" makes every reader reach for a flow. Don't.
- Matching is **not** a flow: N:1 collapse breaks `in-degree = out-degree` at every merge.
- **10.8%** of A-junctions with ≥3 arms have *some* arms matched and some NO_MATCH — conservation is **unsatisfiable by construction**.
- Unit capacities on B-edges would forbid the **dominant legal pattern** (51.2% multi-claim).
- And the problem is APX-hard (§7.3), so a poly-time MCF formulation would imply P = NP.

*(Flow survives only as a Lagrangian slave: given prices, "pick a max-profit set of pairwise-disjoint claims on B-edge b" is weighted interval scheduling = a shortest path on a path-DAG. Documented as an escape hatch in §9.4; not built.)*

### 2.5 Top-k by exclusion re-runs (Yen-style) — **near-empty; do not build**

Measured over all 3611 A-edges (ban the winner's highest-A-coverage B-edge, re-run): **mean 1.26 routes per A-edge**; 2677 (74%) yield exactly **one**. Diagnosed on a 300-sample: **298/300** re-runs return *zero B traversal* (NO_MATCH) despite a median of 8 remaining candidate B-edges. **Banning the spine destroys the corridor; there is no second corridor to find.** An assignment problem whose candidate set has one element is a threshold wearing a solver's clothes.

### 2.6 Terminal-row top-k (free) — **degenerate**

The last DP row holds the exact optimum ending at *every* vertex, so "best per terminal B-edge" is free. It enumerates **tails, not routes**: verified on A-edge 1262, the top-5 were `[518,519,520]`, `+2440`, `+1273`, `+1801`, `+2440,1272` — prefix-identical, several of which the existing zero-traversal trim collapses back together.

### 2.7 k-best Dijkstra inside the DP — **enumerates warpings, not routes**

The DP state graph is cyclic (in-row "H" moves follow B arcs). True k-best needs each state popped up to k times, `D` becoming `(N,V,k)`, `back[]` carrying `(move, pred, rank)` — ~60–100 lines in *each* of the two DPs, memory ×k — and it returns k best *warpings*, overwhelmingly the same route with a stall shifted by one point. Dedup-and-keep-popping has no bound. **Don't.**

### 2.8 A support / continuity term in the global objective — **kills the architecture and buys nothing**

- **It destroys decomposition unconditionally.** A-adjacency *is* the road network: the coupling graph with a support term is **one component of 98.8%** of matched A-edges, at *every* prune radius including δ=0.
- **It does not solve.** The pairwise linearisation generates **302 202 aux variables / 313 118 rows**; HiGHS **did not finish in 13 minutes**.
- **And it is worth nothing.** Measured at the raw independent argmin, with continuity tested on B *positions* (not nodes): **median junction gap = 0.00 m**, 77.0% ≤1 m, 92.3% ≤5 m. The entire dynamic range available to the factor — the best gap reduction obtainable by re-labelling *either* side over the *whole* candidate set — has **median 0.00 m**, and only **4.33%** of adjacent pairs could improve by >5 m. Of 753–826 "broken" adjacency pairs, **98.1%** have both routes terminating within 15 m (median ~1 m) of the shared junction, **zero** have a route ending >30 m away, and **genuine route divergence = 0 pairs**.

The apparent 10.4–12.7% "topological inconsistency" is an **artifact of the wrong test** (whole B-edges instead of offsets) plus the **stub artifact** (§4.3). **Fix the artifacts; do not model the phantom.**

*This is the deep result, and it is the paper's headline:* **the exact local DTW cost is precisely what makes global propagation unnecessary.** Hootenanny needs iterative score propagation because its unary is ambiguous and needs neighbours to disambiguate it. Ours is not.

### 2.9 Loopy belief propagation — **dominated on both sides**

Measured (λ=10, τ=15): loopy min-sum BP (damped, 300 it) = 904 930 in **6.0 s**; exact per-component after persistency = **904 926 in 2.0 s**; a **two-sweep greedy ICM** = 905 349 in **0.2 s**, capturing **98.5%** of the exact gain. BP is dominated above by exact inference and below by the dumbest possible baseline.

**Discipline this imposes on everything downstream: any solver must be benchmarked against 2-sweep ICM and against a plain drift gate. A solver that beats "do nothing" but not "do the dumbest thing" is not a contribution.**

### 2.10 Edge exclusivity / `resolve(one_to_one)` on this data — **catastrophic**

Would reject 51.2% of claimed B-edges, of which 1727 of 2190 shared are **correct N:1 tilings**. True 1:1 is **1.9%** of the network; the `match_type == "1:1"` label (52% of A-edges) is a **selection artifact** — it only means "the route has one B-edge", and those A-edges have median length 26.6 m sitting inside B-edges of median length 111.4 m, with median B-usage 21.9%. **`resolve(one_to_one)` must never be applied to OSM↔NVDB.**

### 2.11 Hard twin consistency — **infeasible on real data**

**333 of 1434 (23.2%)** NVDB physical roads are one-way-only while OSM carries both directions ⇒ **31 of 1693 OSM twin pairs are provably unsatisfiable** — there is no B-edge for the reverse to land on. A hard equality makes the model **infeasible**. Soft, and **the residual *is* the `oneway_mistagged` product** (§8).

---

## 3. Architecture

```
                 A (directed, cyclic)        B (directed, cyclic, twins)
                         │                            │
   ┌─────────────────────▼────────────────────────────▼──────────────────┐
   │  CANDIDATES   matcher.generate_candidate_pairs   (ST_DWithin + NEW  │
   │               bearing gate)  → mean 8.25 → ~1.0 B-edges per A-edge  │
   └─────────────────────┬───────────────────────────────────────────────┘
                         │
   ┌─────────────────────▼───────────────────────────────────────────────┐
   │  LOCAL ORACLE (STAGE 1, unchanged DP)   graph_dtw.match_edge_to_bgraph
   │    · exact against the whole local CYCLIC B-graph (in-row Dijkstra) │
   │    · NEW: vert_pos_m → edge_b_from_m / edge_b_to_m  (D6)            │
   │    · NEW: stub trim; u→twin(u) arc removed; dp_cost exposed         │
   └─────────────────────┬───────────────────────────────────────────────┘
                         │            ┌── MILESTONE 1 ENDS HERE, and it already
                         │            │   beats the status quo on its own.
   ┌─────────────────────▼───────────────────────────────────────────────┐
   │  STAGE 2  CANDIDATE LATTICE   graph_dtw_align(entry_edge=…)         │
   │    one DP per entry B-edge; exit partition free from the final row  │
   │    → routes_alts  (rank, dp_cost, cost_m2, claims, intervals)       │
   └─────────────────────┬───────────────────────────────────────────────┘
                         │
   ┌─────────────────────▼───────────────────────────────────────────────┐
   │  GATE   reconcile.dee_persistency()  — 0.3 s, EXACT, ~90 LOC        │
   │    frozen% + active component sizes = the instance-hardness meter.  │
   │    If frozen% > 85 on a HARD tile → stop. Ship the certificate.     │
   └─────────────────────┬───────────────────────────────────────────────┘
                         │
   ┌─────────────────────▼───────────────────────────────────────────────┐
   │  STAGE 3  GLOBAL SELECTION   reconcile.resolve_network()            │
   │    ILP over (A-edge, route) + NULL:                                 │
   │      min  Σ c(a,r)·x  +  Σ τ·L_a·x_null  +  μ·Σ w_atom·s_atom       │
   │      s.t. Σ_r x[a,r] + x[a,∅] = 1                     (GUB)         │
   │           Σ_{claims ⊇ atom} x − s_atom ≤ 1     (interval packing)   │
   │    HiGHS (scipy.optimize.milp). No support term. No flow.           │
   └─────────────────────┬───────────────────────────────────────────────┘
                         │
        routes_long (selected) · routes_summary (+triage) · routes_alts · issues
```

---

## 4. Stage 1 — the local layer (Milestone 1)

Four changes inside `graph_dtw.py`. None touches the recurrence. **Together they beat the status quo with no global layer at all.**

### 4.1 `_assemble_route` — the refactor everything sits on

The ~230-line post-DP assembly tail (`graph_dtw.py:693-921`: grouping, zero-traversal trim, per-edge `a_len`/`b_len` attribution, overhang/overlap part, metrics) runs **once**, on the single argmin path, with `pairs` / `step_e` / `warping_all` as closure state. Emitting *k* routes means running it *k* times. Extract it first, behind a regression pin.

```python
# graph_dtw.py
def _assemble_route(
    pairs: List[Tuple[int, int]],
    gb: LocalBGraph,
    ax: np.ndarray, ay: np.ndarray,
    a_is_node: List[bool],
    *,
    trim_ends_m: float,
    stub_trim: bool = True,
    seg_records: Optional[List[dict]] = None,
    dp_cost: float,
) -> Dict[str, Any]:
    """Pure. (warping path, local graph) -> {route, route_edges, metrics, warping}.
    No closure state, no DP. Callable once per candidate route."""
```

**Budget it at a week, not two days.** `improvement_plan.md:224` is right. Pin it first: `tests/test_assemble_route_pin.py` asserts column-**subset** equality (D7) of `routes_long` / `routes_summary` on Sundbyberg before and after.

### 4.2 Sub-edge intervals — `edge_b_from_m` / `edge_b_to_m` (D6, Phase 2)

**This is the single highest-leverage change in the whole spec.** Without arc-length offsets on B, *neither conflict nor continuity is expressible at all*.

`LocalBGraph` gains one field:

```python
@dataclass
class LocalBGraph:
    ...
    vert_pos_m: np.ndarray     # float, length V: arc-length position of vertex v ALONG ITS OWN B-edge
    edge_twin: List[Optional[int]]   # edge_index -> edge_index of its reverse twin, if present (§4.4)
```

The arc-lengths already exist inside `_node_projection_pool` (`graph_dtw.py:113-119`, the `(s, is_node)` pool) and are **discarded** when converting to xy at `:134-138`. Either keep them, or cumsum each edge's contiguous vertex block in `build_local_digraph` (both work; the cumsum is 4 lines).

Then, in the per-group loop of `_assemble_route` (today `graph_dtw.py:770-805`), the group's entry vertex `pairs[k][1]` and exit vertex `pairs[j][1]` are **already in scope**:

```python
"edge_b_from_m": float(gb.vert_pos_m[pairs[k][1]]),
"edge_b_to_m":   float(gb.vert_pos_m[pairs[j][1]]),
```

Well-defined because **B never moves backwards inside an edge** (forward-only arcs, `graph_dtw.py:268-271`).

**Test oracle, free:** `dag_dtw.parts_from_matching` **already emits `b_from_m` / `b_to_m`** (plus `b_head_m` / `b_tail_m`, `drift_m`, `bearing_diff_deg`). Mode 3 is a *working reference implementation* of exactly this logic. Pin the new Mode-1 columns against Mode-3 parts on any A-edge both can match. That turns Phase 2 from a 3-day guess into a 2-day port with a golden.

**Acceptance:** intervals reproduce `edge_matched_len` to <0.1 m on all route rows; middle-of-route members are exactly `[0, edge_len]`.

### 4.3 Stub trim

**Measured:** 27–28% of all route members (1282–1566 of 4741–5538) are B-edges traversed for **<10% of their length AND <10 m**, carrying only **1.44%** of matched length. Position: **789–1061 first, 493–505 last, ZERO middle.** Middle members are 100%-used at p10, p50 *and* p90. This is a pure **free-entry/free-exit boundary artifact** (row 0 is free-entry, `graph_dtw.py:641-645`; termination is free-exit).

```python
def _stub_trim(route_edges: List[dict], *, max_used_pct: float = 10.0,
               max_matched_len_m: float = 10.0) -> List[dict]:
    """Drop LEADING and TRAILING route members with edge_b_used_pct < max_used_pct
    AND edge_matched_len < max_matched_len_m. Never touches interior members."""
```

**Why this must land before any solver:** of the 220 decision changes a full global solver produces, **114 are "extend the route to reach the junction" and 12 are "trim the stub"** — 126 of 220 are this artifact. **Build the solver first and the solver gets credited with the trim's wins.**

### 4.4 The U-turn arc — a live correctness bug

`docs/graph_dtw_matching.md:60` and `:368` claim a U-turn is **"structurally impossible"**. **That is false.** The stated mechanism (vertices owned by one edge; an edge that only *ends* at a junction has no outgoing arc) does forbid dipping onto a *side* edge and returning. It does **not** forbid the out-and-back hairpin onto the **reverse twin**, because `end(u) → start(twin(u))` is a legitimate head-to-tail inter-edge arc — the twin starts exactly where `u` ends.

**Measured: 18 routes contain both twins of one physical B road, immediately consecutive**, on A-edges that are *not* closed loops (`source != target`). Drift median 38.0 m — but **2 of the 18 have drift <5 m and pass every threshold silently**.

Fix, in `build_local_digraph`'s inter-edge arc loop (`graph_dtw.py:275-285`):

```python
for eu, u in enumerate(ends):
    for ew, w in enumerate(starts):
        if eu == ew:
            continue
        if edge_twin[eu] == ew:      # NEW: u -> reverse(u) is a hairpin, not a junction crossing
            continue
        ...
```

This needs the **B twin map** plumbed into `build_local_digraph` (NVDB `original_edge_id`; generally, a `b_twin: Mapping[b_edge_id, b_edge_id]` argument, `None` = no twins known).

**Exactness is preserved:** removing an arc from `G_B` leaves the Dijkstra-based DP exact on the restricted graph.

Then **fix the two doc sentences.** Anyone who reads the code will check.

### 4.5 Expose the DP objective

**Hazard:** the reported `dtw_distance` is the **raw mean of the per-step emissions** (`graph_dtw.py:825`; in segment mode the mean of the middle-to-middle state costs, `:860`). The DP minimised a weighted **sum** (`graph_dtw.py:546-548`, `alpha`/`beta`), available today **only** as `dbg['final_cost']` (`:691`, `:451`). They disagree, and they disagree in a way that matters: **the mean is length-normalised, the sum is not.**

Promote it to a first-class returned field `dp_cost`, and state in the schema which quantity every downstream consumer uses. The ILP's assignment cost is defined in §6.2 and it is **neither** of them raw.

### 4.6 Bearing pre-filter on candidates (cheap, large)

In `matcher.generate_candidate_pairs` (`matcher.py:226`), add a bearing gate:

```python
def set_parameters(self, ..., max_bearing_diff_deg: Optional[float] = 45.0):
```

**Measured:** cuts the candidate set from **mean 8.25 → ~1.0** directed B-edges per A-edge (≈ halves DP cost), and across **4042** (A-edge, physical-B-road) plausible pairs it passed **both** NVDB twins in **ZERO** cases and exactly one twin in all 4042. **This is the entire undirected reduction's twin-consistency constraint, delivered for free, with no solver** (§8).

---

## 5. Stage 2 — the candidate lattice

### 5.1 The mechanism (3 lines)

Row 0 is free-entry: `D[0][v] = e0[v]` for all `v` (`graph_dtw.py:641-645`). **Mask it** to the vertices of one B-edge and the DP returns the *exact* optimum over routes **starting** on that edge. The final row `D[N-1][·]` already holds the exact optimum ending at *every* vertex, so the **exit partition is free**.

```python
# graph_dtw.py
def graph_dtw_align(
    coords_a, gb, *,
    emission="point", bearing_weight=0.0, alpha=1.0, beta=1.0,
    trim_ends_m=0.0,
    entry_edge: Optional[int] = None,     # NEW: mask row 0 to this B-edge's vertices
    exit_edges: Optional[Sequence[int]] = None,  # NEW: return one backtrack per exit B-edge
    dbg=None,
): ...
```

```python
def match_edge_to_bgraph(
    coords_a, b_edges, *, ..., b_twin: Optional[Mapping[Any, Any]] = None,
    lattice: bool = False,        # NEW: return the (entry, exit) lattice, not one route
    k_max: int = 8,               # NEW: keep the k cheapest distinct edge-sequences
) -> Dict[str, Any]:
    """lattice=False -> {"route": ..., "metrics": ..., "warping": ...}   (today's contract)
       lattice=True  -> {"routes": [RouteCand, ...]}  ranked by dp_cost, rank 0 == today's route."""
```

### 5.2 Measured

| | Sundbyberg |
|---|---|
| distinct routes per A-edge, uncapped | **mean 45.7**, median 33, p90 108, max 531 |
| capped at `k_max=8` | mean 6.76 routes, of which **4.44 distinct edge-sequences** |
| **rank-0 == today's `match_routes` top-1** | **3466 / 3466 matched A-edges (100%)** ← *this is the regression pin, for free* |
| wall time, 20 cores | **28.1 s** (vs 7.0 s baseline) = 4× wall / **22× core** (156 ms/A-edge core-time) |
| margin to next-best route, per metre of A | p10 0.011, **p50 0.427**, p90 27.4 m — **52% of A-edges have a second route within 0.5 m/m** |

That last row is the justification for the lattice: **the local cost genuinely under-determines the route for half the network**, and it is the *only* generator on the table that produces distinct corridors rather than tail-variants (§2.5–2.7).

### 5.3 Cost control

At county scale, restrict the entry masks to B-edges within `max_distance` of A's **first point** (typically 2–3, not 9). The exit partition stays free. Roughly halves the lattice cost with no change to the exit fan.

---

## 6. Stage 3 — `network_matching/reconcile.py`

### 6.1 Data structures

```python
# network_matching/schema.py   (NEW; D6 asks for this — constants, so probes can't drift)
EDGE_B_FROM_M = "edge_b_from_m"
EDGE_B_TO_M   = "edge_b_to_m"
ROUTES_LONG    = (...)   # ordered tuple of column names
ROUTES_SUMMARY = (...)
ROUTES_ALTS    = (...)
```

```python
# network_matching/reconcile.py
@dataclass(frozen=True)
class Claim:
    source_id: Any
    dest_id:   Any        # DIRECTED B-edge id
    seq:       int        # position in the route -- (source_id, dest_id) is NOT a key (§6.7)
    from_m:    float
    to_m:      float

@dataclass(frozen=True)
class RouteCand:
    source_id:  Any
    rank:       int              # 0 == today's argmin route
    entry_edge: Any
    exit_edge:  Any
    dp_cost:    float            # the DP objective (weighted sum). EXACT.
    drift_m:    float            # mean emission (== today's dtw_distance)
    overlap_pct: float
    bearing_diff: float
    dest_ids:   Tuple[Any, ...]
    claims:     Tuple[Claim, ...]
    cost_m2:    float            # the ASSIGNMENT cost -- see §6.2

@dataclass(frozen=True)
class Atom:
    dest_id: Any
    lo_m: float
    hi_m: float
    members: Tuple[Tuple[Any, int], ...]   # (source_id, rank) whose claim covers this atom
```

### 6.2 The objective — and the one honest unit

`dtw_distance` is a **mean**; `dp_cost` is a **sum** over states. Summing means across A-edges over-weights short edges; summing DP sums over-weights long ones (they accrue more states). The assignment cost is the **drift integrated along A**:

```
c(a, r)  =  L_a · [ overlap(a,r) · drift(a,r)  +  (1 − overlap(a,r)) · τ ]        [m²]
c(a, ∅)  =  L_a · τ
```

- Units **metre-of-A × metre-of-drift** — an extensive *measure*, so it composes additively over tiles.
- `τ` = **the price of not matching a metre of A** (metres of drift). With no coupling this reproduces *exactly* the drift gate — the right sanity property.
- The `(1 − overlap)·τ` term puts `overlap_pct` **into the objective** instead of leaving it a post-hoc filter. This is precisely what lets the global layer **reselect** rather than only delete — the exact deficiency of `resolve_routes` today (`matcher.py:1147-1165`: it can only blank a row to NO_MATCH).
- **State which cost you are summing, in the paper.** If you sum the mean, you are *no longer minimising what the DP minimised*. That is a defensible choice; it is not a silent one.

### 6.3 Constraints

**GUB (one decision per A-edge)** — feasibility is therefore free; `x[a,∅]` conflicts with nothing, so **the ILP is never infeasible**:

```
Σ_{r ∈ K(a)} x[a,r]  +  x[a,∅]  =  1
```

**C2 — interval packing on B (the *only* hard structural constraint).** Build the arrangement of all candidate claim intervals on each *directed* B-edge; its **atoms** (elementary intervals between breakpoints) give a purely **linear** packing constraint — no quadratic terms, no per-pair aux variables:

```
for each atom (b, [lo,hi)) with ≥2 possible claimants:
      Σ_{(a,r) : claim(a,r) ⊇ atom}  x[a,r]  −  s[b,atom]  ≤  1
      s[b,atom] ≥ 0,   objective term  μ · (hi − lo) · s[b,atom]
```

`Σ μ·w·s` is exactly `μ × ∫_B max(0, multiplicity − 1) db` = **μ × total duplicated B-metres**. Measured: **6968 atoms** with ≥2 possible claimants (median 11 members).

**The slack is not softness for its own sake — it is the declared exception.** An A-side dual carriageway collapsing onto a single B centreline *legitimately* overlaps. A **hard** disjointness constraint plus a hand-drawn "physical-collapse sibling" exemption list would destroy correct matches; a **priced** slack surfaces them as a paid, reported exception. `μ = ∞` recovers hard disjointness if you ever want it. Measured: **133 of 6968 atoms retain paid slack** at τ=12, μ=10 — those are the genuine 2:1 collapses, and they become `route_conflict` issues.

*(Equivalent, fewer rows: encode the maximal-clique facets of the per-B-edge interval graph via a left-to-right sweep — 819 cliques replaced 1424 pairwise constraints in the measured instance. Same polytope; see §7.3 for why it is integral.)*

**C1 — support/continuity: NOT IN THE MODEL.** See §2.8. If a reviewer insists, the affordable form is a **candidate-generation prior** (bias the entry mask of `a'` toward the exit edge of `a`'s selected route), not an objective term — but then exactness is gone.

**C3 — twin consistency: soft, and only when the candidate bearing gate hasn't already done it.** See §8.

**C4 — conservation: absent.** See §2.4.

**C6 — U-turns: not a constraint at all.** It is an arc deletion **inside** the DP (§4.4).

**C7 — direction agreement: already structural** (forward-only B arcs).

### 6.4 Provably safe pruning (persistency)

Two exact dominance rules, both from **monotonicity of the penalty** (adding a claim can never *reduce* anyone's overlap):

1. **Null dominance.** Prune any `r` with `c(a,r) ≥ τ·L_a`. Switching to `∅` never *increases* the penalty, so `r` can never beat NULL.
2. **DEE bound.** Prune `r` if `c(a,r) − c(a,r*) > μ · Λ(a)`, where `Λ(a)` = the max B-metres any candidate of `a` claims (≈ `L_a`). Switching back to the local argmin `r*` can save at most `μ·Λ(a)` in penalties; if the local cost gap exceeds that, `r*` strictly improves any solution using `r`.

**Measured** (δ = μ = 10 m/m): `|K|` 6.8 → 4.6, ILP 35 343 → **19 703 vars**, **0.74 s**, **same optimum**.

### 6.5 The solve

```python
def build_claims(routes_alts: pd.DataFrame) -> List[RouteCand]: ...
def build_atoms(cands: Sequence[RouteCand], *, min_atom_m: float = 0.0) -> List[Atom]: ...
def prune_persistent(cands, atoms, *, tau: float, mu: float) -> List[RouteCand]: ...
def coupling_components(cands, atoms) -> List[List[Any]]: ...   # A-edge ids per component
def solve_ilp(cands, atoms, *, tau: float, mu: float,
              time_limit_s: float = 60.0) -> "Solution": ...    # scipy.optimize.milp / HiGHS
def resolve_network(routes_alts: pd.DataFrame,
                    routes_summary: pd.DataFrame,
                    routes_long: pd.DataFrame,
                    *, tau: float, mu: float,
                    action: str = "report",       # "report" | "apply"
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (routes_summary, routes_long, issues). action='report' writes diagnostics only."""
```

**Solve the tile as ONE ILP.** Do **not** decompose as a solve strategy — see §6.7. `scipy.optimize.milp` ships HiGHS; scipy is already in `requirements.txt` (bump `>=1.7` → `>=1.9`, ship as an optional `[solve]` extra per the lazy-heavy-deps convention). **No new dependency.**

**Measured, whole of Sundbyberg (τ=12 m, μ=10 m/m, |K|≤8):**

| | |
|---|---|
| model | 35 343 vars (28 375 binary), 10 916 rows, 122 775 nnz |
| HiGHS | **proven OPTIMAL (`mip_rel_gap = 0`) in 0.4–2.2 s** |
| LP relaxation | **ZERO fractional binaries** at every τ ∈ {6,8,10,12,16,20}; LP bound within **0.12%** of the MILP optimum |
| effect vs local argmin | 2541 keep · **255 re-selected** (7.4%) · **800 → NULL** · 133 atoms with paid slack |
| residual span conflicts | **802 → 0** |

### 6.6 The gate — ship the probe before the solver

```python
def dee_persistency(cands, atoms, adjacency=None, *, tau, mu) -> "PersistencyReport":
    """Goldstein dead-end elimination (exact). Returns frozen labels + a CERTIFICATE:
       no global optimum of the stated model uses an eliminated label.
       .frozen_pct, .active_ids, .component_sizes, .max_cyclomatic"""
```

~90 LOC, **0.3 s**, exact. Measured: **frozen 88.3%** of A-edges at λ=10 (97.1% at λ=1); the residual falls into **167 components of which 158 are trees**, max 16 variables.

**`frozen_pct` is the instance-hardness meter, and it answers `improvement_plan.md` Phase 8's own gate with a number, with no labels and no solver.** Ship it standalone (M3). Then:

> **THE DECISION.** Every number in this spec comes from **one dense urban municipality** where only **11.1%** of A's length has more than one plausible corridor. Point the probe at a **Stockholm-county tile with motorways, dual carriageways, grade separation and long parallel frontage roads** — the one regime nobody has measured, and the only regime where a global layer would earn its keep. **If `frozen_pct` stays > 85%, write the NO-GO, close the track, and ship the certificate. Only if it collapses toward 50–60% do M4–M5 get funded.**

Ship the **2-sweep ICM** alongside it as the mandatory baseline (§2.9).

### 6.7 Decomposition: **the plan's premise was false**

`improvement_plan.md` D10 assumes "conflict components are typically 2–6 A-edges; exhaustive + greedy suffices; the MILP rung is cut." **This is measured on the top-1 selection — i.e. it presumes the answer a solver is supposed to find.** A solver may move any variable to any candidate, so the sound coupling graph is over the **whole candidate set**:

| coupling graph | components > 1 | largest | % of A in comps > 6 |
|---|---|---|---|
| full top-8 lattice | 66 | **1343** | **91.3%** |
| pruned δ ≤ 10 m/m (the safe prune) | 269 | **83** | — |
| top-1 only (what D10 measured) | 389 | 13 | 4.5% |
| **with a support term, at ANY prune radius** | 2 | **3554 = 98.8%** | — |

Exhaustive enumeration is dead (`7.2^83` for the largest safe-pruned component; `3^246` with continuity). **The ILP is not the heavy option — it is 0.74 s and it is *less code* than exhaustive + greedy.**

**Component decomposition is available as a *parallelisation*, not as the solve.** Under the safe prune: 269 non-trivial components, max 83, median 3, embarrassingly parallel. Useful for incremental re-solves (re-solve only the components a changed A-edge touches). Not needed for throughput.

**Key hazard:** `(source_id, dest_id)` is **NOT unique** in `routes_long` — routes can re-enter a B-edge (1–2 cases measured; groups are consecutive runs). **Key every claim on `(source_id, dest_id, seq)`**, or the ledger silently drops or double-counts re-entries. Likewise `edge_b_used_pct` is **capped at 100** (`graph_dtw.py:802`), so over-traversal is invisible in that column — **use the intervals, never the pct.**

---

## 7. Exactness — precisely what is proved and what is not

### 7.1 Exact, provable

| claim | why |
|---|---|
| **The local cost of every candidate route.** | Entry-masking row 0 restricts the DP's *initial state set*; the recurrence, the emission, the in-row Dijkstra (which is what makes a **cyclic** B target exact) and `alpha`/`beta` are untouched. The final row is the exact optimum ending at each vertex. **Verified empirically:** rank-0 of the lattice reproduces the library's current top-1 for **3466/3466** matched A-edges. |
| **Deleting the `u → twin(u)` arc.** | Removes an arc from `G_B`. The DP is exact on the restricted graph. |
| **The two persistency prunes** (§6.4) and **Goldstein DEE** (§6.6). | Dominance / persistency proofs: they remove only labels that appear in **no** optimal solution of the stated objective. Not heuristics. |
| **The global selection, given the candidate sets.** | ILP solved to `mip_rel_gap = 0` by HiGHS. The LP bound is a per-instance **certificate**. |
| **The per-B-edge packing polytope is integral, in principle.** | The maximal-clique/vertex incidence matrix of an **interval graph** has the **consecutive-ones property** ⇒ **totally unimodular**. So the packing constraints contribute **zero** integrality gap on their own. |

### 7.2 Heuristic — say it plainly

- **The candidate cover is a restriction, not a k-best list.** The lattice is `{optimal route for each (entry-edge, exit-edge) pair}`. Routes optimal for *no* (entry, exit) pair are absent. Exit granularity can be refined edge→vertex for free; entry granularity edge→vertex costs V DP runs. **The honest claim is "optimal over the entry/exit-optimal candidate cover", not "globally optimal".**
- **Top-k recall is UNMEASURED.** Whether the true route is in the lattice can only be established against labels (the D3 campaign). *Every* exactness claim is conditional on that, and it must be stated first, not last.
- **τ and μ are calibration, not theory.** τ ∈ [6, 20] moves NULLs from 1067 to 567. Take τ from `thresholds.suggest_thresholds`; set μ from the cost of a duplicated attribute transfer; **print both in every output.**
- **LP integrality is empirical, not proven.** TU holds per B-edge; the GUB rows coupling *across* B-edges break the guarantee in theory. Observed 0 fractional binaries in 9/9 runs. If a pathological tile fractures, HiGHS branches and it still solves — you lose the "LP == ILP" speed story, **not correctness**.
- **Tiling is approximate at the seams** (§9.3). Do not repeat the design's own error of certifying a halo with top-1 statistics.

### 7.3 Where the hardness is (state it; don't hide it)

**The global problem is NP-hard, and APX-hard, and this must appear in the paper.** Two independent reductions:

1. Restrict to `μ = 0`, `|K(a)| = 2`, every route claiming a single interval on a single B-edge. That is **exactly the weighted Job Interval Selection Problem** (jobs = A-edges, each with a set of intervals; pick ≤1 per job; selected intervals pairwise disjoint; maximise weight `τ·L_a − c(a,r) ≥ 0`). JISP is NP-hard and **MAX-SNP-hard with as few as 2 intervals per job** (Spieksma 1999; Chuzhoy–Ostrovsky–Rabani).
2. With a continuity penalty `y ≥ x_i + x_j − 1`, `g > 0`, the pairwise potential is **supermodular** (`θ(0,0)+θ(1,1) = g > 0 = θ(0,1)+θ(1,0)`), so binary MAP inference on the cyclic A-line-graph is NP-hard by reduction from **MAX-CUT**. *(A second reason not to have that term.)*

**The polynomial core:** with `|K(a)| = 1`, `μ = 0` and single-B-edge routes, each B-edge decouples and the problem is **maximum-weight independent set in an interval graph** = weighted interval scheduling = a shortest path on a path-DAG, `O(m log m)`.

**So: NP-hard in theory, integral in practice — and the reason is structural, not lucky.** The packing block is TU; all the integrality gap can come only from (i) job choice `|K| ≥ 2`, (ii) multi-B-edge routes coupling TU blocks through the GUB rows, (iii) supermodular terms (which we don't have). On real data the median A-edge has **exactly one plausible corridor** (mean 1.02), so most variables are fixed by their unary alone. **Measured LP–IP gap: 0.0067%; 6 fractional A-edges out of 3611.**

---

## 8. The undirected reduction

**Reduce by twinning, and note that the reduction is nearly free on real data.**

1. **Twin each undirected edge into two opposing arcs** on both sides. This is already the shape of the data: NVDB ships `original_edge_id` + `is_reverse`; OSM ships `is_reverse`. The DP's forward-only arc construction (`graph_dtw.py:268-271`) already assumes it — which is why no backward arcs are ever synthesised.
2. **Run the directed pipeline unchanged.**
3. **Twin consistency** — `route(ι_A(a)) = ι_B(reverse(route(a)))`, where `ι_B` swaps each B-edge for its twin and maps each interval `[f,t]` on `b` to `[len(b)−t, len(b)−f]` on `ι(b)`.

**And it enforces itself:**
- The **bearing pre-filter** (§4.6) passed **both** NVDB twins in **0 of 4042** plausible (A-edge, physical-B-road) pairs, and exactly one twin in all 4042. **The A-arc's traversal direction determines which B-twin is admissible.** That is the constraint, delivered by the candidate set, with no solver.
- Unaided, twin consistency **already holds for 1390 of 1693 (82.1%)** OSM twin pairs, with 1 outright contradiction and 18 partial overlaps.
- In the **pure undirected↔undirected** case you may go further and enforce it by **variable identification** — `x[a,r] ≡ x[ι(a), ι(r)]` — which *halves* the binary count and makes violation structurally impossible.

**But in the directed↔directed case (OSM↔NVDB, the real one) twin consistency must stay SOFT:**
- **333 of 1434 (23.2%)** NVDB physical roads are one-way-only while OSM carries both directions ⇒ **31 of 1693 twin pairs are provably unsatisfiable.** A hard equality (or identification) makes the model **infeasible**.
- Model it as a slack `ν` in the ILP, or simply as a post-solve check, and **emit every residual violation as a `oneway_mistagged` issue.** **The residual IS the product.**
- **Free repair while you're there:** **57 twin pairs have exactly one side matched** ⇒ propagate `ι(route)` to the unmatched twin as a candidate. ~10 lines, 57 recovered A-edges.

**Two hazards for the twin map, both real:**
- **A-side twin keying is not free.** 19 OSM `(osm_id, node-pair)` groups carry **four** directed edges — two geometrically distinct parallel arcs of the same way between the same two nodes (e.g. 28.8 m and 62.0 m), each with its own twin; 58 ordered node-pairs carry parallel arcs. Neither export has an explicit undirected id (NVDB has `original_edge_id`; OSM has only a boolean `is_reverse`). **Pair on `(osm_id, node-pair, length)`, or have duckOSM export an undirected id** — otherwise the U-turn fix and the twin layer silently mis-pair on those 19 groups.
- **B-side connectivity is purely geometric.** `sweden_edges.csv` has no `source`/`target`; B adjacency is a 0.75 m endpoint snap. Report any continuity number with a **snap-tolerance sensitivity band** (0.75 → 8.0 m moved "broken" pairs only 753 → 683, so it is *not* a snapping defect — but say so *with the band*).

**Structural theorem, worth stating in the paper (the σ-cut).** For *any* total order σ on A's nodes, `A⁺ = {(u,v) : σ(u) < σ(v)}` and `A⁻ = {(u,v) : σ(u) > σ(v)}` are **both acyclic** and cover A losslessly (verified: 1961 / 1987 arcs, both DAGs, 3948/3948 covered, 0 self-loops). And **the two twins of a street always land in opposite halves** (0 violations). This is the cleanest statement of *why* no DAG decomposition can work: **the obstruction is not cyclicity in general — it is that twin 2-cycles are constitutive of a road network.** Cite it as the negative result; do not build on it.

---

## 9. Complexity and the route to county scale

### 9.1 Measured (Sundbyberg: 3948 directed A-edges, 2535 directed B-edges, 20 cores)

| stage | cost |
|---|---|
| candidate join (`ST_DWithin` 30 m) | 0.3 s |
| **baseline `match_routes` (today)** | **7.0 s wall** (1.8 ms/A-edge) |
| Stage 2: entry/exit lattice, `k_max=8` | **28.1 s wall** (156 ms/A-edge **core**-time) |
| interval + atom build (6968 atoms) | ~1 s |
| DEE persistency probe | **0.3 s** |
| **Stage 3: ILP, 35 343 vars / 10 916 rows** | **0.4–2.2 s, proven optimal** |
| same, after the safe prune (19 703 vars) | **0.74 s** |

**The solver is free. The oracle is the bottleneck.**

### 9.2 Asymptotics

- Stage 2: `O(n_cand)` DP runs per A-edge, each `O(N·V + N·E·log V)` — i.e. today's per-A-edge cost **× (mean candidate count + 1) ≈ ×10**, embarrassingly parallel (the joblib fan-out already exists at `matcher.py:873-880`).
- Interval/atom build: sort + sweep per B-edge, `O(Σ_b m_b log m_b)` — **linear in claims**.
- ILP: `O(Σ_a |K(a)|)` columns, `O(n + #atoms)` rows. Worst-case exponential; empirically LP + trivial branch-and-bound.

### 9.3 Stockholm county (~500k directed A-edges ≈ 127× Sundbyberg)

| | |
|---|---|
| matching (k=1) | ~15 min on 20 cores |
| **Stage 2 lattice** | 500k × 0.156 s = 78 000 core-s ≈ **65 min on 20 cores** ← **the bottleneck.** Halve it by restricting entry masks to B-edges within `max_distance` of A's *first* point. |
| DEE probe | ~40 s |
| ILP | ~4.5 M binaries / 1.4 M rows if monolithic. **Tile.** |

**Tiling — and be honest about it.** Tile at ~10k A-edges with a **1 km halo**; solve each tile's ILP; keep only interior A-edges' decisions.

- **Why the halo works:** a candidate coupling *edge* has bounded geometric length (≈ `2 · max_distance + max A-edge length` ≈ 60–100 m), so a 1 km halo captures **every constraint incident to an interior A-edge**.
- **What it does NOT give you:** coupling *components* can chain across the boundary (on the candidate lattice, one component spans 91% of the municipality — §6.7). **So per-tile optima are not guaranteed to compose into the global optimum.** The claim is **"approximate at the seams, with a detectable and boundable seam set"** — *not* "exact".
- **Mitigation:** any component that touches the halo boundary is *detected* and escalated: re-solve on the union of the two tiles (or run a second, offset tiling). Report `seam_component_count` in the output. On Sundbyberg-like density, top-1-selected conflict components have bbox diagonal p50 111 m / max 548 m, so escalations are rare — but **do not certify the halo with top-1 statistics** (that is the error this spec is correcting).
- The objective is an additive **measure** (m·m), so per-tile objectives *do* compose arithmetically even when the argmin doesn't.

**Solver headroom, testable today:** replicate the Sundbyberg ILP ×127 as independent blocks — **1 481 074 variables / 1 084 199 constraints solved by HiGHS in 25.1 s.** *(Caveat, stated: independent blocks are trivially decomposable by HiGHS, so this bounds solver *throughput*, not the coupling structure of a real county. It is a sanity floor, not a proof.)*

**Memory:** `routes_alts` grows ×|K| (~0.7 M → ~5 M rows for the county). Keep it in a **separate frame** (D7 already forbids alternatives in `routes_long`) and drop it after the solve.

**Fix before county scale, regardless of this spec:** the joblib worker currently pickles the entire `LocalBGraph` and the full warping back from every process (`matcher.py:35`, `graph_dtw.py:1015`) only for `matcher.py:899-933` to drop them. At `k_max=8` that waste multiplies. **Return the assembled rows from the worker, not the graph.**

### 9.4 Escape hatch (documented, not built)

If a tile ever becomes too large for the ILP: dualise the GUB constraints. The B-side slave — "given prices, pick a max-profit set of pairwise-disjoint claims on B-edge `b`" — is **exactly weighted interval scheduling**, solved exactly in `O(m log m)` as a shortest path on a path-DAG. That is a solver-free, embarrassingly parallel, **anytime lower bound**. This is the only legitimate role for "flow" in this problem.

---

## 10. Reuse of `graph_dtw` / `dag_dtw`

### Reused **unchanged** — the exact local cost is preserved bit-for-bit

| what | where |
|---|---|
| `_node_projection_pool`, `build_local_digraph`, `LocalBGraph` | `graph_dtw.py:52-299` |
| both DPs (point `:617-691`, segment `:301-458`), incl. the **in-row Dijkstra** that makes cyclic B targets exact | `graph_dtw.py` |
| `matcher.generate_candidate_pairs`, the joblib fan-out (`matcher.py:873-880`), `from_wkt_csv` / CRS handling | `matcher.py` |
| `thresholds.suggest_thresholds` → supplies **τ** | `thresholds.py` |
| `bgraph_prep._collect_endpoints` + cKDTree → the pattern for a B-node table, if ever needed | `bgraph_prep.py:23-52` |
| `conflation_issues/conflation_issues.schema.yaml` → where the residuals land | — |
| `scipy.optimize.milp` (HiGHS) — **already in the env** | scipy ≥ 1.9 |

### `dag_dtw`'s role: **an oracle, not an engine**

**`dag_dtw` is NOT used to match a network.** It cannot be (§2.1–2.3): its source must be a DAG (`_validate` raises `NotADAG`), and V4 + `k_min=1` make it unsafe on data with a 34.8% scope difference.

But it earns its keep once:

> **`dag_dtw.parts_from_matching` ALREADY EMITS `b_from_m` / `b_to_m`** (plus `b_head_m` / `b_tail_m`, `drift_m`, `bearing_diff_deg`, `a_from_m` / `a_to_m`). **It is a working reference implementation and a test oracle for exactly the sub-edge interval logic Mode 1 lacks.** Milestone 1's interval columns are pinned against it.

**Mode 3 keeps its own scope and its own paper claim, unchanged: exact whole-DAG matching for DAG sources.** Nothing in this spec touches that proof. What this spec removes is the *motivation* to extend it to cyclic sources.

### Modified

| file | change |
|---|---|
| `graph_dtw.py` | `LocalBGraph.vert_pos_m`, `LocalBGraph.edge_twin`; drop `u → twin(u)` inter-edge arc; `graph_dtw_align(entry_edge=, exit_edges=)`; `_assemble_route(...)` extraction; `_stub_trim(...)`; `dp_cost` promoted to a returned field; `match_edge_to_bgraph(b_twin=, lattice=, k_max=)` |
| `matcher.py` | bearing gate in `generate_candidate_pairs` / `set_parameters`; k-loop in `_graph_dtw_group`; `compute_route_lattice()` → `routes_alts`; **worker returns assembled rows, not the graph** |
| **`schema.py` (NEW)** | column-name constants (D6) |
| **`reconcile.py` (NEW, ~500 LOC)** | claims / atoms / persistency / DEE / components / ILP / `resolve_network` / issue emission |
| **`issues.py` (NEW, per D4)** | `oneway_mistagged`, `route_conflict`, `low_confidence_match` |
| `docs/graph_dtw_matching.md` | **correct the false "U-turns are structurally impossible" claim** (`:60`, `:368`) |
| `docs/improvement_plan.md` | **rewrite D10; fix D11** (§14) |
| `requirements.txt` / `pyproject.toml` | `scipy>=1.9` as an optional `[solve]` extra |

---

## 11. Output schema

### `routes_long` — one row per selected route-member (D7: **no alternatives here, ever**)

Existing 16 columns unchanged, **plus**:

| column | type | meaning |
|---|---|---|
| `edge_b_from_m` | float | arc-length start of the traversed span on this **directed** B-edge (**D6**) |
| `edge_b_to_m` | float | arc-length end of that span (**D6**) |
| `stub_trimmed` | bool | this member survived the trim (always `True` in the frame; the column records that the trim ran) |

**Key: `(source_id, dest_id, seq)`.** `(source_id, dest_id)` is **not** unique (cyclic re-entry).

### `routes_summary` — one row per A-edge

Existing 13 columns unchanged, **plus** (written at resolve time, per D7):

| column | type | meaning |
|---|---|---|
| `dp_cost` | float | the DP objective (weighted sum) of the selected route. **Not** `dtw_distance`. |
| `assign_cost_m2` | float | `c(a,r)` — the drift integrated along A (§6.2) |
| `n_alternatives_considered` | int | `|K(a)|` after the safe prune |
| `runner_up_margin_m` | float | `c(a, r₂) − c(a, r₁)`, normalised per metre of A. The natural confidence feature. |
| `triage` | str | `OK` / `REVIEW` / `NO_MATCH_SCOPE` / `NO_MATCH_QUALITY` (**D1** — never encoded into `match_type`, **D2**) |
| `reconcile_action` | str | `KEEP` / `RESELECT` / `NULLED` / `RESCUED` |
| `reconcile_margin_m` | float | reduced cost of the runner-up (the LP dual certificate) |
| `conflict_len_m` | float | B-metres of this route's claims that sit in a paid-slack atom |
| `frozen_by_dee` | bool | this A-edge's label is **provably** fixed in every global optimum |

### `routes_alts` — NEW frame, one row per candidate route (dropped after the solve)

`source_id, rank, entry_edge, exit_edge, dp_cost, assign_cost_m2, dtw_distance, overlap_pct, bearing_diff, n_edges, dest_ids, claims_json, selected(bool)`

### `issues` — per `conflation_issues.schema.yaml`

`issue_type ∈ {oneway_mistagged, route_conflict, low_confidence_match, scope_gap}`, `source_id`, `dest_id`, `evidence` (the paid slack in metres / the unsatisfiable twin / the runner-up margin).

### Run manifest (every output)

`tau`, `mu`, `k_max`, `max_distance`, `max_bearing_diff_deg`, `snap_tolerance_m`, `step_meters`, `emission`, `alpha`, `beta`, `lp_gap`, `n_fractional_binaries`, `frozen_pct`, `seam_component_count`. **τ and μ are modelling choices; a result that does not carry them is not reproducible.**

---

## 12. What this does to the paper's exactness claims

**Mode 3's DAG-DTW proof stands, unchanged, for DAG sources.** It does **not** transport to a whole network, and the paper must not let it look as though it does — anyone who reads `_validate`'s `NotADAG` will check.

**The sentence the paper carries for the network setting is exactly this, and no stronger:**

> *Exact local cost per candidate route — an unmodified dynamic program, exact against a cyclic target graph — plus a provably optimal global **selection** over that candidate set, with a per-instance LP optimality certificate and a persistency proof covering 88–97% of edges. This is conditional on the true route lying in the candidate cover, an assumption we state and, absent labels, do not verify.*

**Never write "globally optimal."** A cyclic source has no order, so V1-monotonicity has nothing to range over, and a global warping of a road network **is not a well-defined object**. Say *that*, and say *why* — the σ-cut theorem (§8) and the twin-2-cycle barrier (§2.2) give the exact reason no decomposition rescues it.

This is **strictly stronger** than Hootenanny's ~10 iterations of score propagation (no objective, no bound, no certificate, and an edge-level conflict model that would reject 51.2% of correct N:1 tilings), and **strictly weaker** than Mode 3's DAG proof. Conflating the two would be the single worst thing this project could do to its own credibility.

**And be honest about the size of the win.** The global layer changes **4.65–7.4%** of decisions on Sundbyberg; **~90% of those are "delete an over-claiming junk match"**, which a drift-gated triage achieves more cheaply, and **genuine re-routes are 7–13 A-edges (0.2–0.36%)**. **The sellable claims are structural, not accuracy:**

1. a **measure-injective** output — no B-metre claimed twice (today **572 B-edges are over-claimed** and nothing adjudicates them), which is what makes downstream attribute transfer (`speed`, AADT) *well-defined*;
2. an **optimality certificate** per tile/component;
3. a **per-decision margin** with a dual certificate.

Hootenanny structurally cannot produce any of the three. That is a real and defensible win. **"A large accuracy jump" is not, and nothing in the data supports promising one.**

**Finally, do not let a near-empty conflict set be read as "the matcher is 99.4% right."** With no ground truth it means **"the matcher is 99.4% self-consistent"** — a much weaker claim. Of the residual conflicting A-edges, most have exactly **one** plausible corridor and may be **legitimate N:1 matches** (OSM splitting a road NVDB keeps whole; a service road alongside one centreline). **A solver that "resolves" them by forcing exclusivity would make the output WORSE.** Only the D3 label campaign settles it.

---

## 13. Milestone plan (one developer)

> **Milestone 1 alone beats the status quo.** Everything after M3 is *gated on a measurement*, not on enthusiasm.

### **M1 — Local truth (~8–10 days). Ships value with no global layer.**

1. **Regression pin** `routes_long` / `routes_summary` on Sundbyberg (column-**subset** equality, D7). *(0.5 d)*
2. **Extract `_assemble_route(pairs, gb, …)`** from `graph_dtw.py:693-921`. No behaviour change. *(4 d — this is the honest bulk; `improvement_plan.md:224` is right and "2 days" is not.)*
3. **Sub-edge intervals** (D6/Phase 2): `LocalBGraph.vert_pos_m` + `edge_b_from_m` / `edge_b_to_m`. **Pin against `dag_dtw.parts_from_matching`.** *(2 d)*
4. **Stub trim** in the same tail. *(1 d)*
5. **U-turn fix**: plumb `b_twin` into `build_local_digraph`, drop the `u → twin(u)` arc; **correct `docs/graph_dtw_matching.md:60,:368`.** *(1 d)*
6. **Expose `dp_cost`** as a first-class field. *(0.5 d)*

**DoD:** intervals reproduce `edge_matched_len` to <0.1 m; interior route members are exactly `[0, edge_len]`; the 18 twin hairpins are gone; the 572 over-claimed B-edges become a **measurable overlap in metres**; ~27% of route members (all boundary stubs, 0 interior) are gone; the pinned frames are unchanged on pre-existing columns.
**Why it beats the status quo alone:** attribute transfer (Phase 3) becomes *exactly* definable; the validation map gets true intervals; a real correctness bug is fixed; and ~126 of the 220 changes a full solver would make are already made, deterministically.

### **M2 — Candidate hygiene + triage (~3 days)**

Bearing gate in `generate_candidate_pairs` (mean 8.25 → ~1.0 candidates; halves DP cost; **makes twin consistency self-enforcing**). The `triage` column (D1) with a **scope** class — 34.8% of A-edges have no NVDB corridor at all (77.5% of `service`, 76.5% of `living_street`). Fix A-side twin keying: `(osm_id, node-pair, length)`, **not** `is_reverse`.
**DoD:** candidate count histogram matches; `triage=NO_MATCH_SCOPE` recovers the `service`/`living_street` population; any customer-facing coverage figure states the scope difference explicitly.

### **M3 — THE GATE (~3 days). Do not skip.**

Port `reconcile.dee_persistency` (~90 LOC, exact, 0.3 s) + the **2-sweep ICM baseline** + a plain-τ-gate baseline. Emit `frozen_pct`, active component sizes, max cyclomatic number.
Reproduce on Sundbyberg: frozen 88.3%, 167 components (158 trees), max 16.
**Then run it on a Stockholm-county tile WITH motorways, dual carriageways and parallel frontage roads.**

- **`frozen_pct` > 85% ⇒ STOP.** Write the NO-GO into `improvement_plan.md`, close deferred GR M3/M4 (12–16 days saved), ship the certificate, and go do attribute transfer and the BC pilot.
- **`frozen_pct` collapses ⇒** you have found the regime, on evidence. Fund M4–M6.

*This milestone answers `improvement_plan.md` Phase 8's own question with a number, with no labels, no ground truth, and no solver.*

### **M4 — The candidate lattice (~5 days, gated)**

`entry_edge` mask (3 lines at `graph_dtw.py:641-645`), exit partition of the final row (free), k-loop in `matcher._graph_dtw_group`, new `routes_alts` frame, `runner_up_margin_m` / `n_alternatives_considered` into `routes_summary`. Worker returns rows, not the graph.
**DoD:** rank-0 reproduces today's top-1 for **100%** of matched A-edges (verified: 3466/3466); mean |K| ≈ 6.8 at `k_max=8`; ≤30 s wall on 20 cores for Sundbyberg.

### **M5 — `reconcile.py`: the exact global solve (~5 days, gated)**

Atom arrangement over B; the two persistency prunes; ILP → HiGHS; `resolve_network(action="report")` **first**, `action="apply"` behind a flag.
**DoD:** proven optimal (`gap = 0`) for Sundbyberg in <5 s; fractional-binary count logged; output is interval-disjoint except at paid slacks (802 conflicts → 0); the changed decisions dumped as a **reviewable diff**; **beats the 2-sweep ICM and the τ-gate baselines, or says so.**

### **M6 — Twin layer + issues (~3 days)**

Soft twin term / post-solve check; twin propagation repair (57 pairs); paid slack → `route_conflict`; unsatisfiable twin (31 pairs) → `oneway_mistagged`; runner-up margin → `low_confidence_match`.
**DoD:** recovers the curated Sundbyberg issues. **Fix D11 first** — 6 of the 9 are **OSM-side**, not NVDB-side (§14).

### **M7 — County scale (~4 days)**

Tile at ~10k A-edges with a 1 km halo; escalate seam-touching components; report `seam_component_count`. Re-run **the coupling-graph and ambiguity measurement, not just timing** — this is a gate, not a formality. Sanity floor: the ×127 block replicate (25 s).
**DoD:** lattice ≈ 65 min on 20 cores; ILP ≈ 15 s total; mean |K| after prune and max component size reported for the county.

### **M8 — Honesty pass (~3 days, non-negotiable, gated on D3 labels)**

Measure **top-k recall** (is the true route in the lattice?). Measure the ILP against the τ-gate and the ICM. Write §12's exactness section verbatim. **If the ILP does not beat the gate, say so in the paper and ship the gate — the structural wins (measure-injectivity, certificate, margin) stand either way.**

### Optional probe (2 days, any time): chain-merged Mode-1 sources

`matcher.py:847-866` already builds tasks as `(id_a, coords_a, b_edges)` where `coords_a` is an **arbitrary polyline**, and graph-DTW already aligns any polyline against the whole cyclic B-graph. So a path of consecutive A-edges is just **one** `graph_dtw` task with concatenated geometry and the union of the members' candidates, with the route cut back to A-edge boundaries by arc-length. ~20 lines. No DAG, no V4, no feasibility cliff, NO_MATCH preserved.
It buys **longer source context** — the one lever that actually disambiguates, since the 2063 single-B-edge-route A-edges have median length **26.6 m** sitting inside B-edges of median length **111.4 m**. It buys **disambiguation, not consistency** (pass-through chains hold only 0.9% of junction pairs). **Nobody has measured it. Measure it before believing it.**

---

## 14. Corrections this spec forces on `docs/improvement_plan.md`

**D10 is REFUTED.** *"Conflict components are typically 2–6 A-edges; exhaustive + greedy suffices; the MILP rung is cut."* That is measured on the **top-1 selection**, which presumes the answer. On the real candidate lattice the largest coupling component is **1343** (unpruned) / **83** (safe-pruned) / **246** (with continuity). Exhaustive enumeration is dead. **The MILP rung is un-cut, and it is *less code and faster* than the exhaustive+greedy it was meant to avoid.** *(D10's recorded linearisation note is correct and should be kept: support bonuses need `y ≤ x_i, y ≤ x_j`; conflict/penalty terms need `y ≥ x_i + x_j − 1`. This spec uses only the latter.)*

**D11 is FACTUALLY WRONG.** It states *"All 9 curated `oneway_mistagged` issues are NVDB-side"* and sets the detector direction on that basis. The curated file has **6 OSM-side** (Mönstringsvägen, Arrendevägen_40, Milstensvägen, Lönnvägen_1983, Älvängsvägen, Ripvägen) and **3 NVDB-side** (Löfströms_Allé, Björkhagsvägen, Kyrkogårdsvägen). **A detector built to D11 fails its own acceptance test on two thirds of its cases.**

**Phase 8's gate is now pre-answered.** *"If conflicts are mostly selection errors, GR M3/M4 re-enter; if candidate errors, top-k + solver is wasted."* Measured three independent ways: **85.8%** of conflicting A-edges had ≤1 plausible corridor (no alternative existed to switch to); **77.8%** of conflict pairs are "one good edge vs one junk edge"; genuine re-routes are **7–13 of ~3500**. **These are CANDIDATE errors.** The fix is upstream (Phase 2 intervals + Phase 4 triage + the stub trim) — which is exactly M1–M2 of this plan. **The solver remains gated on M3's hard-tile measurement.**

**"Cyclic-exactness M2 — SCC survey (GO/NO-GO gate)" closes with a NO-GO** (§2.1), citable.

**Turn restrictions:** OSM `restriction` relations are in **neither** export. Turn-restriction legality **cannot be enforced or evaluated on this data at all**. Say that in the paper; do not imply support.

---

## 15. Risks

| # | risk | mitigation |
|---|---|---|
| **R1** | **Sundbyberg only.** One dense urban municipality; only 11.1% of A's length has >1 plausible corridor; no motorways, no dual carriageways, no long frontage roads. **Every number here could be a statement about one easy instance.** | **This is what M3 is for.** `frozen_pct` on a hard tile decides the whole track, for 3 days. |
| **R2** | **No ground truth. "Conflict" ≠ "error"; "optimal" ≠ "correct".** The ILP NULLs 800 and re-routes 255; flipped edges get *worse* local drift (median +0.10 m) to buy consistency. Not one of them is verifiably right. | `action="report"` before `action="apply"`. D3 labels gate M8. Claim structure, not accuracy. |
| **R3** | **Top-k recall is unmeasured.** The candidate cover is a restriction; the true route may not be in it. | Measure it against labels (M8). State it **first** in the paper, not last. |
| **R4** | **τ and μ are calibration.** NULLs swing 1067 → 567 over τ ∈ [6,20]. | τ from `thresholds.suggest_thresholds`; both printed in the run manifest. |
| **R5** | **Cheap baselines are embarrassingly strong.** 2-sweep ICM captures 98.5% of the exact gain; a drift gate catches 16/18 U-turns and most of the NULLs. | **Benchmark every solver against ICM and the τ-gate.** A solver that beats "do nothing" but not "do the dumbest thing" is not a contribution. |
| **R6** | **Build order is load-bearing.** If the solver ships before the stub trim, the solver gets credited with the trim's wins (126 of 220 changes). If attribute transfer ships before intervals, it transfers attributes across 27% spurious claims. | M1 first. Non-negotiable. |
| **R7** | **The 800 NULLs are a SCOPE difference, not a defect** (NVDB does not carry `service`/`living_street`). | Any coverage figure must state the scope difference explicitly or it will be read as a miss. |
| **R8** | **LP integrality is empirical.** | If a tile fractures, HiGHS branches. Failure mode is time, not correctness — and the LP bound tells you *before* you spend it. |
| **R9** | **A "resolver" that forces exclusivity on legitimate N:1 matches makes the output WORSE.** | The B-overlap constraint is a **priced slack**, never a hard exclusivity with an exemption list. Ship as a **flagger** before a resolver. |

---

## 16. One-paragraph summary for the paper's introduction

Conflating two whole road networks cannot be posed as a single warping: both networks are one giant strongly connected component (94.7% of OSM Sundbyberg's nodes), a cyclic source admits no order, and monotonicity has nothing to range over. Decomposition does not rescue it — SCC condensation contracts 96.2% of the network into a single super-node that has no geometry to warp onto, and no DAG decomposition of any kind can contain both directions of a two-way street, because a twin pair is a directed 2-cycle and 86.5% of arcs are half of one. We therefore split the problem: an **exact local oracle** (graph-DTW, which aligns one directed source edge against the whole local *cyclic* target graph, solving the within-row advance by Dijkstra) supplies a lattice of candidate routes with exact, unmodified DP costs; a **global selection layer** then picks one route (or none) per source edge by minimising the drift integrated along the source, plus a priced penalty on duplicated target measure, subject to sub-edge interval disjointness on the target. The result is measure-injective by construction, solved to proven optimality with an LP certificate in seconds per municipality, and — this is the finding we most want to report — **the exact local cost is precisely what makes global message-passing unnecessary**: a persistency filter freezes 88–97% of edges with a proof, and the entire global layer changes the corridor decision for 0.2% of the network. The methods that need iterative score propagation need it because their local score is ambiguous. Ours is not.