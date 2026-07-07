# DAG-DTW: Matching a Source DAG to a Directed Network

This document specifies **DAG-DTW**, the next generalization of
[graph-DTW](graph_dtw_matching.md). Graph-DTW aligns **one A-edge** (a single directed path) to
the local directed graph of B-edges. DAG-DTW aligns a **whole source DAG** — a connected,
*acyclic*, topologically-ordered set of A-edges (a junction neighbourhood, a branching corridor) —
to the same directed B-network, in **one joint solve**, so that A-edges meeting at a junction map
to a **consistent** place in B.

> **Status: design / algorithm spec.** This is the algorithm the implementation
> (`network_matching/dag_dtw.py`, forthcoming) will follow. It is written to be validated first on
> hand-built **DAG test cases** (see §4) — small topologically-ordered edge sets, no loops — used
> to debug the DP exactly as `tests/test_graph_dtw.py` debugs graph-DTW.

Builds on: [`graph_dtw_matching.md`](graph_dtw_matching.md) (the single-edge DP) and
[`weighted_emission.md`](weighted_emission.md) (the `point` / `segment` local cost).

---

## 1. Why generalize from a path to a DAG

Today `match_routes` matches **each A-edge independently** (`matcher.py` groups candidates by
`id_a` and fans each out as its own task). Near a junction the several A-edges are aligned in
isolation, and nothing forces them to agree on **where that junction lands in B**. Two edges that
physically meet at one A-node can be assigned to two *different* B-vertices — an inconsistency that
only a joint solve can rule out.

DAG-DTW aligns the connected source subgraph at once. The single win, stated precisely:

> **Junction consistency.** Every A-vertex `a` is assigned exactly **one** B-vertex `φ(a)`, so all
> A-edges incident to `a` share the same B-location there. Splits and merges are matched coherently
> instead of edge-by-edge.

Everything else — the projection-enriched pools, forward-only B arcs, the `point`/`segment`
emission, the no-U-turn guarantee — carries over unchanged from graph-DTW.

---

## 2. The source becomes a DAG

Ordinary DTW has a source that is a **total order**: points `a_0 … a_{N-1}` swept left to right,
where the only predecessor of `a_i` is `a_{i-1}`. Graph-DTW kept that linear source and generalized
only the **target** to a graph. DAG-DTW generalizes the **source**:

- The source A-edges are oriented (by travel direction) and **stitched head-to-tail** at shared
  endpoints — exactly as `build_local_digraph` stitches B — into a local directed graph `GA`.
- `GA` is required to be **acyclic**: a **DAG**. Acyclicity is what gives a **topological order**
  `a_0, a_1, …` in which every arc points from a lower to a higher index (**predecessors before
  successors**). This is the direct generalization of DTW's left-to-right sweep: the total order
  `0,1,2,…` becomes a *partial* order, laid out by a topological sort.
- Like B, **every A-vertex belongs to exactly one A-edge** (`vert_edge`); junction endpoints are
  kept as separate coincident vertices joined by inter-edge arcs, so a DP state always knows which
  A-edge it is on.
- `GA` has one or more **sources** (in-degree 0 — where a match may begin) and one or more
  **sinks** (out-degree 0 — where a match may end).

**The topological order is laid out in three blocks — `[all sources] [the middle] [all sinks]`.**
This is always achievable: sources have no incoming arcs, so they can always form a contiguous
**prefix**; sinks have no outgoing arcs, so they can always form a contiguous **suffix**; the two
never conflict and the rest fills the middle (an isolated vertex — both source and sink — goes in
the sources block). The three blocks map one-to-one onto the DP's three phases (§3):

```
topological order:  [ all sources ] [ ──── the middle ──── ] [ all sinks ]
DP phase:             free-entry        min-sum propagation      terminate
                      seed E(a,v)       (§3 recurrence)          read C_total (§3.3)
```

It is the exact DAG widening of graph-DTW's own axis, where **row 0** is the single source (free
entry) and **row N−1** the single sink (termination), with the middle rows propagating.

Note the deliberate **asymmetry** with the target:

| side | may contain cycles? | swept how |
|------|--------------------|-----------|
| **A** (source, `GA`) | **no — must be a DAG** | **topological order** (§3) |
| **B** (target, `GB`) | yes (loops, roundabouts) | **Dijkstra** within each A-state (§3.1) |

A is acyclic *by requirement*, so its axis is swept in one topological pass — no Dijkstra needed on
the A side. B may be cyclic, so the horizontal (B-advance) relaxation is still a Dijkstra, exactly
as in graph-DTW §3.1.

```
Source DAG GA (topological order a0 < a1 < … ; every arc goes forward):

            ┌────────► (branch: two exits share the junction)
   a0 ──► a1
            └────────►

Target GB (directed, may cycle):     B1 ──► B2 ──► B3
                                            └────► B5
```

---

## 3. The dynamic program

Let `a` range over the vertices of `GA` in **topological order**, and `v` over the vertices of
`GB`. Write `Apred(a)` for the in-neighbours of `a` in `GA` (its DAG-predecessors) and `Bpred(v)`
for the in-neighbours of `v` in `GB`. Let `E(a, v)` be the **local cost** (emission) of pairing
A-vertex `a` with B-vertex `v` — the same models as graph-DTW: `point` = `dist(a, v)`, `segment` =
the middle-to-middle segment distance `+ λ·Δbearing` (see [weighted_emission.md](weighted_emission.md)).

`D[a][v]` is the minimum cost to align the source **down to `a`**, ending at B-vertex `v`, with
**every A-edge above `a` covered**. This last clause is what separates DAG-DTW from a shortest path
through the DAG, and it dictates the combination operators:

```
D[a][v] = E(a, v) + min(

       min          D[a][v'],                                     # (H) B-advance, A STAYS at a
    v' ∈ Bpred(v)

         Σ        1/outdeg(a') ·      min           D[a'][v']     # (A) A-advance from predecessors
    a' ∈ Apred(a)                v' ∈ Bpred(v) ∪ {v}
)
```

These are DTW's three moves again, split into the two branches of the outer `min`:

- **(H) horizontal — B advances, A stays.** `min over v' ∈ Bpred(v) of D[a][v']`: still on the
  **same** A-vertex `a`, step one B-arc `v'→v`. Staying on `a` crosses **no** A-edge, so this term
  carries **no split factor and no sum** — it is a plain `min`. It is what lets one A-vertex ride a
  *run* of B-vertices (B sampled finer than A). It self-references `D[a][·]`, so it is resolved by
  the within-`a` **Dijkstra** of §3.1 (giving the multi-step B-run), each B-step re-paying `E(a,·)`.
- **(A) vertical + diagonal — A advances from its predecessors.** For each incoming A-edge, choose
  where the predecessor sat: `v' = v` (A advanced, B stayed — *vertical*) or `v' ∈ Bpred(v)` (both
  advanced — *diagonal*); `∪ {v}` folds the two into one `min`.

The (A) term carries the two DAG-specific pieces:

- **`Σ` over `a' ∈ Apred(a)`** — **coverage**: every A-edge flowing into `a` must be aligned, none
  discarded, so at a merge you **add** both approaches' costs. A `min` here would optimise one
  incoming branch and leave the other **unmatched** — wrong, because the goal is to align the
  *whole* DAG, not to find one best path through it. **Sum over branches, min over B-positions.**
- **`1/outdeg(a')` — the split factor.** A vertex divides its accumulated cost **equally among its
  outgoing edges**, so the cost is *conserved* as it flows downstream. Without it, a shared prefix
  feeding several sinks would be counted once per sink; with it, each vertex's `E` contributes
  **exactly once** across all sinks (§3.3). At chain/merge vertices `outdeg = 1` (factor `1`,
  nothing changes); only splits divide.

So on a **chain** (`|Apred(a)| = 1`, `outdeg = 1`) this reads
`E(a,v) + min(D[a][v'] , D[a'][v] , D[a'][v'])` — graph-DTW's exact three moves
(horizontal / vertical / diagonal). The DAG only adds the `Σ` + split on the (A) branch.

It reduces correctly at the two ends:

- **Chain** (`|Apred(a)| = 1`): the (A) sum is a single term, so it collapses to graph-DTW's
  vertical+diagonal, and the (H) term supplies the horizontal — full graph-DTW. The `Σ` only ever
  *differs* from a min at a **merge**.
- **Source** (`Apred(a) = ∅`): the (A) sum is empty (`0`), leaving `D[a][v] = E(a, v)` seeded, then
  the (H) Dijkstra spreads along B — **free entry** at every A-source, the DAG analog of graph-DTW's
  free row 0. No special case needed.

Other properties:

- **Sweep order.** Process A-vertices in **topological order**. When `a` is reached every `a' ∈
  Apred(a)` is already final, so each (A) summand reads a finished value — no iteration to
  convergence, because `GA` is acyclic. (This is exactly why the source must be a DAG.) The (H)
  term is within `a` and is resolved by that A-vertex's own Dijkstra (§3.1).
- **Termination — total cost at the sinks (§3.3).** Because the split factor conserves the
  cost-flow, the total map-match cost is the sum over **sinks** of `min_v D[t][v]`, and it equals
  `Σ over A-vertices E(a, φ(a))` — every edge counted **once, for any DAG shape** (diamonds
  included).
- **Exactness — cost vs. labelling.** The split factor makes the **cost total** exact on any DAG.
  What remains for **reconvergent** DAGs (diamonds) is the **labelling**: a shared junction must
  resolve to a single `φ`, which the forward mins do not by themselves couple — it is fixed at
  backtrack (§3.2), and globally-optimal joint labelling of a reconvergence is future work.

### 3.1 The one-step term becomes a Dijkstra (B may cycle, B may be denser)

The summed inner term reads only **already-finalised** A-predecessors, so its one-step form is
computed directly. But two things make it a shortest-path within the A-state `a`, not a lookup:
`GB` may **cycle** (no topological order on the B side), and B may be **denser than A** (one A-arc
should be free to ride a run of B-vertices). Both are handled by fixing `a` and relaxing with
**Dijkstra** over `GB`'s non-negative arc weights, exactly as graph-DTW §3.1 — with the difference
that the seed is now the **summed** predecessor contribution, not a single row:

```
seed[v] = E(a, v) + Σ_{a'∈Apred(a)} D[a'][v]/outdeg(a')   # summed, split-scaled incoming branches
D[a][·] = seed[·];  push all (seed[v], v)
pop (c, u); for each GB arc u -> w:  cand = D[a][u] + E(a, w)     # let a run of B pay E(a,·)
                                     if cand < D[a][w]: D[a][w] = cand; push
```

(The diagonal `v'∈Bpred(v)` choices are subsumed: a one-arc B-walk from a predecessor's vertex is
just the first Dijkstra relaxation.) So the whole algorithm is **one topological sweep of A**, and
**one Dijkstra per A-vertex** over B — the exact structure of graph-DTW, with graph-DTW's "row `i`"
generalised to "A-vertex `a` in topological order," and its single-row seed generalised to the
**sum over incoming A-branches**.

### 3.2 From a warping *path* to a warping *DAG* — junction consistency

Graph-DTW backtracks a single warping **path**. DAG-DTW backtracks a warping **DAG**: one monotone
alignment per source→sink route of A, **sharing state at common A-vertices**. Backtracking:

The backtrack runs in **reverse topological order** (successors before predecessors) and enforces
the **monotone-forward rule** — *every* GA arc `a → a'` must map to a **forward** B-step
`φ(a) → φ(a')` (reachable along GB arcs; never backward, never to a disconnected vertex):

1. From each **sink** `t`, take `φ(t) = argmin_v D[t][v]` (free choice at the end).
2. For every non-sink `a` (all its successors already fixed), pick the **cheapest** `φ(a) = v`
   (min `D[a][v]`) **subject to `v` forward-reaching every successor's `φ`**. At a **split** this
   forces the junction to a *common B-ancestor* of its branches — so a junction can never spill
   onto a cross road, and coincident junction vertices are consistent by construction. (Without
   the constraint, resolving coincident junction vertices independently produces a **backward
   step** under perturbation — a real bug the sequence tests catch.)
3. Each A-edge's matched B-route is read off its stretch of the warping DAG (grouping consecutive
   steps by `vert_edge`), and a leading/trailing single-vertex **junction touch** on a neighbouring
   B-edge is trimmed, so the route lists only the edges the A-edge actually traverses.

The trade-off is honest: forcing the junction to the common ancestor can *raise the drift* (the
spilled match was cheaper pointwise), but it guarantees a **valid monotone sequence** — the rule
matters more than the pointwise minimum.

**Arc-length re-match (jump-free positions).** The DP + backtrack above decide the *topology* —
which B-edges each A-edge maps to. A final pass then decides the *position*: each A-vertex is
placed at its **arc-length fraction along its route's B-polyline** (snapped to the nearest route
vertex). This is because pure point-to-point picks the *nearest* B-vertex per A-vertex, which
under a large offset compresses A onto part of a B-edge and produces a **jump** at the junction
(the coincident A-vertices land far apart in B — graph-reachable but discontinuous). Re-placing by
arc length makes the B-position advance *proportionally* to A, so the sequence is jump-free; drift
becomes a uniform offset rather than a low-but-discontinuous one. The **no-teleport** rule
(§ sequence tests) checks exactly this.

**Tree vs. reconvergence.** When `GA` is a **tree / polytree** (branches never rejoin — the common
junction-neighbourhood case once edges are oriented by travel direction and a small neighbourhood
is taken), the per-A-vertex `φ` is globally optimal and the backtrack above is exact. When `GA`
**reconverges** (a *diamond*: split then merge), the merge vertex is reached by two branches that
must **agree** on `φ(merge)`; the forward DP already pins `φ(merge) = argmin_v D[merge][v]`, and
both branches back-trace from that shared label — an explicit agreement point. (The split factor
already makes the *cost total* exact here (§3.3); what remains is only globally-optimal *labelling*
at the reconvergence — future work. The debug cases in §4 start as trees, then add a diamond to
exercise the merge rule.)

### 3.3 The objective — total map-match cost

The algorithm minimises the **total match cost of the whole DAG**: the sum of the local costs over
every matched step, i.e. every A-vertex paired with its assigned B-vertex `φ(a)`:

```
C_total = Σ over A-vertices a   E(a, φ(a))          # each A-edge's drift, summed, once
```

The **split factor `1/outdeg`** is exactly what lets you read this off the DP at the terminals:
because it conserves the cost-flow (from each vertex, `1/outdeg` of the value goes to each
successor — a uniform walk whose weight sums to 1 at the sinks, since `GA` is acyclic), each
vertex's `E` reaches the sinks with total weight 1. Hence

```
C_total  =  Σ over sinks t   min_v D[t][v]          # exact for ANY DAG shape (diamonds included)
```

with **no double-counting** of shared prefixes. Worked check — Y-split (`a0→a1`, `a1→a2`,
`a1→a3`), every vertex drifting 0.2, `outdeg(a1)=2`:

```
D[a0]=0.2   D[a1]=0.4   D[a2]=0.2+½·0.4=0.4   D[a3]=0.4
Σ sinks {a2,a3} = 0.8 = 0.2·4 = C_total        (a naive sum without the split gives 1.2 — wrong)
```

**Reported quality metric.** `C_total` is a *count-weighted sum* (more vertices ⇒ larger), so —
exactly like graph-DTW's raw `D` value — it is **not** the number you report. The reported quality
is the **average drift** `C_total / (number of matched steps)` (meters, comparable to graph-DTW's
`avg_distance` and the `resolve_routes` thresholds), plus the **per-A-edge** breakdown (each
A-edge's own avg/max/min drift and coverage — the `routes_long` slice) and **coverage %**.
Junction consistency is a structural guarantee, not part of the cost.

---

## 4. The DAG test — how the algorithm is debugged

A **DAG test** is a small, hand-built source: **a set of directed edges in topological order, with
no loop** (an acyclic edge list, predecessors before successors). It plays the same role for
DAG-DTW that `tests/test_graph_dtw.py`'s hand-built B-edge lists play for graph-DTW — it exercises
one behaviour at a time, in a plain meter CRS, with no DuckDB or real data. Each test fixes an
expected `φ` at the junctions and an expected per-A-edge route, so the DP can be debugged exactly.

The starter ladder (each a topologically-ordered edge list):

| DAG test | shape | what it isolates |
|----------|-------|------------------|
| `chain` | `a0→a1→a2` (one path) | **must reproduce graph-DTW exactly** — a single-path DAG is the base case |
| `y_split` | `a0→a1`, then `a1→a2` **and** `a1→a3` | a branch: two exits share junction `a1`; `φ(a1)` is one B-vertex for both |
| `merge` | `a0→a2` **and** `a1→a2` | two approaches meet at `a2`; both must agree on `φ(a2)` |
| `diamond` | `a0→a1`, `a0→a2`, `a1→a3`, `a2→a3` | split **and** re-merge — exercises the reconvergence / agreement rule (§3.2) |
| `stub_branch` | main path + a short dead-end spur off a junction | a spur that leaves a junction but has no B continuation |

Invariants every DAG test asserts:

- **Single-path equivalence.** On `chain`, DAG-DTW returns byte-for-byte the graph-DTW route.
- **Junction consistency.** Every junction A-vertex maps to exactly one B-vertex; the A-edges
  meeting there share it.
- **Topological monotonicity.** Along every source→sink path the alignment advances monotonically
  in both `GA`'s topological order and `GB`'s arcs (no backward step on either side).
- **Acyclicity guard.** A cyclic source edge list is rejected (`NotADAG`) rather than silently
  looping — the source **must** be a DAG.

---

## 5. Worked example (a Y-split)

One A-corridor that **splits** into two exits at a junction, over a B-network that splits the same
way:

```
Source DAG GA (topological order a0 < a1 < a2 < a3):

   a0 ─(A_main)─► a1 ─(A_left)──► a2
                    └─(A_right)─► a3

Target GB:

   B0 ─(B_main)─► B1 ─(B_left)──► B2
                    └─(B_right)─► B3
```

- Topological sweep: `a0, a1, a2, a3`. At `a1` (the junction) the DP settles
  `φ(a1) = argmin_v D[a1][v]` — the B-junction `B1`.
- Backtracking from sink `a2` fixes the route `A_main → A_left ≈ B_main → B_left`, memoising
  `φ(a1)=B1`; backtracking from sink `a3` **reuses** `φ(a1)=B1` and yields
  `A_main → A_right ≈ B_main → B_right`.
- `A_main` is matched **once** (shared prefix), and both branches agree that the split happens at
  `B1`. Edge-by-edge matching guarantees none of this.

The debug view (forthcoming, mirroring `scripts/graph_dtw_debug_viz.py`) draws `GA` as several
coloured A-edges, the joint correspondence, and `φ` at each junction.

---

## 6. Output — two faces of one solve

- **Debug / playground face** — the full **A-subgraph → B-subgraph correspondence**: the label map
  `φ` (A-vertex → B-vertex), each A-edge's B-route, and the DP internals (cost tables, the warping
  DAG, per-junction `φ`), for inspection and for the debug figure.
- **Real / pipeline face** — the existing `routes_long` / `routes_summary` schema, one B-route per
  A-edge as today, **plus** an `a_edge_id`-within-DAG column and a `junction_consistent` flag. A
  DAG match still "divides per A-edge," so downstream code is unchanged; the only difference is the
  routes were computed **jointly** and are guaranteed junction-consistent.

---

## 7. Scope of this version

- **Source must be a true DAG** (acyclic). It is **derived from the A-edge table** by orienting and
  stitching shared endpoints; a **cyclic** local component (blocks, roundabouts — common in real
  road networks) is **detected and falls back to per-edge graph-DTW** (logged), so the pipeline
  never crashes and the DAG DP stays strictly acyclic.
- **Target B** is the same forward-only directed graph as graph-DTW; it may cycle (handled by the
  per-A-vertex Dijkstra).
- **Directed A → B** only (no symmetric B→A reconciliation).
- **Tree / polytree** source DAGs are handled exactly; **reconvergent** DAGs (diamonds) use the
  merge-agreement rule of §3.2 (exact joint optimisation of reconvergence is future work).
- **Junction consistency is now enforced by a monotone backtrack (§3.2).** The reverse-topological
  backtrack forces every junction to a common B-ancestor of its branches, so the matched sequence
  is always a **valid monotone forward B-walk** (no backward or disconnected step, no spill onto a
  cross road) — validated on clean and rigidly-shifted DAGs by
  [`scripts/dag_dtw_validate.py`](../scripts/dag_dtw_validate.py) and the sequence-rule tests.
  Implemented in [`network_matching/dag_dtw.py`](../network_matching/dag_dtw.py); demo in
  [`notebooks/dag_dtw_playground.ipynb`](../notebooks/dag_dtw_playground.ipynb).
- Cost is **count-weighted** (inherited from graph-DTW): route choice depends on `step_meters`
  density; a length-weighted objective remains future work.

---

## 8. Relationship to the rest of the library

| Concept | graph-DTW (today) | DAG-DTW (this doc) |
|---------|-------------------|--------------------|
| source | one directed **path** (`coords_a`) | a directed **DAG** `GA` (topologically ordered A-edges) |
| source sweep | left → right (`i-1`) | **topological order** (`Apred(a)`) |
| predecessor combine | trivial (one predecessor) | **min-sum**: `Σ` over `Apred(a)` (cover all branches), `min` over B-positions |
| target | directed graph `GB` (Dijkstra) | directed graph `GB` (Dijkstra) — unchanged |
| result | one B-route per A-edge | junction-consistent B-route per A-edge + label map `φ` |
| entry / exit | free entry row 0 / exit last row | free entry at **sources** / exit at **sinks** |
| local cost | `point` / `segment` (unchanged) | `point` / `segment` (unchanged) |
| primitive | `match_edge_to_bgraph` | `match_dag_to_bgraph` (forthcoming) |

DAG-DTW is a strict superset: give it a single-path DAG and it **is** graph-DTW.

---

## 9. Implementation & optimization

In a real source DAG the overwhelming majority of vertices are **interior, in-degree 1 /
out-degree 1** — the projection + gap-fill points along an edge (§2). Only a handful are
**junctions** (sources, sinks, branches, merges). The algorithm is built to pay the DAG price only
at the junctions.

### 9.1 Interior chains *are* graph-DTW

On a degree-1 chain the §3 recurrence collapses:

- one predecessor ⇒ the `Σ` over `Apred(a)` is a **single term** (no summing);
- `outdeg(a) = 1` ⇒ the split factor is **`1`** (no dividing).

So an interior vertex runs **exactly graph-DTW's three linear moves** (horizontal / vertical /
diagonal). The DAG-specific machinery — the `Σ` and the `1/outdeg` split — fires **only** at the
`O(#junctions)` branch/merge vertices. The interior is neither the complex part nor an extra cost;
it is the code that already exists.

### 9.2 Exploit linearity, but keep **one joint DP** — never collapse a chain to a route

The interior being graph-DTW-shaped (§9.1) does **not** license running the existing
`match_edge_to_bgraph` on each macro-edge and stitching the routes. That standalone primitive takes
the `argmin` at its end and returns a **single route per edge** — discarding the cost-over-all-B
vector and choosing each edge's B-alignment **independently**. Stitching those is exactly the
per-edge matching DAG-DTW exists to replace: it loses the shared `φ` at junctions and the min-sum /
split joint optimisation. **If that were accurate there would be no reason to match a DAG at all** —
so it is not a valid optimisation, it is the bug.

The correct implementation is **one joint DP over the whole DAG** (§3). What §9.1 buys is only
*cheapness*, not decomposition: on an interior chain the per-vertex step is the plain linear
recurrence (no sum, no split), but the **full cost vector `D[a][·]` keeps flowing** and the `Σ` +
`1/outdeg` split at junctions stays live throughout. Concretely:

- propagate the cost **vector** through chains, and combine **vectors** at junctions (sum the
  incoming, split the outgoing) — only the final backtrack turns vectors into routes;
- you may reuse graph-DTW's **inner machinery** as shared helpers — the projection-pool builder and
  the per-A-vertex Dijkstra over `GB` — but **not** the route-returning primitive as a black box on
  sub-edges.

So chain-contraction is fine purely as an internal DP *organisation* (fewer bookkeeping points),
but it must carry the vector and keep the junction combination in the loop — it is the joint DP,
not independent stitching.

### 9.3 Band the B-candidates per A-vertex (the real FLOP cut)

The genuine cost is `|A-vertices| × Dijkstra over GB`. But an interior A-vertex only ever matches
B-vertices **near it** — a far B-vertex has a huge `E(a, v)` and can never win. So restrict each
A-vertex's states to a **band** of nearby B-vertices (Sakoe-Chiba-style corridor: k-nearest, or
`E` below a threshold). That turns `|A| × |B|` into `|A| × (band width)` — near-lossless, and it
is where the speedup actually comes from. Per chain, carry only the **local `GB`** (the B-edges
near that chain) and merge into the full local graph only at junctions.

### 9.4 Cheapest lever — interior node count

Two existing knobs trade interior nodes for accuracy: `step_meters` (gap-fill density) and
`min_pool_gap_m` (sliver removal). Coarser sampling = fewer interior vertices = faster. This is
tuning, not structure, but it is the quickest win if a DAG is very large.

**Recommended build order.** Build the **one joint DP** directly (point-to-point, §3) — do **not**
route each edge and stitch (§9.2). Add **banding** (§9.3) if profiling shows the interior Dijkstras
dominate, and use **sampling density** (§9.4) as a tuning fallback. The only decomposition that is
ever valid is the lossless vector-propagation of §9.2, which *is* the joint DP.
