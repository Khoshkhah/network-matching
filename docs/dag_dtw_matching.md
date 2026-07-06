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
D[a][v] = E(a, v) +        Σ                min           D[a'][v']
                      a' ∈ Apred(a)   v' ∈ Bpred(v) ∪ {v}
```

Read it as **min-sum message passing** — two *different* combinations, on purpose:

- **`min` over `v' ∈ Bpred(v) ∪ {v}`** — a **choice** of where the predecessor `a'` sat in B:
  the same vertex `v` (A advanced, B stayed — the *vertical* move) or one arc back `v' ∈ Bpred(v)`
  (both advanced — the *diagonal*). Cheapest option wins, so `min`. (`∪ {v}` folds vertical and
  diagonal into one term.)
- **`Σ` over `a' ∈ Apred(a)`** — **coverage**: every A-edge flowing into `a` must be aligned, none
  discarded, so at a merge you **add** both approaches' costs. A `min` here would optimise one
  incoming branch and leave the other **unmatched** — wrong, because the goal is to align the
  *whole* DAG, not to find one best path through it. This is the crux: **sum over branches, min
  over B-positions.**

It reduces correctly at the two ends:

- **Chain** (`|Apred(a)| = 1`): the sum is a single term, so `Σ` and `min` coincide and the
  recurrence collapses to graph-DTW. The sum only ever *differs* from a min at a **merge**.
- **Source** (`Apred(a) = ∅`): the empty sum is `0`, leaving `D[a][v] = E(a, v)` — **free entry**
  at every A-source (any B-vertex), the DAG analog of graph-DTW's free row 0. No special case
  needed.

Other properties:

- **Sweep order.** Process A-vertices in **topological order**. When `a` is reached every `a' ∈
  Apred(a)` is already final, so each summand reads a finished value — no iteration to convergence,
  because `GA` is acyclic. (This is exactly why the source must be a DAG.)
- **B denser than A.** The inner term steps B by **one** arc (`Bpred(v)`). To let one A-arc ride a
  *run* of B-vertices (graph-DTW's *horizontal* move, when B is sampled finer than A), that
  one-step `min` generalises to a within-`a` shortest-B-**walk** — the Dijkstra of §3.1 — each
  intermediate B-vertex re-paying its `E(a, ·)`. Same idea, multi-step.
- **Termination — at the sinks.** A single path terminates at one point (`argmin_v D[N-1][v]`). A
  DAG has several **sinks**; because the sum already forced every branch to be covered, the result
  is read at **each sink `t`** as `argmin_v D[t][v]` (§3.2 assembles them into one consistent
  labelling).
- **Exactness.** Min-sum is **exact on a tree / forest** (a Y-split is an out-tree, a merge is an
  in-tree — the common junction cases). A **diamond** (split then re-merge) makes the sum
  double-count the shared ancestor above the split; reconvergent DAGs need a shared-prefix
  correction (future work — see §3.2).

### 3.1 The one-step term becomes a Dijkstra (B may cycle, B may be denser)

The summed inner term reads only **already-finalised** A-predecessors, so its one-step form is
computed directly. But two things make it a shortest-path within the A-state `a`, not a lookup:
`GB` may **cycle** (no topological order on the B side), and B may be **denser than A** (one A-arc
should be free to ride a run of B-vertices). Both are handled by fixing `a` and relaxing with
**Dijkstra** over `GB`'s non-negative arc weights, exactly as graph-DTW §3.1 — with the difference
that the seed is now the **summed** predecessor contribution, not a single row:

```
seed[v] = E(a, v) + Σ_{a'∈Apred(a)} D[a'][v]    # A-advance part of every incoming branch, at v
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

1. From each **sink** `t`, take `φ(t) = argmin_v D[t][v]`.
2. Trace predecessors as in graph-DTW (recording which move won), but **memoise `φ(a)` per
   A-vertex**: the first time a backtrack reaches A-vertex `a` it fixes `φ(a)`; any other branch
   that reaches `a` **reuses** that `φ(a)`. This is what makes the assignment **single-valued per
   A-vertex** — the junction-consistency guarantee — for free.
3. Each A-edge's matched B-route is read off its own stretch of the warping DAG, exactly as
   graph-DTW reads a route from a warping path (grouping consecutive steps by `vert_edge`).

**Tree vs. reconvergence.** When `GA` is a **tree / polytree** (branches never rejoin — the common
junction-neighbourhood case once edges are oriented by travel direction and a small neighbourhood
is taken), the per-A-vertex `φ` is globally optimal and the backtrack above is exact. When `GA`
**reconverges** (a *diamond*: split then merge), the merge vertex is reached by two branches that
must **agree** on `φ(merge)`; the forward DP already pins `φ(merge) = argmin_v D[merge][v]`, and
both branches back-trace from that shared label — an explicit agreement point. (Exact joint
optimisation of a reconvergent DAG has the usual shared-ancestor subtlety; the debug cases in §4
start as trees to validate the core, then add a diamond to exercise the merge rule.)

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
