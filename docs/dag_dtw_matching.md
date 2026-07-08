# DAG-DTW: Matching a Source DAG to a Directed Network

This document specifies **DAG-DTW**, the generalization of
[graph-DTW](graph_dtw_matching.md). Graph-DTW aligns **one A-edge** (a single directed path) to
the local directed graph of B-edges. DAG-DTW aligns a **whole source DAG** — a connected,
*acyclic*, topologically-ordered set of A-edges (a junction neighbourhood, a branching corridor) —
to the same directed B-network, in **one joint solve**, so that A-edges meeting at a junction map
to a **consistent** place in B.

Builds on: [`graph_dtw_matching.md`](graph_dtw_matching.md) (the single-edge DP) and
[`weighted_emission.md`](weighted_emission.md) (the `point` / `segment` local cost).

Implementation: [`network_matching/dag_dtw.py`](../network_matching/dag_dtw.py) (matcher),
[`network_matching/dag_conditioning.py`](../network_matching/dag_conditioning.py) (exact reference
solvers), demo [`notebooks/dag_dtw_playground.ipynb`](../notebooks/dag_dtw_playground.ipynb).

---

## Status

DAG-DTW is a **point-to-point v1** that is implemented and validated; a segment+bearing emission and
an α coverage weight are also shipped. The exact reconvergence solvers exist as reference/validation
code but are **not** wired into the matcher (§3.2b explains why). Maturity by component:

| Component | Section | Status | Where |
|-----------|---------|--------|-------|
| Point-to-point joint DP (topological sweep + per-vertex Dijkstra) | §3, §3.1 | **shipped** | `dag_dtw.py` |
| Forward–backward joint junction resolution | §3.2a | **shipped** | `dag_dtw.py` |
| Reachability-guarded backtrack + arc-length re-match | §3.2, §3.2c | **shipped** | `dag_dtw.py` |
| Horizontal emission weight `α` (1:N coverage cost) | §3.4 | **shipped** | `dag_dtw.py` |
| Segment-to-segment emission with bearing (the direction fix) | §3.5 | **shipped** | `dag_dtw.py` |
| `require_tree` guard (assert a forest / polytree) | §7 | **shipped** | `dag_dtw.py` |
| Exact conditioning — recursive vertex-cut / feedback-vertex-set | §3.2b | **reference & cross-validation only, not wired in** | `dag_conditioning.py` |
| Globally-optimal joint diamond labelling *inside* the matcher | §3.2b | **future work** (point-mode diamonds need §3.5, not conditioning) | — |

Everything carries over unchanged from graph-DTW: the projection-enriched candidate pools, the
forward-only B arcs, the `point`/`segment` emission, and the no-U-turn guarantee.

---

## 1. Why generalize from a path to a DAG

Today `match_routes` matches **each A-edge independently** (`matcher.py` groups candidates by
`id_a` and fans each out as its own task). Near a junction the several A-edges are aligned in
isolation, and nothing forces them to agree on **where that junction lands in B**. Two edges that
physically meet at one A-node can be assigned to two *different* B-vertices — an inconsistency that
only a joint solve can rule out.

DAG-DTW aligns the connected source subgraph at once. The single guarantee it buys:

> **Junction consistency.** Every A-vertex `a` is assigned exactly **one** B-vertex `φ(a)`, so all
> A-edges incident to `a` share the same B-location there. Splits and merges are matched coherently
> instead of edge-by-edge.

---

## 2. The source becomes a DAG

Ordinary DTW has a source that is a **total order**: points `a_0 … a_{N-1}` swept left to right,
where the only predecessor of `a_i` is `a_{i-1}`. Graph-DTW kept that linear source and generalized
only the **target** to a graph. DAG-DTW generalizes the **source**:

- The source A-edges are oriented (by travel direction) and **stitched head-to-tail** at shared
  endpoints — exactly as `build_local_digraph` stitches B — into a local directed graph `GA`.
- `GA` is required to be **acyclic** — a **DAG**. Acyclicity is what gives a **topological order**
  `a_0, a_1, …` in which every arc points from a lower to a higher index (**predecessors before
  successors**). This is the direct generalization of DTW's left-to-right sweep: the total order
  `0,1,2,…` becomes a *partial* order, laid out by a topological sort.
- Like B, **every A-vertex belongs to exactly one A-edge** (`vert_edge`); junction endpoints are
  kept as separate coincident vertices joined by inter-edge arcs, so a DP state always knows which
  A-edge it is on.
- `GA` has one or more **sources** (in-degree 0 — where a match may begin) and one or more
  **sinks** (out-degree 0 — where a match may end).

The topological order is laid out in three blocks — **`[all sources] [the middle] [all sinks]`**.
This is always achievable: sources have no incoming arcs, so they can always form a contiguous
**prefix**; sinks have no outgoing arcs, so they can always form a contiguous **suffix**; the two
never conflict and the rest fills the middle (an isolated vertex — both source and sink — goes in
the sources block). The three blocks map one-to-one onto the DP's three phases (§3):

```
topological order:  [ all sources ] [ ──── the middle ──── ] [ all sinks ]
DP phase:             free-entry        min-sum propagation      terminate
                      seed E(a,v)       (§3 recurrence)          read C_total (§3.3)
```

This is the exact DAG widening of graph-DTW's own axis, where **row 0** is the single source (free
entry) and **row N−1** the single sink (termination), with the middle rows propagating.

The two sides are deliberately **asymmetric**:

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

## The problem — a constrained optimization over valid matchings

Everything from §3 onward is an **algorithm**. This section states the **problem** those algorithms
solve, defined with no reference to any algorithm, to `φ`, or to an optimum: a constrained
optimization whose *feasible set* is the **valid matchings** and whose *objective* is total drift.
It is the algorithm-independent specification — the DP (§3), the forward–backward junction resolution
(§3.2a), and the conditioning solvers (§3.2b) are just methods that solve it.

**Objects.** Source DAG `GA = (V_A, →_A)` (§2, acyclic); target digraph `GB = (V_B, →_B)` (may
cycle). `Apred(a)` / `Asucc(a)` are the immediate in-/out-neighbours of an A-vertex `a`; `Bpred(v)` /
`Bsucc(v)` likewise for a B-vertex `v`. `Src_A` / `Snk_A` are the sources / sinks of `GA`. `E(a,v) ≥ 0`
is the emission (local cost) of pairing `a` with `v` (`point` or `segment`, §3.5).

**The matching.** A matching is a **relation** `M ⊆ V_A × V_B` — `(a,v) ∈ M` means "A-vertex `a` is
matched to B-vertex `v`" — with `M(a) = { v : (a,v) ∈ M }`. A relation (not a function) so it can be
**many-to-many**: a vertex may match a *run* of the other side, in either direction.

**The optimization problem.**

```
minimize     C(M) = Σ_(a,v)∈M  E(a, v)        # total drift; §3.3 defines the reported total, §3.4 the 1:N discount
over         M ⊆ V_A × V_B
subject to   M is a VALID warping   —   (V1)–(V4) below
```

### The feasible set — a *valid warping* (local, algorithm-independent)

`M` is valid iff it satisfies all four, using **only** immediate neighbours (`Apred/Asucc`,
`Bpred/Bsucc`) and membership. Two of them are mirror images, and we name them by **which neighbours
they inspect** — deliberately avoiding "forward/backward," which flips meaning depending on whether
you mean the direction you *look* or the direction flow *moves*: **(V2) is the *predecessor* rule**
("is each cell *fed*?") and **(V3) is the *successor* rule** ("does each cell *continue*?").

**(V1) No cross (monotone).**
```
∀ (a,v) ∈ M,  ∀ a⁻ ∈ Apred(a),  ∀ v⁺ ∈ Bsucc(v):    (a⁻, v⁺) ∉ M
```
If `a⁻ →_A a` and `v →_B v⁺` are arcs, you may not match `a` to the earlier `v` while its
DAG-predecessor `a⁻` sits on the later `v⁺` — that pair is an inversion.

**(V2) Predecessor rule — every cell is *fed*.**
```
∀ (a,v) ∈ M :
    [ ∃ v⁻ ∈ Bpred(v) : (a, v⁻) ∈ M ]                                            (i)  rode B inside a's run
  ∨ [ ∀ a⁻ ∈ Apred(a) : ( (a⁻, v) ∈ M ) ∨ ( ∃ v⁻ ∈ Bpred(v) : (a⁻, v⁻) ∈ M ) ]  (ii) every incoming arc feeds it
```
Either `(a,v)` is **interior** to `a`'s B-run — case (i), it has a matched B-predecessor at the same
`a` — or it is the run's **entry**, and then case (ii) forces **every** DAG-predecessor `a⁻` to feed
it (held at `v`, or advanced `v⁻→v`). The universal `∀ a⁻` is what makes a **merge** be fed by *all*
approaches and forbids an orphan entry (a run start reachable by nothing). A **source**
(`Apred(a)=∅`) satisfies (ii) **vacuously** — its entry is free, so no separate boundary exemption is
needed.

**(V3) Successor rule — every cell *continues*.** The exact mirror of (V2):
```
∀ (a,v) ∈ M :
    [ ∃ v⁺ ∈ Bsucc(v) : (a, v⁺) ∈ M ]
  ∨ [ ∀ a⁺ ∈ Asucc(a) : ( (a⁺, v) ∈ M ) ∨ ( ∃ v⁺ ∈ Bsucc(v) : (a⁺, v⁺) ∈ M ) ]
```
Either `(a,v)` continues inside `a`'s run, or it is the run's **exit** and **every** outgoing arc
carries it on — forcing a **branch** down *every* exit and forbidding an orphan exit; a **sink** is
vacuous. (V3) is **not** redundant on a DAG (unlike on a chain, where the predecessor rule (V2) alone
forces the successor rule): a run can continue in `B` while *also* crossing a source arc, producing incomparable cells
that (V1) cannot see — so the exit must be pinned explicitly.

> **(V2)+(V3) subsume "connected B-run, no hole."** The *rode-B* cases (i) are exactly within-run
> connectivity, so no separate contiguity rule is needed. The one residue is that `M(a)` must be a
> **simple** path — no internal *fork* of the run in a branching `GB` — worth asserting separately in
> that rare case.

**(V4) Boundary.** `M(a) ≠ ∅` for every `a` (the whole source is covered); entries lie at `Src_A`,
exits at `Snk_A` (pinned) — or, for free-ends, exactly one unconstrained entry/exit per source→sink
chain.

### Why this form

- **Purely local — reachability is never a primitive.** Every constraint reads only
  `Apred/Asucc/Bpred/Bsucc` and "is this pair in `M`?". Global monotonicity is recovered as the
  **transitive closure** of the local one-step moves (advance-A along a GA arc, advance-B along a GB
  arc, or both) — it is *derived*, not checked. This is exactly why the target `GB` may **cycle**: a
  loop is traversed one forward arc at a time and only an *immediate* reversal is a cross (V1); a
  reachability / `≤` formulation would be ill-defined on a cyclic `GB` (it is the same reason §3.1
  runs **Dijkstra** on B rather than a topological order).
- **Many-to-many vs. the shipped `φ`.** The problem is over a general relation `M`. The shipped
  matcher returns the **single-valued** restriction — a function `φ: V_A → V_B`, `|M(a)| = 1` — which
  is why `φ` gives one B-vertex per A-vertex and the 1:N coverage lives on the edge **route**, not on
  `φ` (§6.1). `check_sequence_rules` (`dag_dtw.py`) is precisely (V1) plus the continuity rules
  (V2)+(V3) checked on that single-valued `M`; a general `check_matching_rules(M, GA, GB)` would check
  all four on any relation.
- **Reduces to classic DTW.** If `GA` is a chain and `GB` a path, (V1) is DTW's monotonicity, (V2)
  the predecessor rule (its continuity), (V4) its boundary — the ordinary warping path; on a chain
  (V3), the successor rule, is *implied* by (V2) and is redundant. DAG-DTW is that rule **lifted from
  two total orders to (DAG source, digraph target)**: `pre`/`post` become neighbour *sets*, and the
  successor rule — free on a chain — becomes a **genuine extra constraint (V3)** because of branches,
  the same "cover every branch" the DP's `Σ over Apred(a) / Asucc(a)` enforces (§3).

---

## 3. The dynamic program

Let `a` range over the vertices of `GA` in **topological order**, and `v` over the vertices of
`GB`. Write `Apred(a)` for the in-neighbours of `a` in `GA` (its DAG-predecessors) and `Bpred(v)`
for the in-neighbours of `v` in `GB`. Let `E(a, v)` be the **local cost** (emission) of pairing
A-vertex `a` with B-vertex `v` — the same models as graph-DTW: `point` = `dist(a, v)`, `segment` =
the middle-to-middle segment distance `+ λ·Δbearing` (see [weighted_emission.md](weighted_emission.md)).

`D[a][v]` is the minimum cost to align the source **down to `a`**, ending at B-vertex `v`, with
**every A-edge above `a` covered**. That last clause — *cover everything above `a`*, not just find a
cheap path to it — is what separates DAG-DTW from a shortest path through the DAG, and it dictates
the combination operators:

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
  carries **no split factor and no sum** — a plain `min`. It lets one A-vertex ride a *run* of
  B-vertices (B sampled finer than A). It self-references `D[a][·]`, so it is resolved by the
  within-`a` **Dijkstra** of §3.1 (giving the multi-step B-run), each B-step re-paying `E(a,·)`.
- **(A) vertical + diagonal — A advances from its predecessors.** For each incoming A-edge, choose
  where the predecessor sat: `v' = v` (A advanced, B stayed — *vertical*) or `v' ∈ Bpred(v)` (both
  advanced — *diagonal*); `∪ {v}` folds the two into one `min`.

The (A) term carries the two DAG-specific pieces:

- **`Σ` over `a' ∈ Apred(a)` — coverage.** Every A-edge flowing into `a` must be aligned, none
  discarded, so at a merge you **add** both approaches' costs. A `min` here would optimise one
  incoming branch and leave the other **unmatched** — wrong, because the goal is to align the
  *whole* DAG, not to find one best path through it. *Sum over branches, min over B-positions.*
- **`1/outdeg(a')` — the split factor.** A vertex divides its accumulated cost **equally among its
  outgoing edges**, so the cost is *conserved* as it flows downstream. Without it, a shared prefix
  feeding several sinks would be counted once per sink; with it, each vertex's `E` contributes
  **exactly once** across all sinks (§3.3). At chain/merge vertices `outdeg = 1` (factor `1`,
  nothing changes); only splits divide.

So on a **chain** (`|Apred(a)| = 1`, `outdeg = 1`) this reads
`E(a,v) + min(D[a][v'] , D[a'][v] , D[a'][v'])` — graph-DTW's exact three moves
(horizontal / vertical / diagonal). The DAG only adds the `Σ` + split on the (A) branch.

**It reduces correctly at both ends:**

- **Chain** (`|Apred(a)| = 1`): the (A) sum is a single term, so it collapses to graph-DTW's
  vertical+diagonal, and the (H) term supplies the horizontal — full graph-DTW. The `Σ` only ever
  *differs* from a min at a **merge**.
- **Source** (`Apred(a) = ∅`): the (A) sum is empty (`0`), leaving `D[a][v] = E(a, v)` seeded, then
  the (H) Dijkstra spreads along B — **free entry** at every A-source, the DAG analog of graph-DTW's
  free row 0. No special case needed.

**Sweep order.** Process A-vertices in **topological order**. When `a` is reached every
`a' ∈ Apred(a)` is already final, so each (A) summand reads a finished value — no iteration to
convergence, because `GA` is acyclic. (This is exactly why the source must be a DAG.) The (H) term
is within `a` and is resolved by that A-vertex's own Dijkstra (§3.1).

**Termination.** Because the split factor conserves the cost-flow, the DP optimum is
`Σ over sinks t   min_v D[t][v]` and it equals `Σ over A-vertices E(a, φ(a))` — every edge counted
**once, for any tree-shaped DAG**. The realized total (the number actually reported) is defined in
§3.3; on a reconvergent diamond the two differ, and §3.3 is authoritative.

**Exactness — exact on trees, a strict *underestimate* on diamonds.** On a **tree** the recurrence
is exact: predecessors have **independent** subtrees, so the `min` *inside* the `Σ` equals the `min`
of the whole sum. On a **reconvergent** DAG (a diamond) it is a strict **underestimate**. The two
arms of the diamond each take their own `min` over the **shared ancestor**'s position — the `min`
sits *inside* the `Σ` — so the arms may pick **inconsistent** labels for that ancestor, and

```
D[t][v] = E + ( min_s arm_a(s) ) + ( min_s arm_b(s) )   ≤   E + min_s ( arm_a(s) + arm_b(s) )
          └──────────── the recurrence ────────────┘        └──────── the true cost ────────┘
```

by `min-inside-a-sum ≤ min-of-the-sum`. Worse, the wrong value **propagates**: anything downstream of
the diamond builds on it, so the whole table above a diamond is corrupted.

> **The `1/outdeg` split factor does *not* fix this.** It conserves the shared ancestor's *emission*
> (so a *consistent* labelling isn't double-counted) — a **cost-conservation** device. The underestimate
> here is a **labelling-consistency** failure: nothing stops the two arms from choosing *different*
> `s`. With `E(s,·)=0` the split factor changes nothing and `D[t]` is still below the true cost.

The true diamond cost needs the `min` taken **outside** the sum — `min over the shared ancestor of
(arm_a + arm_b)`, i.e. holding that ancestor fixed across both arms — which is **conditioning**
(§3.2b). **Consequence:** at a diamond, `D` (and hence `dp_cost`, §3.3) is **not** a valid cost to
backtrack; only the conditioned value is. On a tree there is no shared ancestor, nothing to condition,
and `D` is exact and trace-able. So the recurrence above is the correct forward table **only on a
tree** (or the tree parts of a DAG); every reconvergence must be conditioned (§3.2b) *in the forward
pass*, before anything is stacked on top of it.

### 3.0a Enforcing the (V3) successor rule in the forward pass

The recurrence above reads **predecessors** (`Σ over Apred`), so it coordinates **merges** — the
**(V2) predecessor rule**. On its own it does **not** coordinate **branches**: if each successor of a
branch `a` independently pulls its own best `v'` from `D[a][·]`, two successors can leave `a` at
*different* B-vertices — a **(V3) successor-rule** violation (a branch "left at two points"; see the
feasible-set section). Fixing this needs no cost trick — only a change in *how the pass expands*:
**process each vertex together with all its successors**, committing `a`'s single exit and expanding
**every** successor from it. Written as a successor-oriented sweep (reverse topological order,
successors first):

```
F[a][v] = E(a, v) + Σ over a_k ∈ Asucc(a)  min over w ∈ {v} ∪ Bsucc(v)  F[a_k][w]
                    └──────── every successor a_k expands from a's ONE position v ────────┘
```

Because every successor `a_k` is forced to start at `v` or one B-arc past it, all outgoing roads leave
the junction from the **same** point — **(V3) holds by construction**, with no post-hoc repair and
**independent of the cost split**. Backtrack from a source's `argmin_v F[source][v]`, pushing each
vertex's committed `v` to all its successors.

This is the exact **mirror** of the predecessor sum: `Σ over Apred` (the recurrence above) coordinates
merges (V2); `Σ over Asucc` here coordinates branches (V3). A single forward-processing sweep carries
only **one** of the two — this successor form is exact for a source **out-tree** (branches, no merges);
a neighbourhood with **both** junction types needs both directions (the backward pass `B`, §3.2a,
supplies the merge half). Reference implementation: `forward_successor_dp` in
[`dag_dtw.py`](../network_matching/dag_dtw.py), whose `M` is `check_matching_rules`-clean on (V3).

### 3.1 The one-step term becomes a Dijkstra (B may cycle, B may be denser)

The summed inner term reads only **already-finalised** A-predecessors, so its one-step form is
computed directly. But two things make it a shortest-path within the A-state `a`, not a lookup:
`GB` may **cycle** (no topological order on the B side), and B may be **denser than A** (one A-arc
should be free to ride a run of B-vertices). Both are handled by fixing `a` and relaxing with
**Dijkstra** over `GB`'s non-negative arc weights, exactly as graph-DTW §3.1 — the only difference is
that the seed is now the **summed** predecessor contribution, not a single row:

```
seed[v] = E(a, v) + Σ_{a'∈Apred(a)} D[a'][v]/outdeg(a')   # summed, split-scaled incoming branches
D[a][·] = seed[·];  push all (seed[v], v)
pop (c, u); for each GB arc u -> w:  cand = D[a][u] + E(a, w)     # let a run of B pay E(a,·)
                                     if cand < D[a][w]: D[a][w] = cand; push
```

(The diagonal `v'∈Bpred(v)` choices are subsumed: a one-arc B-walk from a predecessor's vertex is
just the first Dijkstra relaxation.) So the whole algorithm is **one topological sweep of A** and
**one Dijkstra per A-vertex** over B — the exact structure of graph-DTW, with graph-DTW's "row `i`"
generalised to "A-vertex `a` in topological order," and its single-row seed generalised to the
**sum over incoming A-branches**.

### 3.2 Junction consistency — from a warping *path* to a warping *DAG*

Graph-DTW backtracks a single warping **path**. DAG-DTW backtracks a warping **DAG**: one monotone
alignment per source→sink route of A, **sharing state at common A-vertices**.

**The monotone-forward rule.** The backtrack runs in **reverse topological order** (successors
before predecessors) and requires that *every* GA arc `a → a'` maps to a **forward** B-step
`φ(a) → φ(a')` — reachable along GB arcs, never backward, never to a disconnected vertex.

**The core difficulty: junction labels must be chosen *jointly*.** It is tempting to think that once
you have `φ(j)` at every junction the rest is easy — and the rest *is* easy (a chain between two
fixed B-points is a tiny fixed-endpoint DP, §3.2c). **The hard part is getting the `φ(j)` right**,
because the labels are coupled: two routes that pass through the same junction `j` may each, on their
own, prefer a *different* B-vertex there, but `j` is one point and can hold one label. Choosing each
junction's label in isolation (`argmin` per junction) does **not** guarantee that every chain
between two junctions still runs forward. Under a rigid shift — which drifts everything by roughly
the same amount and creates many **near-ties** — independent tie-breaks at neighbouring junctions go
inconsistent and the chain between them can no longer advance → a **backward step**. So junction
labelling is a **joint** discrete optimisation, not a bag of independent `argmin`s. §3.2a solves it
(exactly on trees); §3.2b solves the reconvergent case.

**Backtrack (the shipped approximation).**

1. From each **sink** `t`, take `φ(t) = argmin_v D[t][v]` (free choice at the end).
2. For every non-sink `a` (all its successors already fixed), pick the **cheapest** `φ(a) = v`
   **subject to `v` forward-reaching every successor's `φ`**. At a **split** this forces the junction
   to a *common B-ancestor* of its branches — so a junction can never spill onto a cross road, and
   coincident junction vertices are consistent by construction. (Without the constraint, resolving
   coincident junction vertices independently produces a **backward step** under perturbation — a
   real bug the sequence tests catch.)
3. Each A-edge's matched B-route is read off its stretch of the warping DAG (grouping consecutive
   steps by `vert_edge`), and a leading/trailing single-vertex **junction touch** on a neighbouring
   B-edge is trimmed, so the route lists only the edges the A-edge actually traverses.

Forcing the junction to the common ancestor can *raise the drift* (the spilled match was cheaper
pointwise), but it guarantees a **valid monotone sequence** — the rule matters more than the
pointwise minimum.

**Arc-length re-match (jump-free positions).** The DP + backtrack decide the *topology* — which
B-edges each A-edge maps to. A final pass then decides the *position*: each A-vertex is placed at its
**arc-length fraction** between an entry and an exit point on its route's B-polyline (snapped to the
nearest route vertex). Pure point-to-point picks the *nearest* B-vertex per A-vertex, which under a
large offset compresses A onto part of a B-edge and produces a **jump** at the junction (coincident
A-vertices land far apart in B — graph-reachable but discontinuous). Re-placing by arc length makes
the B-position advance *proportionally* to A, so the sequence is jump-free and drift becomes a
uniform offset rather than a low-but-discontinuous one. Two rules keep it faithful:

- **Free entry / exit.** The endpoint is pinned to the route boundary only at an interior
  **junction**; at a DAG **source** / **sink** it is FREE — it projects onto the route and may land
  in the *middle* of a B-edge (exactly graph-DTW's free entry/exit).
- **Boundary yield.** When consecutive A-edges over/undershoot the junction, one's route ends with
  the B-edge the other's starts with; that shared boundary edge is given to whichever A-edge covers
  it with **more** vertices, and the other yields it. Otherwise the junction-end pins to the *far*
  end of the shared edge — a backward step.

> **Known limit — topology, not position.** Under a large **lateral** shift a branch edge can
> physically overlap a neighbouring trunk B-edge, so the DP mis-assigns the *topology* (a branch
> grabs the trunk). The re-match cannot repair a wrong topology; the validation flags the resulting
> backward step rather than hiding it. This is the nearest-vs-corresponding limit, fixed by the
> direction term of §3.5, **not** by the junction machinery.

**Tree vs. reconvergence.** When `GA` is a **tree / polytree** (branches never rejoin — the common
junction-neighbourhood case), the per-A-vertex `φ` is globally optimal and the backtrack is exact.
When `GA` **reconverges** (a *diamond*: split then merge), the merge vertex is reached by two
branches that must **agree** on `φ(merge)`; §3.2a pins them to a common label, and §3.2b makes that
exact.

### 3.2a Joint junction resolution — the forward–backward pass

This is the **shipped** junction solver (it replaced the greedy per-sink backtrack). It is **exact
on tree-shaped source DAGs** and is what runs in `match_dag_to_bgraph`.

**The problem it fixes.** Two routes source₁→sink₁ and source₂→sink₂ pass through the **same
junction `j`**. Optimised *independently*, route₁ wants `j` at one B-vertex and route₂ at a
**different** one. A greedy per-sink backtrack finishes each sink on its own (`argmin` per sink) and
traces back, so the two traces **collide at `j`** and demand two different `φ(j)`; forcing one label
breaks the other route → a backward step. *You cannot assemble the whole matching from each sink's
own best pieces — the pieces disagree where they overlap.*

**The fix — score each junction once, for everyone through it.** Keep two cost tables:

- **Forward `D[a][v]`** (§3–§3.1): cheapest cost to align everything *upstream* of `a`
  (sources → `a`) with `a` at B-vertex `v`.
- **Backward `B[a][v]`**: reverse **both** graphs — flip `GA` (sinks become sources) and reverse
  `GB`'s arcs — and run the *same* DP. Cheapest cost to align everything *downstream* of `a`
  (`a` → sinks) with `a` at `v`. The forward pass sums over predecessors with the `1/outdeg` split;
  the backward pass sums over successors with the symmetric `1/indeg` split, so the conserved
  cost-flow is preserved in both directions.

Then pin every junction jointly:

```
φ(j) = argmin over v of   D[j][v] + B[j][v] − E(j, v)
```

The `− E(j, v)` removes the double count — the local cost of `j` at `v` sits in *both* tables. This
value is the best whole-DAG cost among matchings that keep `j` at `v`; its minimiser is the label
**all** routes through `j` agree on.

The shipped backtrack scores by `D[a][v] + B[a][v] − E(a,v)` in reverse-topological order **subject
to** the chosen `v` still reaching every successor's `φ` (the same reachability guard as §3.2). The
guard is what makes the joint choice beat a *pure* per-junction `argmin(D+B−E)`: the pure version
drops the constraint and regresses `merge`/`y_split` under a shift (a shifted source edge collapses
onto a nearest cross road). Over a 512-config perturbation sweep the backward step is **eliminated on
tree-shaped source DAGs** (`chain`, `merge` clean; `y_split` clean except one extreme case); only the
reconvergent **`diamond`** still fails — the documented caveat, handled by §3.2b/§3.5.

> **Do NOT read the total cost off `D+B` at a junction.** The tempting identity
> `C_total = D[j][φ(j)] + B[j][φ(j)] − E(j, φ(j))`, and even the conserved-flow reading
> `Σ_sinks min_v D[sink][v]`, are **exact only on a tree**. On a **reconvergent diamond** they are
> wrong: `D` and `B` each conserve the flow *on their own* (`1/outdeg` forward, `1/indeg` backward),
> but **combining them at one junction** mis-counts the split-then-remerge structure — the term gets
> weighted by *both* splits and they do not cancel. (Measured on a perturbed diamond: `D+B−E` at the
> split gave 112.8 while `Σ_sinks min D` gave 96.9; they agree exactly on every tree.) These are
> tree-only shortcuts / sanity checks. The realized sum (§3.3) is the definition of the reported
> total for any shape.

### 3.2b Reconvergent DAGs — exact conditioning (two reference solvers)

On a **tree**, §3.2a is exact and the story ends there. On a **reconvergent** DAG it is not, because
message passing (the `D+B−E` argmin) is exact only when the graph has no *undirected* cycle — and a
DAG has one exactly where two directed paths from a common ancestor **re-meet** at a common
descendant. The smallest is the diamond (`split j1 → {up, down} → merge j2`, an undirected cycle
`j1 — up — j2 — down — j1`); real networks can have several, or nested ones. On such a cycle the two
sides share **both** endpoints, so forward–backward lets each side pick its own label where they
re-meet, and they disagree there → a backward step. The reachability guard of §3.2a is not enough
because the loop constrains the two junctions **both** ways at once.

Two **exact** solvers handle this, both in
[`dag_conditioning.py`](../network_matching/dag_conditioning.py) as
`conditioned_labels(..., method="recursive" | "fvs")`. They are **not wired into
`match_dag_to_bgraph`** (see *Why not wired in* below); they exist as an exact reference for a
genuine loop and as **mutual validation** of each other.

**The shared exact base — the forest solver.** Conditioning is only exact if the *forest* base
solver is. `_forest_solve` labels a forest with **min-sum belief propagation** on the polytree:
unary potential = per-vertex drift + folded pinned-boundary reachability; pairwise potential = the
directed reachability constraint on each A-arc. BP is exact on any tree, so each connected component
is solved to its **global** optimum. (An earlier version used the §3.2a reachability-guarded
backtrack as the base; the cross-check caught that it is only a *heuristic* — on `double_diamond` the
two methods disagreed, 29.09 vs 29.33, because they condition on different vertices and a heuristic
base depends on that choice. With the exact BP base both agree exactly, `double_diamond → 28.93`,
below either heuristic value.) Both solvers minimise the **same** realized cost with this **same**
exact base, so on any DAG they return **equal-cost** labellings — that identity is the whole point of
keeping two.

Removing a set of vertices and pinning them to fixed B-labels turns the loop into a forest the base
solver can finish. The two methods differ only in *which* vertices they remove and *how many at
once*:

#### Method 1 — recursive minimum vertex cut (`method="recursive"`)

**What it does.** Labels **one small cut at a time, recursively.** A *minimum vertex cut* is a
**separator** — the smallest vertex set whose removal **disconnects** the graph into independent
pieces; in a local junction neighbourhood it is usually a **single** junction. Enumerate just that
one cut's nearby B-candidates, pin it, and recurse into the now-independent components:

```
solve(G):
    if G is a forest:                       # no undirected cycle
        return forest_solve(G)              # exact BP base case
    S = a minimum vertex cut of G           # smallest set that disconnects G — usually ONE junction
    best = ∞
    for each label combination s of S:      # ~12 B-candidates NEAR the cut vertex, not all of GB
        pin S = s
        cost = E(S = s) + Σ over components of (G − S)   solve(component)   # pieces are independent
        keep the cheapest s by REALIZED cost (§3.3)
    return the winning labels
```

**Cost.** `(#candidates) ^ w`, where `w` is the size of the **largest single cut** — usually 1.
Conditioning on a separator makes the sides genuinely independent, so each side is recursed
**separately** rather than enumerated together. For a chain of `k` diamonds this is `~k · #cand`
(**linear** in the number of loops).

**Still joint where it must be.** Within one level `S` is enumerated *together* (each `s` a full
assignment to the cut) and scored by the realized cost of the whole subtree below it; it never pins a
cut vertex to its own individually-best label and moves on (that re-creates the disagreement one
level up). The cut is small, so "enumerate it jointly" is cheap. For a single diamond, `S = {j1}`,
~12 tries, each leaving a tree — equivalently the 2-D `(φ(j1), φ(j2))` search
`D_up-to-j1[v1] + up(v1→v2) + down(v1→v2) + B_below-j2[v2]`.

#### Method 2 — one-shot feedback vertex set (`method="fvs"`)

**What it does.** A *feedback vertex set* `F` is the smallest vertex set whose removal makes `GA` a
**forest** (`F = ∅` ⇔ already a tree ⇔ §3.2a is exact). This method finds the minimum `F`, then
**enumerates every label combination for the whole set at once** — the Cartesian product over each
`f ∈ F`'s nearby B-candidates — pins `F` to each combination, solves the forest `GA − F` with the
same BP base, and keeps the cheapest:

```
solve(G):
    F = minimum feedback vertex set of G      # smallest set whose removal leaves a forest
    if F is empty: return forest_solve(G)
    best = ∞
    for each joint assignment f of ALL of F:  # product of ~12 candidates per f  →  12^|F| combos
        pin F = f
        cost = E(F = f) + forest_solve(G − F)
        keep the cheapest f by REALIZED cost (§3.3)
    return the winning labels
```

**Cost.** `(#candidates) ^ |F|` — **exponential** in the number of loops, because the whole set is
labelled jointly instead of one separator at a time.

**Role.** FVS *describes* the problem cleanly (are we a forest yet?) but is a **bad way to label** a
large `F`. It is kept as the simple, obviously-correct reference that the efficient recursive solver
must match. `double_diamond` (`|F| = 2`) is the smallest case that actually separates the two
methods, so the cross-validation exercises it specifically.

#### Why not wired into the matcher

Applied to the synthetic `diamond` under a shift, conditioning does **not** help and even scores
*worse* than the shipped heuristic — because **the diamond's failures are not loop failures.** Pin
the split `j1` to the *exact* B-split and solve the branches: in **point mode** both A-branches
*still* collapse onto the nearer B-edge, because that genuinely costs **less**, and the correct split
never appears at *any* cut label. That is the **nearest-vs-corresponding** limit (§3.2, §3.5), not
reconvergence; the exact solver only confirms the collapse *is* the true minimum-drift optimum in
point mode. The real cure is a **direction term** (segment/bearing, §3.5), not conditioning. So the
shipped matcher keeps the §3.2a reachability-guarded backtrack, and the conditioning solvers live
alongside as exact reference + cross-validation.

> **Scope & residual limit.** Local junction neighbourhoods have tiny cuts (size 0–1) and few
> candidates (`CAND_K = 12`), so conditioning is a handful of extra forest-solves — cheap and
> **exact** for a true loop inconsistency. (Finding a *minimum* cut/FVS is NP-hard in general, but
> the local graphs are small; every undirected cycle in a DAG passes through a branch/merge junction,
> so the search is restricted to those.) `scripts/dag_conditioning_validate.py` and
> `tests/test_dag_conditioning.py` check the two methods against each other on the scenarios, on
> `double_diamond`, across a 225-config sweep (225/225 agree), and against an independent brute-force
> optimum.

### 3.2c Extracting the matching — anchors, chains, and joint labels

Once the junction labels are fixed, the matching is read off in two parts. The split matters because
it locates the difficulty precisely: **the chains are trivial; the labels are the whole problem
(§3.2).**

**Anchors & chains (structure).** The **anchors** are the non-interior vertices — sources (in-deg 0),
sinks (out-deg 0), branches (out-deg > 1), merges (in-deg > 1). They cut `GA` into **chains**:
maximal runs of degree-(1-in, 1-out) vertices between two anchors.

**Chain fill-in (the easy part).** Given an anchor `a₀` pinned to `v₀` and `a₁` to `v₁`, the chain
between them is a linear A-path; align it to the best monotone B-walk **from `v₀` to `v₁`** with a
tiny fixed-endpoint DP (seed at `v₀`, force the last vertex to `v₁`, backtrack). Optimal, monotone,
and jump-free **by construction** — *once the endpoints are right*.

**The hard part — jointly-consistent anchor labels.** The problem is not "what is the best label for
junction `j`?" but: *choose B-labels for all anchors at once so that (a) each is cheap and (b) every
chain between two anchors can still run forward (`φ(a₀)` reaches `φ(a₁)` along GB arcs).* This is met
two ways:

- **Cheap (shipped, exact on trees):** the reverse-topological backtrack of §3.2a, scoring by
  `D+B−E` **subject to the reachability constraint**. That constraint is precisely a cheap
  enforcement of joint consistency — it discards the inconsistent tie-breaks — which is why it beats
  the "pure" per-junction `argmin` and why trees come out clean.
- **Exact (for loops):** on a diamond the reachability guard is not enough, because split and merge
  constrain each other *both* ways around the loop. §3.2b's cutset conditioning fixes one junction,
  turning the loop into a tree where the guard *is* enough, then tries its candidates and keeps the
  best.

**Assemble.** `φ` = the jointly-consistent anchor labels ∪ the chain fill-ins. Read each A-edge's
route off its chain's B-walk; total cost = sum of chain costs + each anchor's `E` once (§3.3).

> The current shipped code carries the §3.2 arc-length re-match rather than the exact fixed-endpoint
> chain DP. These are **not cost-interchangeable**: the chain DP recovers the *charged* 1:N coverage
> and yields `dp_cost`, the geometric re-match does not. §3.2d is the integrated `φ → M` procedure
> that closes this gap.

### 3.2d From labels to the matching — how `M` is computed exactly

The DP gives a *label per A-vertex*, `φ` (cost `C_total`), but the quantity it **minimizes** is the
cost of a *warping relation* `M` **with its 1:N coverage** (cost `dp_cost`). `φ` and `M` are
different objects, and turning one into the other is the whole algorithm. This section is the exact
procedure. **Input:** the forward table `D`, backward table `B`, emission `E` (all already computed,
§3–§3.1); `GA` is a tree.

| | what it is | cost |
|---|---|---|
| **`φ` — labels** | one B-vertex per A-vertex | `Σ_a E(a, φ(a))` = `C_total` (a *subsample* drift) |
| **`M` — the warping** | the relation, **incl. the 1:N coverage** | `Σ_(a,v)∈M E(a, v)` = `dp_cost` |

**Notation.** `Bpred(v)` / `Bsucc(v)` = immediate in-/out-neighbours of `v` in `GB`. An **anchor** is
a source, sink, branch (out-deg > 1) or merge (in-deg > 1); anchors cut `GA` into **chains** =
maximal runs of degree-(1-in,1-out) A-vertices between two anchors. `M(a) = { v : (a,v) ∈ M }` is
`a`'s **run** (the B-vertices it rides).

#### Stage 1 — pin the anchors (commit ONE optimum)

Assign a B-label `φ(a)` to every anchor, in **reverse-topological order** (sinks first), threading a
*single* optimum — never an independent per-vertex `argmin` (that mixes optima and crosses, §3.2a):

```
for anchor a in reverse topological order:              # its successor anchors are already committed
    for v in argsort_v ( D[a][v] + B[a][v] − E(a,v) ):  # cheapest whole-tree cost with a pinned at v, first
        if for every chain a … a' to an already-committed anchor a':   v forward-reaches φ(a') in GB:
            φ(a) = v ;  break                            # first feasible = cheapest consistent label
```

`D[a][v]+B[a][v]−E(a,v)` is the **min-marginal** (best whole-tree cost given `a` at `v`); the
reachability filter keeps the picks inside one optimum. Only anchors are pinned here — the chain
interiors are decided next.

#### Stage 2 — fill each chain by a pinned-end DP (this IS the coverage)

For a chain `a₀ → a₁ → … → a_k` whose ends are anchors already pinned to `v₀ = φ(a₀)` and
`v_k = φ(a_k)`, run a small DTW pinned at **both** ends and **backtrack it**:

```
# forward pass — d[i][v] = min cost to align a₀..a_i with a_i's run EXITING at B-vertex v
d[0][v] = E(a₀, v₀)  if v == v₀  else +∞                          # pin the start at v₀
for i = 1 … k:
    d[i][v] = E(a_i, v) + min(
        min over v'∈Bpred(v)      d[i][v'] ,                     # (H) a_i rides B  (v'→v, same A-vertex; Dijkstra §3.1)
        min over v'∈Bpred(v)∪{v}  d[i-1][v'] )                    # (A) advance from a_{i-1} (v'=v vertical, v'→v diagonal)

# backtrack from the PINNED end (a_k, v_k) — every step emits one matched pair
cur = (a_k, v_k)
while cur ≠ (a₀, v₀):
    (a_i, v) = cur ;  add (a_i, v) to M
    if the (H) term won at d[i][v]:  cur = (a_i, v')             # v' the Bpred used — a_i ALSO rides v'
    else (the A term won):           cur = (a_{i-1}, v')         # step back to the previous A-vertex
add (a₀, v₀) to M
```

The backtrack emits, for each A-vertex, **every** B-vertex it rides — that set is `M(a_i)`, the 1:N
coverage — and each emitted pair `(a_i, v)` is charged `E(a_i, v)`. (With `α < 1`, weight the *(H)*
steps' emission by `α`, §3.4.)

#### Assemble and score

```
M = ⋃ over all chains (their backtracked pairs)      # anchors are shared between adjacent chains → appear once
cost(M) = Σ_(a,v)∈M  E(a, v)  =  dp_cost
```

**Worked micro-chain.** `a₀ → a₁ → a₂`, anchors `a₀@v₀` and `a₂@v₃`, over a *denser* B-path
`v₀ → v₁ → v₂ → v₃`. The pinned-end DP may put `a₁` on the run `{v₁, v₂}` (a 1:N cover):

```
M    = { (a₀,v₀), (a₁,v₁), (a₁,v₂), (a₂,v₃) }
cost = E(a₀,v₀) + E(a₁,v₁) + E(a₁,v₂) + E(a₂,v₃)          # the TWO a₁ terms are the coverage cost
```

`φ` alone would record only `(a₁,v₁)` **or** `(a₁,v₂)` — one B-vertex — dropping the other coverage
term. That dropped term is exactly the `C_total` vs `dp_cost` difference.

**Why it is correct.** Every emitted step is an `H` (advance B, stay on `a`) or `A` (advance A) move,
so `M` obeys (V1)–(V4) with no repair. On a tree the anchors cut `GA` into **independent** chains, so
the sum of each chain's optimal pinned-end DP (anchors counted once) is the global optimum —
`cost(M) = dp_cost` exactly. The one non-obvious step: **Stage 2 recovers the coverage by
*backtracking a DP*, not by geometry.**

> **Shipped status.** The code does Stage 1 (guarded backtrack) but replaces **Stage 2 with the
> arc-length re-match** (§3.2) — geometry, not the pinned-end DP — and reports `C_total = Σ_a E(a, φ)`,
> the *subsample* drift, not `dp_cost`. Implementing the Stage-2 chain DP is what produces `M`
> explicitly, makes `cost(M) = dp_cost` provable, and lets `check_matching_rules(M)` (`dag_dtw.py`)
> certify (V1)–(V4) directly.

### 3.3 The objective — total map-match cost

The **total match cost of the whole DAG** is the sum of the local costs over every matched step —
each A-vertex paired with its **final** assigned B-vertex `φ(a)`:

```
C_total = Σ over A-vertices a   E(a, φ(a))          # the REALIZED cost of the returned matching
```

Compute it from the `φ` you actually return, **after** the joint junction resolution (§3.2a/§3.2c)
and any re-match — i.e. sum the per-A-vertex drifts. That is the honest number, consistent with the
reported `avg_drift = C_total / (matched steps)`, and it is correct for **any** DAG shape. It needs
no cut vertex and no `D+B` identity.

> **Do not report the raw DP optimum as the total.** The forward split factor conserves the
> cost-flow, so `Σ over sinks t   min_v D[t][v]` equals the cost of the DP's *discrete,
> unconstrained* optimum — every sink free to pick its own cheapest **sampled** B-vertex. This is
> **not a bound** on the realized cost in either direction: the continuous arc-length re-match can
> place a vertex *between* samples and come in **below** it (clean `y_split`: `Σ_sinks min D` = 11.6
> vs realized 11.5), while joint-consistency under a shift pushes the realized cost far **above** it
> (shift-4 `y_split`: 61.7 vs 108.3; diamond 96.9 vs 183.6). And `Σ_sinks min D` is itself exact only
> on a tree (§3.2a). So treat `Σ_sinks min D` as a *DP diagnostic* (`res["dp_cost"]`), never as the
> cost; the realized sum above is the definition.

The conserved-flow identity still makes a clean **worked check on a tree** — a Y-split (`a0→a1`,
`a1→a2`, `a1→a3`), every vertex drifting 0.2, `outdeg(a1)=2`, all labels consistent:

```
D[a0]=0.2   D[a1]=0.4   D[a2]=0.2+½·0.4=0.4   D[a3]=0.4
Σ sinks {a2,a3} = 0.8 = 0.2·4        (a naive sum without the split gives 1.2 — wrong)
```

— here it coincides with the realized total because a clean tree's DP optimum *is* consistent.

**Reported quality metric.** `C_total` is a *count-weighted sum* (more vertices ⇒ larger), so — like
graph-DTW's raw `D` value — it is **not** the number you report. The reported quality is the
**average drift** `C_total / (matched steps)` (meters, comparable to graph-DTW's `avg_distance` and
the `resolve_routes` thresholds), plus the **per-A-edge** breakdown and **coverage %**. Junction
consistency is a structural guarantee, not part of the cost.

### 3.4 Horizontal emission weight `α` — cheaper 1:N coverage, no laundering

The **(H) horizontal** move is "A stays at `a` while B advances" — how a *single* A-vertex covers a
*run* of B-vertices (a 1:N match). The plain recurrence pays `E(a, ·)` at **every** covered
B-vertex, so the cost of one A-point matching a B-stretch grows **linearly with how finely B is
sampled** — an arbitrary quantity. The goal is to *decrease the cost of one A-point matching many
B-points.*

> **Rejected first idea — discount the carried cost.** Multiplying the *carried* horizontal cost
> (`… + min(α·D[a][v'], (A))`) saturates the coverage cost but, for `α < 1`, makes the horizontal
> step's effective edge weight **negative** (a low-drift B-vertex can *lower* `D`), so the DP
> "launders" cost by wandering through cheap B-vertices. It re-weights the alignment by *recency* and
> shifted the `diamond` `avg_drift` 0.55 → 0.74. Rejected.

**The fix — discount the EMISSION, and only on a horizontal step.** Weight `E(a, v)` by `α`, but
apply the discount **only when the vertex is reached by extending coverage** (the min came from the
horizontal `D[a][v']`); a genuinely new match (reached by A-advance) pays full `E`:

```
D[a][v] = α · E(a, v) + min(
      (H)  min_{v' ∈ Bpred(v)}  D[a][v'],                                       # B-advances, A STAYS
      (A)  Σ_{a' ∈ Apred(a)}  (1/outdeg(a')) · min_{v' ∈ Bpred(v) ∪ {v}} D[a'][v']   # A-advances
)
      with  α = horizontal_weight (≤ 1)  if the min is the (H) term (came from D[a][v']),
            α = 1                          if the min is the (A) term (a new A-vertex's first match).
```

Unrolling a coverage `v₀ → … → v_k` (drift `δ`): `v₀` is entered by A-advance (full `δ`), the rest
by horizontal (`α·δ` each), so the coverage cost is `δ · (1 + α·k)`:

| B-run length k | `α = 1` | `α = 0.5` | `α = 0` |
|---|---|---|---|
| 1 (1:1) | δ | δ | δ |
| 6 | 7δ | 4δ | δ |
| 30 | 31δ | 16δ | δ |

So `α = 1` is today's per-B-point charge; `α < 1` discounts each *extra* covered B-vertex; `α → 0`
charges the coverage essentially **once** (sampling-independent). Crucially the discount is a
**non-negative local emission** (`α·E ≥ 0`), added per step — `D` never decreases along a coverage,
so there is **no laundering** and the routing (the `min(H, A)` decision on carried cost) is
unchanged; only the *emission charged* differs.

**Implementation.** With `α = 1` the recurrence is bit-for-bit today's (the Dijkstra horizontal is
untouched). For `α ≠ 1` the emission depends on which move wins, so the (H) pass is resolved in
**B-topological order** (compute `h = min D[a][v']` and `A`; the winner sets `α`, then
`D = α·E + min(h, A)`); a cyclic local B-graph falls back to bounded iterative relaxation.

**What it changes.** `α` lives inside the DP's *decision* cost (`D`, `B`), so it shifts **which
alignment** `φ` is chosen — always toward *more* 1:N coverage (verified: on `diamond`/`double_diamond`
under shift, `α = 0.3` extends routes like `A_up: [B_up] → [B_up, B_up2]`). The **reported**
`total_cost` / `avg_drift` stay the **raw** `Σ drift` of that chosen `φ` (still meters, still
comparable to graph-DTW's `avg_distance`) — `α` is not folded into the reported metric, only into the
routing decision. On a plain 1:1 corridor (no coverage choice) `α` therefore changes nothing.

**Trade-offs.** (1) Very small `α` makes extra coverage nearly free, so an A-vertex can
**over-cover** (grab more B than it should — seen on the shifted `diamond`); keep `α` comfortably
above 0 unless pay-once is truly wanted. (2) It is **orthogonal** to the junction-label and
nearest-vs-corresponding problems (§3.2, §3.5) — it only reshapes 1:N coverage. Default `1.0`
(bit-for-bit today's result); reach for `α < 1` only when 1:N cost scaling with B-sampling density is
the problem.

### 3.5 Segment-to-segment emission with a bearing term — the direction fix

Point mode's emission `E(a,v) = dist(a,v)` is blind to heading, so under a lateral shift a branch
collapses onto the *nearest* B-edge rather than the *corresponding* one (the diamond, §3.2, §3.2b).
The fix is a **direction term**, added exactly the way graph-DTW's `emission="segment"` does it, and
gated behind the same second mode (`emission="point" | "segment"`, default `"point"`).

**Per-vertex segment.** Each vertex owns the segment starting at it along its own edge: vertex `v` on
edge `e` pairs with `w`, its same-edge successor (`vert_edge[w] = e`). Define `mid(v) = ½(v + w)` and
the **compass bearing** `bear(v) = (deg·atan2(Δx, Δy) + 360) mod 360` (graph-DTW's convention
verbatim — `0° = north`, clockwise). The **last vertex** of an edge (no same-edge successor) falls
back to its incoming segment `u → v`; a degenerate 1-vertex edge falls back to `mid=v, bear=0`.

**The emission** (`λ = bearing_weight`, `circ(θ,φ) = min(|θ−φ|, 360−|θ−φ|) ∈ [0,180]`):

```
E_point(a, v)   = |a − v|                                              # today's, verbatim
E_segment(a, v) = |mid(a) − mid(v)|  +  λ · circ(bear(a), bear(v))     # middle-to-middle + heading
```

**What changes and what does not.** *Only the emission row* changes. The `(a,v)` **vertex DP state**
and **every** piece of junction machinery — forward `D`, backward `B`, the joint `D+B−E` reachability
backtrack, `α`, `require_tree`, the arc-length re-match — and **the entire output dict** (`phi`,
`routes`, `routes_detail`, …) are **untouched**: segment mode returns the *same output shape* as point
mode, only a better alignment. The emission is symmetric, so forward and backward share it (reversing
both graphs rotates every bearing by 180°, leaving `circ` unchanged). The reported `avg_drift` stays
the **raw point distance** of the chosen `φ` (decision-only, like `α`); `λ` and the middle-to-middle
basis transfer directly from graph-DTW, so its tuned `λ ≈ 1–5` applies.

**Guarantees & the diamond.** `emission="point"` (default) is **bit-for-bit** today's result.
`emission="segment", λ = 0` is deliberately **under-constrained** (middle-to-middle only, no heading)
and is *not* claimed equal to point mode — exactly as in graph-DTW. With `λ > 0` the diamond
resolves: `A_up` (heading up) prefers `B_up` over the nearer-but-downward `B_dn`, because a
wrong-heading match now costs `λ·(90–180)` "metres", dwarfing the few-metre distance saving. This is
the actual cure for the nearest-vs-corresponding limit — orthogonal to `α` (§3.4, coverage) and to
the junction machinery (§3.2a).

> **Measured effect & the rotation caveat.** Over the perturbation sweep, restricted to **lateral
> shift + noise** (`rot = 0`, the realistic regime), `segment@λ = 3` reduces the reconvergent
> failures without touching trees — `double_diamond` 14→8, `diamond` 9→7 of 32;
> `chain`/`y_split`/`merge` stay 0. The one honest caveat: the bearing term is, by design, sensitive
> to a **relative rotation between A and B**. Real conflation matches the *same* roads (A and B differ
> by a metre, not a heading), so this is harmless; but the synthetic sweep *rotates A while leaving B
> fixed*, which manufactures a global heading offset the term (correctly) penalises — so the `rot ≠ 0`
> rows are adversarial to *any* heading model and should not be read as a regression. Tune `λ` to the
> expected A↔B rotation noise (0 if it can be large).

---

## 4. The DAG test — how the algorithm is debugged

A **DAG test** is a small, hand-built source: **a set of directed edges in topological order, with
no loop** (an acyclic edge list, predecessors before successors). It plays the same role for
DAG-DTW that `tests/test_graph_dtw.py`'s hand-built B-edge lists play for graph-DTW — it exercises
one behaviour at a time, in a plain meter CRS, with no DuckDB or real data. Each test fixes an
expected `φ` at the junctions and an expected per-A-edge route, so the DP can be debugged exactly.

The starter ladder (each a topologically-ordered edge list, in `network_matching/dag_synthetic.py`):

| DAG test | shape | what it isolates |
|----------|-------|------------------|
| `chain` | `a0→a1→a2` (one path) | **must reproduce graph-DTW exactly** — a single-path DAG is the base case |
| `y_split` | `a0→a1`, then `a1→a2` **and** `a1→a3` | a branch: two exits share junction `a1`; `φ(a1)` is one B-vertex for both |
| `merge` | `a0→a2` **and** `a1→a2` | two approaches meet at `a2`; both must agree on `φ(a2)` |
| `diamond` | `a0→a1`, `a0→a2`, `a1→a3`, `a2→a3` | split **and** re-merge — exercises the reconvergence / agreement rule (§3.2) |
| `double_diamond` | two diamonds in series, joined by a chain (`\|F\| = 2`) | the case that separates the two conditioning solvers (§3.2b) |

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

- Topological sweep: `a0, a1, a2, a3`. At `a1` (the junction) the DP settles `φ(a1)` at the
  B-junction `B1` (jointly, via §3.2a).
- Backtracking from sink `a2` fixes the route `A_main → A_left ≈ B_main → B_left`, memoising
  `φ(a1)=B1`; backtracking from sink `a3` **reuses** `φ(a1)=B1` and yields
  `A_main → A_right ≈ B_main → B_right`.
- `A_main` is matched **once** (shared prefix), and both branches agree the split happens at `B1`.
  Edge-by-edge matching guarantees none of this.

The debug view (mirroring `scripts/graph_dtw_debug_viz.py`) draws `GA` as several coloured A-edges,
the joint correspondence, and `φ` at each junction.

---

## 6. Output

`match_dag_to_bgraph` returns a dict. Alongside the raw `phi` / `a_vertex_match` / `total_cost` /
`avg_drift` / `dp_cost` / `GA` / `GB` / `sources` / `sinks` (and, with `debug=True`, the `D` / `B`
cost tables), the accurate, downstream-ready face is **`routes_detail`** — one entry per A-edge.

### 6.1 `routes_detail` — per-A-edge route with score and partial-coverage geometry

The plain `routes` field gives only the *ordered B-edge ids* per A-edge. `routes_detail` adds the
**score** and — crucially — **exactly where the route begins and ends** on its boundary B-edges,
which are typically only *partially* covered:

```python
res["routes_detail"][a_edge_id] = {
    "route":        [b1, b2, ..., bk],     # ordered B-edges (== res["routes"][a_edge_id])
    "avg_drift":    float,                  # score: mean per-point drift (m) over this A-edge
    "max_drift":    float,                  #        worst per-point drift (m)
    "n_points":     int,                    # A-vertices on this edge
    "covered_len_m": float,                 # B-length actually traversed by this A-edge (m)
    "start": {"b_edge": b1, "t": 0.0–1.0, "xy": (x, y)},   # WHERE the route begins on b1
    "end":   {"b_edge": bk, "t": 0.0–1.0, "xy": (x, y)},   # WHERE the route ends on bk
    "edges": [                              # per B-edge in the route, in order
        {"b_edge": b, "t_from": 0.0–1.0, "t_to": 0.0–1.0, "cover_pct": float,
         "avg_drift": float, "xy_from": (x, y), "xy_to": (x, y)},
        ...
    ],
}
```

- **`t` is the fractional arc-length position along that B-edge** (`0` = the B-edge's start, `1` =
  its end), computed by `shapely` `project(..., normalized=True)` in the B-edge's **own travel
  direction**. So `start.t = 0.0` means the route begins exactly at `b1`'s start; `start.t = 0.30`
  means it begins **30 % of the way along `b1`** (a partial first edge). `end.t` likewise pins the
  partial last edge. This is the "number between zero and one" *and* the map location (`xy`).
- **Only the first and last B-edge can be partial.** Interior B-edges are fully covered
  (`t_from = 0.0`, `t_to = 1.0`, `cover_pct = 100`); a single-edge route (`k = 1`) carries the
  route on one edge from `start.t` to `end.t`.
- **`cover_pct`** per edge `= (t_to − t_from)·100` — how much of that B-edge this A-edge uses.
- **Directed & jump-free by construction:** the fractions come from the final arc-length-re-matched
  `φ` (§3.2), so `t` increases monotonically along the route.

Computed purely from the returned `φ` and the B-edge geometries — no extra DP. `res["routes"]`
stays as the terse id-only list for callers that don't need the detail.

---

## 7. Scope of this version

- **Source must be a true DAG** (acyclic). It is **derived from the A-edge table** by orienting and
  stitching shared endpoints; a **cyclic** local component (blocks, roundabouts — common in real
  road networks) is **detected and falls back to per-edge graph-DTW** (logged), so the pipeline
  never crashes and the DAG DP stays strictly acyclic.
- **Target B** is the same forward-only directed graph as graph-DTW; it may cycle (handled by the
  per-A-vertex Dijkstra).
- **Directed A → B** only (no symmetric B→A reconciliation).
- **Tree / polytree** source DAGs are handled **exactly** (§3.2a). **Reconvergent** DAGs (diamonds)
  use the same joint resolution, exact on trees; a globally-optimal joint diamond labelling *inside
  the matcher* is future work — the exact conditioning solvers of §3.2b exist as reference and
  confirm the point-mode diamond failure is nearest-vs-corresponding (fixed by §3.5), not a loop bug.
- **`require_tree` option** — pass `require_tree=True` to assert the source has **no undirected
  loop** (a forest / polytree — no reconvergence). The matcher verifies `GA`'s undirected skeleton
  is acyclic (cyclomatic number `E − V + C = 0`) and raises **`NotATree`** on any diamond. Because
  trees are solved **exactly** (§3.2a) and never hit the nearest-vs-corresponding diamond limit,
  this guarantees the exact regime for callers who know their source is a tree. It composes with both
  emission modes (`point`/`segment`) and with `α`. Default `False` (any DAG).
- Cost is **count-weighted** (inherited from graph-DTW): route choice depends on `step_meters`
  density; a length-weighted objective remains future work.

Junction consistency is enforced by the monotone reachability-guarded backtrack (§3.2), validated on
clean and rigidly-shifted DAGs by [`scripts/dag_dtw_validate.py`](../scripts/dag_dtw_validate.py) and
the sequence-rule tests.

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
| local cost | `point` / `segment` | `point` / `segment` (unchanged) |
| primitive | `match_edge_to_bgraph` | `match_dag_to_bgraph` |

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
