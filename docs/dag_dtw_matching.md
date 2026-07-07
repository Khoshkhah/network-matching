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
placed at its **arc-length fraction** between an entry and an exit point on its route's B-polyline
(snapped to the nearest route vertex). This is because pure point-to-point picks the *nearest*
B-vertex per A-vertex, which under a large offset compresses A onto part of a B-edge and produces
a **jump** at the junction (coincident A-vertices land far apart in B — graph-reachable but
discontinuous). Re-placing by arc length makes the B-position advance *proportionally* to A, so
the sequence is jump-free; drift becomes a uniform offset rather than a low-but-discontinuous one.

Two rules make the re-match faithful:

- **Free entry / exit.** The endpoint is pinned to the route boundary only at an interior
  **junction**; at a DAG **source** / **sink** it is FREE — it projects onto the route and may
  land in the *middle* of a B-edge (exactly graph-DTW's free entry/exit, so the source doesn't
  have to match the start of a B-edge).
- **Boundary yield.** When consecutive A-edges over/undershoot the junction, one's route ends with
  the B-edge the other's starts with; that shared boundary edge is given to whichever A-edge
  covers it with **more** vertices, and the other yields it. Otherwise the junction-end would pin
  to the *far* end of the shared edge — a **backward step**.

The **no-teleport** / monotone rules (§ sequence tests) check both. *Known limit:* under a large
**lateral** shift a branch edge can physically overlap a neighbouring trunk B-edge, so the DP
mis-assigns the topology (e.g. a branch grabs the trunk); the re-match cannot repair a wrong
topology, and the validation flags the resulting backward step rather than hiding it.

**Tree vs. reconvergence.** When `GA` is a **tree / polytree** (branches never rejoin — the common
junction-neighbourhood case once edges are oriented by travel direction and a small neighbourhood
is taken), the per-A-vertex `φ` is globally optimal and the backtrack above is exact. When `GA`
**reconverges** (a *diamond*: split then merge), the merge vertex is reached by two branches that
must **agree** on `φ(merge)`; the forward DP already pins `φ(merge) = argmin_v D[merge][v]`, and
both branches back-trace from that shared label — an explicit agreement point. (The split factor
already makes the *cost total* exact here (§3.3); what remains is only globally-optimal *labelling*
at the reconvergence — future work. The debug cases in §4 start as trees, then add a diamond to
exercise the merge rule.)

### 3.2a Joint junction resolution — the forward–backward pass

> **Status: implemented** (replaces the greedy backtrack of §3.2). Over a wide perturbation sweep
> (512 configs) the backward step is **eliminated on tree-shaped source DAGs** — `chain`, `merge`
> clean, `y_split` clean except one extreme case; only the reconvergent **`diamond`** still fails
> (the documented caveat). The label is chosen by a reverse-topological backtrack that scores by
> `D[a][v] + B[a][v] − E(a,v)` **subject to** the chosen `v` still reaching every successor's `φ` —
> the joint score resolves the shared-junction disagreement, and the reachability constraint keeps
> a shifted source edge from collapsing onto a nearest cross road (a *pure* per-vertex `argmin(D+B)`
> drops the constraint and regresses that case).

**The real cause of the backward step.** Two routes source₁→sink₁ and source₂→sink₂ pass through
the **same junction `j`**. Optimised *independently*, route₁ wants `j` at one B-vertex and route₂
wants it at a **different** one — but `j` is a single point and can hold only one label. The greedy
backtrack finishes each sink on its own (`argmin` per sink) and traces back, so the two traces
**collide at `j`** and demand two different `φ(j)`. Forcing them to one label breaks the other
route → a **backward step**. *You cannot assemble the whole matching from each-sink's-own-best
pieces* — the pieces disagree where they overlap. (This is distinct from the topology mismatch note
above; it happens even when every A-edge maps to the right B-edge.)

**The fix — score each junction once, for everyone through it.** Keep two cost tables:

- **Forward `D[a][v]`** (already computed, §3–§3.1): cheapest cost to align everything *upstream*
  of `a` (sources → `a`) with `a` at B-vertex `v`.
- **Backward `B[a][v]`**: reverse **both** graphs — flip `GA` (sinks become sources) and reverse
  `GB`'s arcs — and run the *same* DP. This is the cheapest cost to align everything *downstream*
  of `a` (`a` → sinks) with `a` at `v`. The forward pass sums over predecessors with the
  `1/outdeg` split; the backward pass sums over successors with the symmetric `1/indeg` split, so
  the conserved cost-flow (§3.3) is preserved in both directions.

Then pin every junction jointly:

```
φ(j) = argmin over v of   D[j][v] + B[j][v] − E(j, v)
```

The `− E(j, v)` removes the double count — the local cost of `j` at `v` sits in *both* tables. This
value is the best whole-DAG cost among matchings that keep `j` at `v`; its minimiser is the label
**all** routes through `j` agree on.

**Assembling the whole matching.** Once every junction's `φ` is fixed **consistently** (the hard
part — see §3.2c), the junctions **cut the DAG into simple chains** between now-fixed points, and
each chain is an ordinary single-path alignment between two *known* B-vertices (a source/sink end
stays **free**). *If* the junction labels are jointly consistent, there are no conflicts left and
stitching the chains yields `φ` for all A-vertices — but that "if" is the whole difficulty: choosing
the labels so every chain still fits is a **joint** decision, not a per-junction `argmin` (§3.2c).
On a tree this is achievable (exactly, or cheaply via the reachability guard); a **diamond** is
where it needs the cutset trick of §3.2b.

**Whole-DAG cost — compute it from the final matching, not from `D+B`.** The robust total is the
**realized cost of the final consistent labels**:

```
C_total = Σ over A-vertices   E(a, φ(a))      # φ = the FINAL consistent labels; always correct
```

This needs no cut vertex and no `D+B` identity, and is right for **any** shape.

> **Do NOT read the total off `D+B` at a junction.** The tempting identity
> `C_total = D[j][φ(j)] + B[j][φ(j)] − E(j, φ(j))` — and even the conserved-flow reading
> `Σ_sinks min_v D[sink][v]` — are **exact only on a tree**. On a **reconvergent diamond** they are
> wrong: `D` and `B` each conserve the flow *on their own* (`1/outdeg` forward, `1/indeg` backward),
> but **combining them at one junction** mis-counts the split-then-remerge structure — it gets
> weighted by *both* splits and they don't cancel. Verified on a perturbed diamond: `D+B−E` at the
> split gave 112.8 while `Σ_sinks min D` gave 96.9 (they agree exactly on every tree). So these are
> tree-only shortcuts / sanity checks; the realized sum above is the definition. It is the same
> reconvergence flaw that breaks the *labelling* (§3.2b) — here it corrupts the *cost*.

### 3.2b Reconvergent DAGs — recursive minimum-vertex-cut conditioning

> **Status: implemented as two EXACT, cross-validating reference solvers** in
> `network_matching/dag_conditioning.py` (`conditioned_labels(..., method="recursive" | "fvs")`).
> They are **not wired into `match_dag_to_bgraph`** — see the finding below — but they exist as the
> exact joint solver for a genuine loop and, more importantly, as a **mutual validation**: the
> recursive minimum-vertex-cut and the one-shot minimum-FVS share an **exact min-sum BP forest
> solver** (below), so on *any* DAG they must return **equal-cost** labellings. `scripts/
> dag_conditioning_validate.py` + `tests/test_dag_conditioning.py` check this on the scenarios, on a
> `double_diamond` (`|F| = 2`, the case that actually separates the two methods), across a
> perturbation sweep (225/225 agree), and against an independent brute-force optimum.
>
> **Why not wired in — the finding.** Applied to the synthetic `diamond` under shift it does **not**
> help and even scores *worse* than the shipped heuristic, because **the `diamond`'s failures are NOT
> loop failures.** Pin the split `j1` to the *exact* B-split and solve the branches: point mode
> *still* collapses **both** A-branches onto the nearer B-edge, because that genuinely costs **less**,
> and the correct split never appears at *any* cut label. That is the **nearest-vs-corresponding**
> limit (§3.2c note), not reconvergence; the real fix needs a **direction term** (segment/bearing).
> The exact solver only confirms this — the collapse *is* the true minimum-drift optimum in point
> mode. So the shipped matcher keeps the §3.2a reachability-guarded backtrack; the conditioning
> solvers live alongside as exact reference + validation.
>
> **The exact forest solver (why the two methods provably agree).** Conditioning is only exact if the
> *forest* base solver is exact. The §3.2a reachability-guarded backtrack is a **heuristic** — and the
> cross-check caught it: on `double_diamond` the two methods *disagreed* (29.09 vs 29.33) because they
> condition on different vertices and the heuristic's output depends on that choice. Replacing it with
> **min-sum belief propagation** on the polytree (unary = drift, folded pinned boundaries; pairwise =
> the directed reachability constraint; exact on any tree) makes the forest solve a true global
> optimum, and then both decompositions agree exactly (`double_diamond` → 28.93 for both, *below*
> either heuristic value). This is the honest lesson of the whole §3.2 arc: **the labels are the hard
> part, and "conditioning" buys exactness only on top of an exact forest solver.**

**The general problem.** Choosing the junction labels is a joint discrete optimisation (§3.2c).
Forward–backward solves it **exactly on a tree** and only on a tree, because message passing is
exact only when the graph has no undirected cycle. A DAG has an undirected cycle exactly where two
directed paths from a common ancestor **re-meet** at a common descendant — a **reconvergence**. The
smallest one is the diamond (`split j1 → {up, down} → merge j2`, an undirected cycle
`j1 — up — j2 — down — j1`); real networks can have several, or nested ones. On any such cycle the
two sides share **both** endpoints, and forward–backward lets each side pick its own label where
they re-meet (the backward table `B[j1][v]` sums the branches as if each could choose its own `j2`),
so they disagree there → a backward step. The reachability guard of §3.2a is not enough because the
loop constrains the two junctions **both** ways at once.

**Two framings — FVS *describes* the problem, a recursive vertex-cut *solves* it.** A **feedback
vertex set** `F` (the smallest vertex set whose removal makes `GA` a forest) is the clean way to
*state* when we are done: `F = ∅` ⇔ a tree ⇔ the shipped solver is already exact. But it is a **bad
way to label**: labelling `F` means enumerating **every combination of labels for the whole set at
once** — `(#candidates)^|F|`, which blows up as the number of loops grows. *How do you label a big
`F`?* You don't want to.

The efficient method labels **one small cut at a time, recursively.** A **minimum vertex cut**
(separator) is small — usually a **single** junction — and removing it **splits `GA` into
independent pieces**. So enumerate just that one cut, and recurse into the now-independent pieces:

```
solve(G):
    if G is a forest:                      # no undirected cycle  (FVS empty)
        return forward–backward + reachability guard        # base case — the SHIPPED solver
    S = a minimum vertex cut of G          # smallest vertex set that disconnects it — usually 1 junction
    best = ∞
    for each label combination s of S:     # ~5 B-candidates NEAR each cut vertex, not all of GB
        pin S = s
        cost = E(S = s) + Σ over components of (G − S)   solve(component)   # pieces are independent
        keep the cheapest s by REALIZED cost (§3.3)
    return the winning labels
```

**Why recursion beats labelling `F` at once.** Each level labels only **one small cut**, never the
whole feedback set, and conditioning on a separator makes the sides **genuinely independent**, so
they are solved (and recursed) **separately**. Complexity is therefore `(#candidates)^w` where `w`
is the **largest single cut** (usually 1) — *not* `(#candidates)^|F|`. A chain of `k` diamonds:
FVS-at-once is `(#cand)^k` (exponential); the recursive cut splits at each middle junction and
solves each diamond alone → `~k · #cand` (**linear**). Same exactness, tiny per-step labelling.

**Still joint where it must be — never greedy.** Within one level, `S` is enumerated *together*
(each `s` a full assignment to the cut) and scored by the realized cost of the whole subtree below
it; we do **not** pin a cut vertex to its own individually-best label and move on (that re-creates
the disagreement one level up — a cut vertex's best label depends on the pieces it joins). The cut
is small, so "enumerate it jointly" is cheap. (A single diamond: `S = {j1}`, ~5 tries, each leaving a
tree; equivalently the 2-D `(φ(j1), φ(j2))` search
`D_up-to-j1[v1] + up(v1→v2) + down(v1→v2) + B_below-j2[v2]`.)

**Graceful degradation.** The base case *is* the shipped code: a tree hits `solve`'s first branch
and returns immediately. Only genuine loops trigger conditioning, and only over a handful of nearby
candidates for one cut vertex at a time.

**Scope & residual limit.** Local junction neighbourhoods have tiny cuts (size 0–1) and few
candidates, so this is a handful of extra forest-solves — cheap and **exact** for the loop
inconsistency. (Finding a *minimum* cut/FVS is NP-hard in general, but the local graphs are small; a
greedy separator — e.g. the lowest-`|F|` junction on a cycle — is fine.) The residual
**nearest-vs-corresponding** error under an *extreme* shift (a branch physically lying on the wrong
B-edge) is a separate point-mode limit that conditioning *reduces* (it tries alternative labels) but
does not remove without a direction term.

### 3.2c Extracting the matching — the hard part is *jointly-consistent junction labels*

> **The correction that matters.** It is tempting to think "once we have `φ(j)` at every junction,
> the rest is easy." **The chains are easy; getting the `φ(j)` right is the hard part.** A chain is
> a straight piece of A between two *fixed* B-points — a tiny, safe DP. But the junction labels must
> be chosen **jointly**, so that every chain between them still runs *forward*. Choosing each
> junction's label on its own (`argmin(D+B−E)` per junction) does **not** guarantee that, and under
> a shift it fails (below). So the real work is the labels, not the fill-in. This section states the
> problem that way; the shipped code approximates it, §3.2b solves the loop case.

**The two parts.**

- **Anchors & chains (structure).** The **anchors** are the non-interior vertices — sources
  (in-deg 0), sinks (out-deg 0), branches (out-deg > 1), merges (in-deg > 1). They cut `GA` into
  **chains**: maximal runs of degree-(1-in, 1-out) vertices between two anchors.
- **Chain fill-in (the easy part).** Given an anchor `a₀` pinned to `v₀` and `a₁` to `v₁`, the
  chain between them is a linear A-path; align it to the best monotone B-walk **from `v₀` to `v₁`**
  with a tiny fixed-endpoint DP (seed at `v₀`, force the last vertex to `v₁`, backtrack). Optimal,
  monotone, jump-free **by construction**. This is the step we both assumed was the whole problem —
  and it *is* trivial, *once the endpoints are right*.

**The hard part — the labels must be jointly consistent.** The problem is not "what is the best
label for junction `j`?" It is:

> Choose B-labels for **all** anchors **at once** so that (a) each is cheap and (b) **every chain
> between two anchors can still run forward** (`φ(a₀)` reaches `φ(a₁)` along GB arcs).

Per-junction `φ(j) = argmin_v (D[j][v] + B[j][v] − E)` optimises each junction **in isolation**. On
clean data the isolated optima happen to agree. But a rigid shift drifts *everything* by roughly the
same amount, creating many **near-ties** — two B-spots for a junction cost almost the same. Each
junction then breaks its tie independently, two neighbours break theirs *inconsistently*, and the
chain between them can no longer go forward → a **backward step**. (Verified: the pure per-junction
`argmin` regresses `merge`/`y_split` under perturbation where the joint choice does not.) So the
junction labels are a **joint** decision, not a bag of independent ones.

**How the problem is met — cheap vs exact.**

- **Cheap (shipped, clean on trees):** a reverse-topological backtrack that scores by
  `D[a][v]+B[a][v]−E` **subject to a reachability constraint** — the chosen `v` must still reach
  every successor's `φ`. That constraint is *precisely* a cheap enforcement of joint consistency: it
  throws away the inconsistent tie-breaks. It is why the messy-looking guard beats the "pure"
  per-junction `argmin`, and why trees come out clean. *(The current code also carries an arc-length
  re-match; once the labels are jointly consistent that step can be replaced by the exact
  fixed-endpoint chain DP above — a cleanup, not a correctness fix.)*
- **Exact (for loops):** on a **diamond** the reachability guard is not enough, because the split
  and merge constrain each other *both* ways around the loop. §3.2b's **cutset conditioning** is the
  exact joint solution: fix one junction, which turns the loop into a tree where the guard *is*
  enough, try its candidates, keep the best.

**Assemble.** `φ` = the jointly-consistent anchor labels ∪ the chain fill-ins. Read each A-edge's
route off its chain's B-walk; total cost = sum of chain costs + each anchor's `E` once (§3.3).

**Bottom line.** *The chains were never the problem. The junction labels are — and they must be
chosen together.* The reachability guard is the cheap joint solver (trees); cutset conditioning is
the exact one (loops).

### 3.3 The objective — total map-match cost

The **total match cost of the whole DAG** is the sum of the local costs over every matched step, i.e.
every A-vertex paired with its **final** assigned B-vertex `φ(a)`:

```
C_total = Σ over A-vertices a   E(a, φ(a))          # the REALIZED cost of the returned matching
```

Compute it from the `φ` you actually return, **after** the joint junction resolution (§3.2a/§3.2c)
and any re-match — i.e. sum the per-A-vertex drifts. That is the honest number, consistent with the
reported `avg_drift = C_total / (matched steps)`.

> **Do not report the raw DP optimum as the total.** The forward split factor conserves the
> cost-flow, so `Σ over sinks t   min_v D[t][v]` equals the cost of the DP's *discrete, unconstrained*
> optimum — every sink free to pick its own cheapest **sampled** B-vertex. This is **not a bound** on
> the realized cost, in either direction: the continuous arc-length re-match can place a vertex
> *between* samples and come in **below** it (clean `y_split`: `Σ_sinks min D` = 11.6 vs realized
> 11.5), while joint-consistency under a shift pushes the realized cost far **above** it (shift-4
> `y_split`: 61.7 vs 108.3; diamond 96.9 vs 183.6). And `Σ_sinks min D` is itself exact only on a
> tree — on a diamond the conserved-flow reading and the `D+B−E`-at-a-junction reading (§3.2a)
> disagree with each other too. So treat `Σ_sinks min D` as a *DP diagnostic* (`res["dp_cost"]`),
> never as the cost; the realized sum above is the definition.

The conserved-flow identity still makes a clean **worked check on a tree** — Y-split (`a0→a1`,
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
sampled** — an arbitrary quantity. We want to *decrease the cost of one A-point matching many
B-points.*

**Why not discount the carried cost.** The first idea — multiply the *carried* horizontal cost,
`… + min(α·D[a][v'], (A))` — saturates the coverage cost (nice) but, for `α < 1`, makes the
horizontal step's effective edge weight **negative** (a low-drift B-vertex can *lower* `D`), so the
DP "launders" cost by wandering through cheap B-vertices. Verified real: it re-weights the alignment
by *recency* and shifted `diamond` `avg_drift` 0.55 → 0.74. Rejected.

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
`D = α·E + min(h, A)`); a cyclic local B-graph falls back to bounded iterative relaxation. The step
adds a non-negative emission, so — unlike the carried-cost form — plain forward relaxation is valid.

**Trade-offs.** (1) It changes the cost **meaning**: `C_total` is no longer `Σ drift` and stops being
comparable to graph-DTW's `avg_distance` / the `resolve_routes` thresholds. (2) Very small `α` makes
extra coverage nearly free, so an A-vertex can **over-cover** (grab more B than it should) — keep `α`
comfortably above 0 unless pay-once is truly wanted. (3) It is **orthogonal** to the junction-label
and nearest-vs-corresponding problems (§3.2) — it only reshapes 1:N coverage *within* an edge.
Default `1.0`; reach for `α < 1` only when 1:N cost scaling with B-sampling density is the problem.

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
