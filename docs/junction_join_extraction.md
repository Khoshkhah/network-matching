# Junction-Join Extraction — forward-only, exact

> **Status: specified and PROTOTYPE-VERIFIED** — `scripts/junction_join_prototype.py`: root-table
> minimum == brute-force optimum on **96/96** runs (chain, y-split, split-under-split, the canonical
> merge shape × 4 in-domain weights, + 80 random subdivided-polytree runs), root through-cost ==
> honest `C(M)` exactly, every returned `M` valid. **Implemented as `extract_join`** alongside the
> branching `extract`; the two cross-validate each other (below).
> A second extraction for Tree-DTW, alternative to the
> §5 branching exploration of `docs/tree_dtw_matching.md`: instead of enumerating candidate
> relations and judging them, it computes the **optimal labels for all sinks and split vertices
> directly**, by a recursive **table join over the split hierarchy** — using only the forward table
> `D`/`bpD`. Everything here uses the notation of `docs/tree_dtw_matching.md` (§3 objective, §4
> forward table). The merge bookkeeping is resolved (§5); the exactness claim is scoped by **cell
> resolution** (§6a) — the one caution to keep in mind.

## 1. The enabling facts

* **Between junctions the table is deterministic.** A chain vertex has one predecessor, so from any
  cell the `bpD` walk upward is forced (cover pairs walk the run, the advance pair steps to the
  parent). Consequently: from any **sink label**, the walk induces **exactly one** label at every
  ancestor junction — write `ind_U(S, v)` for the label induced at junction `U` by sink `S` at `v`.
* **The free variables are the sink labels only.** A consistent assignment of sink labels pins every
  split and every chain vertex; the whole relation `M` follows by pointer-walk.
* **The split coupling is a join condition.** (V3) demands all branches below a split `U` agree on
  `U`'s label — i.e. their induced labels **join** on `U`. No search is needed to enforce it.
* **The split factor makes the cost additive.** The forward pass divided `U`'s cone by
  `1/outdeg(U)` across its branches (§4.1), so per-branch costs recombine without double counting
  (the identity in §3 below).

## 2. The object — every table is a sink-type table

A **table** for a vertex `X` maps `label → through-cost`, where the through-cost is the cost of
everything that flows through `X` (its own cone plus every joined branch below it), with `X` at that
label. Each row also carries the **pinned labels** of the sinks and splits already folded into it.

A sink `S` starts with the trivial table `{v : D[S][v]}` (no pinned columns). The recursion's
invariant: **the output of a join is again a sink-type table** — after joining, the split *is* a
sink of the reduced graph, so tables are type-uniform and the recursion closes on itself.

## 3. The algorithm

Sweep the splits in **reverse topological order** (deepest first). At split `U` with `outdeg(U) = k`,
whose branches lead to already-tabled nodes `T₁ … T_k`:

1. **Induce.** For every label `v` of every `Tᵢ`, walk `bpD` up to `U`: `u = ind_U(Tᵢ, v)`. Rows
   whose walk touches a forbidden cell are dropped.
2. **Join on `u`.** Given the shared label `u`, the branches are independent, so per branch keep
   `bestᵢ(u) = min { cost_{Tᵢ}(v) : ind_U(Tᵢ, v) = u }` (remember the arg-row: the winning label of
   `Tᵢ` and everything it had pinned). The join is a **contraction** — table sizes never multiply.
3. **The new table for `U`:**

   $$\text{through}_U(u) \;=\; \sum_{i=1}^{k} \text{best}_i(u)$$

   with the pinned columns of all winning rows attached, plus `U = u` itself. **No subtraction is
   needed against the implemented table**: each `bestᵢ(u)` carries `D[U][u]/k` (the split factor),
   so the sum reassembles `U`'s cone exactly once —

   $$\sum_i \big(\text{branch}_i + \tfrac{D[U][u]}{k}\big) = \sum_i \text{branch}_i + D[U][u].$$

   *(Equivalent bookkeeping: if per-branch costs carried `U`'s cone unscaled, the correction would
   be `Σᵢ bestᵢ(u) − (k−1)·D[U][u]` — the `− cost(U)` form.)*
4. **Reduce.** Delete the joined branches; `U` with its new table is a sink of the reduced graph.
   A label of `U` with no surviving row in *some* branch has no consistent completion — it simply
   has no row (if **no** label survives, the component is infeasible: raise, as always).

At the **root** (last surviving table of the component) the cost column **is the total decision cost
`C(M)`** of the whole component. Take the minimum row: it pins every sink and every split. Commit
those labels; every chain vertex follows deterministically by the pointer walk; coverage runs are
pulled by the cells the connections reference (as in §5); the final `M` is judged by `check_rules`
exactly as always (the join optimizes the decision cost — validity of `M` remains the judge's word,
e.g. on cyclic-B targets).

### Equivalent one-step formulation (per-vertex sweep)

The same recursion can be run **one vertex at a time** in reverse topological order, with no
split-jumping: at out-degree 1, the child's table is **re-keyed** one pointer hop upward (contract
rows landing on the same label — cost unchanged, the child's cost already contains this vertex's
cone); at out-degree `k`, the `k` re-keyed tables **join** as above. Same tables, same arithmetic,
finer steps — use whichever formulation is more convenient to implement.

## 4. Worked example — the y-split, real numbers

`A: 0→1→{2,3}`, `B: s→j→{u,d}`, `r = 20`, `α = β = 1`. After the forward pass (coupling included;
vertex 1's survivors = `{j}`):

```
vertex 1 (split):  j: D=1.00                       (s, u, d forbidden)
sink 2:            u: D=1.00   j: D=11.91  d: D=12.00     all induce 1→j
sink 3:            d: D=1.00   j: D=12.43  u: D=13.00     all induce 1→j
```

Join at `U = 1` (`k = 2`): `best₂(j) = 1.00` (label `u`), `best₃(j) = 1.00` (label `d`) →

```
table(1):  label j :  through = 1.00 + 1.00 = 2.00    pinned: {2: u, 3: d}
```

Identity check: `Σ branchᵢ + D[1][j] = (0.5 + 0.5) + 1.00 = 2.00` ✓. The reduced graph `0→1` has no
further split; the root table is `table(1)`; its minimum (only) row pins `1→j, 2→u, 3→d`; the walk
adds `0→s`. Result `M = {(0,s), (1,j), (2,u), (3,d)}`, total cost `2.00 = Σ E` — the known optimum.

## 5. Merge bookkeeping — RESOLVED: consumed-once, no division

A sink below a merge sits under **two** splits (one above each merge arm) — the sharing question.
The resolution, verified numerically: **each table is consumed by exactly one later join.** When the
second split's branch walks down and reaches the shared region, it finds the **collapsed table** of
the first join and induces its own label from that table's **recorded interior cells** (e.g. the
merge cell, recorded when the first join's walk passed it) — the polytree message flow. Costs then
recombine exactly through the split factors alone; **no `cost/indeg` division is needed**.

Verified on the canonical shape (`U → x → m ← z ← V`, sinks below `m` and on both other branches):
junction-join == brute force at all four in-domain weights (cost 6.500), and across 80 random
subdivided polytrees with natural merges — **96/96 agreement overall, 0 disagreements**.

## 6a. Exactness scope — cell resolution

The join works **vertex to vertex**: a table row is one label per vertex, and its history is the
stored `bpD` walk. But some moves live **between cells inside one vertex** — the (H) coverage runs.
Those intra-vertex alternatives (where a run starts, which of a parent's run cells a child connects
through) are frozen to the single stored history per label; the join is therefore **exact over the
stored-history family**, not over all valid relations. The branching `extract` explores exactly
those cell-level alternatives — so the two engines are complementary:

* **join**: global junction coupling, exact within its family, polynomial, no caps;
* **branching**: cell-resolution run alternatives, best-of-enumerated, capped.

Measured (structured 384-case sweep, both engines on the same tables): join valid **379/384**,
branching valid **376/384** (the join also succeeds on the deep×dense cases where branching hits its
state cap); **61 cost divergences where the branching beat the join — ALL in dense-target (coverage)
regimes, 0 elsewhere, present even at `α = β = 1`** — i.e. purely the cell-resolution gap, not a
weight effect.

**Cross-validation practice**: run both, take the cheaper valid `M`; a divergence *is information* —
it flags a coverage-regime case. The suite pins the verified invariant: a join loss to the branching
**must involve coverage** (`test_extractions_cross_validate`); a loss without coverage would be a
real exactness bug.

**Future refinement** (if the gap matters on real data): lift the join to cell resolution by keying
table rows on the run's **boundary cells** (entry, exit) instead of a single label — the join logic
is unchanged, the tables get one extra dimension.

## 6b. Complexity & properties

| | junction-join | §5 branching exploration |
|---|---|---|
| result | **exact decision-cost optimum** over the coupled label space | best of the enumerated candidates |
| mechanism | per-split table join (contraction) | branch on every alternative, judge filters |
| caps | none (polynomial: `O(Σ_labels · path-length)` walks, linear joins) | `max_states` (raises on excess) |
| needs §4.1a forbidding | no (join enforces the coupling itself; forbidding remains as pruning) | relies on it for shared exits |
| validity | judged on final `M` (unchanged) | judged on final `M` (unchanged) |

Both end at the same judge; they can coexist (junction-join as the label-finder, branching as a
fallback/cross-check) or the join can replace the branching once verified.

## 7. Verification results

`scripts/junction_join_prototype.py` (prototype + brute force over all sink-label combinations,
reconstruction by bp up-flood, honest `C(M)` costing):

| cases | weights | result |
|---|---|---|
| chain, y-split, split-under-split, merge shape | 4 in-domain `(α,β)` each | **16/16 agree**, root through == `C(M)` exactly, all `M` valid |
| 40 random subdivided polytrees (natural merges) | 2 weights each | **80/80 agree**, 0 infeasible, 0 skipped |

Library wiring done: `extract_join` (judge unchanged — root rows tried cheapest-first, first
valid wins), suite 191 passing incl. `test_extract_join_exact_on_merge_shape` (vs brute force) and
the cross-validation tests; dual-engine envelope: join 379/384, branching 376/384, divergences
confined to coverage regimes (§6a). Both engines stay — they cross-validate each other.

---

## 8. Cell-Level Join — full resolution (IMPLEMENTED: `extract_cell`)

The §6a gap closed at the source: run the same recursion with **cells, not vertex labels**, as the
propagating object. No stored history is replayed — the engine needs only `prepare`'s `E` (and
`forbidden` as pruning); `D`/`bpD` are not consulted. Intra-vertex moves are first-class variables.

### 8.1 The E-multiplier ledger

Every cell's emission enters the total exactly once, weighted by the move that enters the cell:

| move into the cell | multiplier |
|---|---|
| source free entry / advance (one B-arc from the parent's cell) | **1** |
| stall (some parent arm holds the same cell) | **β** |
| cover (same vertex, one B-arc — a run cell) | **α** |

No `1/outdeg` fractions appear anywhere: those exist only inside *stored* `D` values (shared cones);
the fresh upward pass shares nothing — plain sums at split cells, consumed-once at merges.

**Deferred entry.** Whether a vertex's *entry* cell pays `1` or `β` depends on the parent's cell
(stall ⇔ the entry lies in some parent's run) — so each vertex's entry-`E` is **deferred to the
step that connects it to its parent**, and at a merge to the **last** arm (any arm stalling ⇒ `β`).

### 8.2 The sweep — how data flows cell to cell

**The data packet — one row.** Every vertex's table is a set of rows; a row is the unit that flows:

```
row = ( entry,     the cell of THIS vertex where its parent will connect (the upward interface)
        value,     cost of everything below, per the ledger -- this vertex's entry-E NOT yet paid
        pending,   {(merge-vertex, its entry-cell) -> stall-flag} : deferred merge entries
        cells )    {vertex -> its run} : the piece of M built so far -- M travels WITH the row
```

Vertices are processed in reverse topological order, each exactly once. Three flow operations:

**PASS — a chain vertex `X`, one child `c`.** For each choice of `X`'s entry `e` and run
`R = e→…→u`: take a child row whose entry `ce` connects at the run **end** — `ce == u` (stall) or
`ce ∈ Bsucc(u)` (advance) — and **pay the child's deferred entry now** (`β·E(c,ce)` on stall,
`1·E(c,ce)` on advance: stall-vs-advance is the parent's call, which is why the child could not
price its own entry). Add `α·E` per cover cell of `R`; `cells ← child.cells + {X: R}`; `X`'s own
entry-E stays unpaid (deferred to *its* parent). Keep the cheapest row per `(entry, pending)` key —
the contraction.

**MERGE of data — a split vertex `X`, children `c₁ … c_k`.** As PASS, but **all k children must
connect at the same run end `u`** — that *is* (V3), per cell. Combining one row per child:
`value` = Σ child values + Σ their just-paid entries + the run's covers; `pending` = union (flags
OR'd; two branches carrying the *same* merge-vertex with *different* entry cells ⇒ the combination
is **discarded**); `cells` = union of the k maps (disjoint — different subtrees). Merging data in a
cell = *sum the values, union the cells, reconcile the pendings, at one shared cell.*

**SPLIT of data — a merge vertex `m` serving two parent arms.** Consumed-once; the data is never
duplicated. The **first** parent line absorbs everything — value and cells — but does **not** pay
`m`'s entry-E: the row gains `pending[(m, entry)] = did-this-arm-stall`. Every **other** parent
line receives only an *interface*: rows with `m`'s possible entry cells, `value = 0`, empty cells,
and the same pending key with its own arm's stall flag. Splitting data = *full data up exactly one
arm; the other arms carry only the coordination key.* This is why no `1/indeg` division exists —
nothing was counted twice to begin with.

**Cell removal (pre-pass).** Before the sweep: one **reverse search from all sink cells** over the
cell-move graph (cover reversed inside a vertex, advance/stall reversed across edges); every cell
the exploration never sees is **removed, in every role** (entries, run ends, intermediates alike) —
an unseen cell cannot appear on any chain to a sink; runs may not cover removed cells either
(`sink_reachable`). It over-approximates exact usefulness, which is what makes it safe; the
upstream mirror (no path from sources) stays pruned by the `D < ∞` filter. A vertex with no
surviving cell ⇒ immediate, precisely located infeasibility.

**Termination and the root join.** The sweep finishes when it reaches the **sources**; each source
pays its own entry-E in full (free entry, `w = 1`) and its rows become **root tables** — one root
per source line (several exactly when merges split the lines). The **root join** then picks one row
per root: legal only if the pendings **agree on every merge's entry cell**; flags OR; each pending
`(m, e)` finally pays its deferred `E(m,e)` once — `β` if *any* arm stalled, else `1`. The joined
value is the complete `C(M)` (self-checked: `value == C(M)` holds on every verified case).

**`M` is never extracted — it travels with the rows.** The winning row's `cells` map *is* the
relation: `M = {(vertex, cell) for each vertex's run}`. No traceback, no pointer walk, no
reconstruction, no gap-fill. The final `M` goes to `check_rules` — the judge's role unchanged.
Worked micro-example (y-split): sink rows carry `{2:(u,)}` and `{3:(d,)}`; the split's row merges
them into `{1:(j,), 2:(u,), 3:(d,)}` while paying `E(2,u) + E(3,d)`; the source's row adds
`{0:(s,)}`, pays `E(1,j)`, then its own `E(0,s)` — the root row's value is the full cost and its
cells map is the full `M`.

### 8.3 Expected invariants (the three-way cross-validation)

`C(cell-join) ≤ C(branching)` **always** — no coverage exception (this is what closes §6a) — and
`C(cell-join) ≤ C(vertex-join)`; all three valid or loudly infeasible. Verified below.

### 8.4 Verification results (`scripts/cell_join_prototype.py`)

| check | result |
|---|---|
| tiny dense-B cases (chain, y-split) vs **full-space** brute force (all entry+run combinations) | equal to the digit, 3 weights each |
| mini-merge (pending-separator path) vs branching & vertex-join | equal, 3 weights |
| 30 random mini polytrees vs full-space brute force | **30/30 exact & valid** |
| `value == C(M)` self-check | passes on every case |
| §6a closure — divergence hunt (200 random dense-B cases) | 4 cases where **cell-join < vertex-join**; in each, cell-join == branching, and where the full brute is computable, **cell-join == full-space optimum** (e.g. 15.382 vs vertex-join 15.407) |

The cell-level engine is exact over the **full** cell-level space — runs included — and closes the
§6a gap by construction and by measurement.

### 8.5 Library wiring — `extract_cell`

Implemented as `extract_cell(A, B, α, β, run_cap=8, max_rows=50000)`: deterministic ordering
throughout, loud caps (`run_cap` bounds cover runs, `max_rows` raises — never truncates silently),
joined rows tried cheapest-first with the unchanged `check_rules` judge, and the root join
**contracts per pending-key** after each fold (only the pendings matter for future folds — the
blowup on dense merge structures collapses to `Π |merge entries|`). Suite: 220 passing, including
equality with the full-space brute force on tiny dense cases **in both point and segment mode**
(the segment check runs on the line graphs with the bearing term active), the pinned divergence
case (cell strictly beats the vertex join and equals the full-space optimum), and the three-way
harness in both modes. One-call pipeline entry: `match_tree(A, B, r, α, β, mode, engine)`
(`engine="all"` runs all three and returns the cheapest valid matching).

**Structured envelope (384 cases, all three engines on the same tables):**

| engine | valid |
|---|---|
| branching `extract` | 376/384 (state-cap on deep×dense) |
| vertex `extract_join` | 379/384 |
| **cell `extract_cell`** | **384/384** |
| invariant `C(cell) ≤ C(branching)` and `≤ C(vertex-join)` | **384/384** |

The cell engine is strictly dominant on this envelope: valid everywhere, never costlier than either
other engine, no caps hit. The three-way cross-validation is the standing harness
(`test_three_way_cross_validation`, `scripts/test_tree_point.py`).
