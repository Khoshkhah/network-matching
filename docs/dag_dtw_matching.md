# DAG-DTW: Matching a Directed Tree or DAG to a Directed Network

DAG-DTW aligns a **directed source DAG** — a road structure that branches, merges, and may reconverge (a tree is the special case with no reconvergence) — onto a **directed target network**, generalizing DTW from sequences to DAGs. The output is a **matching relation** `M ⊆ V(A) × V(B)`: valid by the four warping rules (§3), selected by direct cost. The algorithm is **one forward table** (with the split coupling built in, §4.1a) plus an **anchored extraction** over it (§5); no backward pass is needed to match. Two modes share all of it: **point** (vertices of `A`,`B`) and **segment** (the same algorithm on the line graphs `L(A)`,`L(B)`, §8).

| piece | status |
|---|---|
| forward table with V3 coupling (§4) | implemented — `forward` |
| extraction (§5, §10) | **three cross-validating engines**: `extract` (branching, §5), `extract_join` (vertex join, §10.1), **`extract_cell`** (cell-level join — exact over the full space, 384/384 on the envelope; §10.2) |
| validation & diagnostics (§6) | implemented |
| segment mode (§8) | implemented & verified (three-way + full-space brute on the line graphs, bearing active); a merge+split junction's `L(A)` cluster is a reconvergent DAG — accepted natively, three engines valid & agreeing on it |
| known validator limit on cyclic B (§7) | documented, pinned by test |
| **DAG sources** (reconvergences) | accepted by default — only a **directed cycle** is rejected (`NotADAG`); `extract_cell` verified exact — 195/195 vs full-space brute (§10.4); other engines: judged cross-checks on DAGs |
| DuckDB pipeline (§9) | `DuckDBMapMatcher.match_dag()` — same sources/CRS handling as Modes 1–2, returns `(dag_long, dag_summary)` DataFrames |

---

## 1. Inputs

* **A — the source**, a **directed acyclic graph**: forks, merges, and reconvergences (diamonds, divided roads) are all legal; only a **directed cycle** is rejected (`NotADAG`). A tree is the special case with no reconvergence — on reconvergent sources only the cell-level engine carries the exactness claim (§10.4). Vertices carry coordinates `x, y`; a **junction is a vertex** (split = out-degree > 1, merge = in-degree > 1). `Apred/Asucc` are the immediate neighbours.
* **A must be subdivided**: at least one interior point on every real edge. This is what makes a split's children pairwise incomparable (§4.0) and a merge's parents childless-siblings — the ordering and coupling guarantees rest on it.
* **B — the target**, any directed network; **it may cycle** (roundabouts, grids). `Bpred/Bsucc` are its neighbours.
* **Candidates.** Each A-vertex `a` gets a radius-gated candidate set `cand(a) = {v ∈ V(B) : ‖a−v‖ ≤ r}` (`r = match_radius_m`; if fewer than `k_min` fall inside, the `k_min` nearest are kept). Each pair `(a, v)` is a **cell**; cells hold the emission `E`, the forward cost `D`, its back-pointer `bpD`, and a `forbidden` flag (§4.1a). The whole algorithm runs on these cells.
* **Emission.** `E(a, v) = ‖pos(a) − pos(v)‖ + λ·circ(bearing(a), bearing(v))` — the bearing term only when both nodes carry a `bearing` (segment mode); `circ` is the circular degree difference in `[0,180]`, `λ = bearing_weight`.

## 2. The Structural Core: Independence

On a **tree**, a merge's incoming branches live in **disjoint upstream subtrees** — if they shared an ancestor, that ancestor would fork and rejoin, an undirected cycle, which a tree forbids. So each branch can be optimized independently and their costs **added** at the merge — this is why one number per cell suffices, and why the forward sum at a merge is exact (§4.1). On a **reconvergent DAG** the branches *can* share an ancestor, so the forward sum double-counts it — which is exactly why on DAG sources only the cell engine, which never reads `D`'s values, keeps the exactness claim (§10.4).

A split is the mirror **with a twist**: its branches are disjoint *downstream*, but they **share the split vertex**. The forward recurrence alone fills each branch independently, so nothing forces them to agree on the split's cell — that agreement is enforced **during the build** by the forbid-and-rebuild step (§4.1a).

## 3. The Objective

A **matching** is a relation `M ⊆ V(A) × V(B)`; `M(a) = {v : (a,v) ∈ M}` is the run of target cells `a` covers (a singleton in the 1:1 case). The goal:

$$\text{minimize}\; C(M) = \sum_{(a,v) \in M} w(a,v)\,E(a,v) \qquad \text{subject to } M \text{ a valid warping (V1)–(V4)},$$

with `w = 1` on a 1:1 advance, `w = α` on a 1:N coverage cell, `w = β` on an N:1 stall. **Domain: `α ∈ (0, 1]`, `β ∈ [1, ∞)`** (defaults `α = β = 1`): covering a finely-sampled target may be *discounted*, stacking several source vertices on one target cell is *never* discounted — only penalized — so the matcher spreads the source rather than collapsing it. The reported drift stays the raw `Σ E` of the chosen matching. `M` is **valid** iff:

**(V1) Monotonicity — no cross.**
```
∀ (a,v) ∈ M,  ∀ a⁻ ∈ Apred(a),  ∀ v⁺ ∈ Bsucc(v):    (a⁻, v⁺) ∉ M
```
**(V2) Predecessor rule — every cell is fed.**
```
∀ (a,v) ∈ M :
    [ ∃ v⁻ ∈ Bpred(v) : (a, v⁻) ∈ M ]                                             (i)  continues a's run
  ∨ [ ∀ a⁻ ∈ Apred(a) : ( (a⁻, v) ∈ M ) ∨ ( ∃ v⁻ ∈ Bpred(v) : (a⁻, v⁻) ∈ M ) ]   (ii) every predecessor feeds it
```
**(V3) Successor rule — every cell continues.** The mirror of (V2) over `Asucc`/`Bsucc`.
**(V4) Full coverage.** `M(a) ≠ ∅` for every `a`; a source's entry and a sink's exit are free, interior junctions are pinned.

(V2) keeps a merge from being entered at two points, (V3) keeps a split from being left at two points, (V1) forbids a backward step. These are properties of `M` alone — `check_rules(M, A, B)` tests exactly them, restricted to neighbours present in `M`.

---

## 4. The Forward Table

One pass builds everything the extraction needs: the cost `D`, the back-pointers `bpD`, and the `forbidden` flags.

### 4.0 Vertex Order — Longest-Path Layering

The build needs an order in which **every split's children are complete, as a group, before anything downstream of them is filled** — so the group can be reconciled (§4.1a) while nothing has read it yet. A plain topological sort does not guarantee this (`J, b₁, x, b₂` is topological, yet `x` — a successor of `b₁` — precedes the sibling `b₂`). Longest-path layering does:

1. sweep `A` topologically; a **source** gets `L = 0`; every other vertex gets `L(v) = max_{p∈Apred(v)} L(p) + 1`;
2. order vertices by `L` ascending (ties by id, for determinism).

`L` strictly increases along every edge, so the order is topological; on a **subdivided** source a split's children each have the split as sole predecessor, hence are pairwise incomparable and share layer `L(split)+1`, ahead of all their successors; a merge lands after **all** its branches (`L = max + 1`). Example:

```text
S → a₁ → J ─→ b₁ ────────→ M → d₁ → T          L:  S=0  a₁=1  J=2   b₁=b₂=3
             └→ b₂ → c₁ ──↗                        c₁=4  M=5  d₁=6  T=7
```

*(Implemented as `layer_order(A)`; identical on `L(A)` for segment mode.)*

### 4.1 The Recurrence — `D`, its Back-Pointers, and Coverage

`D[a][v]` = minimum cost of matching `a`'s **upstream cone** with `a` pinned at `v`. For each predecessor `p`, its two ways into `v` (all look-ups intersected with the gated candidate sets; a cell pruned by `r` is `∞`):

$$\text{step}_p = \min_{x \in Bpred(v)} D[p][x] \quad(\text{advance}),\qquad \text{stall}_p = D[p][v] \quad(\text{already on } v).$$

$$D[a][v] = \min\begin{cases}
E(a,v) + \sum_{p} \tfrac{\text{step}_p}{\text{outdeg}(p)} & \text{(D) every branch advances — full } E \\[4pt]
\beta E(a,v) + \min_{q}\big[\tfrac{\text{stall}_q}{\text{outdeg}(q)} + \sum_{p\neq q}\tfrac{\min(\text{stall}_p,\,\text{step}_p)}{\text{outdeg}(p)}\big] & \text{(V) ≥1 branch stalls — N:1, } \beta E \\[6pt]
\alpha E(a,v) + \min_{v' \in Bpred(v)} D[a][v'] & \text{(H) 1:N coverage along B, } \alpha E
\end{cases}$$

* A **source** has an empty sum: `D = E` (free entry). At a **merge** the sum is finite only if *every* branch reaches `v` — **(V2) folded into the cost**; the split factor `1/outdeg(p)` counts a shared point once. `α = β = 1` reproduces the unweighted cost bit-for-bit; the `min_q` in (V) forces a genuine stall so `β` is never charged on a pure advance.
* **`bpD[a][v]` records exactly the cells the value came from** — the classic DTW arrow generalized to a list: `[]` for a source (free entry); `[(p, x_p) per predecessor]` for an advance/stall (a merge's coupling in one list); `[(a, v')]` — a **same-vertex pair** — for a coverage step (`a` extends its run one B-arc). The move type is read off *whose* vertex appears; no tags.
* **(H) is a within-row fixed point, iterated to convergence.** Unlike (D)/(V), which read only predecessor rows (final under the sweep), (H) reads *other cells of the same row*; B carries no order of its own, so the row is **relaxed until nothing changes**, lowering `D` and repointing `bpD` **in the same step** (the pointer always names the cell that produced the value). `αE ≥ 0` makes this a monotone descent to the unique least fixed point — correct even on a cyclic B, where any single pass would leave cells un-relaxed.

#### Deterministic argmin (Part 4b)

Every argmin above breaks **exact-cost ties** by one fixed total order on B's vertices (`_b_order`, sorted by id): among equal cells the smallest-order one wins, in every pass identically. This changes only which of two *equal* cells is stored — costs and optimality are untouched — and makes the tables invariant to B's insertion order.

### 4.1a The V3 Coupling — Forbid-and-Rebuild

Filled independently, two children of a split can link back to **different** cells of the split — a (V3) break the forward sum cannot see (each `min_v D` places the split wherever is cheapest *for that branch alone*; summing such minima can put one vertex in two places at once — a phantom no matching realizes). The coupling closes this **during the build**, in the §4.0 layer order. It acts on **cells, never on vertices**, so it is identical in both modes and indifferent to how many parents a child has.

Each cell carries a **`forbidden`** flag: once set, **no back-pointer may link to that cell** — every place the recurrence reads a neighbour cell (advance source, stall source, same-row coverage source) skips it. Per split `a`, children `a₁, a₂, …` (one layer):

1. **Build** each child's row with the recurrence, skipping forbidden cells.
2. **Forbid non-shared exits.** As each child completes — *including the first* — mark forbidden every cell of `a` it does **not** link to. Allowed exits = `∩ᵢ links(aᵢ)`.
3. **Rebuild whole rows.** A newly-forbidden cell is dead for **all** siblings, past and future; every earlier child that linked it re-runs its **entire row** under the current flags (the within-row (H) chains make a single-cell patch impossible).
4. **Iterate to the fixed point** — a rebuilt row may re-link to an exit some sibling doesn't share; the forbidden set grows monotonically, so at most `|cand(a)|` rounds.

At the fixed point **every surviving exit of every split is linked by all its children** — each surviving option is (V3)-valid; keeping several options is legitimate (the extraction chooses among them). If a split's exits empty out, there is no V3-valid warping inside the gate: raise the feasibility error (*increase `match_radius_m`*).

Worked trace — split `a`, children `a₁, a₂`, `cand(a) = {v₁, v₂, v₃}`:

| round | event | forbidden | allowed |
|---|---|---|---|
| 1 | `a₁` links `{v₁,v₂}` → forbid `v₃` | `{v₃}` | `{v₁,v₂}` |
| 1 | `a₂` links `{v₁}` → forbid `v₂` | `{v₂,v₃}` | `{v₁}` |
| 2 | `a₁` had linked `v₂` → rebuild `a₁`'s row (skips `v₂,v₃`) → re-links via `v₁` → fixed point | `{v₂,v₃}` | **`{v₁}`** |

**Invariant** (`check_split_exits`): every surviving exit of every split is linked by every child, none links a forbidden cell, survivors non-empty.

*(Implemented as `forward(A, B, α, β)` — this IS the algorithm's forward pass; the name is historical. `forward()` is the uncoupled recurrence, kept only for the §6 diagnostics.)*

---

## 5. The Extraction — Anchored Enumeration over the Forward Table

The extraction uses the forward table only, through **two pointer types**: the stored `bpD` (upstream) and its **reverse** (downstream — cell `(p,x)` lists every cell whose `bpD` names it). No backward table.

```text
anchor ← the vertex with the FEWEST non-forbidden cells        (per weakly-connected component)
for each non-forbidden cell v₀ of the anchor (deterministic order):
    explored ← {(anchor, v₀)}
    leg 1:  from the anchor cell, follow the FORWARD pointers (the reverse of bpD) as far as
            they go — add every cell reached
    leg 2:  from the anchor cell, follow the BACKWARD pointers (the stored bpD) as far as
            they go — add every cell reached
    repeat: find an explored cell from which ONE of the two directions still yields an
            unexplored cell; explore that direction from it as far as it goes; add
    until   no explored cell yields anything new
    M(v₀) ← the explored cells;   touching a forbidden cell anywhere ⇒ label INVALID → next label
    C(M)  ← Σ w(a,v)·E(a,v)                      # w ∈ {1, α, β}, read off M and the graphs alone
return the cheapest M;  every label invalid ⇒ feasibility error (increase r)
```

Each leg is **maximal in a single direction** — the alternation happens between legs, never inside
one. Conjecture (to verify empirically): at the fixed point, each explored cell is productive in at
most **one** of the two directions — the other direction only re-reaches what the arriving leg
already collected.

**Branching — decided.** When a step offers **several cells of the same child vertex** (e.g. a
stall cell and an advance cell both pointing at the explored cell), the exploration does **not**
choose: it **branches** — one candidate continuation per option, every alternative kept alive until
a decision is forced. Same-vertex cover pairs are *not* alternatives (they are one run and are
pulled in by whichever cells the connections actually reference); only **entry cells of a child**
branch. Each completed branch is one candidate relation `M`.

**The judge.** Every candidate from every label goes to the same judge: candidates violating
(V1)–(V4) are discarded — validity *is* the definition of a matching — and among the valid ones the
cheapest `C(M)` wins. If no candidate of any label survives, raise the feasibility error. The
branching is capped (default 4096 states, `max_states`) and **exceeding the cap raises** — never a
silent truncation.

**Further engines (§10).** The junction-join extractions compute the optimal labels by recursive
table joins — forward-only, polynomial, no caps: the vertex-level join (§10.1, exact over the
stored-history family) and the cell-level join (§10.2, exact over the full space).
**`extract_cell`** (§10.2) is the **cell-level join** — built from scratch upstream out of `E`
alone, exact over the *full* space including coverage runs; on the structured envelope it is valid
384/384 and never costlier than either other engine. The three engines **cross-validate** (§10.3):
run them, take the cheapest valid `M`; `C(cell) ≤ C(branching)` and `C(cell) ≤ C(vertex-join)`
always — a violation is an exactness bug by definition (pinned in the suite).

## 6. Verification & Diagnostics

`check_rules(M, src, tgt)` — **the judge**: V1–V4 on the final relation, on the same graph the match lives on (`A,B` or `L(A),L(B)`), each rule restricted to neighbours present in `M`. Everything else below is tooling that verifies the machinery, not part of matching:

* **`validate_tables`** — replays every finite cell's back-pointer chain and checks it is a legal partial warping in isolation.
* **6b — the backward table and cross-table agreement.** `backward()` fills the mirror table `B[a][v]` (downstream cone: `pred→succ`, `Bpred→Bsucc`, `outdeg→indeg`, reverse sweep, same `α/β/E`) and `extract_two_table` is the older two-table traceback over `bpD`+`bpB` (seed at a joint `D+B−E` argmin, flood both pointer sets, pivot-path coverage gap-fill). Both are kept **as diagnostics**: `check_reciprocity` demands that every source edge the forward pointers thread, the backward pointers thread back identically — on the committed matching only (off the optimum the two tables optimise differently-pinned subproblems and legitimately disagree). If the coupled pass runs, run it **before** `backward` — the backward pass respects the `forbidden` flags, so its pointers never target dead cells.
* **6c — reachability.** Each table's back-pointers must reconstruct the tree's own source↔sink structure: walking `bpD` from every finite sink cell (branching at every entry) must reach exactly the sink's ancestor sources; mirror for `bpB`. Empirically clean across every sweep, including under the V3 coupling — table-level structure is never the failure site.
* **6d — complementarity.** Read alone, the *uncoupled* forward table can violate V3 (it is split-optimistic) and the backward table V2 — `check_forward` / `check_backward_v2` surface exactly where. Under weighting they fire by design; this is what motivated the §4.1a coupling, which closes the forward table's half (`check_split_exits` is its invariant).

## 7. Guarantees & Limits

* **Valid by construction, where construction reaches**: (V2) is enforced by the forward sum, (V3) by the §4.1a coupling, (V4) by full-coverage extraction; every candidate `M` is costed directly on the relation.
* **Best of the enumerated labels** — the extraction returns the cheapest matching among the anchor's `≤ |cand(anchor)|` labels. This is deliberate (generation is cheap and honest scoring judges); it is **not** a proven global optimum. The two-table traceback's old "exact optimum" claim was disproven (its arbitrary seed is unsound at a merge — `D+B−E` is not constant across vertices); the anchored extraction fixed that case.
* **Feasibility, never silent breakage**: an unreachable vertex, an emptied split, or an all-invalid enumeration raises `ValueError` telling you to increase `match_radius_m`.
* **Validator limit on cyclic B**: on a 2-cycle `p⇄q` the local (V1) predicate cannot orient a step — the *smallest* case is a 2-vertex chain over `p⇄q`, where the geometrically correct matching (and every separating alternative, in both directions) is flagged. This is a property of the predicate, not of the matcher; pinned by `test_smallest_invalid_output_two_cycle`.
* **The extraction never returns an invalid matching**: candidates violating (V1)–(V4) are discarded by the judge, so the result is valid-by-check or the call raises (no-valid-candidate / state-cap — both explicit). Structured point-mode envelope: 367/384 valid returns, 17 refusals, 0 invalid outputs (`scripts/test_dag_point.py`; dense-target + heavy-shift regimes refuse).
* **Complexity**: table `O(|A| × band)` plus the per-row (H) relaxation and the (rare, bounded) §4.1a rebuilds; extraction `O(labels × |A|)`; all linear-ish in practice.

## 8. Segment Mode — the Same Algorithm on the Line Graphs

A point-mode *state* is a vertex pair; dressing its cost with a heading term does not make it segment matching — on an N:1 stall there is no target segment and the heading is silently free. True segment matching makes the **state a segment pair**: run the identical algorithm on `L(A) = line_digraph(A)` vs `L(B) = line_digraph(B)`, whose nodes are the directed arcs, each carrying its **midpoint** as `x, y` and its **bearing**:

| point (§1–§7) | segment |
|---|---|
| vertex `a` / `v` | arc `s = (t→h)` / `e = (u→v)` |
| `E = ‖a − v‖` | `E = ‖mid(s) − mid(e)‖ + λ·circ(bear(s), bear(e))` |
| `Apred/Bpred` | arc adjacency in `L(A)` / `L(B)` |

Everything — layering, the recurrence, the coupling, the extraction, `check_rules` — is byte-for-byte the point-mode code on the lifted graphs; every state pays its emission (a stall costs `βE`, never zero — the heading cannot be bypassed). A junction is a vertex, so the line graph connects real segments directly; there are no stitch connectors to park on. A merge+split vertex becomes a bipartite cluster in `L(A)` — a reconvergent DAG, accepted natively (all three engines return valid, agreeing matchings on it; only the cell engine carries the exactness claim there, §10.4). The matching is emitted and validated **on the arcs** (`M_seg ⊆ E(A) × E(B)`); any per-point view is derived convenience, never validated in its place.

## 9. Implementation Notes (Parts)

* **Part 1 — cells.** `prepare(A, B, r, k_min, bearing_weight)` validates inputs, gates candidates by KD-tree, and stores each vertex's row on the node:
  `A.nodes[a]["cand"][v] = {"E": …, "D": inf, "bpD": [], "B": inf, "bpB": [], "forbidden": False}`
  (`B/bpB` are filled only by the diagnostic backward pass.)
* **Part 1.3 — feasibility.** `r` must cover the largest A↔B drift (NVDB↔OSM: 10–20 m). A gate-severed chain leaves rows all-`∞`; a coupled infeasibility shows up as a severed (`None`) pointer; both raise `ValueError`, never a broken match.
* **Function map.**

  | algorithm step | function |
  |---|---|
  | candidates & cells (§1) | `prepare` |
  | vertex order (§4.0) | `layer_order` |
  | forward pass incl. V3 coupling (§4.1–§4.1a) | `forward` |
  | extraction (§5) | `extract` (branching) · `extract_join` (vertex join) · `extract_cell` (cell join) |
  | one-call pipeline | `match_dag(A, B, r, α, β, mode, engine)` — `engine="all"` = cheapest valid of the three |
  | DuckDB pipeline | `DuckDBMapMatcher.match_dag(alpha, beta, engine, bearing_weight, …)` — Mode-1/2 input system (WKT CSV / geofiles / DuckDB tables, lon-lat → UTM), converts to `networkx` via `edges_to_digraph` (densify + junction-snap), runs segment mode, returns `(dag_long, dag_summary)` DataFrames |
  | geometry → graph | `edges_to_digraph(edges, step_meters, snap_decimals)` — densified polylines (supplies the §1 subdivision), junction-snapped; arcs carry `road_id` + `seq` |
  | judge (§6) | `check_rules` |
  | diagnostics (§6) | `backward`, `extract_two_table`, `validate_tables`, `check_reciprocity`, `check_reachability`, `check_forward`, `check_backward_v2`, `check_split_exits` |
  | segment lift (§8) | `line_digraph` |

* **Playground** — `notebooks/dag_dtw_playground.ipynb` (interactive Plotly scenarios, the historical failure demos and their fixes). **Cross-validation sweep** — `scripts/test_dag_point.py` (all three engines over structure × density × shift × noise × weights).

---

## 10. The Join Engines — exact extraction by table joins

Two further extraction engines share §5's judge and cost; both are **joins over tables**, not
searches. The **cell-level join** (`extract_cell`) is the primary engine — exact over the full
space; the vertex-level join and the §5 branching exploration remain as cross-checks.

### 10.1 The vertex-level join (`extract_join`)

Between junctions `bpD` chains are deterministic: each sink label induces exactly one label at
every ancestor junction. So the sink labels are the only free variables, and (V3) at a split is a
**join condition** — all branches must induce the same split label. Every table is a *sink-type*
table (`label → through-cost` + pinned labels): sweep splits deepest-first; per branch keep the
best row per induced label (a contraction, never a cross-product); sum — the split factor makes
the sum exact (`Σᵢ (branchᵢ + D[U][u]/k) = Σᵢ branchᵢ + D[U][u]`); the joined table *is* a
sink of the reduced graph, so the recursion is type-uniform; a merge's table is **consumed once**
(a second split reaching the shared region continues through the collapsed table's recorded
interior cells — no cost division). The root table's minimum pins every sink and split.

**Scope — cell resolution.** Rows are vertex labels with *stored* histories, so intra-vertex
alternatives (where a run starts, which run cell a child connects through) are frozen: the vertex
join is exact over the **stored-history family** only. Divergences from the cell engine occur
exactly in dense-target coverage regimes, at any weights — which motivates:

### 10.2 The cell-level join (`extract_cell`) — the primary engine

Built **from scratch upstream out of `E` alone** — the stored `D`/`bpD` propagation is never
consulted (it froze the very choices being optimized, and its values carry shared-cone `1/outdeg`
fractions); `forbidden` and `D < ∞` serve only as pruning.

**The data packet — one row:**

```
row = ( entry,     the cell of THIS vertex where its parent will connect (the upward interface)
        value,     cost of everything below, per the ledger -- this vertex's entry-E NOT yet paid
        pending,   {(merge-vertex, its entry-cell) -> stall-flag} : deferred merge entries
        cells )    {vertex -> its run} : the piece of M built so far -- M travels WITH the row
```

**The E-multiplier ledger** — every cell's emission enters exactly once, weighted by the move that
enters the cell: **1** (source free entry / advance), **β** (stall — some parent arm holds the same
cell), **α** (cover — a run cell). No fractional factors anywhere: plain sums at split cells,
consumed-once at merges. **Deferred entry**: whether a vertex's entry cell pays `1` or `β` is the
parent's call, so its entry-`E` is paid at the step connecting it to its parent — at a merge, at
the root join, `β` if **any** arm stalls.

**Three flow operations** (reverse topological sweep, one vertex at a time):

* **PASS** (chain vertex `X`, child `c`): per entry `e` and run `R = e→…→u`, take a child row whose
  entry `ce` connects at the run **end** (`ce == u` stall / `ce ∈ Bsucc(u)` advance), **pay the
  child's deferred entry now**, add `α·E` per cover cell, `cells ← child.cells + {X: R}`; contract
  per `(entry, pending)`.
* **MERGE of data** (split `X`, children `c₁ … c_k`): as PASS with all `k` children connecting at
  the **same** run end — per-cell (V3). Sum the values, union the cells (disjoint subtrees),
  reconcile the pendings (same merge-vertex with different entry cells ⇒ the combination is
  discarded; stall flags OR).
* **SPLIT of data** (merge `m` serving several parent arms): **consumed-once** — the first arm
  absorbs value and cells but leaves `m`'s entry-`E` unpaid, gaining
  `pending[(m, entry)] = did-this-arm-stall`; every other arm receives only an *interface* (entry
  cells, value 0, the same pending key with its own flag). Nothing is counted twice — which is why
  no `1/indeg` exists.

**Cell removal (pre-pass)**: one reverse search from **all sink cells** over the cell-move graph
(cover reversed inside a vertex, advance/stall reversed across edges); every cell never seen is
removed, in every role; runs may not cover removed cells. A vertex with no surviving cell ⇒
immediate, precisely located infeasibility. The `D < ∞` filter prunes the upstream mirror.

**Termination and the root join**: sources pay their own entry-`E` (full — free entry) and become
root tables; the root join combines one row per root (pendings must agree on every merge's entry
cell; flags OR; each deferred merge entry paid once), **contracting per pending-key after each
fold**. The joined value equals `C(M)` exactly. **`M` is never extracted** — the winning row's
`cells` map *is* the relation; no traceback, no reconstruction, no gap-fill. Rows are tried
cheapest-first; the first `M` passing `check_rules` wins (judge unchanged). Loud caps only
(`run_cap` bounds cover runs; `max_rows` raises, never truncates).

### 10.3 Verification & the standing cross-validation

| check | result |
|---|---|
| tiny dense-B cases vs **full-space** brute force (all entry+run combinations), point **and** segment (bearing active) | equal to the digit |
| random mini polytrees vs full-space brute | exact & valid, every case |
| structured 384-case envelope (`scripts/test_dag_point.py`, all three engines) | branching 376/384 · vertex join 379/384 · **cell join 384/384** valid |
| invariant `C(cell) ≤ C(branching)` and `≤ C(vertex join)` | **384/384** — pinned in the suite: a violation is an exactness bug by definition |
| pinned divergence case | cell strictly beats the vertex join *and* equals the full-space optimum |

The three engines cross-validate: run them (`match_dag(engine="all")`), take the cheapest valid
`M`; `value == C(M)` is self-checked.

### 10.4 DAG sources (accepted by default)

The tree property was only ever needed for the **forward sum's** cost exactness (§2) — and the cell
engine never reads `D`'s values, so it carries no tree dependence: on a diamond, the two arms hold
the merge's pending separator and must agree when they meet at the **shared ancestor's join** — the
same consumed-once mechanism, resolving earlier than at the root. `prepare`/`match_dag` accept any
**subdivided DAG** by default (only a directed cycle is rejected — `NotADAG`). Verified: 75 jittered
diamonds × 3 weights + 120 random subdivided reconvergent DAGs = **195/195 exact** vs the
full-space brute force, all valid. Scoping: only `extract_cell` carries the exactness claim on DAG
sources — the other engines consume `D` values (tree-only arguments) and act as judged cross-checks
there; the coupling's forbidden-pruning is empirically safe on DAGs, theoretically unproven.
