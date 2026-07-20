# DAG-DTW: Matching a Directed Tree or DAG to a Directed Network

DAG-DTW aligns a **directed source DAG** — a road structure that branches, merges, and may reconverge (a tree is the special case with no reconvergence) — onto a **directed target network**, generalizing DTW from sequences to DAGs. The output is a **matching relation** `M ⊆ V(A) × V(B)`: valid by the four warping rules (§3), selected by direct cost. The algorithm is **one forward table** (with the split coupling built in, §4.1a) plus the **cell-level join extraction** (§5): a **backward pass** from the sinks builds per-vertex tables out of the emissions alone — the forward table serves it as pruning — and the selection just takes the cheapest valid root row. A vertex-level join (§10) exists only to cross-validate it. Two modes share all of it: **point** (vertices of `A`,`B`) and **segment** (the same algorithm on the line graphs `L(A)`,`L(B)`, §8).

| piece | status |
|---|---|
| forward table with V3 coupling (§4) | implemented — `forward` |
| extraction (§5) | **`extract_cell`** — the cell-level join: backward table build (§5.2) + cheapest-valid selection (§5.3); **exact over the full space**, 384/384 valid on the envelope; implemented as the per-cell cell-DAG sweep (`docs/cell_dag_extraction.md`) |
| cross-validation engine (§10) | `extract_join` (vertex-level join) — kept **only** to validate/cross-validate §5; `C(cell) ≤ C(join)` pinned |
| validation & diagnostics (§6) | implemented |
| segment mode (§8) | implemented & verified (cross-engine + full-space brute on the line graphs, bearing active); a merge+split junction's `L(A)` cluster is a reconvergent DAG — accepted natively, both engines valid & agreeing on it |
| known validator limit on cyclic B (§7) | documented, pinned by test |
| **DAG sources** (reconvergences) | accepted by default — only a **directed cycle** is rejected (`NotADAG`); `extract_cell` verified exact — 195/195 vs full-space brute (§10.2); the vertex join: judged cross-check on DAGs |
| DuckDB pipeline (§9) | `DuckDBMapMatcher.match_dag()` — same sources/CRS handling as Modes 1–2, returns `(dag_long, dag_summary)` DataFrames |

---

## 1. Inputs

* **A — the source**, a **directed acyclic graph**: forks, merges, and reconvergences (diamonds, divided roads) are all legal; only a **directed cycle** is rejected (`NotADAG`). A tree is the special case with no reconvergence — on reconvergent sources only the cell-level engine carries the exactness claim (§7, §10.2). Vertices carry coordinates `x, y`; a **junction is a vertex** (split = out-degree > 1, merge = in-degree > 1). `Apred/Asucc` are the immediate neighbours.
* **A must be subdivided**: at least one interior point on every real edge. This is what makes a split's children pairwise incomparable (§4.0) and a merge's parents childless-siblings — the ordering and coupling guarantees rest on it.
  * `edges_to_digraph` supplies that subdivision via `_densify`, which places stations every ~`step_meters` **and, by default (`keep_vertices=True`), keeps the polyline's own vertices**. Uniform-only sampling keeps a vertex just by coincidence, so a cell can span a corner and the straight line between its endpoints **cuts** it. That matters because callers read the cells' lengths as *distances along the road* — `parts_from_matching`'s `a_from_m`/`b_from_m` stations and `b_len_m`. With the vertices kept, every cell lies inside one original segment, so it is straight and its chord **is** its arc, making those stations exact by construction. (A 90° corner, `[(0,0),(7,0),(7,8)]` at `step=15`: uniform-only returns just the two endpoints and measures 10.63 against a true 15.0. On real data, one 38.91° vertex on a 1082.66 m road cost 0.85 m.)
  * The trade-off, and why it is a parameter rather than unconditional: the objective is **count-weighted** (§3 — one emission term per matched cell, regardless of its length), so cell spacing is a *matching* decision, not only a geometric one. Keeping vertices makes cells uneven, giving a finely-digitized bend more terms than a coarsely-drawn straight of equal length. Pass `keep_vertices=False` for the strictly uniform subdivision. Changing this changes which routes win, so a tuned configuration should be re-tuned after switching.
* **B — the target**, any directed network; **it may cycle** (roundabouts, grids). `Bpred/Bsucc` are its neighbours.
* **Candidates.** Each A-vertex `a` gets a radius-gated candidate set `cand(a) = {v ∈ V(B) : ‖a−v‖ ≤ r}` (`r = match_radius_m`; if fewer than `k_min` fall inside, the `k_min` nearest are kept). Each pair `(a, v)` is a **cell**; cells hold the emission `E`, the forward cost `D`, its back-pointer `bpD`, and a `forbidden` flag (§4.1a); the concrete storage layout is in §4. The whole algorithm runs on these cells.
* **Emission.** `E(a, v) = ‖pos(a) − pos(v)‖ + λ·circ(bearing(a), bearing(v))` — the bearing term only when both nodes carry a `bearing` (segment mode); `circ` is the circular degree difference in `[0,180]`, `λ = bearing_weight`.

## 2. The Structural Core: Independence

On a **tree**, a merge's incoming branches live in **disjoint upstream subtrees** — if they shared an ancestor, that ancestor would fork and rejoin, an undirected cycle, which a tree forbids. So each branch can be optimized independently and their costs **added** at the merge — this is why one number per cell suffices, and why the forward sum at a merge is exact (§4.1). On a **reconvergent DAG** the branches *can* share an ancestor, so the forward sum double-counts it — which is exactly why on DAG sources only the cell engine, which never reads `D`'s values, keeps the exactness claim (§7, §10.2).

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

### The Data Structure — Rows on the `networkx` Nodes

There is no matrix, array, or table class. Both `A` and `B` are plain `networkx.DiGraph` objects: a vertex's node-attribute dict carries its float coordinates `x`, `y` (segment-mode nodes add `bearing`), and adjacency (`Apred/Asucc`, `Bpred/Bsucc`) is read live off `A.predecessors` / `A.successors` / `B.predecessors` / `B.successors` — never copied. The forward table itself is stored **on `A`'s nodes**: `prepare` (§1) gives every A-vertex one **row**, a dict keyed by its gated candidate B-vertices, whose values are the cells:

```python
A.nodes[a]["cand"] = {
    v: {"E": float,               # emission E(a, v) -- §1, fixed at prepare()
        "D": float,               # forward cost -- inf until §4.1 fills it
        "bpD": list,              # back-pointer cells: [] | [(p, x_p), ...] | [(a, v')]
        "B": float, "bpB": list,  # diagnostic backward mirror (§6b) -- untouched here
        "forbidden": bool}        # §4.1a role-aware flag: a True cell cannot be a run END
    for v in cand(a)              # ONLY the radius-gated candidates
}
```

So the math's `D[a][v]` is literally `A.nodes[a]["cand"][v]["D"]`, and a **cell** is one inner dict. Three consequences worth naming:

* **Sparsity is the gate.** A row holds only `cand(a)`; every neighbour look-up intersects with the row's keys, so a pair pruned by `r` needs no sentinel — **absence is the `∞` cell**. This is what makes the table `O(|A| × band)` rather than `O(|A| × |B|)`.
* **`bpD` stores cell coordinates, not tags.** Each entry is an `(A-vertex, B-vertex)` pair naming the cell the value came from (§4.1): `[]` source, one pair per predecessor for advance/stall, the same-vertex pair `[(a, v')]` for coverage. A severed reference is `(p, None)` — the coupled-infeasibility signature (§9, Part 1.3).
* **Mode-blind.** Segment mode stores the identical structure on `L(A)`'s nodes, whose ids are arc pairs `(u, v)` carrying midpoint + bearing (§8) — the row/cell layout never changes, only the node ids and the emission.

`forward(A, B, α, β)` mutates these rows in place and returns `A`; nothing is written anywhere else (`B`/`bpB` fill only if the diagnostic `backward` pass runs).

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
* **`bpD[a][v]` records exactly the cells the value came from** — the classic DTW arrow generalized to a list: `[]` for a source (free entry); `[(p, x_p) per predecessor]` for an advance/stall (a merge's coupling in one list); `[(a, v')]` — a **same-vertex pair** — for a coverage step (`a` extends its run one B-arc). The move type is read off *whose* vertex appears; no tags. Viewed as a graph over the cells — an edge `(a,v) → (p,x)` per stored pair, arrows pointing **upstream** as stored — the pointers always form a **DAG**: advance/stall arrows descend `A`'s layers (§4.0), and a cover chain strictly descends in `D` — exact ties descend the fixed `_b_order` (Part 4b) — so no pointer chain can loop, even over a cyclic `B`. Its topological order runs sink cells → source cells (its transpose's, source → sink, is the §4.0 fill order). But it is one *chosen* history per cell, not the space of legal moves (that is the §5.0 graph, which *can* cycle within a row).
* **(H) is a within-row fixed point, iterated to convergence.** Unlike (D)/(V), which read only predecessor rows (final under the sweep), (H) reads *other cells of the same row*; B carries no order of its own, so the row is **relaxed until nothing changes**, lowering `D` and repointing `bpD` **in the same step** (the pointer always names the cell that produced the value). `αE ≥ 0` makes this a monotone descent to the unique least fixed point — correct even on a cyclic B, where any single pass would leave cells un-relaxed.

#### Deterministic argmin (Part 4b)

Every argmin above breaks **exact-cost ties** by one fixed total order on B's vertices (`_b_order`, sorted by id): among equal cells the smallest-order one wins, in every pass identically. This changes only which of two *equal* cells is stored — costs and optimality are untouched — and makes the tables invariant to B's insertion order.

### 4.1a The V3 Coupling — Forbid-and-Rebuild

Filled independently, two children of a split can link back to **different** cells of the split — a (V3) break the forward sum cannot see (each `min_v D` places the split wherever is cheapest *for that branch alone*; summing such minima can put one vertex in two places at once — a phantom no matching realizes). The coupling closes this **during the build**, in the §4.0 layer order. It acts on **cells, never on vertices**, so it is identical in both modes and indifferent to how many parents a child has.

Each cell carries a **`forbidden`** flag — **role-aware**: once set, the cell can no longer be a **run END**, so every place a NEIGHBOUR row attaches to it (advance source, stall source) skips it; it remains a legal same-row **coverage source**, because a within-row cover step merely passes through the cell on the way to the run's real end. Per split `a`, children `a₁, a₂, …` (one layer):

1. **Build** each child's row with the recurrence, skipping forbidden cells.
2. **Forbid unusable exits.** As each child completes — *including the first* — mark forbidden every cell of `a` the child **cannot use**: no β-stall entry (the cell is not among the child's own candidates) *and* no advance entry (no B-arc from the cell into any of the child's candidates). Allowed exits = `∩ᵢ feasible(aᵢ)` (`_feasible_links`) — a **feasibility** intersection, never an optimality one. Which exits a child's *cheapest* row happens to link (`_links`, its optimal back-pointers) is no ground to kill an exit: the coupling is **pruning** for the §5 extraction, and the extraction optimizes over the survivors itself. *(The earlier rule — forbid whatever the child's optimal row did not link — could empty a split that had a valid warping: each child's cheap private entries forbade the one shared cell, monotonically and irrecoverably. Counterexample pinned in `tests/test_dag_couple_feasibility.py`.)*
3. **Rebuild whole rows.** A newly-forbidden cell is dead for **all** siblings, past and future; every earlier child whose row **linked** it re-runs its **entire row** under the current flags (the within-row (H) chains make a single-cell patch impossible) — the rebuild keeps back-pointers off dead cells; it no longer decides what dies.
4. **Iterate to the fixed point** — feasibility per (child, split) is static, so the forbidden set grows monotonically and settles in at most `|cand(a)|` rounds; rebuilds only restore row/bp consistency under the final flags.

At the fixed point **every surviving exit of every split is usable by all its children** — each surviving option admits a (V3)-valid attachment; keeping several options is legitimate (the extraction chooses among them). If a split's exits empty out, some child has **no entry from any exit cell** — no cell can serve as the split's **run end** inside the gate: raise the feasibility error (*increase `match_radius_m`*).

> **The flag is role-aware.** `forbidden` means "**not a valid run END**" — never "not a valid place". V3 binds only the cell a vertex's run *stops* on (§3: the rule is vacuous for a cell the vertex itself continues past), so a cell no child can follow may still be **required** as the split's run *entry/interior*, with the coverage chain walking on to a usable end (canonical case: continuity from the parent forces the run to enter at exactly that cell — pinned in `tests/test_dag_role_aware_forbid.py`). Enforcement is therefore split by role: a flagged cell takes **no attachment** (advance/stall source for a neighbour row, END-state row, sink seed) but remains a legal **coverage source** and reachability through-cell. Historical note: enforcing the flag in every role turned feasible instances infeasible and displaced optima (4/120 random tree cases, worst +24%, 1 spurious raise — under the old optimality rule identically, which forbade a strict superset).
>
> **Open (deeper, pre-existing): validity-blind contraction in §5.** `extract_cell` keeps only the *cheapest* row per pending-signature before the terminal `check_rules` judge; on **cyclic** targets a cheap-but-V1-invalid row can evict the valid row sharing its signature → spurious "no valid root row" or a displaced optimum (~2% of adversarial random cases; both enforcement variants; the role change nets 18+1 vs 22+3 per 900). Repro + dissections: `scripts/repro_contraction_eviction/`. Fix direction: validity-aware or top-K contraction. Related: the diagnostic engines (`extract_join`, two-table/reciprocity) still treat cover-through-flagged pointers as severed and disagree with the primary engine on the run-through family — diagnostic-only, but permanent until aligned.

Worked trace — split `a`, children `a₁, a₂`, `cand(a) = {v₁, v₂, v₃}`, `a₁` able to use `{v₁, v₂}`, `a₂` only `{v₁}`:

| round | event | forbidden | allowed |
|---|---|---|---|
| 1 | `a₁` can use `{v₁,v₂}` → forbid `v₃` | `{v₃}` | `{v₁,v₂}` |
| 1 | `a₂` can use `{v₁}` → forbid `v₂` | `{v₂,v₃}` | `{v₁}` |
| 2 | `a₁`'s row had **linked** `v₂` → rebuild `a₁`'s row (skips `v₂,v₃`) → its pointers re-route via `v₁` → fixed point | `{v₂,v₃}` | **`{v₁}`** |

**Invariant** (`check_split_exits(A, B)`): every surviving exit of every split is usable by every child, no child's row links a forbidden cell, survivors non-empty.

*(Implemented as `forward(A, B, α, β)` — the production forward pass, recurrence and coupling fused: its loop is `refill(a)` then `_couple(A, B, a, built, refill)` for every vertex in layer order, and it owns the `forbidden` flags (reset first, set as it goes). There is no uncoupled variant.)*

> **What the coupling does and does not settle.** The forbid is a **feasibility** intersection, so it removes only exits some child *cannot* use. Where several exits are usable by every child — the normal case — it keeps them all and each child's row independently links its own cheapest, so the forward table's own `bpD` trace **can place a split on two cells**: a V3 violation by design, reported by `check_forward_v3` and left for the extraction to resolve. Measured on the `map-conflation` hourglass: 0 violations on 10 synthetic cases and on lines `100042`/`100341`, but **2** on `102752` and **3** on `100350` — in every case a split with `outdeg = 2` whose 44–66 exits were all feasible for both children, so `check_split_exits` passes and the coupling has nothing to forbid. Choosing among *possible* exits needs the children's choices priced jointly — see `profiled_forward_table.md`.

---

## 5. The Extraction — the Cell-Level Join (`extract_cell`)

The extraction is a **join over tables built backward from the sinks**, not a search: §5.0 defines
the space it works in (the cell-move graph), §5.1 the tables (a dynamic program over that space),
§5.2 the backward pass that fills them, §5.3 the selection — just take the cheapest valid root row.
The implementation is the **per-cell sweep** specified in `docs/cell_dag_extraction.md` — the same
rows and ledger at cell granularity, with inbox-push freeing (peak memory = the sweep frontier),
implicit runs (no `run_cap`), and early discharge of merge pendings.
Everything is built from scratch upstream, out of `E` alone: the stored `D`/`bpD` propagation is
**never consulted** — it froze the very choices being optimized, and its values carry shared-cone
`1/outdeg` fractions; the §4 table serves only as **pruning**: a `forbidden` or `D = ∞` cell can
appear in no row. The cost is §3's `C(M)`, the judge §6's `check_rules`; exactness over the full
space — coverage runs included — is verified against full-space brute force (§10.2).

### 5.0 The Cell-Move Graph — the Space the Extraction Works In

Everything in §5 happens on one object: the **directed graph of cells**. Its nodes are §1's cells
`(a, v)`; its edges are the three legal moves (the same three the §4.1 recurrence uses):

| move | edge | meaning |
|---|---|---|
| **cover** | `(a, v) → (a, w)`, `w ∈ Bsucc(v)` | `a` extends its run one B-arc — stays inside the row |
| **stall** | `(p, v) → (c, v)`, `c ∈ Asucc(p)` | child `c` enters on its parent's cell (N:1) |
| **advance** | `(p, v) → (c, w)`, `c ∈ Asucc(p)`, `w ∈ Bsucc(v)` | child `c` enters one B-arc ahead (1:1) |

Two structural facts (both verified by `scripts/dag_cell_graph_probe.py`, across the scenario zoo,
segment mode, and 20 random polytrees over cyclic targets):

* **Graded by `A`.** Stall/advance edges strictly increase `A`'s topological layer; cover edges
  never leave their row. Hence **every directed cycle is confined to a single row** and projects to
  a directed cycle of `B` inside `cand(a)`.
* **DAG ⟺ acyclic rows.** The cell graph is a DAG **iff every row-induced target subgraph
  `B[cand(a)]` is acyclic**: always when `B` is a DAG; still true on a cyclic `B` whenever the
  radius gate cuts each cycle (a roundabout wider than `2r`); false when a whole B-cycle fits
  inside one gate (the §7 two-cycle). Where cycles exist they are harmless to the pass — runs are
  **simple** cover paths (§5.2), so the part ever walked is loop-free.

**A matching is an embedding of `A` into this graph**: each vertex maps to a **cover path** (its
run `M(a)`), and each A-edge `(p, c)` is realized by exactly **one stall/advance edge** from `p`'s
run *end* to `c`'s run *entry*. (V1)–(V4) say exactly that such an embedding exists and covers
every vertex; `C(M)` charges each cell by the edge that enters it (`1` advance, `β` stall, `α`
cover; a source's entry free). The extraction is therefore a **minimum-cost embedding** problem —
*not* a shortest path (a path cannot fork at a split, nor make two arms agree at a merge), which is
why §5.1–§5.3 are a dynamic program over per-vertex tables rather than a graph search.

The graph is kept **implicit** in the code — no cell `DiGraph` is materialized: the §5.2 pre-pass
walks its edges reversed (`_cell_reachable`), the cover recursion of the per-cell sweep walks its
cover edges one arc at a time, and the child-connection test is exactly "does a stall/advance edge
exist here" (`docs/cell_dag_extraction.md` §3).

**Not the back-pointer graph.** The stored `bpD` arrows also form a graph over these same cells —
always a DAG (§4.1) — but it is a smaller, different object: per cell, exactly the one incoming
arrow set the forward argmin committed to, every alternative discarded. The extraction searches the
full move graph precisely because that frozen history can miss the optimum (the §10.2 pinned
divergence case: the optimal coverage run is one the pointers never stored); engines that walk the
stored pointers are exact only over the stored-history family (§10.1).

### 5.1 The Data Structure — DP Tables of Rows

Plain Python, ephemeral, nothing on the graphs: `tables` is a local dict, one entry per A-vertex,
each a **list of 4-tuples**, discarded when the call returns (contrast §4, whose table persists on
`A`'s nodes). `networkx` is only *read* here — adjacency plus the §4 rows for `E` and the pruning
flags.

```python
tables[X] = [                # one DP table per A-vertex; rows sorted cheapest-first
    (entry,    # B-vertex id -- the cell of X where its parent will connect
     value,    # float       -- cost of everything at/below X, per the §5.2 ledger
     pending,  # dict        -- {(merge-vertex, its entry-cell): stall-flag}
     cells),   # dict        -- {A-vertex: its run, an ordered tuple of B-vertices}
    ...
]
```

One row = one **fully-decided embedding of `X`'s downstream cone** into the §5.0 graph, summarized
by what the rest of the algorithm still needs. Read each field as the answer to a question:

* **`entry` — where does upstream connect?** The single cell of `X` exposed to `X`'s parent (the
  start of `X`'s run). Everything else about the cone is internal and already decided.
* **`value` — what does the cone cost?** Every cell at/below `X` paid per the ledger, **except**
  two deferred items: `X`'s own entry-`E` (whether it costs `1` or `β` is the *parent's* choice of
  edge, so the parent pays it on connection) and the entries listed in `pending`.
* **`pending` — what is still unresolved?** One key per merge below that this line has touched:
  the merge's chosen entry cell (other parent arms **must** present the same cell, or the
  combination dies) and a flag saying whether some arm stalled onto it (decides `1` vs `β` when it
  is finally paid, at the root join).
* **`cells` — the answer so far.** The embedding itself: every decided vertex mapped to its run
  (`run[0]` is its committed entry). **`M` travels with the row** — the winning row's `cells` map
  *is* the relation; no traceback, no reconstruction, no gap-fill.

**The invariant:** after a vertex is processed, its table holds — per `(entry, frozenset(pending))`
key — the **cheapest** embedding of its cone with that interface. Two rows agreeing on the
interface are interchangeable to everything upstream, so only the cheaper survives. This
**contraction** is the polynomial bound — it is what keeps the join a join rather than a
cross-product of subtree choices.

### 5.2 The Backward Pass — Building the Tables

A **reverse-topological sweep of `A`** — sinks first, sources last, every child's table complete
before its parent's is built. (Unrelated to the §6b diagnostic `backward()` table, which mirrors
`D` on the graph; this pass builds §5.1's row tables.)

**Cell removal (pre-pass)**: one reverse search from **all sink cells** over the cell-move graph
(cover reversed inside a vertex, advance/stall reversed across edges); every cell never seen is
removed, in every role; runs may not cover removed cells. A vertex with no surviving cell ⇒
immediate, precisely located infeasibility. The `D < ∞` filter prunes the upstream mirror.

**The E-multiplier ledger** — every cell's emission enters exactly once, weighted by the move that
enters the cell: **1** (source free entry / advance), **β** (stall — some parent arm holds the same
cell), **α** (cover — a run cell). No fractional factors anywhere: plain sums at split cells,
consumed-once at merges. **Deferred entry**: whether a vertex's entry cell pays `1` or `β` is the
parent's call, so its entry-`E` is paid at the step connecting it to its parent — at a merge, at
the root join, `β` if **any** arm stalls.

**The sweep step**, per vertex `X`: for each surviving **entry** cell `e` and each **cover run**
`R = e→…→u` inside `cand(X)` (simple directed B-paths over surviving cells; the implementation
never enumerates them — runs grow one arc per cover edge, uncapped):

1. start the row at `Σ α·E` over `R`'s covered cells (`e`'s own `E` stays unpaid — deferred);
2. **connect every child at the run's end `u`**: a child row may connect iff its entry `ce`
   satisfies `ce == u` (**stall**) or `ce ∈ Bsucc(u)` (**advance**) — i.e. iff a §5.0 stall/advance
   edge `(X, u) → (c, ce)` exists — nothing else;
3. **pay the child's deferred entry now** — `β·E(c, ce)` on a stall, `1·E(c, ce)` on an advance —
   unless the child is a merge, whose entry defers again (*SPLIT of data* below);
4. cross-combine the children's options: pendings must union without conflict (the same
   merge-vertex with two different entry cells kills the combination; stall flags OR), cells union;
5. **contract** the finished rows per `(entry, pending)`; `max_rows` **raises**, never truncates.

The step is one and the same at every vertex; what varies is the child structure — three shapes:

* **PASS** (chain vertex, one child): the step verbatim.
* **MERGE of data** (split `X`, children `c₁ … c_k`): all `k` children connect at the **same** run
  end — per-cell (V3) by construction. Values sum, cells union (disjoint subtrees), pendings
  reconcile.
* **SPLIT of data** (merge `m` serving several parent arms): **consumed-once** — the first arm to
  reach `m` absorbs its table (value and cells) but leaves `m`'s entry-`E` unpaid, gaining
  `pending[(m, entry)] = did-this-arm-stall`; every other arm receives only an *interface* (entry
  cells, value 0, the same pending key with its own flag). Nothing is counted twice — which is why
  no `1/indeg` exists. This is also why the engine carries **no tree dependence** (§2): on a
  reconvergent source (a diamond) the arms' pendings must agree when they meet at the shared
  ancestor's join — the same mechanism, resolving earlier than at the root.

**Worked trace** — the split scenario (`A: 0→1, 1→{2,3}` over `B: s→j, j→{u,d}`, each B-vertex
0.5 m off its A-twin, so every aligned `E = 0.5`), swept in reverse topological order `2, 3, 1, 0`:

```text
tables[2] = [(u, 0, {}, {2:(u,)}), …]     a sink: one row per entry — nothing below, nothing paid yet
tables[3] = [(d, 0, {}, {3:(d,)}), …]
tables[1]:  entry j, run (j,) → run end is j;  child 2 connects by ADVANCE j→u: pay 1·E(2,u),
            child 3 by ADVANCE j→d: pay 1·E(3,d)   (a split: both children at the SAME run end)
            → (j, 1.0, {}, {1:(j,), 2:(u,), 3:(d,)})                      # 0.5 + 0.5
tables[0]:  entry s, run (s,);  child 1 connects by ADVANCE s→j: pay 1·E(1,j)
            → (s, 1.5, {}, {0:(s,), 1:(j,), 2:(u,), 3:(d,)})
root join:  0 is the only source → pay its own entry E(0,s)  →  value 2.0 = C(M) exactly
selection:  the row is valid (V1–V4) → M = its cells map; committed = the runs' first cells
```

(Each table also holds the costlier rows for the other entries/runs — dropped here for space; the
contraction already discarded everything dominated per interface. A **merge** below would differ in
one way: its entry would not be paid at connection but appear as a `pending` key, settled once at
the root join — the *SPLIT of data* bullet above.)

**Feasibility is located, loud**: a vertex with no surviving cell (pre-pass ∩ `D < ∞`) or no
feasible row raises `ValueError` naming the vertex — increase `match_radius_m`.

### 5.3 The Selection — Cheapest Valid Root Row

The backward pass ends with one table per **source**; each source pays its own entry-`E` (full —
free entry). The **root join** folds the root tables into single whole-graph rows — pendings must
agree on every merge's entry cell (flags OR), contracting per pending-key after each fold — then
**pays every deferred merge entry once** (`β` if any arm stalled). The joined `value` now equals
`C(M)` exactly, self-checked.

Selection is exactly that — select the best of the backward pass's result: rows are tried
**cheapest-first**; the first whose `cells`-map relation passes `check_rules` (the judge, §6) is
returned — `M` is read straight off the row, `committed[a] = run[0]`. Rows failing (V1)–(V4) are
skipped; if none survives, the feasibility `ValueError` is raised. The judge's word is final: the
extraction never returns an invalid matching.

## 6. Verification & Diagnostics

`check_rules(M, src, tgt)` — **the judge**: V1–V4 on the final relation, on the same graph the match lives on (`A,B` or `L(A),L(B)`), each rule restricted to neighbours present in `M`. Everything else below is tooling that verifies the machinery, not part of matching:

* **`validate_tables`** — replays every finite cell's back-pointer chain and checks it is a legal partial warping in isolation.
* **6b — the backward table and cross-table agreement.** `backward()` fills the mirror table `B[a][v]` (downstream cone: `pred→succ`, `Bpred→Bsucc`, `outdeg→indeg`, reverse sweep, same `α/β/E`) and `extract_two_table` is the older two-table traceback over `bpD`+`bpB` (seed at a joint `D+B−E` argmin, flood both pointer sets, pivot-path coverage gap-fill). Both are kept **as diagnostics**: `check_reciprocity` demands that every source edge the forward pointers thread, the backward pointers thread back identically — on the committed matching only (off the optimum the two tables optimise differently-pinned subproblems and legitimately disagree). If the coupled pass runs, run it **before** `backward` — the backward pass respects the `forbidden` flags, so its pointers never target dead cells.
* **6c — reachability.** Each table's back-pointers must reconstruct the tree's own source↔sink structure: walking `bpD` from every finite sink cell (branching at every entry) must reach exactly the sink's ancestor sources; mirror for `bpB`. Empirically clean across every sweep, including under the V3 coupling — table-level structure is never the failure site.
* **6d — complementarity.** Read alone, the *uncoupled* forward table can violate V3 (it is split-optimistic) and the backward table V2 — `check_forward` / `check_backward_v2` surface exactly where. Under weighting they fire by design; this is what motivated the §4.1a coupling, which closes the forward table's half (`check_split_exits` is its invariant).

## 7. Guarantees & Limits

* **Valid by construction, where construction reaches**: (V2) is enforced by the forward sum, (V3) by the §4.1a coupling, (V4) by full-coverage extraction; every candidate `M` is costed directly on the relation.
* **Exact over the full cell-level space** — the root join's value equals `C(M)` by the §5.2 ledger, verified to the digit against full-space brute force on trees and reconvergent DAGs (§10.2). On DAG sources the claim is `extract_cell`'s alone — the vertex join consumes `D` values, whose exactness argument is tree-only (§2, §10.1), and the coupling's forbidden-pruning is empirically safe there but theoretically unproven. The two-table traceback's old "exact optimum" claim was disproven (its `D+B−E` seed is unsound at a merge) — it survives only as the §6b diagnostic.
* **Feasibility, never silent breakage**: an unreachable vertex, an emptied split, a vertex with no surviving cell, or an all-invalid root join raises `ValueError` telling you to increase `match_radius_m`.
* **Validator limit on cyclic B**: on a 2-cycle `p⇄q` the local (V1) predicate cannot orient a step — the *smallest* case is a 2-vertex chain over `p⇄q`, where the geometrically correct matching (and every separating alternative, in both directions) is flagged. This is a property of the predicate, not of the matcher. On that smallest case the cell join **refuses loudly** (its root contraction leaves only the flagged row); the vertex join returns the valid stall, so `engine="all"` still succeeds — pinned by `test_two_cycle_judge_prefers_valid`.
* **The extraction never returns an invalid matching**: rows violating (V1)–(V4) are skipped by the judge, so the result is valid-by-check or the call raises (no-valid-row / `max_rows` — both explicit). Structured point-mode envelope: 384/384 valid returns, 0 invalid outputs (`scripts/test_dag_point.py`).
* **Complexity**: table `O(|A| × band)` plus the per-row (H) relaxation and the (rare, bounded) §4.1a rebuilds; extraction: per-cell states with the pending-signature contraction bounding each table (`max_rows` raises, never truncates; runs implicit, no cap; merge pendings discharged at the arms' first common ancestor, so chained merges stay linear); peak memory is the sweep frontier. All linear-ish in practice.

## 8. Segment Mode — the Same Algorithm on the Line Graphs

A point-mode *state* is a vertex pair; dressing its cost with a heading term does not make it segment matching — on an N:1 stall there is no target segment and the heading is silently free. True segment matching makes the **state a segment pair**: run the identical algorithm on `L(A) = line_digraph(A)` vs `L(B) = line_digraph(B)`, whose nodes are the directed arcs, each carrying its **midpoint** as `x, y` and its **bearing**:

| point (§1–§7) | segment |
|---|---|
| vertex `a` / `v` | arc `s = (t→h)` / `e = (u→v)` |
| `E = ‖a − v‖` | `E = ‖mid(s) − mid(e)‖ + λ·circ(bear(s), bear(e))` |
| `Apred/Bpred` | arc adjacency in `L(A)` / `L(B)` |

Everything — layering, the recurrence, the coupling, the extraction, `check_rules` — is byte-for-byte the point-mode code on the lifted graphs; every state pays its emission (a stall costs `βE`, never zero — the heading cannot be bypassed). A junction is a vertex, so the line graph connects real segments directly; there are no stitch connectors to park on. A merge+split vertex becomes a bipartite cluster in `L(A)` — a reconvergent DAG, accepted natively (both engines return valid, agreeing matchings on it; only the cell engine carries the exactness claim there, §7). The matching is emitted and validated **on the arcs** (`M_seg ⊆ E(A) × E(B)`); any per-point view is derived convenience, never validated in its place.

## 9. Implementation Notes (Parts)

* **Part 1 — cells.** `prepare(A, B, r, k_min, bearing_weight)` validates inputs, gates candidates by KD-tree, and stores each vertex's row on the node in the §4 layout — only `E` is filled; `D/bpD` are placeholders for the forward pass, `B/bpB` for the diagnostic backward pass.
* **Part 1.3 — feasibility.** `r` must cover the largest A↔B drift (NVDB↔OSM: 10–20 m). A gate-severed chain leaves rows all-`∞`; a coupled infeasibility shows up as a severed (`None`) pointer; both raise `ValueError`, never a broken match.
* **Function map.**

  | algorithm step | function |
  |---|---|
  | candidates & cells (§1) | `prepare` |
  | vertex order (§4.0) | `layer_order` |
  | forward pass incl. V3 coupling (§4.1–§4.1a) | `forward` |
  | extraction (§5) | `extract_cell` — the cell-level join (per-cell sweep, `docs/cell_dag_extraction.md`) |
  | cross-validation engine (§10) | `extract_join` — the vertex-level join |
  | one-call pipeline | `match_dag(A, B, r, α, β, mode, engine)` — `engine="cell"` (default) · `"join"` · `"all"` = cheapest valid of the two |
  | DuckDB pipeline | `DuckDBMapMatcher.match_dag(alpha, beta, engine, bearing_weight, …)` — Mode-1/2 input system (WKT CSV / geofiles / DuckDB tables, lon-lat → UTM), converts to `networkx` via `edges_to_digraph` (densify + junction-snap), runs segment mode, returns `(dag_long, dag_summary)` DataFrames |
  | geometry → graph | `edges_to_digraph(edges, step_meters, snap_decimals)` — densified polylines (supplies the §1 subdivision), junction-snapped; arcs carry `road_id` + `seq` |
  | judge (§6) | `check_rules` |
  | diagnostics (§6) | `backward`, `extract_two_table`, `validate_tables`, `check_reciprocity`, `check_reachability`, `check_forward`, `check_backward_v2`, `check_split_exits` |
  | segment lift (§8) | `line_digraph` |

* **Playground** — `notebooks/dag_dtw_playground.ipynb` (interactive Plotly scenarios, the historical failure demos and their fixes). **Cross-validation sweep** — `scripts/test_dag_point.py` (both engines over structure × density × shift × noise × weights). **Cell-graph probe** — `scripts/dag_cell_graph_probe.py` (materializes the §5.0 graph; verifies the grading, DAG ⟺ acyclic-rows, and that every returned `M` embeds).

---

## 10. Cross-Validation — the Vertex-Level Join (`extract_join`)

`extract_join` is kept **only to validate and cross-validate** the §5 extraction — it is never the
engine of record. It shares §3's cost and §6's judge, but joins at vertex resolution, which caps
its exactness (§10.1); the standing results it anchors are in §10.2.

### 10.1 The vertex-level join

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
join is exact over the **stored-history family** only. Divergences from §5's cell engine occur
exactly in dense-target coverage regimes, at any weights — the reason it is a cross-check, not the
extraction. It also consumes `D` values, whose exactness argument is tree-only, so on reconvergent
DAG sources it is a judged cross-check even within its family (§7).

### 10.2 The standing cross-validation

| check | result |
|---|---|
| tiny dense-B cases vs **full-space** brute force (all entry+run combinations), point **and** segment (bearing active) | cell join equal to the digit |
| random mini polytrees vs full-space brute | cell join exact & valid, every case |
| **reconvergent DAG sources** — 75 jittered diamonds × 3 weights + 120 random subdivided reconvergent DAGs vs full-space brute | cell join **195/195 exact**, all valid |
| structured 384-case envelope (`scripts/test_dag_point.py`, both engines) | vertex join 379/384 · **cell join 384/384** valid |
| invariant `C(cell) ≤ C(vertex join)` whenever both succeed | pinned in the suite: a violation is an exactness bug by definition |
| pinned divergence case | cell strictly beats the vertex join *and* equals the full-space optimum |

The engines cross-validate: `match_dag(engine="all")` runs both and returns the cheapest valid
`M`; `value == C(M)` is self-checked.

## 11. Edge-Level Outputs — `dag_long`, `dag_summary`, `dag_parts`

The DuckDB pipeline (`DuckDBMapMatcher.match_dag`) aggregates the arc-level matching `M` back to
**input-edge level**. Three tables:

- **`dag_long`** — one row per (A-edge, B-edge) the matching connects: `source_id, dest_id, seq,
  n_pairs, avg_dist_m, avg_bearing_diff`. Note this grain *collapses re-entry*: if a route leaves
  a B-edge and later returns to it, both visits land in one row.
- **`dag_summary`** — one row per A-edge: `source_id, dest_ids, n_dest, n_parts, n_pairs,
  avg_dist_m, avg_bearing_diff, match_type` (`1:1` / `1:N_ROUTE`).
- **`dag_parts`** (returned when `parts=True`) — the full per-edge decomposition, one row per
  **part**.

### 11.1 Parts

A **part** is a maximal run of consecutive A-arcs (by `seq` along the A-edge) matched to the
same B-edge. Parts are ordered along the A-edge (`part` = 1, 2, …); the same `dest_id` may appear
in more than one part (re-entry). When a single A-arc matches two B-edges (an advance across a
B-junction mid-arc), the arc closes the current part and opens the next — within one arc, pairs
are ordered by walking B's chain, continuing the previous part's B-edge first (ordering by
`dest_id` would compare *ids*, not geometry: `"10" < "2"`).

**The rows partition the A-edge.** `a_from_m`/`a_to_m` are contiguous and non-overlapping across
an edge's rows, and span it end to end — so `Σ a_len_m` is the A-edge's length and `Σ a_pct` is
100. An arc whose pairs straddle two parts is attributed wholly to the **earlier** one (its
pairs still count in both parts' `n_pairs` and scores; only the *span* is assigned once).

**Non-overlap at the route's begin/end (`part_type`).** Every A-arc is matched by (V4), so an
A-edge that starts before or ends after what B covers cannot show a *gap* — instead its terminal
arcs **stall**: several consecutive A-arcs at the edge's begin/end all match the same single
B-arc (B does not advance). The decomposition detects these terminal stall runs and emits the
surplus arcs (all but the geometrically true one) as their own rows with `part_type = "head"` /
`"tail"`; the matched parts between them are `part_type = "match"` and their scores exclude the
overhang pairs. On A-edges interior to a larger source DAG, a `head`/`tail` row means B did not
advance across the edge boundary (an N:1 compression) rather than a true network overhang.

| column | meaning |
|---|---|
| `source_id`, `part`, `part_type`, `dest_id` | A-edge, 1-based part order along it, `match`/`head`/`tail`, matched B-edge |
| `a_from_m`, `a_to_m`, `a_len_m`, `a_pct` | span of the A-edge covered by this part (meters along A / % of A's length) |
| `n_pairs` | matched (A-arc, B-arc) pairs in the part |
| `n_a_arcs`, `n_b_arcs` | distinct arcs on each side |
| `drift_m`, `drift_max_m` | mean / max midpoint distance over the part's pairs (pure geometry — the `bearing_weight` term is never mixed in) |
| `bearing_diff_deg` | mean circular bearing difference over the part's pairs |
| `b_from_m`, `b_to_m`, `b_len_m` | used span along the B-edge, and the B-edge's total length |
| `b_head_m`, `b_tail_m` | the **non-overlapping** begin/end of the B-edge: `b_from_m` and `b_len_m − b_to_m` — the parts of B before/after the used span |

### 11.2 Reading the table

- **A-side non-overlap** is in the `head`/`tail` rows (and summarized per edge as
  `dag_summary.a_head_m`/`a_tail_m`); **B-side non-overlap** is in `b_head_m`/`b_tail_m` — for
  the route as a whole, the B network before the entry point is the first part's `b_head_m` and
  the B network after the exit is the last part's `b_tail_m`.
- **Whole-edge scores are yours to compose.** The natural aggregate over the `match` parts is
  the length-weighted mean, e.g. `edge_drift = Σ(a_len_m · drift_m) / Σ(a_len_m)` and likewise
  for `bearing_diff_deg`; or gate per part first (`drift_m ≤ τ`) and score
  `matched_pct = Σ(a_len_m | pass) / A-length`, discounting by `a_head_m + a_tail_m`.
- **Consistency:** `Σ n_pairs` over an edge's rows (all `part_type`s) equals
  `dag_summary.n_pairs`; `dag_summary.n_parts` counts the `match` rows; `Σ a_len_m` is the
  A-edge's length (§11.1). An edge whose entire matching is a single stall run — B never advances
  at all — is left as one `match` part (there is no interior to separate a head/tail from).
- **Arc identity.** `edges_to_digraph` numbers arcs per input edge id, gapless, **continuing
  across rows that share an id** (a multipart geometry exported as several rows), so
  `(road_id, seq)` is unique. Hand-built line graphs violating that raise `ValueError` rather
  than silently corrupting spans.

The standalone helper is `dag_dtw.parts_from_matching(M, LA, LB)` (segment mode only): it
requires the line-graph nodes to carry `road_id`/`seq` (attached by the pipeline) and `length`
(attached by `line_digraph`), and returns the rows as plain dicts.
