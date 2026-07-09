# Tree-DTW on networkx — the rebuild

A rebuild of the matcher (see `docs/tree_dtw_matching.md` for the algorithm core — objective, the four
rules V1–V4, the forward `D` / backward `B` tables, the joint extraction) on top of **networkx**. What
changes is the *representation and storage*, not the mathematics:

| aspect | old (`tree_dtw.py`, `LocalBGraph`) | rebuild (this doc, `tree_dtw_nx.py`) |
|---|---|---|
| source A / target B | custom `LocalBGraph` built from `[(road, coords)]` edge-lists | plain **`networkx.DiGraph`** passed in directly |
| a road/edge id | `vert_edge` on every vertex, `edge_ids` | *(none — a graph is only vertices + edges)*; optional `road_id` edge attribute for caller-side grouping |
| a junction | several **coincident** per-road vertices joined by zero-length **stitch** edges | **one vertex** (split = out-degree > 1, merge = in-degree > 1) |
| candidates for a vertex | **all** B-vertices (dense `|V(A)|×|V(B)|` table) | **radius-gated**: only B-vertices within `r` (`match_radius_m`, default 20 m) |
| where the DP table lives | dense numpy arrays `D`, `B`, `bp` | **on the node** — each A-vertex stores its own candidate table |
| segment mode | hand-built arc line-graph, stitches contracted | **`nx.line_graph`** (verified to give exactly the directed line graph) |
| validation | `check_tree_rules` on the point `M` (also used for segment — the bug) | V1–V4 on the **graph** (point) or **line-graph** (segment); no point conversion |

The build is done **part by part**, each independently verifiable: (1) representation + candidates,
(2) emission `E`, (3) forward `D`, (4) backward `B`, (5) extraction, (6) validation.

---

## The optimization problem

A **matching** is a relation `M ⊆ V(A) × V(B)` (point mode) — or `M ⊆ E(A) × E(B)`, i.e.
`V(L(A)) × V(L(B))`, in **segment** mode. `(a, v) ∈ M` means source vertex `a` is matched to target
vertex `v`; write `M(a) = { v : (a,v) ∈ M }` (a singleton in the 1:1 case, a run under 1:N coverage).
The matcher solves

$$
\text{minimize}\quad C(M) \;=\; \sum_{(a,v)\in M} w(a,v)\,E(a,v)
\qquad\text{subject to } M \text{ a valid warping (V1)–(V4),}
$$

where `E(a,v)` is the emission (Part 2) and the per-move weight `w` is `1` on a 1:1 advance, `α` on a
1:N coverage step, `β` on an N:1 stall (`α = horizontal_weight`, `β = vertical_weight`; `α = β = 1`
gives plain `Σ E`). **Candidate gating** (§1.2) sets `E(a,v) = ∞` for `‖a − v‖ > r`, so the sum ranges
only over gated pairs. Because A is a tree, the exact minimum is reached by one forward pass (`D`), one
backward pass (`B`), and one traceback (Parts 3–5) — no search over the exponentially many warpings.

## Validity — the four rules (V1)–(V4)

Neighbour sets are plain graph adjacency: `pred(a) = A.predecessors(a)`, `succ(a) = A.successors(a)`,
`Bpred(v) = B.predecessors(v)`, `Bsucc(v) = B.successors(v)`. **Segment mode uses the identical
formulas with `A, B` replaced by `L(A), L(B)`.** `M` is a **valid warping** iff:

**(V1) Monotonicity — no cross.**
```
∀ (a,v) ∈ M,  ∀ a⁻ ∈ pred(a),  ∀ v⁺ ∈ Bsucc(v) :    (a⁻, v⁺) ∉ M
```
You may not match `a` to the earlier `v` while a predecessor `a⁻` sits on the later `v⁺` — that runs backward.

**(V2) Merge — every cell is fed.**
```
∀ (a,v) ∈ M :
    [ ∃ v⁻ ∈ Bpred(v) : (a, v⁻) ∈ M ]                                              (i)  continues a's run
  ∨ [ ∀ a⁻ ∈ pred(a) : ( (a⁻, v) ∈ M ) ∨ ( ∃ v⁻ ∈ Bpred(v) : (a⁻, v⁻) ∈ M ) ]     (ii) every predecessor feeds it
```
The `∀ a⁻` bites at a **merge**: all incoming arcs must arrive at the same `v`. A source (`pred(a)=∅`) satisfies (ii) vacuously.

**(V3) Split — every cell continues.** The mirror, over successors:
```
∀ (a,v) ∈ M :
    [ ∃ v⁺ ∈ Bsucc(v) : (a, v⁺) ∈ M ]                                              (i)  continues a's run
  ∨ [ ∀ a⁺ ∈ succ(a) : ( (a⁺, v) ∈ M ) ∨ ( ∃ v⁺ ∈ Bsucc(v) : (a⁺, v⁺) ∈ M ) ]     (ii) every successor carries it on
```
The `∀ a⁺` bites at a **split**: all exits must leave from the same `v`. A sink (`succ(a)=∅`) satisfies (ii) vacuously.

**(V4) Full coverage.**
```
∀ a ∈ V(A) :  M(a) ≠ ∅
```
Nothing is left unmatched; a source's entry and a sink's exit are **free** (may land mid-edge); every interior junction is **pinned**.

(V1) forbids a backward step; (V2) keeps a merge from being entered at two points; (V3) keeps a split
from being left at two points; (V4) forces full coverage. On a tree all four are simultaneously and
exactly satisfiable. `check_rules` (Part 6) tests exactly these, evaluating each `∀`/`∃` **restricted to
neighbours that appear in `M`** so that an unmatched branch (a target vertex no source covers, allowed
by V4) does not false-fire.

---

## Part 1 — Representation & candidate gating

### 1.1 Input graphs

* **A** (source tree) and **B** (target network) are `networkx.DiGraph`.
* Every node carries float coordinates in attributes **`x`, `y`**.
* Edges are the directed **segments**; a node's coordinates are its geometry (segments are straight
  between endpoints). Node ids may be any hashable value.
* **A must be a tree**: its underlying *undirected* graph is acyclic (`nx.is_forest(A.to_undirected())`).
  A directed reconvergence (a diamond) has an undirected cycle and is rejected — `NotATree` — because it
  breaks the junction-independence the algorithm relies on (main spec §2, §7). B has no such restriction.
* A **junction is a vertex**: `out_degree > 1` is a split, `in_degree > 1` is a merge, both at once is a
  merge+split. Nothing is "coincident"; there are no stitches.

### 1.2 Candidates — radius-gated, stored on the node

For an A-vertex `a`, a **candidate** is a B-vertex `v` it may match to. We gate by distance:

* Parameter **`r = match_radius_m`** (default **20 m**).
* `cand(a) = { v ∈ V(B) : ‖a − v‖ ≤ r }`, found with a KD-tree over B's vertex coordinates.
* **Non-empty guarantee (feasibility fallback).** If fewer than `k_min` (default 1) B-vertices lie
  within `r`, include the `k_min` nearest anyway, so no row is ever empty for a purely-geometric reason.

Each A-vertex stores its **own candidate table** as a node attribute — this *is* that vertex's row of
the DP tables, filled progressively by later parts:

```python
A.nodes[a]["cand"] = {
    v: {"E": ‖a − v‖,          # emission (part 2)
        "D": +inf, "bpD": [],  # forward cost + back-pointers (part 3)
        "B": +inf, "bpB": []}  # backward cost + back-pointers (part 4)
    for v in cand(a)
}
```

Part 1 fills only `E` (the geometric distance); `D/bpD/B/bpB` are placeholders until parts 3–4.

### 1.3 Feasibility rule

Radius gating can make the *coupled* DP infeasible even when every row is individually non-empty: the
warping is a chain, and `D[a][v]` needs a predecessor candidate in its reach `{v} ∪ B_pred(v)`. If `r`
is smaller than the true A↔B drift somewhere along a path, that chain breaks and every cell of some
vertex becomes `∞`. So:

* `r` should be **≥ the largest expected A↔B drift** (synthetic tests: < 2 m; NVDB↔OSM: 10–20 m).
* After the DP (parts 3–5), if any A-vertex has **no finite** `D+B` entry, **raise**
  `ValueError("vertex … unreachable within r=…; increase match_radius_m")` — never return a broken match.
* The per-vertex check above is **not sufficient at a merge/split**, whose arms are only coupled during
  the traceback: a vertex can have a finite `D+B` at its arg-min yet the coupled optimum still runs
  through an infeasible cell, recorded as a **severed back-pointer** (a `None` cell reference). The
  extraction (Part 5) guards this — a `None` while following `bpD`/`bpB` raises the same feasibility
  `ValueError` rather than dereferencing the missing cell (which previously crashed with `KeyError`).

### 1.4 What Part 1 delivers

`prepare(A, B, r)`: validate the inputs (both `DiGraph`; every node has `x,y`; `A` is a forest), build
the KD-tree on B, and populate `A.nodes[a]["cand"]` with the gated candidate set and its emission `E`.
Verified by printing the candidate table for each vertex on a chain, a split, and a merge.

---

## Part 2 — Emission `E` and the node-attribute contract

Both modes run the **same** DP on a graph whose nodes carry a **position** and — for segment mode — a
**bearing**. A segment is just a point-mode node *plus a bearing*:

| mode | graph | a node is… | node attributes it must carry |
|---|---|---|---|
| **point** | `A`, `B` | a vertex | `x, y` (position, meters) |
| **segment** | `L(A)`, `L(B)` | a segment `(u, v)` | `x, y` = **segment midpoint**, `bearing` = **segment compass bearing** |

One emission formula serves both, using whatever the nodes carry:

```
E(a, v) = ‖pos(a) − pos(v)‖  +  λ · circ(bearing(a), bearing(v))     ← 2nd term only if both carry `bearing`
```

* `pos(n) = (x, y)`, `‖·‖` Euclidean (meters).
* `bearing` = compass bearing `(deg·atan2(Δx, Δy) + 360) mod 360`, 0° = north;
  `circ(θ, φ) = min(|θ − φ|, 360 − |θ − φ|) ∈ [0, 180]`; `λ = bearing_weight`.
* **Point mode** — nodes carry only `x, y`, so `E = ‖a − v‖` (no bearing term); this is what Part 1
  already fills.
* **Segment mode** — each L-node carries `x, y` = midpoint and `bearing`, so
  `E = ‖mid − mid‖ + λ · circ(bearing)`.
* **Gating uses `pos`** (§1.2): a candidate's `pos` within `r` of the source's `pos` — i.e.
  midpoint-to-midpoint distance in segment mode.
* `E` is symmetric under reversing both graphs (every bearing rotates 180°, `circ` unchanged), so the
  forward and backward passes share it.

Stored in `A.nodes[a]["cand"][v]["E"]` (point) / `L(A).nodes[s]["cand"][e]["E"]` (segment).

---

## Part 3 — Forward table `D` (upstream cost) — on the node

`D[a][v]` = minimum cost of matching the **upstream cone** of `a` (a and its ancestors) with `a` pinned
at candidate `v`; `bpD[a][v]` records the cells it was computed from. Both live in
`A.nodes[a]["cand"][v]`. A is swept in **topological order** (sources first). For each predecessor
`p ∈ pred(a)` write its two ways into `v` (both read from `p`'s gated candidate table):

```
step_p  = min_{x ∈ Bpred(v) ∩ cand(p)} D[p][x]      # p advances one B-arc into v
stall_p = D[p][v]   (∞ if v ∉ cand(p))              # p already sits on v
```

`D[a][v]` is then the cheapest of a full-cost advance (D), a **β**-discounted stall (V), and an
**α**-discounted coverage step (H):

```
D[a][v] = min {
  (D)  E(a,v) + Σ_{p∈pred(a)} step_p / outdeg(p)                                                   full E
  (V)  β·E(a,v) + min_{q∈pred(a)} [ stall_q/outdeg(q) + Σ_{p≠q} min(stall_p, step_p)/outdeg(p) ]    β·E
  (H)  α·E(a,v) + min_{v'∈ Bpred(v) ∩ cand(a)} D[a][v']                                             α·E
}
```

* **α = `horizontal_weight`** discounts a **1:N coverage** step — one source vertex spanning a B-run (line H).
* **β = `vertical_weight`** discounts an **N:1 stall** — an extra source vertex stacking on the same `v`
  (line V). The inner `min_q` **forces at least one predecessor to stall**, so β is never taken on a pure
  advance ((D) already covers that at full `E`); `a`'s emission is discounted **once**, however many
  branches stack (main §4.1).
* `α = β = 1` (defaults) is **bit-for-bit** the unweighted point cost; use `α, β ∈ (0, 1]`.
* **Source** (`pred(a)=∅`): empty sum → `D[a][v] = E(a,v)` (free entry).
* **Merge** (`|pred(a)| > 1`): the predecessor **sum** is finite only if every predecessor can reach `v`
  — that is (V2), folded into the cost; `1/outdeg(p)` splits a shared point's cost once across its
  branches (main §4.1).

**Back-pointer `bpD[a][v]`** records exactly the cells that produced the winning line, so the move type
is read back with no separate tag:

| winning line | `bpD[a][v]` | reads as |
|---|---|---|
| source | `[]` | free entry, nothing before it |
| (D)/(V) advance-or-stall | `[(p, x_p), …]` — the chosen cell per predecessor (`x_p = v` if it stalled) | every predecessor feeds `v` |
| (H) coverage | `[(a, v')]` — **same** source `a`, one B-arc before `v` | `a` extends its run onto `v` |

All predecessor / B-neighbour look-ups are **intersected with the gated candidate sets**: a candidate
pruned by `r` has `D = ∞`, so it can't be a `step`, `stall`, or coverage source; a chain the gate has
severed leaves every cell `∞` and trips the feasibility rule (§1.3). Line (H) is a within-row
shortest-path relaxation over B restricted to `cand(a)` — topological when B is acyclic, Dijkstra otherwise.

---

## Part 4 — Backward table `B` (downstream cost) — on the node

`B[a][v]` mirrors Part 3 over the **downstream cone** (a and its descendants): the identical three-way
`min` — same **α** (coverage) and **β** (stall) weights, same emission `E` — with A and B **reversed**:
`pred → succ`, `Bpred → Bsucc`, `outdeg → indeg`, swept in reverse topological order. Concretely
`step_p = min_{x ∈ Bsucc(v) ∩ cand(p)} B[p][x]`, `stall_p = B[p][v]`, over `p ∈ succ(a)`. Stored in
`A.nodes[a]["cand"][v]` as `B`, `bpB`. This is the split coupling (V3) exactly as `D` is the merge
coupling (V2) (main §4.2, §6).

---

## Part 5 — Extraction → `M`

The joint traceback (main spec §5), reading only the stored back-pointers.

1. **Seed.** Pick any still-uncommitted A-vertex `r` and its
   `v* = argmin_{v ∈ cand(r)} ( D[r][v] + B[r][v] − E(r,v) )` (feasibility rule §1.3 if none is finite).
   Commit `r → v*`. This is the **only** arg-min in the whole extraction.
2. **Flood via back-pointers.** From a committed `(c, v)` walk the coverage run, then commit each
   predecessor in `bpD[c][·]` and each successor in `bpB[c][·]`, repeating until the queue drains.
   Following predecessors (up), successors (down), and siblings (down from a shared parent) reaches the
   seed's **entire weakly-connected component** — and since a connected tree *is* one component, a single
   seed commits **every** vertex. The flood never crosses into a disconnected part (no back-pointer spans
   the gap). **Coverage is read from the forward COVER chain only** (never both chains — that was the old
   over-assignment) — but a 1:N run can be recorded on the *backward* chain instead, which the
   forward-only read then **drops**, leaving an uncovered target cell between two committed neighbours.
3. **Re-seed only if needed.** If any A-vertex is still uncommitted, go to step 1. This happens **only
   when A is a forest** (≥ 2 disconnected trees); a single connected tree needs exactly **one** seed.
4. **Coverage gap-fill (§8.6).** Once every vertex is committed, close the dropped-run gaps **from the
   committed pivots, not the cover chains**: for each source edge `p → c`, walk the B-path between
   `committed[p]` and `committed[c]` and assign each still-uncovered cell to the downstream vertex it is a
   candidate of. This partitions correctly — only real gaps *between committed pivots* are filled, never
   the phantom coverage the cover chains also hold — and adds nothing to an already-covered matching
   (zero-regression). *Known residual:* complex **split/merge** coverage gaps (not simple linear ones)
   are not yet closed by this pass.

Output is the relation on the graph, **never converted to another index space**:

* **point mode:** `M ⊆ V(A) × V(B)` (a vertex plus any 1:N coverage run it owns).
* **segment mode:** the same walk on `L(A)` gives `M_seg ⊆ E(A) × E(B)`.

Optional convenience outputs (caller-side, clearly derived, never validated in place of `M`): group `M`
by a `road_id` edge attribute for road-level routes.

---

## Part 6 — Validation — V1–V4 on the graph (never a point conversion)

One checker, `check_rules(M, src, tgt)`, run on the **same graph the match lives on**: `src, tgt = A, B`
for point mode; `src, tgt = L(A), L(B)` for segment mode. Each rule is checked **restricted to
neighbours present in `M`** (so a cone's absent branches don't false-fire):

* **(V1) no cross** — for `(a,v) ∈ M`, no matched predecessor of `a` sits on a successor of `v`.
* **(V2) merge** — `a` continues a run, or every matched predecessor of `a` lands on `v` or a
  predecessor of `v`.
* **(V3) split** — symmetric over successors.
* **(V4) coverage** — every source node appears in `M`.

Segment mode is validated on `L(A)`/`L(B)`, **not** by collapsing `M_seg` to a point matching — that
collapse is what over-assigned coverage and produced phantom crosses. Point and segment validation are
the *same function* on different graphs; neither substitutes for the other.

### 6b Cross-table agreement — reciprocity on `M`

`validate_tables` (§6) checks each `D`/`B` cell is a legal warping **in isolation**; it never checks
that the two independently-computed tables **agree**. `check_reciprocity` does — on the committed
matching only: whenever the **forward** table threads a source edge `p → c`, the **backward** table
must thread the same edge, at the same committed cells.

For a committed vertex `c` (pivot `committed[c]`), let `head(c)` / `tail(c)` be its forward / backward
**advance anchors** — `committed[c]` walked along its own COVER chain (a `bpD` / `bpB` list that is a
single same-source pair `[(c, ·)]`) to the run's start / end. The real advance pointers live there:
`bpD[c][head(c)] = [(p, x_p)…]` (predecessors, `p ≠ c`) and `bpB[c][tail(c)] = [(s, w_s)…]`
(successors, `s ≠ c`). A vertex connects to its **predecessors at its run-start** `head(c)` and to its
**successors at its run-end** `tail(c)`. The invariant, over every source edge `p → c`:

```
(p, tail(p)) ∈ bpD[c][head(c)]   ⟺   (c, head(c)) ∈ bpB[p][tail(p)]
```

Forward "feeds `c` from `p`'s run-end" iff backward "continues `p` into `c` at `c`'s run-start", on the
**same** cells. (With no coverage every anchor equals the pivot, so it reads `(p, committed[p]) ∈
bpD[c][committed[c]] ⟺ (c, committed[c]) ∈ bpB[p][committed[p]]`.) This is the structural twin of the
numeric agreement `g(a) = min_v ( D[a][v] + B[a][v] − E(a,v) )` being **constant** across a component
(§5 seeds one representative precisely because it is): both certify the two passes found the *one*
optimum. `check_reciprocity` returns the offending edges — empty ⇒ agree.

**Only on `M`, never table-wide.** Off the optimum the reciprocity is *false*, and a check over all
finite cells would flag correct tables. `D[a][v]` and `B[a][v]` optimise **differently-pinned**
subproblems (best way *into* `a@v` vs best way *out of* `a@v`); fan-in lets several forward cells name
one predecessor cell whose backward-optimal successor is only one of them. Minimal counterexample — a
plain chain `a₀→a₁` over target `v₀→v₁→v₂`: `bpD[a₁][v₂] = [(a₀, v₁)]` (advance) while
`bpB[a₀][v₁] = [(a₁, v₁)]` (stall), so `(a₁, v₂) ∉ bpB[a₀][v₁]` — both cells correct, reciprocity absent.

**Coverage is excluded.** Same-source COVER pairs `[(c, v′)]` are read from the forward chain only (§5)
and have no backward mirror, so the `head` / `tail` walk consumes them rather than testing them.

A failure is a genuine forward/backward disagreement on the optimum — a bug in one pass, **not**
something to repair by forcing the tables to agree: forcing picks a winner and reintroduces the very
split- / merge-optimism the two-pass design exists to avoid (main §4.2). Fix the pass at its source.

### 6c Table reachability — sources ↔ sinks by back-pointers

`validate_tables` (§6) checks each cell's back-pointer chain is a legal warping; `check_reciprocity`
(§6b) checks the two tables **agree**. This is a third, **per-table structural** test: each table's
back-pointers must reconstruct the tree's own **source ↔ sink reachability**.

A **cell** is a pair `(a, v)` — a source vertex `a` pinned at a target cell `v`. Every back-pointer step
is **cell → cell**. From the tree `A` we know, for every source, the set of **sinks reachable** from it
(its descendant leaves), and for every sink, the **sources** that reach it (its ancestor roots):
`ancestor_sources(t) = (ancestors(t) ∪ {t}) ∩ sources`, `descendant_sinks(s) = (descendants(s) ∪ {s}) ∩ sinks`.

* **Forward `D`** — `bpD` points **upstream** (predecessor cells); a source has `bpD = []`. For **every
  finite cell of every sink** `t`, walk `bpD` from that cell, **branching at every predecessor entry**,
  and collect the vertices of the terminal cells (`bpD = []`). They must equal **exactly
  `ancestor_sources(t)`**.
* **Backward `B`** — `bpB` points **downstream** (successor cells); a sink has `bpB = []`. For **every
  finite cell of every source** `s`, walk `bpB`, branching at every successor, and collect the terminal
  vertices. They must equal **exactly `descendant_sinks(s)`**.

Branching is the whole point: a **split** makes `bpB` fork so one source cell must reach **all** its
downstream sinks; a **merge** makes `bpD` fork so one sink cell must reach **all** its upstream sources.
Because a source vertex may span a 1:N coverage run, a step to a cell with the **same** source vertex (a
COVER pointer `[(a, w')]`) stays on `a`, and a step to a **different** vertex (an ADVANCE) moves on — the
move is cell → cell either way; the reached set is read off the **vertex component** of each terminal cell.

A cell is **invalid** if the walk (i) hits a `None` cell reference — the path is severed (a defensive
guard), or (ii) its reached endpoint set ≠ the tree's required set (a branch is missing or spurious).
`∞`-cost cells are infeasible **by construction** and are skipped.

**Empirically the two passes always pass this** — every finite cell reconstructs exactly the tree's
reachability (0 invalid cells over a 22 500-case α,β × point/segment sweep, including cases where the
*extraction* crashes or returns an invalid matching). So its value is twofold: a **regression guard** on
the table-filling, and a **localiser** — it certifies the known failures (the merge `KeyError`, the
invalid/sub-optimal matchings) live **downstream of the tables**, in the extraction/coverage, not in the
tables' reachability. (The merge `KeyError` is an *extraction* fault: it follows a `None` back-pointer
that sits on an `∞` cell, which this test skips — the table itself is sound.)

`check_reachability(A, which)` runs this for `which ∈ {"D", "B"}` and returns the invalid cells (empty ⇒
the table's back-pointers encode exactly the tree's reachability). Point mode runs on `A`, segment mode
on `L(A)` — same function, different index set.

---

## Segment mode = the same six parts on the line-graph

Segment mode is not separate code — it is Parts 1–6 run on `L(A) = nx.line_graph(A)` against
`L(B) = nx.line_graph(B)`:

* `nx.line_graph` on a `DiGraph` gives exactly the directed line graph: nodes are the edges `(u,v)`,
  and `(a,b) → (c,d)` iff `b == c` (verified). A merge+split vertex becomes a bipartite cluster in
  `L` (an undirected cycle) — harmless: it is one shared pin, not a reconvergence (main §8.4).
* `line_graph` copies **no** attributes, so we attach `x,y`-derived `mid` and `bear` to each L-node
  from the original endpoints.
* Candidates are gated by **midpoint** distance ≤ `r`; `E` is the segment emission (§2).
* Junction stitches do **not** exist in this model (a junction is a vertex), so the old
  stitch-contraction special-case is gone — `L(A)`'s edges already connect the right segments.

Everything else — `D`, `B`, extraction, V1–V4 — is byte-for-byte the point-mode code with `L(A)`,
`L(B)` in place of `A`, `B`.
