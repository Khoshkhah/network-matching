# Cell-DAG Extraction — a Per-Cell Re-Implementation of §5 (design)

A re-implementation of the extraction of `docs/dag_dtw_matching.md` §5 — the backward pass (§5.2)
and the selection (§5.3) — as a **per-cell dynamic program over the cell DAG**, with **state
freeing**. Everything upstream is unchanged: `prepare`, the forward table with its V3 coupling
(§4), and its `forbidden` / `D = ∞` pruning of the extraction. The judge (`check_rules`) and the
§3 cost are unchanged. The contract: **the same matching at the same cost as `extract_cell`**
(`C(M)` equal on every case; `M` bit-equal off exact ties), measured for time and peak memory.

| piece | status |
|---|---|
| design (this doc) | agreed with §5 semantics; implemented |
| motivation — why not a scalar min-DP (§6) | two worked counterexamples, one per sweep direction: forward/splits (§6.1), backward/merges (§6.2) |
| **integration** | **this engine IS `network_matching.dag_dtw.extract_cell`** (2026-07-10) — the default `engine="cell"` of `match_dag` and the DuckDB pipeline; `run_cap` removed from the API; gate passed: full suite 164/164, envelope 384/384 valid + `C(cell) ≤ C(join)` 384/384 |
| benchmark vs the replaced vertex-granularity engine | **run** (§7): parity 16/16; up to 5.5× faster and 35× less peak memory; on chained merges 326×/338× and beyond the old engine's reach entirely; the old engine is preserved verbatim in `scripts/extract_cell_dag.py` as the baseline (`extract_cell_vertex`) |
| early discharge (§3.5) | **implemented** — pending keys paid & dropped at the arms' first common ancestor; kills the chained-merge exponential (§7) |
| tree-with-multiple-merges wall (§8) | **diagnosed + measured (§8.5); fixes in progress** — early discharge cannot fire on a *tree* (a merge's arms share no ancestor), so ≥2 merges give `O(∏ \|cand(m)\|)` (line 102752: `45·45·14`, ~15 s/edge). Forward-table cell pruning (§8.2, **Fix 2, in design**) is the base-shrink and the only lever for all-coupled edges; factoring independent merges (§8.1, Fix 1) gives 16–42× on most; sink-labeling (§8.3) **measured unsafe** (funnels pervasive) |

## 0. Notation — every index used in this doc

| symbol | meaning |
|---|---|
| `X`, `P`, `c`, `m`, `J` | vertices of the **source** `A` (`P` a parent, `c` a child, `m` a merge = in-degree > 1, `J` a split = out-degree > 1) |
| `u`, `v`, `w`, `e`, `ce`, `x` | vertices of the **target** `B` (candidate cells of some A-vertex) |
| `(X, v)` | a **cell**: A-vertex `X` pinned to B-vertex `v` |
| `Apred(X)` / `Asucc(X)` | `X`'s parents / children in `A` |
| `Bpred(v)` / `Bsucc(v)` | `v`'s in- / out-neighbours in `B` |
| `outdeg(X)` | number of `X`'s children in `A` |
| `E(X, v)` | emission of the cell — drift distance (+ bearing term in segment mode) |
| `α`, `β` | the §3 weights: `α` per covered run cell (1:N), `β` per stall (N:1), `1` per advance |

## 1. The Space — the Cell DAG, Swept in Reverse

The DP runs on the **surviving cells**: cells that have `D < ∞` and pass the §5.2
sink-reachability pre-pass (the forward table's role is exactly this pruning — nothing else of it
is read). The `forbidden` flag is **role-aware** (§4.1a): a flagged cell may not be a **run END**
(no END-state rows, no attachment from a child, no sink seed), but it stays in the space as a
cover through-cell — runs may pass through it on the way to a legal end. Edges are §5.0's three
moves (cover / stall / advance), restricted to surviving cells.

* **Order**: one **reverse topological order of the cells** — by §5.0's grading this is "A's layer
  order reversed, refined inside each row by a reverse topological order of `B[cand(X)]`". Every
  state is computed once, after everything it reads.
* **Cyclic-row fallback**: where a directed B-cycle survives inside one row (§5.0: the only place
  cycles exist), the row's SCC is relaxed to a fixed point instead — monotone since `α·E ≥ 0`,
  ties broken by the fixed `_b_order` (§4b). Covering a cell twice is never cheaper, so the
  optimum uses simple runs, as today.

## 2. The Data Structure — Two States per Cell, Each Holding One Table

Each cell `(X, v)` carries **two states**; each state holds one **table**: a dict keyed by the
frozen `pending` set, one row per key, cheapest only.

```python
END[(X, u)]   = { frozenset(pending): (value, pending, cells) }   # X's run ENDS at u
ENTRY[(X, v)] = { frozenset(pending): (value, pending, cells) }   # parent connects at v
```

* **`END[(X, u)]` — "the downstream cone of `X`, given `X`'s run ends at `u`."** This is where
  `X`'s children are attached (§3.2). Its `cells[X] = (u,)` — the run is just its end so far.
* **`ENTRY[(X, v)]` — "the cone of `X`, given the parent connects at `v`."** The run from `v` to
  its eventual end is already decided *inside* the state (§3.1); `v`'s own `E` is **unpaid** — the
  §5.2 ledger's deferred entry, paid by the parent's connecting edge (or at the root join).
* The row payload is §5.1's, minus the `entry` field — the entry **is the cell that owns the
  state**: `value` (cost per the ledger, entry-`E` and pendings unpaid), `pending`
  (`{(merge-vertex, its entry-cell): stall-flag}`), `cells` (`{vertex: its run}` — `M` travels
  with the state, no traceback).
* **Contraction is per cell, per pending-key**: within one state's table, two rows with the same
  pending key are interchangeable upstream — only the cheaper survives. On a merge-free source
  (an out-tree) every pending is empty, so **every table is a single row** and the whole sweep is
  a scalar DP; all row multiplicity comes from merges.
* **Loud cap**: a state's table exceeding `max_rows` pending keys raises — never truncates.
  `run_cap` **does not exist** in this design (§3.1).

## 3. The Three Operations — How a Table Passes, Merges, and Splits

### 3.1 PASS — a cover edge moves the entry hat (within the row)

```
ENTRY[(X, v)]  =  contract(  END[(X, v)]                                   # run ends right here
                           ∪ { shift(row) : row ∈ ENTRY[(X, w)],           # or extends one B-arc
                               w ∈ Bsucc(v) ∩ survivors(X) } )

shift(row):  value += α·E(X, w);   cells[X] = (v,) + cells[X];   pending unchanged
```

The run is never enumerated: it grows one arc per cover edge, so `_cell_runs` and `run_cap`
disappear. The **entry hat moves**: while `w` was the entry it was unpaid; the moment the run
extends past it, `w` becomes a covered cell and pays `α·E(X, w)`; `v` is the new unpaid entry.
Total per the ledger: the final entry paid once (`1`/`β`, by the parent), every other run cell
paid `α` — exactly §5.2.

### 3.2 MERGE of tables — children attach at a run end

`END[(X, u)]` cross-joins the children's `ENTRY` tables, all attached at the **same** cell `u` —
per-cell (V3) by construction, no coupling needed here:

```
END[(X, u)]:  start from the base row (0, {}, {X: (u,)})
              for each child c (a split cross-joins ALL of them here):
                  options = { ENTRY[(c, u)]   paying β·E(c, u)     (STALL edge)
                            , ENTRY[(c, w)]   paying 1·E(c, w),  w ∈ Bsucc(u)  (ADVANCE edge) }
                  # a merge child pays nothing here — see §3.3
                  combos ← combos × options:  values add, cells union (disjoint cones),
                            pendings union — same merge-vertex with two different entry
                            cells kills the combination; stall flags OR
              contract per pending key
```

A sink cell's `END` is just the base row. This is §5.2's *MERGE of data*, moved from
per-vertex-run granularity to one cell.

### 3.3 SPLIT of a table — a merge vertex serves several parent arms (consumed-once)

Identical rule to §5.2, vertex-granular, decided **statically** before the sweep: each merge `m`
gets one **absorbing parent** (deterministic pick, e.g. first in the sweep order); which arm
absorbs is arbitrary — nothing is double-counted either way.

* The absorber's cells consume `ENTRY[(m, ·)]` as ordinary child options (§3.2) — but `m`'s
  entry-`E` stays unpaid; the option gains `pending[(m, ce)] = did-this-arm-stall`.
* Every **other** parent's cells see only **interfaces**: per surviving entry `ce` of `m`, a row
  `(0, {(m, ce): stall-flag}, {})` — no value, no cells, just the separator that the root join
  will match against the absorber's choice.

### 3.4 Selection — unchanged §5.3

Sources pay their own entry-`E` in full (free entry) on their `ENTRY` tables; the root join folds
one table per root, contracting per pending-key after each fold; every deferred merge entry is
paid once (`β` if any arm's flag is set). Rows tried cheapest-first; the first `cells`-map passing
the judge is `M`; none valid ⇒ the feasibility `ValueError`.

### 3.5 When a Cell Decides

A cell never chooses *whether* to decide — one uniform rule runs at every turn. Terminology: one
pending entry's **key** is `(merge-vertex, its entry-cell)`; a row's **signature** is its whole
frozen pending set, `frozenset(pending.items())` — the contraction key of §2.

1. **Merge always.** Cross-join whatever arrived from the children groups, union `cells`, union
   pendings (same merge with two different entry cells kills the combination on the spot).
2. **Decide only within a signature.** Rows with identical signatures are interchangeable given
   the same assumptions — keep the cheapest, drop the rest. Rows with *different* signatures are
   never compared: they pass on side by side as alternatives. A cell decides everything whose
   consequences it can fully see, and postpones exactly the choices that involve an arm it has
   not seen — the pending key *is* the marker of "not my decision yet".
3. **A merge's decision happens at the fold where its last arm joins.** There the key is
   **discharged**: agreement is already checked, the merge's entry is paid once (`β` if any
   arm's flag is set, else `1`), the key is removed — and the surviving rows, signatures now
   equal, become comparable again, so the min applies.

Turn-by-turn on §6.2's preference case (`p₁` the absorbing arm; watch the values — a merge
child's `E` is paid at **discharge**, never at consumption (§3.3), so the costs `1`/`2` are locked
behind the pending until the last arm folds in):

```
(m, w₁), (m, w₂):   base rows (0, {}, {m:(w₁,)}) / (0, {}, {m:(w₂,)}) — E(m,·) unpaid.

(p₁, a):    consumes the merge child  →  pending BORN:
                (0, {(m,w₁): False}, {p₁:(a,), m:(w₁,)})
                (0, {(m,w₂): False}, {p₁:(a,), m:(w₂,)})
            Signatures differ  →  NOT p₁'s decision  →  BOTH rows pass on.

(p₂, b₁):   interface only:   (0, {(m,w₁): False}, {p₂:(b₁,)})
(p₂, b₂):   interface only:   (0, {(m,w₂): False}, {p₂:(b₂,)})

root join   (sources pay their own entries first: p₁ +0;  b₁ +5,  b₂ +0):
            {m@w₁} × {m@w₂}  →  same merge, different cells  →  CHECKED: dead
            {m@w₁} × {m@w₁}  →  agree  →  (5, {(m,w₁): False}, all cells)
            {m@w₂} × {m@w₂}  →  agree  →  (0, {(m,w₂): False}, all cells)
            m's last arm folded  →  keys DISCHARGED (no flag set ⇒ weight 1, not β):
                5 + 1·E(m,w₁) = 6            0 + 1·E(m,w₂) = 2
            no keys left → rows comparable → DECIDE:  min = 2   ✓ the optimum
```

Boundary cases, same rule: on an **out-tree** every pending is empty → one signature everywhere →
every cell fully decides → exactly one row passes on (the §6.4 collapse to the scalar scheme —
the "decision moment" is every cell). On a **reconvergent DAG** the arms meet earlier, at the
shared ancestor's cross-join — conflicts die there, and the prototype **discharges there too**
(*early discharge*: the arms' common ancestors are precomputed statically; the key is paid and
dropped at the first one the sweep reaches, and the rows re-contract). Deferring payment to the
root is equally correct but lets signatures multiply across *chains* of merges — the §7
exponential; early discharge makes them linear.

## 4. Freeing — the Inbox-Push Lifecycle

Nothing waits to be read and nothing is reference-counted. Each cell owns one **inbox**; a cell's
whole life is one turn of the sweep:

1. **Process the inbox** — everything pushed by earlier turns is already there: the children's
   contributions (grouped by child) and the row's cover contributions. Cross-join, contract per
   signature (§3.5), form the own `END`/`ENTRY` rows.
2. **Push** the rows to the readers' inboxes, each already shaped for its reader: *shifted*
   (`+α·E`, run prepended) to the row-predecessor cells (§3.1) · *priced* (`1`/`β` by edge type)
   or *pending-tagged* (merge, §3.3) to the **absorbing** parent's cells with a stall/advance
   edge in (§3.2) · to the root join's inbox if `X` is a source. A merge's *other* arms need no
   push at all — their interfaces are generated locally from the surviving entry set.
3. **Free** — the cell and its rows are dropped in the same turn. Rows are immutable tuples, so a
   push shares references, never copies.

| state | pushed to |
|---|---|
| `END[(X, u)]` | consumed by `ENTRY[(X, u)]` in the same turn — never leaves the cell |
| `ENTRY[(X, v)]` | inboxes of `(X, v⁻)`, `v⁻ ∈ Bpred(v)` in the row (§3.1) · the absorbing parent's cells `(P, u)` with a stall/advance edge into `(X, v)` (§3.2) · the root join, if `X` is a source |

Live memory is therefore exactly the **unprocessed inboxes** — the sweep's current frontier
(largest antichain the order crosses), not, as today, every vertex's table held until the
component ends.

*Honest caveat*: the heavy part of a row is its `cells` payload (`O(cone size)`), which freeing
does not shrink — it shrinks the *number* of live rows. If the benchmark shows `cells` dominating
peak memory, the follow-up fork is back-pointers-plus-traceback instead of M-travels-with-rows —
which conflicts with freeing, so it is decided by the numbers, not here.

## 5. What Changes vs `extract_cell`, What Is Measured

| | the replaced engine (`extract_cell_vertex`, kept in the benchmark script) | this design (= `extract_cell` today) |
|---|---|---|
| granularity | per **vertex**: enumerate `(entry, run)` combos, then contract | per **cell**: contract at every state, runs implicit |
| runs | `_cell_runs`, simple paths, `run_cap` (loud cap) | §3.1 recursion — no enumeration, no cap |
| tables kept | all vertices, until the component ends | frontier only (inbox-push, §4) |
| cross-products | full combo list per vertex before contraction | dominated rows die per cell, before multiplying |
| forward table, coupling, judge, ledger, pending, root join | unchanged | unchanged |

**Benchmark**: `time.perf_counter` + `tracemalloc` peak, current vs new, on scaling families —
dense-B chains (run-heavy), wide trees (frontier width), growing random polytrees (merge/pending
load) — asserting per case: `M` valid, `C(M)` equal to `extract_cell`'s (bit-equal `M` off exact
ties). The verdict decides whether this replaces §5.2's implementation or is archived with its
numbers.

## 6. Motivation — Why Not a Scalar Min-DP over the Cell DAG

The natural first idea (proposed during design, and worth recording because half of it *is* the
forward table): give every cell one aggregated value, sweep the cell DAG in topological order, and
at each cell take the minimum over what arrived, add the own cost, push onward. The scheme exists
in two sweep directions, and **each direction is broken by exactly one junction type** — the one
whose arms the sweep visits *separately*:

| sweep | aligned junction — coupled free, through one shared cell | poison junction — arms never share a cell |
|---|---|---|
| forward (sources → sinks) | merges: the arms meet in one inbox | **splits** — §6.1 |
| backward (sinks → sources) | splits: children combine inside one cell `(J, v)` | **merges** — §6.2 |

The forward variant, written out with every index named:

```
value(a, v) = E(a, v)                        a — the A-vertex being computed
                                             v — the B-cell it is pinned to
    + Σ                                      p — ranges over a's parents:  p ∈ Apred(a)
       p ∈ Apred(a)
         [ min  value(p, x) ] / outdeg(p)    x — ranges over p's cells with an edge into (a,v):
           x                                     x = v            (a STALL edge)
                                                 v ∈ Bsucc(x)     (an ADVANCE edge)
                                             outdeg(p) — p's child count: the cost-sharing
                                                 fraction, so a shared parent is counted once
                                                 when branches are summed again downstream
```

This is exactly §4.1 of the main doc — the recurrence of `forward()`'s `D` — in dataflow form.
It already exists in the pipeline, and it is a **cost bound**, not a matching. The reason is the
independent `min` per branch:

### 6.1 Forward sweep — broken by splits (four vertices)

```
A:   J ──→ c₁            B:   v₁ ──→ u      v₁ ──→ d′         emissions:
     └───→ c₂                 v₂ ──→ u′     v₂ ──→ d          E(J,v₁) = E(J,v₂) = 1
                                                              E(c₁,u) = 0    E(c₁,u′) = 5
                                                              E(c₂,d) = 0    E(c₂,d′) = 5
```

`c₁`'s cheap cell `u` is reachable only out of `v₁`; `c₂`'s cheap cell `d` only out of `v₂`.
Run the scheme (`α = β = 1`; `J` is a source, so `value(J, ·) = E(J, ·)`; `outdeg(J) = 2`):

```
value(J,v₁)  = 1                     value(J,v₂)  = 1
value(c₁,u)  = 0 + value(J,v₁)/2  = 0.5        ← c₁'s min chose parent cell v₁
value(c₁,u′) = 5 + value(J,v₂)/2  = 5.5
value(c₂,d)  = 0 + value(J,v₂)/2  = 0.5        ← c₂'s min chose parent cell v₂
value(c₂,d′) = 5 + value(J,v₁)/2  = 5.5

claimed total = value(c₁,u) + value(c₂,d) = 1.0
```

Now the **real** matchings (rule V3: both children must leave `J` from ONE cell):

```
J on v₁:  M = {(J,v₁), (c₁,u),  (c₂,d′)}   →  C(M) = 1 + 0 + 5 = 6
J on v₂:  M = {(J,v₂), (c₁,u′), (c₂,d)}    →  C(M) = 1 + 5 + 0 = 6
```

The scheme reports **1.0**; the true optimum is **6**. The claimed value placed `J` on `v₁` *and*
`v₂` at once — half of each, courtesy of the `/outdeg` fractions — a **phantom** no matching
realizes (main doc §4.1a). The independent per-branch `min` is the culprit: each child chose the
`J`-cell that suited it alone.

**Swept backward, this same case is computed correctly** — at `(J, v₁)` both children's minima are
conditioned on the one shared cell `v₁` (`value(J,v₁) = 1 + 0 + 5 = 6`, likewise `v₂`), and the
source's last decision returns the true `6` with a consistent `cells` map. Splits are the backward
sweep's *aligned* junction. Its poison is the merge:

### 6.2 Backward sweep — broken by merges (preference, not reachability)

First, what a merge counterexample must survive: the pipeline **keeps the forward table**, and
the §4.1 merge sum makes `D[m][v]` finite only if **every** arm reaches `v` — a cell only one arm
can reach never survives the pruning. **The forward table already does the *reachability* half of
merge coordination** (caught in design review: the naive counterexample, where one arm simply
cannot reach the other's preferred cell, dies at `D = ∞` and proves nothing). What the forward
table cannot prune is a **preference** conflict: both merge cells fully reachable by both arms —
finite `D`, nothing to prune — while the arms' local minima prefer different ones:

```
A:   p₁ ──→ m ←── p₂        B:   a  → w₁,  a  → w₂      emissions:  E(p₁,a)  = 0
                                 b₁ → w₁,  b₂ → w₂                  E(p₂,b₁) = 5   E(p₂,b₂) = 0
cand(p₁) = {a}   cand(p₂) = {b₁, b₂}   cand(m) = {w₁, w₂}           E(m,w₁)  = 1   E(m,w₂)  = 2
```

Forward-table check — nothing is pruned: `D[m][w₁] = 1 + D[p₁][a] + D[p₂][b₁] = 1 + 0 + 5 = 6`
and `D[m][w₂] = 2 + 0 + 0 = 2`, both finite (both arms reach both cells); `A` has no split, so
the coupling forbids nothing. Run the scheme by its own rules, sinks first:

```
value(m, w₁) = 1                value(m, w₂) = 2

(p₁, a):   options for child m out of a:   w₁ → 1,   w₂ → 2
           min  →  (value 1, cells {p₁: a, m: w₁})
           ⚠ the (a, m@w₂) row at value 2 is DISCARDED — the damage is done here

(p₂, b₁):  only w₁  →  (6, {p₂: b₁, m: w₁})
(p₂, b₂):  only w₂  →  (2, {p₂: b₂, m: w₂})

LAST DECISION (combine the sources, ids checked):
           (a, m@w₁) × (b₂, m@w₂)   →   m: w₁ ≠ w₂  →  conflict, dead
           (a, m@w₁) × (b₁, m@w₁)   →   agree       →  C(M) = 0 + 5 + 1 = 6   ← returned
```

The true optimum is `M = {(p₁,a), (p₂,b₂), (m,w₂)}` at `C(M) = 0 + 0 + 2 = 2`. The scheme returns
a **valid but 3× costlier** matching — silently, no error — because `(p₁, a)`'s min baked in
`m@w₁` before `p₂`'s cheap arm (which only pairs with `w₂`) had been seen anywhere. Without the
id check it is worse still: the two conflicting rows combine into the `m`-on-two-cells phantom.

Why no cell-level trick can save it — the exact contrast with §6.1's backward success: at the
split, both children were combined *inside one cell's computation* (`(J, v₁)` conditioned both).
Here the two arms consume `m`'s result in **two different cells**, `(p₁, a)` and `(p₂, b₂)`,
computed with no knowledge of each other; their only meeting point is the last decision, and by
then the row the combination needed is gone. (A second, independent coordination problem hides in
the same example: where the arms *do* agree, `m`'s `E` must still be counted once, not once per
arm — the consumed-once rule, §3.3.)

The minimal fix, read straight off the example: `(p₁, a)` must keep **both** rows — `(1, m@w₁)`
and `(2, m@w₂)` — labelled by the cell of `m` each assumed, so the last decision can select the
agreeing cheap pair (`w₂` + `w₂` → `C(M) = 2`, correct). "One kept row per assumed cell of the
shared merge" **is** `pending` (§2), the whole of it. In one line: the forward table covers merge
**reachability**; `pending` covers merge **preference**.

### 6.3 Appending ids detects the conflict — and cannot repair it

Amend either direction: each cell appends its own choice to the pushed dict, and where branches
meet, the shared ids are compared. In §6.1, `c₁` pushes `(0.5, {J: v₁, c₁: u})`, `c₂` pushes
`(0.5, {J: v₂, c₂: d})` — `v₁ ≠ v₂`, conflict, combination discarded: no phantom escapes. But
**nothing valid is left either**: the compatible alternatives (`c₁` via `v₂` at 3.0, `c₂` via
`v₁` at 3.0) were already discarded by the `min`. §6.2's last decision is the same event
(`m: w₁ ≠ w₂`, the needed `w₂`-row long gone). In both directions, detection comes after the
information needed for repair is gone.

### 6.4 The three exits — there is no fourth

| exit | mechanism | what it is in this project |
|---|---|---|
| retry other options on conflict | backtracking search, worst-case exponential, needs a state cap | the removed branching engine (`extract`, deleted 2026-07) |
| **keep alternatives proactively, keyed by the shared commitment** | one row per key `(shared vertex, its cell)` — polynomial DP | **`pending` — this design (§2–§3)** |
| force agreement globally during the fill | iterative forbid-and-rebuild on the table | the §4.1a coupling inside `forward()` |

The design in §2–§3 is therefore the scalar scheme **completed**: the same aggregated `value`,
the same push, plus the minimum machinery that makes cross-branch combination sound — a row per
pending key. It sweeps **backward** because Mode-3 sources are out-trees — split-rich,
merge-poor — so the poison junction is the rare one: on a merge-free source the keys vanish and
the completed scheme *collapses back to* the scalar one — one row, one float, one partial
matching per cell, no `pending` anywhere.

## 7. Results — the Benchmark Verdict (2026-07-10, `scripts/extract_cell_dag.py`)

Correctness parity first: **16/16 cases agree** with `extract_cell` on cost and on refusals — 12
random polytrees over cyclic targets, the two-cycle refusal (main doc §7), a diamond chain, a
dense chain, an out-tree. Then the scaling families (`α = 0.5`, `tracemalloc` peak, one process):

| family | size (`|A|` / cells) | time cell → dag | peak memory cell → dag | cost |
|---|---|---|---|---|
| dense-B chain (coverage-heavy) | 800 / 7 190 | 0.61 s → 0.11 s (**5.5×**) | 118 MB → 3.4 MB (**35×**) | equal |
| dense-B chain | 400 / 3 590 | 0.20 s → 0.08 s (2.4×) | 30 MB → 1.6 MB (19×) | equal |
| binary out-tree (split-rich, pending-free) | depth 8: 1 021 / 189 269 | 4.08 s → 1.01 s (**4.0×**) | 61 MB → 46 MB (1.35×) | equal |
| diamond chain (one merge per unit) | k=4: 21 / 83 | 0.43 s → 0.001 s (**326×**) | 34 MB → 0.1 MB (**338×**) | equal |
| diamond chain | k=40 / 120 / 400 | cell **refuses** (`max_rows`) · dag 0.02 / 0.08 / 0.66 s | cell 339 MB at the refusal · dag 0.8 / 3.7 / 30 MB | dag-only |

Findings:

* **The space idea works.** Freeing rows with the frontier is worth up to **35×** peak memory on
  coverage-heavy chains — and the gap *widens* with size (2× at n=50, 35× at n=800): `extract_cell`
  holds every vertex's table to the end, the cell DAG holds only unprocessed inboxes.
* **Time wins come from implicit runs.** No `_cell_runs` enumeration, no `run_cap`; the §3.1
  recursion visits each cell once. 4–5.5× on the large cases; small inputs (n≈50) pay a constant
  overhead and can be slightly slower — irrelevant at scale.
* **The chained-merge exponential, diagnosed and killed.** Without early discharge, pending keys
  live to the root, so on a *chain of merges* signatures multiply per merge (`~|cand(m)|` each) —
  **both** engines exceeded `max_rows` at ~6 chained diamonds (size-independent refusal point:
  constant ~2.7 s / 339 MB for `extract_cell` at k = 40, 120, 400). **Early discharge** (§3.5) —
  pay and drop a key at the arms' first common ancestor, statically precomputed, then re-contract
  — makes it linear: k=4 falls from 0.43 s / 34 MB to **1 ms / 0.1 MB**, and k=400 runs in
  0.66 s / 30 MB where `extract_cell` cannot finish at any setting of its caps. The wall is now
  exclusively `extract_cell`'s. (One theoretical note: discharging narrows the judge's fallback
  pool — rows differing only in a discharged merge's placement contract early; no divergence
  observed in the parity suite, including cyclic-B cases.)

Verdict per §5's protocol: better time, much better space, one capability the old engine lacked
(long merge chains), exactness and refusals preserved. **Integrated 2026-07-10** as
`network_matching.dag_dtw.extract_cell` — the gate passed in full: test suite 164/164, the
structured envelope 384/384 valid with `C(cell) ≤ C(join)` 384/384 (`scripts/test_dag_point.py`),
and benchmark parity 16/16 against the replaced engine, which is preserved verbatim as
`extract_cell_vertex` in `scripts/extract_cell_dag.py` so these numbers stay reproducible.

## 8. The tree-with-multiple-merges wall — proposed fixes (not yet implemented)

Early discharge (§3.5, §7) pays a merge's `pending` key at the **first common ancestor of its
arms**. A *reconvergent* source — the §7 diamond chains — has one, so keys die young and the chain
stays linear. A **tree** never does: a merge's arms lead to **disjoint subtrees that never rejoin
going up**, so their only common ancestor is the root. On a tree with **≥2 merges** every key
therefore survives to the root and the signatures **multiply** — cost `O(∏ over concurrently-open
merges of |cand(m)|)`, the exact §7 exponential early discharge was built to kill but **cannot reach
on a tree-shaped `A`**.

This is not hypothetical. The `map-conflation` hourglass source (`local_dag.build_hourglass`) is
**always a tree**; a normal edge has one merge (`J_u`, where the in-tree stubs funnel in), so `∏` is
a single factor and nothing shows. But an edge whose junctions carry several merges hits the wall:
line `102752`'s line-graph is 29 nodes with **3 concurrently-open merges** of 45, 45, 14 cells →
`45·45·14 = 28 350` joint signatures → **13.8 M** `_pend_union` calls, **~15 s for one edge**
(cProfile: `extract_cell` 18 s cumulative, `_fill` 8.4 s, `_pend_union` 5.9 s). The §7 wall, on
trees — which is precisely the shape the hourglass always takes.

Fix 1 (§8.1) and Fix 2 (§8.2) are exact and composable; §8.3 is a rejected alternative. Measurement
(§8.5) then showed Fix 2 is only a *conditional* accelerator (loose bound, unreliable incumbent), which
motivates §8.6 — **attacking the coupling directly**, the robust option that needs no incumbent and no
bound.

### 8.1 Fix 1 — factor independent merges (`O(∏)` → `O(∑)`)

Two concurrently-open merges `m₁, m₂` whose **cones are disjoint** have **independent** cell choices:
the total decomposes `C = C₀ + f₁(m₁-cell) + f₂(m₂-cell)`, so the `min` over the *joint* signature
equals `min f₁ + min f₂` — the product is pure waste. The fix: when a set of open merges is pairwise
cone-disjoint, **do not materialize their joint `pending`**; carry and minimise each key's
alternatives **independently**, recombining only the (additive) costs. Product becomes sum.

**Caveat — nesting.** The fix applies to *disjoint* merges only. Some hourglass merges are
**nested**: `J_v`'s arm runs *through* `J_u` along the shared waist `e`, so their cell choices
interact through `e`'s run and do **not** factor. Fix 1 collapses the disjoint pairs and leaves the
nested ones on the current `pending`. **Step one is therefore a measurement** — over the slow
hourglass edges, of each pair of concurrently-open merges, how many are cone-disjoint vs nested. That
number bounds what Fix 1 alone buys. Exact by construction (an identity on additive, independent
terms).

### 8.2 Fix 2 — forward-table pruning of merge cells (vetted 2026-07-17)

**Admissible branch-and-bound.** `prepare`/`forward` already give every cell a forward cost `D`. Drop
a MERGE candidate cell (`A.in_degree(m) ≥ 2` — the only cells that widen `pending`) whose lower bound
already exceeds a real matching's cost, because no optimal matching can use it. Shrinks the **base** of
the product (`45·45·14` → a small product) without changing the exponent; **composes** with Fix 1 and
is the **only** lever for the all-coupled tail (§8.5: `100350` is invisible to Fix 1 but Fix 2 gives
32×), so it goes first.

**Mechanism.**

- **Lower bound `LB(m,v) = D[m][v]`** — the forward cost *alone* (see "why not `D+B−E`" below).
- **Upper bound `UB` = the incumbent's cost** — a real valid matching from `extract_join` (the §10
  junction-join engine, which has *no* pending blow-up and returns in ~0 ms even on the worst edge):
  `UB = _cost_of(A,B, extract_join(A,B,α,β)[0], α,β)`. On any `extract_join` `ValueError`, `UB = ∞` ⇒
  block nothing (a degenerate incumbent is a no-op, never a corruption).
- **Block** every merge cell with `D[m][v] < ∞` **and** `D[m][v] > UB + ε`, **inside**
  `_cell_reachable`'s reverse BFS (a `blocked` param at its three add-sites) so the single `seen` set
  stays internally consistent; everything downstream reads `seen` unchanged.
- **Default off** (`prune=False` on `extract_cell`/`match_dag`): shipped behaviour is bit-identical
  until the gate (§8.4/§7) is green.

**Exactness — a formal proof.**

*Notation.* $\mathcal{M}$ = the valid matchings (V1–V3 + the run/move structure); for
$M \in \mathcal{M}$, $\mathrm{run}_M(a) \subseteq V(B)$ is $a$'s run, and cell $(a,v)$ is *used by* $M$
iff $v \in \mathrm{run}_M(a)$. Cost $C(M) = \sum_a \sum_{v \in \mathrm{run}_M(a)} w_M(a,v)\,E(a,v)$,
with $E \ge 0$ and every ledger weight $w \in \{1, \beta, \alpha\}$, $\beta \ge 1$, $\alpha \in (0,1]$
— all $> 0$, so **every cost term is $\ge 0$** $(\star)$. Write $c^{*} = \min_{\mathcal{M}} C$ and let
$M^{*}$ be an optimiser. By the construction of `forward()` (§4.1),
$D[a][v] = \min_{W} \Phi(W)$ over the legal **upstream configurations** $W$ of $(a,v)$ — a cell for
$a$ and for each ancestor of $a$, move- and V3-consistent, with $a$'s run ending at $v$ — where
$\Phi(W) = \sum_{(b,x) \in W} \gamma_W(b)\,w_W(b,x)\,E(b,x)$ and
$\gamma_W(b) = \prod_{e:\, b \to a} 1/\mathrm{outdeg}(\mathrm{tail}\,e) \in (0,1]$ (the §4.1
split-sharing; §6.1 shows it under-counts).

*Proposition 1 (forward under-count): $D[a][v] \le C(M)$ for every $M$ using $(a,v)$.* From $M$ build
$\widehat{M}$: keep $M$'s assignment on $a$ and its ancestors, **truncate** $\mathrm{run}_M(a)$ to end
at $v$, and **drop** $M$ on all descendants of $a$. (i) $\widehat{M}$ is a legal upstream config for
$(a,v)$ — its moves are $M$'s (valid), V3-consistent because $M$ is, $a$'s run ends at $v$ — so it is a
feasible point of the $D[a][v]$ minimisation: $D[a][v] \le \Phi(\widehat{M})$. (ii) each $\gamma \le 1$
and each $w\,E \ge 0$, so $\Phi(\widehat{M}) = \sum \gamma\,w\,E \le \sum w\,E$. (iii) $\widehat{M}$'s
terms are a *subset* of $M$'s (truncation only deletes terms; each kept cell keeps its $M$-weight), all
$\ge 0$, so by $(\star)$, $\sum w\,E \le C(M)$. Chaining: $D[a][v] \le C(M)$. $\blacksquare$

*Proposition 2 (incumbent bound): $c^{*} \le \mathrm{UB}$.* $\mathrm{UB} = C(M_J)$ for the incumbent
$M_J \in \mathcal{M}$ (from `extract_join`); $c^{*}$ is the minimum of $C$ over $\mathcal{M} \ni M_J$.
$\blacksquare$

*Theorem (exact prune).* Let
$\text{blocked} = \{\,(m,v) : \mathrm{indeg}(m) \ge 2,\ D[m][v] < \infty,\ D[m][v] > \mathrm{UB}\,\}$ and
$\mathcal{M}' = \{\,M \in \mathcal{M} : M \text{ uses no blocked cell}\,\}$. **(1)** For any cell
$(m,v)$ used by $M^{*}$: Prop 1 gives $D[m][v] \le c^{*}$, Prop 2 gives $c^{*} \le \mathrm{UB}$, so
$D[m][v] \le \mathrm{UB} \Rightarrow (m,v) \notin \text{blocked}$; hence $M^{*} \in \mathcal{M}'$.
**(2)** $M^{*} \in \mathcal{M}' \Rightarrow \min_{\mathcal{M}'} C \le c^{*}$, and
$\mathcal{M}' \subseteq \mathcal{M} \Rightarrow \min_{\mathcal{M}'} C \ge c^{*}$; so
$\min_{\mathcal{M}'} C = c^{*}$, attained by $M^{*}$, and $\mathcal{M}' \ne \varnothing$ — the incumbent
witnesses it: every cell of $M_J$ has $D \le C(M_J) = \mathrm{UB}$ by Prop 1, so $M_J \in \mathcal{M}'$
(the code-level absence of a spurious `ValueError` is the Corollary, part (b)). $\blacksquare$

*Corollary (bit-identical result).* Let $M_{\mathrm{ret}}$ be the matching `extract_cell` returns on
the un-pruned instance. `extract_cell` contracts per pending-signature with a **validity-blind** min and
runs `check_rules` only on the final list, so $M_{\mathrm{ret}}$ is the cheapest *valid* matching among
the surviving signature representatives — *not* provably $c^{*}$ on its own. Assume therefore the
cross-validation invariant $C(M_{\mathrm{ret}}) = C(\text{cell}) \le C(\texttt{extract\_join}) =
\mathrm{UB}$ (gated 384/384; §7, §10.2). **(a)** $M_{\mathrm{ret}}$ is valid with
$C(M_{\mathrm{ret}}) \le \mathrm{UB}$, so Prop 1 applies to it: every cell $(m,v)$ it uses has
$D[m][v] \le C(M_{\mathrm{ret}}) \le \mathrm{UB}$, hence $(m,v) \notin \text{blocked}$ —
$M_{\mathrm{ret}}$ uses no blocked cell, and all its cells survive in `seen`. **(b)** $M_{\mathrm{ret}}$
is the cheapest matching of its own final pending-signature (else a cheaper same-signature matching
would have been the un-pruned representative); removing blocked cells — none of them
$M_{\mathrm{ret}}$'s — creates no new rows and no cheaper same-signature matching, so $M_{\mathrm{ret}}$
remains the pruned finals representative of its signature and, being valid, surfaces. The pruned run
returns the same $M_{\mathrm{ret}}$, with no spurious `ValueError`. $\blacksquare$

*Caveat (reconvergent residual).* Prop 1 bounds only *valid* matchings, so one case is not excluded by
this proof: an *invalid*, blocked-cell-using finals candidate with cost $\le C(M_{\mathrm{ret}})$
sharing a *post-early-discharge* signature with a valid resurfacing row — which requires an
early-discharged (reconvergent) merge. On **tree** sources (the hourglass — §8: early discharge never
fires, so every blocked merge cell stays in the final signature) this cannot arise and bit-parity is
exact up to exact-cost string-ties; on reconvergent DAGs the residual is covered by the §8.4 empirical
parity gate (§7: "no divergence observed").

The proof rests entirely on Proposition 1, whose only non-elementary input is that $D$ **is** the
min-over-configs above (the established semantics of `forward()`); $\widehat{M}$'s feasibility and
$\gamma \le 1$ do the rest.

**Why `D` alone, not `D+B−E`.** The first draft used `D[m][v] + B[m][v] − E[m][v]` (forward + backward,
minus the doubly-counted emission). Two independent adversarial refutations killed it with executed
counterexamples (15/7997 hard `ValueError`s on articulation merges): gluing the forward run (which
*ends* at `v`) to the backward run (which *starts* at `v`) with a single `−E` **over**-counts whenever
`M`'s true run for `m` spans more than one cell (any `α<1` coverage run), so `D+B−E` can *exceed* `c*`
— violating step 1 and pruning the optimum. `D` alone can only *under*-count (the phantom), so step 1
is a **theorem**, not an assumption. `D+B−E` stays permanently out. Sound tightening if `D`-only is
ever too loose: `LB = max(D[m][v], B[m][v])` — the max of two independently-valid floors, no gluing —
which needs `backward()` and a re-run of the gate.

**Measured limitations (2026-07-17) — Fix 2 is a *conditional* accelerator, not a dependable fix.** Two
findings weaken it (correctness is never at risk — the answer is always `extract_cell`'s — but the
*speedup* is loose and often unavailable):

1. **The bound is one-sided.** `D[m][v]` covers only the *upstream* sub-DAG, while `UB` is a whole-DAG
   cost. So `D > UB` fires only for cells whose upstream *alone* already exceeds a full matching — the
   sink-side cells. Source-side cells have tiny `D` and never prune, *however tight `UB` is*. Measured:
   `D`-only 4–45×, and `max(D,B)` only 6×/33×/49×/14× — it never shrinks the hard edges to a handful.
2. **The incumbent is frequently missing.** `extract_join` is the min-frozen engine (§10); its
   contraction can discard the label combination a feasible matching needs, so it fails even when one
   exists. On a 2000-edge sample it failed on 601, of which **536 had a feasible matching it missed**
   (~27% of all edges). On those, `UB = ∞` and Fix 2 prunes nothing.

So Fix 2 helps only when the incumbent exists *and* the loose one-sided bound clears enough cells. That
is why the coupled tail wants a different attack — §8.6.

### 8.3 Alternative considered — sink-cell labeling (conditional)

**The idea (proposed during design).** Key the dedup on the cells assigned to the **sink vertices**
instead of on `pending`. Its premise: *fix the labels of all sink vertices, and the min-cost matching
of the rest is determined* — so keeping one cheapest row per full sink-labeling loses nothing, and if
distinct sink-labelings are fewer than distinct `pending` signatures, it is a **smaller key** than
`pending` and thus a saving.

**When the premise holds.** Exactly when **no merge sits above a target funnel** — i.e. no `B`-cell
`x` is reachable as `w₁→x` *and* `w₂→x` from two different cells `w₁, w₂` of the same merge. Under
that condition each merge-cell choice produces a *distinct* sink configuration, so the sink-label
determines every merge commitment and sink-labeling is a **correct** dedup key (equivalent to
`pending`). If on top of that the target collapses several `pending` signatures onto one sink-label,
it is also **smaller** — the potential win the idea is after.

**When it fails.** The premise breaks the moment a merge sits above a funnel. Then `m@w₁` and `m@w₂`
reach the *same* sinks (same label) but cost differently through the upstream arm, so "cheapest per
sink-label" keeps the downstream-cheaper cell and **silently drops the row the arm needs** — a
§6.2-shaped valid-but-suboptimal answer. Worked case: `A: p₁→m←p₂, m→s`; `B: w₁→x, w₂→x` (the
funnel); `E(m,w₁)=1, E(m,w₂)=2`, `p₂: b₁→w₁ (5), b₂→w₂ (0)`. Both routes end at sink `s@x`; the
optimum is `w₂` at cost **2**, but sink-dedup keeps the downstream-cheaper `w₁` and returns **6**.

**Verdict.** Conditionally correct, not unconditionally. Adopting it *safely* requires **proving the
target funnel-free below every merge** — a checkable property of each `B` window: where it holds it is
a valid, possibly-smaller key; where it does not it is silently wrong. Its intuition — "merge the
redundant rows" — is exactly Fix 1 (§8.1) made rigorous, keyed on cone-independence (**always** safe)
rather than shared sinks (safe only funnel-free). So the recommended shape is Fix 1 as the
unconditional path, with sink-labeling available as a **per-target optimisation behind a funnel-free
check** if the measurement shows the `B` windows satisfy it and it collapses meaningfully more rows.
**Measured (§8.5): unsafe on this target** — funnels are present on nearly every merge (3/3 on three
of four slow edges), so the funnel-free precondition fails. Shelved.

### 8.4 Plan and gate

1. **Measure** — **done (§8.5)**: Fix 1 gives 16–42× on 3 of 4 slow edges but **1×** on the
   all-coupled tail (`100350`); funnels pervasive, so §8.3 is out.
2. **Fix 2** first — base-shrink, unconditional, and the only lever for the all-coupled tail.
   **In progress** (design + exactness verification before any code).
3. **Fix 1** for the disjoint pairs; nested pairs stay on `pending`.
4. **Gate** identical to the §7 integration: cost parity + refusals on the 164-case suite, the
   structured envelope 384/384 with `C(cell) ≤ C(join)`, and benchmark parity against
   `extract_cell_vertex`. Both fixes are exact, so parity must hold to the digit — any divergence is
   a bug in the fix, not a tolerance.

> **Superseded (2026-07-18).** Fixes 1–3 were not implemented. The wall was closed instead by
> `docs/profiled_forward_table.md` — a forward table carrying a cost **per profile** (where the
> upstream splits are placed), which prices a split's children jointly and blocks the V3 phantom at
> construction. It is now `match_dag`'s default engine on width ≤ 2 sources. Measured on these same
> four edges: `100350` **687.7 s / 783 MB → 0.48 s / 15 MB**, and V3 violations 2 → 0 and 3 → 0. The
> per-merge analysis below stands as the diagnosis; only the proposed remedies are superseded.

Status (2026-07-17): wall diagnosed, repro reproducible, **measurement done (§8.5)**; **Fix 2 in
design** (exactness workflow), Fix 1 to follow, sink-labeling shelved (measured unsafe).

### 8.5 Measurement (2026-07-17)

On line `102752` and three other slow edges, with `prepare`+`forward` populating `D`, then per-merge
cell counts, the disjoint/nested split of merge pairs, and a funnel check. **"Factored"** = the row
count Fix 1 would leave: `∑` over the connected components of the merges' *coupling graph* (an edge
joins two merges when one is reachable from the other in the line-graph) of `∏ |cand(m)|` within each
component.

| edge | merges (cells) | full `∏` | Fix 1 factored | reduction | funnel merges |
|---|---|---|---|---|---|
| 102752 | 45, 45, 14 | 28 350 | 675 | **42×** | 3/3 |
| 100042 | 29, 17, 16 | 7 888 | 481 | **16×** | 1/3 |
| 100341 | 20, 19, 19 | 7 220 | 399 | **18×** | 3/3 |
| 100350 | 56, 55, 25 | 77 000 | 77 000 | **1×** | 3/3 |

* **Fix 1 helps substantially on most edges (16–42×)** — one merge is genuinely cone-disjoint and
  factors off. The hourglass merges are *not* all coupled; the independence is real.
* **But Fix 1 is not universal.** Line `100350` (the largest, `77 000`) has all three merges
  *transitively* coupled (two nested pairs chain them into one component), so factoring gives **1×** —
  the worst case is exactly where Fix 1 does nothing.
* **Fix 2 is therefore essential, not optional** — the only lever for the all-coupled tail, and it
  composes with Fix 1 everywhere (shrinks each surviving factor).
* **Sink-labeling (§8.3) is measured unsafe on this target.** Funnels are present on nearly every
  merge (3/3 on three of four edges), so the funnel-free precondition fails.

**Fix 2 effectiveness (D-only LB vs the `extract_join` incumbent).** The incumbent `UB` is obtained in
**~0 ms** on every blow-up edge (`extract_join` has no pending product), and `D > UB` removes cells:

| edge | `UB` | merge cells kept | product | Fix 2 |
|---|---|---|---|---|
| 102752 | 909 | 45→39, 45→39, 14→5 | 28 350 → 7 605 | 4× |
| 100350 | 339 | 25→5, 55→22, 56→22 | 77 000 → 2 420 | **32×** |
| 100042 | 402 | 16→4, 17→4, 29→11 | 7 888 → 176 | 45× |
| 100341 | 410 | 19→12, 19→13, 20→5 | 7 220 → 780 | 9× |

The load-bearing row is **`100350`**: Fix 1 gives `1×` there (all merges coupled), Fix 2 gives **32×**,
confirming Fix 2 is the tail's only lever. Fix 2 is weakest exactly where Fix 1 is strongest (102752:
4× vs Fix 1's 42×) and strongest where Fix 1 is weakest — and the two **compose** on the surviving
factors. The `D`-only pruning is exact (§8.2 proof). But per §8.2's measured limitations, this lever is
loose and not always available — so the coupled tail wants §8.6.

### 8.6 Attack the coupling — eliminate the inner merge (proposed)

The extraction runs on the **line-graph DAG** `A` — **not a tree** (line-graphing the source's
$\text{degree}\ge 3$ vertices creates cycles). Fix 1 factors merges whose cones are disjoint; this
handles a **nested** pair. Recall the doc's convention: a **merge** is $\operatorname{indeg}(m)\ge 2$, a
**split** is $\operatorname{outdeg}(J)\ge 2$. On line 102752 the three merges have
$\operatorname{indeg}\,2,2,3$ and $|\mathrm{cand}|=45,45,14$: two are pairwise disjoint (Fix 1's job),
and the $\operatorname{indeg}\,2$ merge $m_1$ ($|\mathrm{cand}|=45$) is an **ancestor** of the
$\operatorname{indeg}\,3$ merge $m_2$ ($|\mathrm{cand}|=14$), joined by a single directed
**through-path** $P$ of four $\operatorname{indeg}=1,\operatorname{outdeg}=1$ vertices. Measured: deleting
$m_2$ isolates its down-cone, so **$m_2$ is a separator**; $m_1$ is not (line-graph reconvergence).

**Costs live on cells, and we eliminate one variable — not two.** The cost is a sum of per-**cell** terms
$w\cdot E(X,v)$ (§0), and the optimum $C^{*}$ is a min over **every** vertex's cell — *not* over
$(u,v)$; there is no global $C(u,v)$. So eliminate $m_2$'s cell alone. Because $m_2$ separates its cone,
the per-cell partial minimum

$$
\mathrm{Below}(m_2,v)\;=\;\min\big\{\,\text{cost of the sub-matching on } m_2\text{'s down-cone}\ \big|\ m_2\text{'s cell} = (m_2,v)\,\big\}
$$

is well defined — one value per cell $(m_2,v)$. The path $P$ carries that cell up to $m_1$; summing $v$
out gives the **message**

$$
\mu(m_1,u)\;=\;\min_{v}\Big[\ \mathrm{PathCost}\big((m_1,u)\to(m_2,v)\big)\;+\;\mathrm{Below}(m_2,v)\ \Big],
\qquad\text{one value per cell } (m_1,u),
$$

where $\mathrm{PathCost}$ is the DTW cost of warping $P$ between the two endpoint cells ($=\infty$ if
none). $\mu$ is exactly the message $m_1$ already receives from its down-child in the backward sweep —
but with $(m_2,v)$ **summed out at the path** instead of ridden to the root as pending. Fold $\mu$ into
$m_1$'s cell cost and the problem has one fewer merge; finish it as before (the disjoint $45$-merge still
factors by Fix 1). The inner merge's $|\mathrm{cand}(m_2)|$ has collapsed into $\mu$: the coupled joint
$45\cdot 14$ becomes a $45$-entry message.

**Why it is exact.** $\min$ distributes over $+$ across disjoint cell-sets, and $m_2$ reaches the rest of
$A$ only through $P\to m_1$ (separator), so $\mu$ depends on the cell $(m_1,u)$ alone — arity $1$:

$$
\min_{\text{rest},\,v}\big[\,R(\dots,u) + \mathrm{PathCost} + \mathrm{Below}(m_2,v)\,\big]
\;=\;\min_{\text{rest}}\big[\,R(\dots,u) + \mu(m_1,u)\,\big],
$$

valid because $R$ (everything not below $m_2$) has no $v$. This is one step of **min-sum variable
elimination**; operationally it is a new discharge point (§3.5) — sum $m_2$ out at the path, for the case
where the common ancestor a tree would give does not exist. It needs no incumbent and no bound (unlike
Fix 2). Baseline for scale: `extract_cell` carries **5,087** multi-merge signatures of the **28,350**
product (the infeasible $(u,v)$ are already dropped — $\mathrm{PathCost}=\infty$), against **99** for a
lone merge.

**Complexity & scope.** A chain of $k$ nested merges, eliminated inner-first, costs $O(k\,d^{2})$ with
$d=\max_m|\mathrm{cand}(m)|$, versus $d^{\,k}$. Exact whenever each eliminated merge is an arity-$1$
separator. When it is not — merges mutually coupled, or a split $J$ ($\operatorname{outdeg}\ge 2$)
forcing $\mu$ to depend on more than one cell — the message arity equals the **treewidth** of the
coupling graph: general junction-tree elimination, the open case.
