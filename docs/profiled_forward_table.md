# The Profiled Forward Table

Every cell carries a cost **per profile** — per placement of the upstream splits — so a split's
children are priced jointly instead of each choosing independently.

Implemented as `network_matching/profiled.py`. **`match_dag` defaults to `engine="auto"`**, which
dispatches to this engine, to `"cell"`, or to §8's re-based variant according to the source's shape
(§9). All gates green: unit suite 198, envelope 384/384, cyclic-B 731/731 with 0 invalid.
Measurements live in
`report/profiled_forward_table_measurements.md`.

---

## 1. The Problem

`dag_dtw_matching.md` §4.1's forward table computes `D[a][v]` = the cheapest match of `a`'s upstream
cone with `a`'s run ending at `v`. It minimises over **all** upstream configurations, including
invalid ones.

The invalid ones are **phantoms**: one vertex placed on two cells at once. They arise at a **split**
(`outdeg ≥ 2`) because each child's row is filled independently and each picks the split-cell that
suits it alone. §4.1a's coupling removes exits *no child can use*; where several exits are usable by
every child — the normal case — it keeps them all and the children still disagree.

**No single `D` is wrong.** `D[a][v]` equals the minimum over the profiled table at every cell. A
phantom needs two branches disagreeing about one split, so it appears only when cells are
**combined**: `cell_dag_extraction.md` §6.1's `D[c₁][u] = 0.5` and `D[c₂][d] = 0.5` are each correct;
their *sum* is the phantom.

So this design does not make values smaller. **It labels each value with what it assumed**, so values
can be combined consistently.

---

## 2. Definitions

### 2.1 The profiled set `S`

$$S \;=\; \{\, s \in V(A) \;:\; \operatorname{outdeg}(s) \ge 2 \,\} \qquad\text{— the splits.}$$

A vertex's cell can only be disputed if **two downstream branches carry it**, and a branch point is
exactly `outdeg ≥ 2`. A vertex with `outdeg = 1` is seen by one branch and can never be contradicted.

The branches need **not** rejoin: a matching assigns every vertex one run *globally*, so two sinks
with disjoint descendants still conflict at assembly. `S = ∅` ⇒ the design is a no-op.

*Not source cells.* V3 binds the cell a **split's run ends on**. Every split lies downstream of its
source ancestors, so both children of a split inherit the *identical* source-cell assignment — a
source profile cannot see the conflict at all.

### 2.2 A profile

> **A profile is a set of cells — one cell per live split.**

Each element `(J, v)` is a cell in the ordinary sense: an A-vertex paired with a B-vertex. It reads
*"split `J`'s run ends on B-vertex `v`"*. Concretely a `frozenset` of `(A_split, B_vertex)` pairs.

House convention: `a, b, c, p, m, J, X` are A-vertices; `u, v, w, x` are B-vertices.

```python
frozenset({ (J1, u),      # split J1's run ends on B-vertex u
            (J2, w) })    # split J2's run ends on B-vertex w
```

| property | consequence |
|---|---|
| **set, unordered** | two rows agreeing on the same placements collide correctly when contracted |
| **at most one cell per split** | a vertex has one run; `{(J1,u), (J1,w)}` is the contradiction §2.3 rejects |
| **`frozenset`** | hashable, so it can key a dict |
| **`frozenset()` is legal** | no live splits upstream, or all discharged |

**Width** = `|π|`, the number of live keys — *structural*, a property of `A`'s topology. No
hyperparameter sets it. `r`, `k_min`, `α`, `β` change `|cand|`, which drives **multiplicity** (how
many profiles a cell holds), not width.

*Segment mode.* On a line graph every vertex name is itself a `(u, v)` tuple, so both halves become
tuples: `frozenset({(('c','d'), ('u','v'))})`. Same structure, different names.

### 2.3 Consistency

Two profiles are **consistent** iff they agree on every key both name. A vertex has one run, so
disagreement means no matching realises the combination. On split cells, this test **is** V3.

### 2.4 Discharge

> Drop `s` from the profile at vertex `a` **iff `a` post-dominates `s`** — every path from `s` to any
> sink passes through `a`.

Below `a` only one branch continues, so nothing can contradict `s` again and the key is dead weight.
Computed once from a post-dominator tree (`nx.immediate_dominators` on the reversed graph, rooted at a
virtual super-sink).

**Discharge is a minimisation, not a deletion.** Rows that differed only in the dropped key become
identical; keep the cheapest.

```
{J@u}: c1 ┐
{J@v}: c2 ├─ drop J ─→  {} : min(c1, c2, c3)
{J@w}: c3 ┘
```

Dropping a key without taking that min is silent corruption.

Forward mirror of `cell_dag_extraction.md` §3.5's early discharge — *first common ancestor going
backward* is *post-dominator going forward*.

### 2.5 The life of a key

Four events touch a profile; nothing else does.

| event | where | effect |
|---|---|---|
| **born** | at a split `a ∈ S` | add `(a, v)` |
| **carried** | neither split nor discharge point | unchanged — literally the same object |
| **run-end moves** | α-coverage step at a split | overwrite `(a,v') → (a,v)` |
| **merged** | at a merge | arms must agree; union, or drop the pair |
| **discharged** | at the split's post-dominator | remove the key, collapse rows |

Traced on `a → J → {p, q} → m → t`:

```
 a   source                      {}
 J   SPLIT — key born            {J@v}
 p   carries parent's profile    {J@u}, {J@v}, {J@w}
 q   carries parent's profile    {J@u}, {J@v}, {J@w}
 m   MERGE + DISCHARGE J         {}            <- arms agree, then the key dies
 t   nothing live                {}
```

`J`'s key exists **only between `J` and `m`**. That bounded lifetime is what keeps the table small.

---

## 3. The Data Structure

`prepare()` already gives every A-vertex a table of candidate cells. This adds **one key** to each
cell record:

```
A.nodes[a]                       one A-vertex's attributes
    ├── "x", "y", …              geometry
    └── "cand"                   its candidate B-vertices, gated to within r of a
        └── [v]                  the record for cell (a, v)
            ├── "E"      float   emission cost of pairing a with v
            ├── "D"      float   forward cost — ONE number
            ├── "bpD"    list    its back-pointers
            ├── "forbidden"      §4.1a: not a valid run END
            └── "Dp"     dict    <- ADDED:  { profile : (cost, bp) }
```

| field | |
|---|---|
| `profile` | the key — a set of cells (§2.2) |
| `cost` | the cheapest way to reach `(a,v)` under that placement |
| `bp` | `[(vertex, cell, profile), …]` — one triple per predecessor, or a single **same-vertex** triple for a coverage step, mirroring `bpD`'s convention |

**`Dp` does not replace `D`.** The profiled path never reads `D`; it reads `E`, `forbidden`, `Dp`. `D`
survives because `forward()` must run first to produce the `forbidden` flags, and computing `D` is how
it gets them.

A **passthrough** vertex — neither split nor discharge point — does not change the profile, so its
rows keep the **parent's frozenset object** rather than a rebuilt equal one.

---

## 4. Algorithm, Phase 1 — Building the Table

Sweep `A` in the §4.0 layer order. For each vertex `a`, for each candidate cell `v`:

Entry cells for parent `p` into `(a,v)` are `x ∈ ({v} ∪ Bpred(v)) ∩ cand(p)` — a **stall** at `v` or
an **advance** from a B-predecessor. That set is bounded by B's local in-degree (1–3 in a road
network), **not** by `|cand(p)|`.

```
fill(a, v):

  1. FOLD the predecessors, one at a time, keyed by (profile, any-stall):

       for each parent p:
           options = [(entry cell x, is_stall, parent profile pi_p, cost/outdeg(p))]
           for each running combo and each option:
               pi = merge(combo.pi, pi_p)         <- consistency test (§2.3); skip if None
               keep the cheapest per (pi, stall)

  2. PRICE the emission:   beta*E(a,v)  if any parent stalled,  else E(a,v)

  3. COVERAGE (H):  relax within the row to a fixed point —
       cost(a,w) <- cost(a,v) + alpha*E(a,w)   for each B-arc v->w,  per profile

  4. OWN KEY:  if a in S, overwrite the pair (a, v) in every profile

  5. DISCHARGE:  remove every key in drop[a];  rows that now collide keep the cheapest
```

Four points the steps make that prose blurs:

* **The fold is joint, never per-parent.** Minimising each parent separately and then testing the
  winners is the algorithm `cell_dag_extraction.md` §6.2 shows fails.
* **Folding, not enumerating.** The running table is bounded by the number of *consistent* profiles,
  not by the product of the parents'.
* **Step 4 overwrites, it does not accumulate** — the entry means "the cell this split's run currently
  ends on", and a coverage step moves it. Consumers attach at the run end, so this is exactly V3.
* **Step 5 is a min** (§2.4).

A source has no predecessors: `Dp = {pi0: (E, [])}`, with `pi0 = {(a,v)}` if `a ∈ S` else
`frozenset()`.

---

## 5. Algorithm, Phase 2 — The Extraction

At a **sink** `t` the upstream cone is everything above `t`, and the sinks' cones cover `A`. So the
extraction is not a search — it is a **join over profiles**.

### 5.1 The objective

Each sink is a **factor** over its live splits:

$$f_t(\pi) \;=\; \min_{v \in \mathrm{cand}(t)} \widehat{D}[t][v][\pi|_t]
\qquad\text{and}\qquad
C^{*} \;=\; \min_{\pi}\ \sum_{t} f_t(\pi|_t).$$

### 5.2 Why the sinks cannot simply be folded together

Combining sinks pairwise keys the running table by the **union** of their keys. Two sinks sharing no
key always merge successfully, so a pairwise fold enumerates their **cross product** — on a 16-sink
out-tree that exhausts memory while the forward table it reads is a few megabytes.

> **A rejected fix, recorded because it looks right.** "Group sinks that share a key, minimise each
> group, sum" does nothing on an out-tree: every sink descends from the root split, which is never
> discharged, so all sinks share that key and the grouping gives one component.

The elimination in step 2 is the *same* operation the forward table already performs at every merge
(§4): `_merge` for consistency, then the cheapest per key, then discharge. The forward pass stays
small because a key dies at its immediate post-dominator, so no vertex ever holds more than `W` live
keys. A sink has nothing downstream to discharge into, so it arrives holding its full live set and
the join must reconcile every split at once — which is why the extraction, not the table, is the
phase that runs out of memory.

### 5.3 The steps

```
  1. FACTORS      one per sink: its best cell for each profile it can carry

  2. ELIMINATE    for each split J, in MIN-FILL order:
                      collect the factors mentioning J
                      combine them          (consistent pairs, costs summed)
                      minimise J out        -> one factor over the remaining keys
                      one row per key: the cheapest

  3. COMBINE      join whatever factors remain (all keyless now) -> candidate list,
                  sorted by cost

  4. JUDGE        cheapest-first, take the first candidate that
                      covers every vertex  and  passes check_rules

  5. RECONSTRUCT  flood bp from the winning (sink, cell, profile) picks;
                  cover chains expand into the run cells
```

**The order is what bounds it.** Step 2 is variable elimination, so its cost is set by the **induced
width** — the largest key set any single elimination step has to hold. Order chooses that width.
Min-fill eliminates the split whose removal unions the fewest factor scopes; the obvious alternative,
deepest-first, can force a step to hold every split at once:

| line | splits | deepest-first | min-fill |
|---|---|---|---|
| 100350 | 2 | width 2, 3 072 rows | width 2, 3 072 rows |
| 100935 | 5 | width 5, 1 300 068 000 rows | **width 4, 18 572 400 rows** |

Row counts are upper bounds — the product of candidate cells over the step's keys. Joint reachability
prunes them hard in practice (line 102752 carries 5 087 of a 28 350 product), so treat them as a
ratio between orders, not an absolute.

**Rows carry pick chains by reference.** A row is `(cost, picks)`, and a joined row's `picks` is the
concatenation of its two parents'. Building that eagerly costs more than the cost it accompanies —
measured ~16 KB per row, which is what turns a large factor into gigabytes. Rows therefore store a
`(left, right)` pair and the chain is flattened only for the candidate the judge accepts.

### 5.4 Why step 4 exists

`π` enforces V3 and the recurrence enforces V2, but **V1 is not covered**. V1 is the *non-crossing*
rule (`dag_dtw_matching.md` §3):

$$\forall\,(a,v) \in M,\ \forall\,a^- \in \mathrm{Apred}(a),\ \forall\,v^+ \in \mathrm{Bsucc}(v):\quad (a^-, v^+) \notin M$$

— a *predecessor* of `a` may not sit on a *successor* of `v`; the matching must not run backwards. A
cyclic `B` is what makes it bite, because `Bsucc(v)` can wrap around, so a cell that looks earlier is
reachable as a successor and crossing becomes possible. Nothing in the forward pass rules it out, so
it is only detectable once a complete matching exists.

Keeping only the minimum per key lets a cheap V1-invalid row hide a valid costlier one, so the judge
needs **fallbacks** — the *"top-K contraction"* `scripts/repro_contraction_eviction/README.md` asks
for. Where they are kept decides whether they are affordable.

> Factors carry one row per key throughout the elimination, **except the last step**, which retains
> `JUDGE_FALLBACKS = 32`. `_judge_fallbacks(B)` returns 1 when `B` is acyclic, since V1 needs a
> B-cycle to be reachable at all.

A `keep` parameter once applied the multiplier at *every* step, and that is what made it expensive —
it compounds down the chain rather than being paid once:

| re-based extraction | at every step (`keep=512`) | last step only |
|---|---|---|
| line 100350 | 0.67 s · 96 MB | **0.24 s · ~4 MB** |
| line 100935 | 19.36 s · 2522 MB | **4.14 s · 164 MB** |

Applied at the narrowest point the cost is unmeasurable against no fallbacks at all, and it recovers
almost all the reach:

| cyclic-B gate | answered where `extract_cell` refuses |
|---|---|
| fallbacks at every step | 168 |
| **fallbacks at the last step** | **166** |
| none | 0 |

The two missing cases need alternates that a mid-chain step already discarded. Cost parity and
validity are unaffected at every setting (§10) — fallbacks only ever change how many otherwise-refused
cases can be answered, never what an answer is.

### 5.5 Why summing the sinks is exact

The `1/outdeg` split fractions are **not** an approximation once V3 holds — they are exactly the
weights that make a sink-sum count each vertex once. A vertex's emission enters a descendant's cost
scaled by `∏ 1/outdeg` along the connecting path; summed over every path from it to a sink those
factors total **1**. A merge sums its arms without dividing, so mass arriving by two arms recombines
to 1 as well.

On `cell_dag_extraction.md` §6.1 with the split forced onto one cell: `0.5 + 5.5 = 6.0`, the true
cost. The `1.0` that section reports as the failure is `E(J,v1)/2 + E(J,v2)/2` — half of each cell.
**The under-count *is* the V3 violation**, so blocking the phantom fixes the arithmetic too.

---

## 6. Worked Example

One split, one chain — the smallest case where a profile does anything:

```
A:  a ──→ J ──→ c          J is the only split
              └─→ d
B:  u ──→ v ──→ w ──→ x

candidate cells of J :  u, v, w, x
candidate cells of c :  w, x
```

```
cell (c, w)  holds 2 profiles:
     cost   9.891   when   J ends on v
     cost  11.953   when   J ends on w

cell (c, x)  holds 3 profiles:
     cost   9.242   when   J ends on w
     cost  14.764   when   J ends on x
     cost  14.990   when   J ends on v
```

Read `(c, w)` as: *pairing `c` with `w` costs 9.891 **if** `J`'s run ended on `v`, or 11.953 if it
ended on `w`*. Same pairing, two prices, because the upstream differs. The `min` of each list is what
`D` holds.

**Why every row is kept.** If sibling branch `d` can only be matched with `J` on `w`, then `(c, w)`
cannot use its cheapest row at 9.891 — it must use **11.953**, and the optimum is whichever total is
smallest once *both* branches are priced under the *same* placement of `J`. Collapsing the list to its
minimum throws away the row the optimum needs.

**Width > 1.** With a second split downstream (`a → J1 → {c,d}`, `c → J2 → {e,f}`), a cell below both
carries one pair per split:

```
cell (e, x)   live splits: J1, J2      width 2
     cost  14.814   when   J1 ends on v,  J2 ends on x
     cost  16.829   when   J1 ends on w,  J2 ends on x
     cost  17.586   when   J1 ends on v,  J2 ends on w
```

The rows enumerate the combinations — each extra live split **multiplies** the row count rather than
adding to it. That graph is also the out-tree limit in miniature: neither split's branches rejoin, so
no key is ever discharged.

---

## 7. Limits

### 7.1 Out-trees — exponential in depth

Nothing post-dominates on a tree, so **no key is ever discharged** and width equals the depth. A
factor over `depth` keys costs `∏ |cand|` over them, so the whole thing is **exponential in depth** —
the mirror of `pending`'s tree-of-merges wall, and not something the §5.3 elimination removes. What
elimination does remove is the dependence on the *total* split count: intermediate factors are bounded
by depth, so 15 or 63 splits are equally fine as long as the tree is shallow.

Measured on `btree` (`extract_cell` for comparison):

| depth | `\|A\|` | splits | `extract_cell` | profiled | memory |
|---|---|---|---|---|---|
| 3 | 29 | 7 | 0.002 s | 0.034 s | 0.8 MB |
| 4 | 61 | 15 | 0.005 s | 0.344 s | 31.9 MB |
| **5** | 125 | 31 | 0.018 s | — | **MemoryError** after 17 s |
| **6** | 253 | 63 | 0.072 s | — | **MemoryError** after 47 s |

So the usable ceiling is **depth 4**. Before the §5.3 elimination it was depth 3, so that fix bought
one level, not a cure. **§8 removes the limit outright** by changing what a cost means.

**The "fake branch" idea does not transfer.** At a merge, `extract_cell` lets one arm absorb while the
others merely *tag* the merge cell in `pending`, resolving the tag later at the discharge — it works
because the arms **meet**. At a split the branches **diverge**, and on an out-tree they never meet, so
a tag has no point at which it could ever be cashed. Dropping a split's key from one branch makes that
branch's cost independent of the split's placement, which is precisely the joint pricing this design
exists to do. The variant that *is* available — recompute the branch's subtree once per split cell
instead of carrying the key — trades the memory for `|cand(J)|` × subtree work, i.e. the same
exponential paid in time.

**Unexploited:** a split with exactly **one** surviving exit contributes a constant key and could be
dropped from `S` outright. Measured: one such split per `btree` (the root, pruned to a single cell by
§4.1a). A real width reduction, but worth one key out of `depth`.

### 7.2 Other limits

| | |
|---|---|
| **no judge fallbacks** | factors keep one row per key, so a cheap V1-invalid row can hide a valid costlier one (§5.4). Failure mode is a refusal, never a wrong answer — but 168 cyclic-B cases are refused that need not be |
| **no global budget** | `max_profiles` bounds one cell, `max_rows` one factor; neither bounds the aggregate — which is what `btree(5)` exhausts. `max_rows` is now the binding constraint on line `100935` |
| **`rebase`'s quadrant is narrow** | it wins only when *both* pressures are high (§9) — nested splits **and** several concurrently-open merges. On either alone, `"cell"` or this engine beats it |

## 8. Re-basing — removing the depth limit

`PROFILED_REBASE=1`. Default **off**; the path described above is unchanged when it is unset.

### 8.1 The one change

| | §1–§5 (default) | re-based |
|---|---|---|
| `Dp[a][v][π]` | cost of `a`'s **whole upstream cone** | cost **since the last split** |
| the key `π` | **all** live ancestor splits | the **last** split only (a set at merges, one per arm) |
| width on a tree | = depth | **1, at any depth** |
| total | sum over sinks | sum over **segments** |

### 8.2 Why it works

The depth blow-up is not caused by the *screening* — given `J`'s cell, nothing below `J` depends on
where `P` ended. It is caused by the *accounting*: `Dp` is cumulative, so `cost(P → J)` sits inside
**both** of `J`'s children, and if the key for `P` is dropped they minimise that shared cost
independently and can pick **different** cells of `P`. That is the phantom, one level up.

Re-basing puts `cost(P → J)` in exactly one place — `J`'s own segment — so nothing below `J` contains
it and there is no shared quantity left to disagree about. Last-split keying is then exact.

### 8.3 What changes in the algorithm

Only two steps differ; §5.3's elimination and judge are untouched.

```
PHASE 1, new step 6:   at a split, BANK (cost, advance-bp, run cells) as the split's own
                       segment factor, then RESET the accumulator to 0.
                       The cover run is walked at bank time, while this row's pre-reset
                       profile keys are still valid.

PHASE 2, step 1:       factors are one per SPLIT (parent-key x own cell) as well as one
                       per sink.

PHASE 2, step 5:       reconstruction REPLAYS the banked chain -- a pick carries
                       ("SEG", split, run cells, advance-bp) and needs no lookup.
```

That last point is the one that took three attempts. Two earlier versions re-derived the chain by
matching profiles against an assignment recovered from the elimination; both were ambiguous exactly
where `B` has cycles, and each passed one corpus while collapsing the other. **Carry the
back-pointer, never infer it** — the same rule the default path already follows.

### 8.4 What it buys and what it costs

| `btree` depth | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| default: profiles/cell | 14 | 61 | 263 | — | — |
| default | 0.03 s | 0.18 s | 25.7 s | `MemoryError` | `MemoryError` |
| **re-based: profiles/cell** | **5** | **5** | **5** | **5** | **5** |
| **re-based** | 0.03 s | 0.07 s | 0.73 s | 3.6 s | 15.9 s |

Width **1 and 5 profiles per cell, flat from depth 3 to 7**, all costs exact. The depth exponential
is gone.

But re-based is still **16–40× slower than `extract_cell`** on this family and the gap widens with
depth. The *memory* wall is removed; the constant factor is not. `extract_cell` remains the right
engine for out-trees.

**On the hourglass it changes nothing** — width 2 either way, all four edges exact. This buys
out-trees and only out-trees.

### 8.5 Why re-basing paid most for `keep`

Re-basing builds a factor per segment **and** per sink, and eliminates across all of them, where the
default path builds one factor per sink. A per-key row budget therefore multiplied far more
intermediate results here, which is why removing it (§5.4) recovered 5-19x on this path alone and
nothing on the default one.

---

## 9. Choosing an Engine

**`cell` and this engine are not complements.** They fail on *independent* pressures, so a source can
trip both at once — and then neither is usable:

| | few open merges | many open merges |
|---|---|---|
| **shallow splits** | all fast | **profiled** — the hourglass: `cell` 4.8–688 s, profiled 0.1–0.8 s |
| **nested splits** | **cell** — `btree`, `ladder`: `cell` 5 ms, profiled 34 s or worse | **both bad → `rebase`** (§8) |

The two axes, both pure topology and both computed before any cell work:

| | what it measures | what it kills |
|---|---|---|
| `profiled_width(A)` | max **live profile keys** at any vertex (§2.4's discharge applied) | **this engine** — the key grows with nesting depth |
| `merge_pressure(A)` | max **concurrently-open merges**, mirroring `extract_cell`'s own early-discharge rule | **`cell`** — `pending`'s product is over exactly these |

> `merge_pressure` is **not** a merge's in-degree. One merge of in-degree 10 is still *one* open merge
> and costs `pending` one factor. The wall that motivated all of this was **three** concurrently-open
> merges at 45×45×14.

```
W  = profiled_width(A)
Mo = merge_pressure(A)

W <= 2       ->  "profiled"
Mo >= W      ->  "rebase"
otherwise    ->  "cell"

then, if that engine's estimate exceeds max_work:  take the cheapest-estimate engine
                        if that one exceeds too:  refuse
```

### 9.1 The refusal gate estimates the engine it chose

Three estimates, all read off the forward table in under a millisecond:

| | function | models |
|---|---|---|
| `cell_rows` | `predict_work(A)[0]` | the product `pending` enumerates over merges |
| `profiled_rows` | `predict_work(A)[1]` | one cell's `Dp` — a product over the widest live split set |
| `rebase_rows` | `rebase_work(A)` | same sweep, but a split **resets** the live set to `{itself}` |

The order matters: **pick the engine first, then gate on that engine's number.** Gating on
`min(cell, profiled)` was wrong, because neither describes re-basing:

| line 100935 | estimate | verdict |
|---|---|---|
| `cell` | 789 462 244 | hopeless |
| `profiled` | 1 300 068 000 | hopeless |
| **`rebase`** | **331 650** | answers in 0.24 s |

`W = 5, Mo = 5` routes it to `rebase`, but the old gate refused it on the other two engines' numbers
first — a 4 000x misjudgement of the engine it had just chosen. `rebase_work` is lower for exactly the
reason §8 exists: the reset means no vertex carries splits from above it.

**The merge threshold scales with `W`; it is not a constant.** Re-basing wins exactly when **every**
nested split's branch rejoins — then `cell`'s `pending` and this engine's key are *both* maximally
loaded. Below that, some path through the nesting stays merge-free and `cell` remains cheap.

Sweeping the two axes independently (`report/probe_pressure_sweep.py` holds `W ≈ k` while varying how
many branches rejoin, so `Mo ≈ j`):

| `W`, `Mo` | profiled | `cell` | `rebase` | best |
|---|---|---|---|---|
| 5, 0 | 1.770 s | **0.003 s** | 0.015 s | cell |
| 5, 2 | 1.103 s | **0.007 s** | 0.014 s | cell |
| 5, 3 | 1.209 s | **0.037 s** | 0.084 s | cell |
| 5, 4 | 1.204 s | **0.341 s** | 0.383 s | cell |
| **5, 5** | 1.673 s | 2.073 s | **0.294 s** | **rebase** |

`cell` wins at every `Mo < W` and loses only at `Mo = W`. A fixed threshold of `Mo >= 2` misroutes 4
of 11 swept cases — and the misroute is dangerous in one direction, because `rebase` **fails
outright** on the `ladder` family, so sending a `cell`-friendly source there turns a 5 ms answer into
a refusal.

With `Mo >= W` the rule agrees with the measured winner on **20 of 20** families.

Explicit engines remain available: `"profiled"`, `"rebase"`, `"cell"`, `"join"`, `"all"`. Both
`match_dag` and `DuckDBMapMatcher.match_dag` route through one shared `extract_by_engine`, so the
engine set cannot drift between them.

---

## 10. Relation to Existing Machinery

| existing | relation |
|---|---|
| §4.1a forbid-and-rebuild | **prerequisite, not replaced.** It answers *"which exits can no child use?"* from transition existence alone; this answers *"which exit should all children take?"* by pricing them jointly. `forward()` keeps the `forbidden` flags and the feasibility error |
| `pending` (`cell_dag_extraction.md` §2–3) | the same job keyed on merges instead of splits |
| §3.5 early discharge | the backward mirror of §2.4 |
| §8.6 inner-merge elimination | the same min-sum elimination, applied to one merge rather than every split |
| `check_forward_v3` | today a diagnostic; under this design it must return **empty** — an invariant |
