# The Profiled Forward Table

Every cell carries a cost **per profile** — per placement of the upstream splits — so a split's
children are priced jointly instead of each choosing independently.

Implemented as `network_matching/profiled.py`; **not wired into `match_dag`**. Measurements and gate
results live in `report/profiled_forward_table_measurements.md`.

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

### 5.3 The steps

```
  1. FACTORS      one per sink: its best cell for each profile it can carry

  2. ELIMINATE    for each split J, DEEPEST FIRST:
                      collect the factors mentioning J
                      combine them          (consistent pairs, costs summed)
                      minimise J out        -> one factor over the remaining keys
                      keep the `keep` cheapest rows per key

  3. COMBINE      join whatever factors remain (all keyless now) -> candidate list,
                  sorted by cost

  4. JUDGE        cheapest-first, take the first candidate that
                      covers every vertex  and  passes check_rules

  5. RECONSTRUCT  flood bp from the winning (sink, cell, profile) picks;
                  cover chains expand into the run cells
```

**Deepest-first is what bounds it.** When `J` is eliminated the only key its factors still share is
`J`'s **parent** split, so intermediate factors are bounded by the tree's **depth**, not its total
split count.

### 5.4 Why step 2 keeps `keep` rows, and step 4 exists

`π` enforces V3 and the recurrence enforces V2, but **V1 is not covered**. V1 is the *non-crossing*
rule (`dag_dtw_matching.md` §3):

$$\forall\,(a,v) \in M,\ \forall\,a^- \in \mathrm{Apred}(a),\ \forall\,v^+ \in \mathrm{Bsucc}(v):\quad (a^-, v^+) \notin M$$

— a *predecessor* of `a` may not sit on a *successor* of `v`; the matching must not run backwards. A
cyclic `B` is what makes it bite, because `Bsucc(v)` can wrap around, so a cell that looks earlier is
reachable as a successor and crossing becomes possible. Nothing in the forward pass rules it out, so
it is only detectable once a complete matching exists.

Keeping only the minimum per key lets a cheap V1-invalid row hide a valid costlier
one, so the elimination retains the `keep` cheapest and the judge picks the first that survives
`check_rules`. This is the *"top-K contraction"* `scripts/repro_contraction_eviction/README.md` asks
for. `keep` never affects cost parity — only how many otherwise-refused cases can be answered.

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
one level, not a cure.

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
| **`keep`** | a measured plateau, not a proven bound. Failure mode is a refusal, never a wrong answer |
| **no global budget** | `max_profiles` bounds one cell, `max_rows` one factor; neither bounds the aggregate — which is what `btree(5)` exhausts |
| **not adopted** | nothing calls `profiled.py` |

## 8. Relation to Existing Machinery

| existing | relation |
|---|---|
| §4.1a forbid-and-rebuild | **prerequisite, not replaced.** It answers *"which exits can no child use?"* from transition existence alone; this answers *"which exit should all children take?"* by pricing them jointly. `forward()` keeps the `forbidden` flags and the feasibility error |
| `pending` (`cell_dag_extraction.md` §2–3) | the same job keyed on merges instead of splits |
| §3.5 early discharge | the backward mirror of §2.4 |
| §8.6 inner-merge elimination | the same min-sum elimination, applied to one merge rather than every split |
| `check_forward_v3` | today a diagnostic; under this design it must return **empty** — an invariant |
