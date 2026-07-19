# Low-Memory Extraction

How the profiled engine's extraction phase (`extract_profiled`, `_extract_rebased`) is kept inside a
memory budget. Companion to `profiled_forward_table.md`, which specifies the phase itself in §5.

---

## 1. The problem

The forward table is not the phase that runs out of memory. The starting point for everything below —
hourglass line `100935`, whose re-based table builds fine and whose extraction then exhausts the
machine:

| phase | line 100350 | line 100935 |
|---|---|---|
| `forward_profiled` | 0.20 s / 5 MB | 3.7 s / 142 MB |
| `extract_profiled` | 0.10 s / 3 MB | **out of memory** |

§4 has the current numbers.

## 2. Why the extraction is the wide phase

Extraction is **variable elimination** over the split keys — the same operation the forward table
performs at every merge: `_merge` for consistency, keep the cheapest per key, then discharge.

The forward pass stays small because **a key dies at its immediate post-dominator**. No vertex ever
holds more than `W` live keys (`W = 5` on `100935`, and `1` under re-basing).

A sink has nothing downstream to discharge into. Every sink therefore arrives still holding its full
live set, and the join has to reconcile all splits at once. That is the whole difference.

## 3. What bounds it

The cost of an elimination is its **induced width** — the largest key set any single step must hold.
Two things set the memory:

    memory  ~=  (rows in the widest factor)  x  (bytes per row)

and there is a change for each.

### 3.1 Rows — elimination order

Order chooses the width. Measured on factor scopes alone (which splits each factor mentions, no costs
involved, microseconds to compute):

| line | splits | deepest-first | min-fill |
|---|---|---|---|
| 100350 | 2 | width 2, 3 072 rows | width 2, 3 072 rows |
| 100935 | 5 | width 5, 1 300 068 000 rows | **width 4, 18 572 400 rows** |

Deepest-first forces one step to hold all five splits. Min-fill — repeatedly eliminate the split whose
removal unions the fewest factor scopes — never holds more than four, worth **70x** for the same
answer. It never costs anything: on `100350` both orders give width 2.

Row counts are upper bounds, the product of candidate cells over a step's keys. Joint reachability
prunes them hard in practice (line `102752` carries 5 087 of a 28 350 product), so read them as a
ratio between orders, not as absolutes. On `100935` the measured elimination never exceeds 58 989
rows against that 18 572 400 ceiling — width sets the exponent, reachability collapses it.

### 3.1a Rows — never materialise a segment factor

Under re-basing the widest object is not a sink. It is the **segment factor** of a split `J`, keyed by
`_merge(parent_profile, {(J, cell)})` — the cross product of `J`'s parent profiles with `J`'s own
cells. On `100935` that is `seg[v17]`: 1 486 parents x 59 cells = **58 989 rows**, the only factor in
the whole elimination above the shipped `max_rows = 50 000`.

Every one of those rows exists to be minimised away in the very next step, and it never needs to
exist at all, because **everything below `J` is conditionally independent of everything above `J`,
given `J`'s cell**. That is exactly what the re-base reset buys (§8 of `profiled_forward_table.md`):
at a split the profile becomes `{(J, v)}` and the cost accumulator resets, so no branch below `J`
carries any quantity from above it.

So `J`'s elimination fuses into one streaming pass:

```
  1. combine the other factors mentioning J        (small -- elim[v16] is 38 rows)
  2. index those rows by J's cell
  3. stream SEG[J]'s (parent_profile, cell) entries against that index,
     accumulating min into out[parent_profile]
```

Peak becomes the parent-profile count, not the product:

| split | as a standalone factor | fused |
|---|---|---|
| `seg[v17]` | 58 989 | **1 486** |
| `seg[v6]` | 76 | **1** |

Measured end to end on `100935`, peak rows across the whole elimination fall **58 989 -> 1 486**, and
every step's row count equals the number of profiles that vertex actually has.

### 3.2 Bytes per row — pick chains by reference

A row is `(cost, picks)`. A joined row's `picks` was the concatenation of its two parents', built
eagerly — about **16 KB per row**, dwarfing the float it accompanies, and paid by every intermediate
row although only one is ever read.

Rows instead store a `(left, right)` pair, and the chain is flattened only for the candidate the judge
accepts. Leaves are lists of pick triples; internal nodes are 2-tuples, so `_flatten` tells them apart
by type.

Consequence: rows are no longer order-comparable, so candidate lists must sort on `key=lambda r: r[0]`
rather than on the whole tuple.

### 3.2a Bytes elsewhere — the merge memo

`_merge` memoises into a module-level `_MERGE_CACHE`, cleared once per graph. Its docstring assumes
the forward pass's world — *"the profile universe is tiny, a few hundred distinct values"* — which
holds there: the forward fold leaves **2 169** entries.

`_join` breaks that assumption. It calls `_merge` on every key pair across whole factors, and those
pairs are mostly seen once, so the memo becomes a leak. Measured on `100935`:

| | entries | RSS |
|---|---|---|
| after `forward_profiled` | 2 169 | 617 MB |
| after `extract_profiled` | **3 756 561** | 1 114 MB |
| after clearing the cache | — | 630 MB |

So **484 MB of the extraction's 497 MB is the memo**, against ~13 MB of real extraction state. A size
ceiling — stop inserting past a bound, still return the correct result — keeps the forward-pass win
and drops the extraction cost to near zero.

### 3.3 The caps

`max_rows` bounds one factor and `max_profiles` one cell. Both are checked **during** the build, not
after it — a cap applied to an already-materialised product reports the peak instead of bounding it.

## 3.4 Where collapsing to one row per profile is legal

A factor keeps **one row per profile**, discarding which cell each row used. That is sound only where
the profile is a sufficient key for everything that can still observe the row.

| site | key sufficient? | why |
|---|---|---|
| a sink | yes | nothing continues from it; the cell cannot affect anything |
| a split, re-based | yes | the reset makes the profile `{(J, v)}` — the cell **is** the key, and the cost resets |
| an extraction factor | yes | a finished subproblem; only split placements are still observable |
| an interior vertex | **no** | see below |

The interior case fails because a row there still has a future. The recurrence admits a child cell `v`
only from `v` itself or a B-predecessor of `v`:

```python
entries = [v] + list(B.predecessors(v))
```

Keep one cell per profile at the parent and the retained cell may not be in `entries` for a given `v`
at all, so `v` loses its entry entirely. Measured on the 48-case envelope: table rows fall 1 078 164
-> 96 921 (11x), but 5 cases become **infeasible** ("sink has no reachable profile") and 23 of the
43 that still answer are costlier — up to 8x.

The damage is worst where there are fewest splits. On a chain every profile is `frozenset()`, so
"one cell per profile" means one cell per *vertex* and the alignment collapses outright: 15.0 -> 120.0.

## 4. Status

| change | effect | state |
|---|---|---|
| caps checked during build | refuses cleanly instead of `MemoryError` | done |
| one row per key (`keep` removed) | 5x time, 19x memory on the re-based path | done |
| min-fill elimination order | 70x fewer rows at the widest step on `100935` | done |
| pick chains by reference | ~16 KB/row -> pointer | done |
| segment factors streamed, not built | peak 58 989 -> 1 486 rows on `100935` | done |
| `_MERGE_CACHE` ceiling | 484 MB -> ~9 MB | done |
| branch-and-bound extraction | see §5 | **not needed so far** |

Line `100935` **solves at the shipped defaults**, no tuning:

| phase | time | RSS |
|---|---|---|
| `forward_profiled` | 3.90 s | +143.8 MB |
| `extract_profiled` | 0.24 s | **+27.5 MB** |

returning `cost = 441.6883`, `V1/V2/V3 = 0/0/0`, full cover. For scale, extraction was 80.4 s and
8.4 GB three changes earlier.

§5 remains unimplemented and, on the evidence, unnecessary: none of the memory was induced width.
Every cost removed here was building something that was about to be discarded.

> Under `ulimit -v` a large run can **segfault** rather than raise `MemoryError` — the allocation
> fails inside a C path that does not check. Pick-chain nesting is not involved (max depth 5).

## 5. What was not needed

An earlier draft of this document specified a **branch-and-bound** extraction — DFS over split
assignments with an admissible bound, memory `O(#splits)` because it holds a path rather than a
frontier — on the premise that the frontier was the cost and that only not storing it could help.

That premise was wrong, and the trace in §3.1a is what refuted it: the elimination frontier on
`100935` peaks at **1 486 rows**, and every step's row count equals the number of profiles that
vertex actually has. Induced width was never the problem. The memory was a leaked memo (§3.2a) and a
materialised cross product (§3.1a) — both of them building something about to be discarded.

It is recorded here as a refuted design, not a backlog item. If a future source genuinely exhausts
the elimination, measure *what* is large before reaching for it: on every case seen so far the answer
was "something we didn't need to build".
