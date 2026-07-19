# Low-Memory Extraction

How the profiled engine's extraction phase (`extract_profiled`, `_extract_rebased`) is kept inside a
memory budget. Companion to `profiled_forward_table.md`, which specifies the phase itself in §5.

---

## 1. The problem

The forward table is not the phase that runs out of memory. On hourglass line `100935` the re-based
table builds in 3.7 s / 142 MB; the extraction that reads it then exhausts the machine.

| phase | line 100350 | line 100935 |
|---|---|---|
| `forward_profiled` | 0.20 s / 5 MB | 3.7 s / 142 MB |
| `extract_profiled` | 0.10 s / 3 MB | **out of memory** |

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
ratio between orders, not as absolutes.

### 3.2 Bytes per row — pick chains by reference

A row is `(cost, picks)`. A joined row's `picks` was the concatenation of its two parents', built
eagerly — about **16 KB per row**, dwarfing the float it accompanies, and paid by every intermediate
row although only one is ever read.

Rows instead store a `(left, right)` pair, and the chain is flattened only for the candidate the judge
accepts. Leaves are lists of pick triples; internal nodes are 2-tuples, so `_flatten` tells them apart
by type.

Consequence: rows are no longer order-comparable, so candidate lists must sort on `key=lambda r: r[0]`
rather than on the whole tuple.

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
| branch-and-bound extraction | see §5 | not started |

Line `100935` is **solvable but not affordable**. At the 50 000 default the widest factor still
overflows; at `max_rows = 250 000` it returns `cost = 441.6883` in 80.4 s and **8.4 GB**. Before
min-fill and by-reference chains the same cap died at 4 GB with no answer, so those two changes are
what put it in reach — but 8.4 GB is proof the answer exists, not a usable engine. That is the case
§5 exists for.

> Under `ulimit -v` this run **segfaults** rather than raising `MemoryError`: the allocation fails
> inside a C path that does not check. Pick-chain nesting is not involved — measured max depth 5.

## 5. If that is not enough — branch-and-bound

Elimination stores a frontier, and the frontier is the product. A depth-first search stores a path.

- **Variables** the splits, in min-fill order.
- **Bound** for a partial assignment, sum over remaining factors of each one's cheapest row consistent
  with it. Precomputed per-factor minima indexed by `(split, cell)` — thousands of entries, not the
  product.
- **Search** DFS, cells in increasing local cost, prune when `g + h >= incumbent`.
- **Leaf** reconstruct through `_flood`, run `check_rules`. Valid updates the incumbent; invalid keeps
  searching.

Memory becomes `O(#splits)` instead of `O(#profiles)`. Two properties come free:

- **The judge gets its fallbacks back.** Search regenerates alternates on demand, which is what the
  removed `keep` parameter was buying with gigabytes — worth 168 cyclic-B cases.
- **It is anytime.** The first valid leaf is a usable answer; continuing only proves optimality. A
  time budget then yields a result rather than a refusal.

Exactness is preserved: the bound is admissible and the search exhaustive under pruning. The risk is
worst-case exponential time, so it needs `max_seconds` as the backstop, and on `100935` it may return
a valid matching without proving it optimal.

Best-first/A\* on the same bound expands fewer nodes but grows a frontier again, which is the thing
being escaped — hence DFS.
