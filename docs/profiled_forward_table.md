# The Profiled Forward Table

**Status:** implemented as `network_matching/profiled.py` (`forward_profiled` + `extract_profiled`),
**not wired into `match_dag`** — nothing calls it yet. Standalone; merges into `dag_dtw_matching.md`
§4 and `cell_dag_extraction.md` §8 when adopted.

| | |
|---|---|
| the gap it closes (§5.0) | the forward table is **V3-invalid on the two slow hourglass edges** (2 and 3 violations), valid on the other two and on all 10 synthetic cases |
| gate (§7) | unit suite **198 passed**; envelope **384/384** cost parity; cyclic-B **731/731** with **0** invalid and **168** cases answered where `extract_cell` refuses; four hourglass edges all matching |
| speed, real edges | `100350` **687.7 s → 0.44 s**; `100341` 33.4 s → 0.19 s; `102752` 30.0 s → 0.98 s |
| memory (§1.4) | 3–14 MB per edge, everything retained (~279 B/row); no freeing lifecycle needed |
| **known bad case (§5.1)** | a pure out-tree: nothing post-dominates, so no key discharges. `btree(4)` **dies** (`MemoryError`) where `extract_cell` takes 14 ms. The blow-up is in the **sink join**, not the forward table — see §6.1 |

Today's forward table (`dag_dtw_matching.md` §4.1) minimises over **all** upstream configurations,
including phantoms — a vertex placed on two cells at once, courtesy of the `1/outdeg` split
fractions. It is a *bound*, not a matching, and the extraction exists to recover a real matching from
it, paying `pending` rows keyed on merge cells to do so.

This design narrows the minimisation instead. Every cell carries a **profile** — where the upstream
splits are placed — and a cost *per profile*. Parents may only combine when their profiles agree. The
phantom is blocked at construction, and the coupling key moves from **merge** cells to **split**
cells, which on the hourglass is the far cheaper of the two.

---

## 1. The Object

### 1.0 Prerequisite — §4.1a stays

This design **runs on top of** `dag_dtw_matching.md` §4.1a's forbid-and-rebuild coupling, it does not
replace it. `forward()` still owns the `forbidden` flags; the profiled pass reads them and skips a
forbidden cell wherever a row attaches to it, exactly as `_fill_row` does.

The two operate at different levels and both are wanted:

| | §4.1a coupling | this design |
|---|---|---|
| question | which exits can **no** child use? | which exit should **all** children take? |
| evidence | transition existence (`_feasible_links`) — no DP values | cost per split placement |
| effect | deletes impossible cells | prices the possible ones jointly |
| cost | negligible | the profile table |

Keeping §4.1a first is what makes the profiled pass cheap: it removes infeasible cells before any
profile is built, so the candidate sets the fold ranges over are already trimmed. It also still owns
the feasibility error — *"split `p`: no surviving V3 exit within `r`"* — which the profiled pass has no
reason to re-derive.

What §4.1a cannot do is choose among the exits it keeps, and §5.0 measures that gap: on the two slow
hourglass edges every violation is a split whose 41–66 exits are feasible for **both** children, so
the intersection removes nothing and each child links its own cheapest. That choice is what the
profile prices.

| field | today | here |
|---|---|---|
| `D[a][v]` | one value: min over all configs incl. phantoms — a lower bound | **`D̂[a][v][π]`** — min cost over consistent configs with upstream splits placed per `π` |
| `bpD[a][v]` | the argmin's back-pointers | one per `(cell, π)` |

Costs are held **per profile**, never collapsed to a single argmin: different profiles have different
costs and the optimum sometimes needs a non-cheapest one, so a cell that dropped its alternatives
could not be reached under the profile a later tuple requires. §5 shows the full table is affordable.

**Both `D` and `Dp` exist on the record — but the profiled path never reads `D`.** It reads only `E`,
`forbidden` and `Dp`. `D` survives because `forward()` has to run first for the `forbidden` flags
(§1.0), and computing `D` is *how it gets them*: `_couple` uses `_links` — which reads `D` and `bpD` —
to decide which sibling rows to rebuild. So `D` is a by-product of producing the flags, not an input
to this design.

It costs little (7 ms, 0.08 MB on `102752`) so there is no reason to remove it, and the diagnostics
(`extract_join`, `check_forward_v3`, `check_reachability`) still depend on it. But nothing here needs
its values, and a future coupling-only pass that emits the flags without filling `D` would be a valid
simplification.

### 1.1 The profiled set `S`

$$S \;=\; \{\, s \in V(A) \;:\; \operatorname{outdeg}(s) \ge 2 \,\} \qquad \text{— exactly the splits.}$$

A vertex's cell can only be disagreed about if **two distinct downstream branches carry it**, and a
branch point is precisely `outdeg ≥ 2`. A vertex with `outdeg = 1` is seen by one branch, so no
configuration can contradict it.

The branches need **not** rejoin. `cell_dag_extraction.md` §6.1's phantom has `c₁` and `c₂` as
separate sinks with disjoint descendants and is still invalid, because a matching assigns every vertex
one run *globally*. Any definition of `S` requiring a reconvergence point misses it.

### 1.1a The format of a profile `π`

> **A profile is a set of cells — one cell per live split.**

Each element `(J, v)` *is* a cell in the ordinary sense (§1.4: an A-vertex paired with a B-vertex), so
a profile is a **set of cells**, restricted to cells whose A-vertex is a live split. It reads *"this
split's run ends on that B-vertex"*. Concretely it is a `frozenset` of `(A_split, B_vertex)` pairs.

Three properties, all of them consequences of *set*, not *list*:

| | |
|---|---|
| **unordered** | `{(J₁,v), (J₂,x)}` and `{(J₂,x), (J₁,v)}` are the same profile — which is why two rows that agree collide correctly during contraction |
| **at most one cell per split** | a vertex has one run in a matching, so `{(J₁,v), (J₁,w)}` is not a profile — that contradiction is exactly what §1.2's consistency test rejects when merging two arms |
| **no duplicates** | free from set semantics |

*Relation to the source-cell version this grew from:* structurally **identical** — a set of cells, at
most one per keyed vertex. Only two things changed: **which** vertices are keyed (splits, not sources
— §1.1b), and that a split's key **dies** at its post-dominator whereas a source's key would live to
the sinks (§1.1d).

Following the notation used throughout these docs — **`a, b, c, p, m, J, X` are A-vertices; `u, v, w,
x` are B-vertices** (as in `D[a][v]`, `bpD`'s `(p, x)` pairs, and coverage's `[(a, v')]`) — a cell
with two live splits `J₁` and `J₂` holds profiles like:

```python
frozenset({ (J₁, u),      # the run of split J₁ ends on B-vertex u
            (J₂, w) })    # the run of split J₂ ends on B-vertex w
```

The subscripts distinguish two splits and nothing more; `u` and `w` are two B-vertices. Real graphs
use whatever identifiers their builder assigned (see the note below).

| | |
|---|---|
| left of each pair | an **A**-vertex with `outdeg ≥ 2`, i.e. a member of `S` (`J`, `a`, …) |
| right of each pair | a **B**-vertex (`u`, `v`, `w`, …) — where that split's run **ends** (§2: run *end*, not entry) |
| the pair itself | one **cell** `(A_split, B_vertex)` — "cell" always means such a pair |
| how many pairs | `|S ∩ ancestors(a)|` minus everything discharged (§1.3) — the **width** |
| `frozenset()` | legal and common: no splits upstream, or all of them discharged |

> **Reading the dumps in this doc.** The examples below are printed from the benchmark families in
> `scripts/extract_cell_dag.py`, which build **congruent** A and B graphs and name each B-vertex by
> appending an apostrophe to its A counterpart. So `r` is an A-vertex and `r'` is the B-vertex lying
> on top of it; `ru_m` is an A-vertex and `ru_m'` its B counterpart. The apostrophe is the *only*
> thing distinguishing them. That convention is convenient for tests — the correct match is visually
> obvious — but it makes profiles hard to read, so in prose this doc uses the house convention
> (`J`, `a`, … for A-vertices; `u`, `v`, `w` for B-vertices) instead.

`frozenset` rather than `dict` for two reasons: it is **hashable**, so it can key the `Dp` dict; and it
is **order-independent**, so two rows that agree on the same placements collide correctly during
contraction regardless of the order the keys were added.

**Segment mode looks alarming but is the same thing.** On a line graph every vertex name is itself a
`(u, v)` tuple, so both halves of each pair become tuples:

```python
frozenset({ (('ru_m', 'ru'), ("r'", "ru_m'")) })
#            └─ A-segment ─┘  └─ B-segment ─┘
#            the split         where its run ends
```
Both halves are now `(u, v)` tuples because a line-graph vertex *is* an edge of the original graph.
The left tuple is an A-segment (the split), the right is the B-segment its run ends on — the same
`(A_split, B_vertex)` pair as above, with tuple-valued names.

Nothing about the structure changes — only the names. This is what the hourglass uses, which is why
`102752`'s profiles print as nested tuples.

`S = ∅ ⇒ the design is a no-op` — no conflicts exist, every profile is `frozenset()`, and `D̂ ≡ D`.

### 1.1b Why splits, and not source cells

The idea this design grew from was to label each cell with **which source cells it came from**. That
is not what `S` is, and the substitution is deliberate — recorded here because the two are easy to
confuse and the source version fails for a reason that is not obvious.

**A source profile cannot see a V3 violation.** V3 binds the cell a **split's run ends on**. Measured
on all four hourglass edges, **no source is ever a split** (overlap 0 below), so every split lies
strictly *downstream* of its source ancestors — and therefore **both children of a split inherit the
identical source-cell assignment**. Two children can carry the same source profile and still leave the
split from different cells. The conflict is invisible to it.

Splits are the minimal set that does see it: a cell can only be disagreed about when two downstream
branches carry it (§1.1), and a branch point is exactly `outdeg ≥ 2`.

**And it is not even cheaper.** Sources are the more numerous set, and on the tail edge the source
profile is twice as wide:

| edge | `\|LA\|` | sources | splits | overlap | width if splits | width if sources |
|---|---|---|---|---|---|---|
| 102752 | 29 | 4 | 2 | 0 | 2 | 2 |
| 100042 | 26 | 3 | 2 | 0 | 2 | 2 |
| 100341 | 29 | 3 | 2 | 0 | 2 | 2 |
| **100350** | 21 | 4 | 2 | 0 | **2** | **4** |

So the substitution costs nothing and buys correctness. Reproduce with
`report/probe_sources_vs_splits.py`.

### 1.1c The list of profiles at one cell

> The keys of `Dp` at cell `(a,v)` are **every distinct placement of `a`'s live ancestor splits under
> which `(a,v)` is reachable**. Each key stores the **cheapest** cost achieving that placement.

Same cell — different assumptions about what happened upstream. A real dump from the smallest case
that shows it, one split and one chain:

```
A:  a ──→ J ──→ c          J is the only split (outdeg 2)
              └─→ d
B:  u ──→ v ──→ w ──→ x

candidate cells of J :  u, v, w, x
candidate cells of c :  w, x
```

```
cell (c, w)  holds 2 profiles:
     cost   9.891   when   J ends on v
     cost  11.953   when   J ends on w
     min = 9.891    D = 9.891

cell (c, x)  holds 3 profiles:
     cost   9.242   when   J ends on w
     cost  14.764   when   J ends on x
     cost  14.990   when   J ends on v
     min = 9.242    D = 9.242
```

Read `(c, w)` as: *"pairing `c` with `w` costs 9.891 if `J`'s run ended on `v`, or 11.953 if it ended
on `w`."* Same pairing, two prices, because the upstream differs.

**Here a profile happens to be a single pair — because there is only one split.** In general it is
**one pair per live split**. Add a second split downstream and the profiles carry two:

```
A:  a ──→ J₁ ──→ c ──→ J₂ ──→ e          two splits, neither ever discharged
           └───→ d          └───→ f      (their branches never rejoin)
B:  u ──→ v ──→ w ──→ x ──→ y
```

```
cell (c, v)   live splits: J₁          width 1
     cost   9.544   when   J₁ ends on v
     cost  13.075   when   J₁ ends on u

cell (e, x)   live splits: J₁, J₂      width 2
     cost  14.814   when   J₁ ends on v,  J₂ ends on x
     cost  16.829   when   J₁ ends on w,  J₂ ends on x
     cost  17.586   when   J₁ ends on v,  J₂ ends on w
     cost  19.601   when   J₁ ends on w,  J₂ ends on w
```

`(e, x)` sits below both splits, so every one of its rows must say where **both** ended — and the rows
enumerate the combinations. That is the product in §1.1's multiplicity, and why width matters: each
extra live split multiplies the row count instead of adding to it.

This little graph is also the out-tree failure (§5.1) in miniature: `J₁`'s branches (`c`, `d`) never
rejoin and neither do `J₂`'s, so **no key is ever discharged** and the widths only grow downstream.

* **Count** = the cell's *multiplicity*. At most `∏` over live ancestor splits of `|cand(split)|`,
  minus combinations that are unreachable — `J` has 4 candidate cells but only 2 of them can reach
  `(c, w)`, so that cell holds 2 rows, not 4.
* **`min` over the list is exactly `D`** (§2.1), as both cells show.
* **The rows are not local alternatives.** `9.891` is what `(c, w)` costs *given* that the rest of the
  matching also puts `J` on `v`. It is a conditional price, not an option.

**Why every row is kept.** Suppose the sibling branch `d` can only be matched with `J` on `w`. Then
`(c, w)` cannot use its cheapest row at 9.891 — it must use **11.953**, and the global optimum is
whichever total is smallest once both branches are priced under the *same* placement of `J`.
Collapsing this list to its minimum throws away the row the optimum needs. That is what §1's "never
collapsed to a single argmin" means, and why `pending` exists in the engine this replaces.

Reproduce with `report/probe_profile_list.py`.

### 1.1d The life of a profile key — born, carried, merged, discharged

A key is **not** created at a source and carried unchanged to the end. It is **born at a split** and
**dies at that split's post-dominator**. Exactly four events touch a profile, and nothing else does:

| event | where | effect on `π` |
|---|---|---|
| **born** | at a split `a ∈ S` | add `(a, v)` — this run ends on `v` |
| **carried** | any vertex that is neither a split nor a discharge point | unchanged — literally the *same object* (§1.4) |
| **run-end moves** | an α-coverage step at a split | the split's own pair is **overwritten**, `(a,v') → (a,v)` |
| **merged** | at a merge | the arms' profiles must **agree**; union if they do, the pair is dropped if not (§1.2) |
| **discharged** | at a post-dominator of `s` | remove `s` — nothing downstream can contradict it (§1.3) |

Traced on `diamond_chain(2)`, `s → J0 → {x0,z0} → m0 → t0 → J1 → {x1,z1} → m1 → t1`, splits
`{J0, J1}`:

```
vertex  in/out   role                                  profiles at one cell
     s    0/1    source                                {}
    J0    1/2    SPLIT — key born                      {J0@s'}
    x0    1/1    carries parent's profile              {J0@J0'}, {J0@s'}
    z0    1/1    carries parent's profile              {J0@J0'}, {J0@s'}
    m0    2/1    MERGE (arms agree) + DISCHARGE J0     {}
    t0    1/1    nothing live                          {}
    J1    1/2    SPLIT — key born                      {J1@m0'}
    x1    1/1    carries parent's profile              {J1@J1'}, {J1@t0'}
    z1    1/1    carries parent's profile              {J1@J1'}, {J1@t0'}
    m1    2/1    MERGE + DISCHARGE J1                  {}
    t1    1/0    nothing live                          {}
```

Read it as a lifetime: `J0`'s key exists **only between `J0` and `m0`**. Below `m0` it is gone, so
`J1`'s key is the only one live in the second diamond — the width never reaches 2. That is why
`diamond_chain(400)` runs in 0.5 s while `btree(4)` dies: on a tree nothing post-dominates, so no key
is ever discharged and they accumulate to the sinks.

`x0` holding **two** profiles is §1.1c in miniature: `J0`'s run can end on `J0'` *or* on `s'`, and
`x0` is reachable under either, so it stores a row for each.

> **Contrast with a source-cell profile.** There a key is born at a source, is fixed at birth, and
> never dies — so the width is the source count and only grows. Here a key appears at a split, tracks
> that split's run end, and is deleted the moment it can no longer be contradicted. The lifetime is
> what keeps the table small (§5.1), and it is the substantive difference between the two designs
> beyond the correctness argument in §1.1b.

Reproduce with `report/probe_profile_life.py`.

### 1.2 Consistency

Two profiles are **consistent** iff they agree on every key both name; a tuple is consistent iff
pairwise consistent. A vertex has exactly one run in a matching, so disagreement means no matching
realises the combination — sound by construction, and on split cells it **is** V3.

### 1.3 Discharge — the load-bearing part

> Drop `s` from the profile at cell `(a, ·)` **iff `a` post-dominates `s`** — every path from `s` to
> any sink passes through `a`.

From there exactly one branch carries `s`, so no later tuple can contradict it and the key is dead
weight. Static, computed once from a post-dominator tree (`nx.immediate_dominators` on the reversed
graph, rooted at a virtual super-sink).

This is the forward mirror of `cell_dag_extraction.md` §3.5's early discharge — *first common ancestor
going backward* is *post-dominator going forward*. It is not an optimisation: §5.1 shows it is the
difference between exponential and linear.

**Width** — `|π|`, the number of live keys — is therefore **structural**, a property of `A`'s
topology. No matching hyperparameter sets it. `r`, `k_min`, `bearing_weight`, `α`, `β` change
`|cand|`, which drives *multiplicity* (how many profiles a cell holds), not width. Widening `r` for a
hard match grows the cheap dimension.

### 1.4 What a cell stores

**The whole structure**, top to bottom. Everything except the last line already exists — this design
adds exactly one key, `Dp`.

```
A                                    the source graph (networkx DiGraph)
│
└── .nodes[a]                        one A-vertex's attribute dict
    │
    ├── "x", "y"                     its position                       (point + segment)
    ├── "bearing", "length"          segment-mode only                  (line_digraph)
    ├── "road_id", "seq"             provenance, when the caller sets it
    │
    └── "cand"                       ITS CANDIDATE B-VERTICES  — built by prepare(),
        │                            gated to those within `r` of `a`.  Each key here
        │                            is one CELL (a, v).
        │
        └── [v]                      the record for cell (a, v)
            │
            ├── "E"          float   emission cost of pairing a with v
            ├── "D"          float   forward cost   — ONE number
            ├── "bpD"        list    its back-pointers
            ├── "B", "bpB"           the backward (diagnostic) table
            ├── "forbidden"  bool    §4.1a: not a valid run END
            │
            └── "Dp"         dict    ← ADDED BY THIS DESIGN
                └── [profile] → (cost, bp)
```

**Why `"cand"` is its own level.** `A.nodes[a]` is networkx's attribute dict and already holds the
vertex's geometry. B-vertex names are arbitrary — plain strings in point mode (`"s'"`, `"J0'"`) and
**tuples** in segment mode (`("s'", "J0'")`) — so putting cells directly in `A.nodes[a]` would let a
B-vertex named `x`, `y`, `bearing` or `length` silently overwrite the geometry, and would mix string
attribute keys with tuple cell keys in one dict. Keeping them under `"cand"` also makes the candidate
set a unit: `len(cand)` is the cell count, `for v in cand` sweeps cells, `v in cand` tests membership.

**What `Dp` holds.** One entry per profile:

| | |
|---|---|
| `profile` | `frozenset` of `(A_split, B_vertex)` pairs — where the upstream splits sit (format: §1.1a) |
| `cost` | `float` — the value of this row |
| `bp` | `[(vertex, cell, profile), …]` — where the value came from |

`D` answers *"what does this cell cost?"* with a single number. `Dp` answers *"what does it cost
**given where the upstream splits are placed**?"* — one number per placement. Nothing else on the
record changes, and `forward()` keeps filling `D`/`bpD`/`forbidden` exactly as before (§1.0).

`bp` holds one triple per predecessor (advance/stall), or a **single same-vertex triple** for an
α-coverage step — the same convention `bpD` uses, so the move type is read off *whose* vertex appears.

**Worked example** — `btree(2)`, the source `r` (a split, `outdeg = 2`) and its child `ru_m`
(`indeg = 1`, `outdeg = 1`, not a split, nothing to discharge):

| | `r` at cell `r'` | `ru_m` at cell `r'` |
|---|---|---|
| profile | `{r: r'}` | `{r: r'}` |
| cost | `0.4000` | `11.0240` |
| `bp` | `[]` (source, free entry) | `[(r, r', {r: r'})]` |

`r` carries `{r: r'}` because it **is** a split — §2's "own split cell" step writes its own placement
into the key. `ru_m` carries the same profile because it is not a split and discharges nothing: it
simply inherits.

#### The single-parent case — inherit by reference, do not copy

A vertex that is **neither a split nor a discharge point** does not change the profile at all. Its
rows therefore keep the **parent's frozenset object itself**, not a rebuilt equal one:

```
passthrough = (a not in S) and (not drop[a])
if passthrough:  return pi          # same object, no allocation
```

That is the common case — most vertices are `indeg 1, outdeg 1` — and rebuilding an identical
frozenset per cell per profile was the bulk of the memory. Measured on `btree(4)`'s forward table:

| | before | after |
|---|---|---|
| bytes per row | 814 | **279** |
| forward table | 14.36 MB | **4.91 MB** |
| time | 0.205 s | **0.122 s** |

Row *counts* are identical — this is representation only, nothing is dropped.

What each row still costs after that: the dict entry (the key is shared, but a frozenset is re-hashed
on every lookup), the `(cost, bp)` tuple, the `bp` list, and the triple inside it — four objects to
carry one float and one pointer. Remaining reductions, cheapest first, **none yet implemented**:

| | what | cost to do |
|---|---|---|
| intern profiles to ints | a per-graph `frozenset → int` table; keys become small integers, so hashing is an int hash and the dicts shrink | contained — only `_merge`, `remap`, `_flood` touch profile identity |
| flatten `bp` | at `indeg 1` there is exactly one triple; store it directly instead of wrapping it in a list | trivial |
| delegate entirely | a passthrough cell stores only `(parent_vertex, {entry_cell: cost})` and resolves profiles through its parent on demand | invasive — every reader must chase the chain |

### 1.5 Lifetime

**Allocate per cell, keep to the end. There is no row-freeing lifecycle.** What shrinks is the profile
*keys* (§1.3), not the rows — that is where the contraction happens, and it happens during the sweep.

Freeing rows would buy nothing anyway: reconstruction needs **either** the stored `bp` **or** the cost
rows to recompute the argmin from, so one of the two must survive to the end. Keeping both is
affordable at the measured scale — the real forward tables, as built:

| | 102752 | 100042 | 100341 | 100350 |
|---|---|---|---|---|
| profile rows | 57 478 | 12 425 | 12 871 | 28 444 |
| forward table | 14.4 MB | 3.1 MB | 3.4 MB | 10.1 MB |
| build time | 0.97 s | 0.26 s | 0.18 s | 0.40 s |

At ~279 bytes per row (§1.4) that is a few megabytes per edge and no freeing is warranted. Contrast
`cell_dag_extraction.md` §4, where freeing **was** load-bearing (35× peak memory): there the frontier
was the whole table set.

*If it ever does get tight:* store costs, **drop `bp`**, and recompute the argmin along the single
winning chain during reconstruction — re-evaluating one cell's `fill()` is cheap (entry cells bounded
by `1 + |Bpred(v)|`, ≤ ~64 tuples at `indeg 3`). That keeps the smaller half and recomputes the
larger. Not worth building until a measurement demands it.

**The cap does not yet bound memory.** `forward_profiled(max_profiles=…)` refuses when one *cell*
exceeds the limit, but the out-tree blow-up (§5.1) is **aggregate across cells** — `btree(4)` peaks at
~61 profiles per cell against a 50 000 cap, so the guard never fires. A global row budget is what is
needed; until then the refusal is not a memory bound.

---

## 2. The Recurrence

Entry cells for parent `p` into `(a,v)` are `x ∈ ({v} ∪ Bpred(v)) ∩ cand(p)` — a **stall** at `v` or
an **advance** from a B-predecessor. This set is bounded by `1 + |Bpred(v)|`, B's local in-degree
(1–3 in a road network), **not** by `|cand(p)|`. That is what makes a joint min affordable.

```
def fill(a, v):
    rows = {}                                  # π -> (cost, backpointers)

    # (D)/(V) — JOINT over parent tuples AND their profiles, never one min per parent
    for (x_p) in ∏_{p ∈ Apred(a)} ({v} ∪ Bpred(v)) ∩ cand(p):
        for (π_p) in ∏_p keys(D̂[p][x_p]):
            π = merge(π_p)                     # None if two parents disagree on a shared s ∈ S
            if π is None: continue             # ← the consistency test (§1.2)
            w = E(a,v) if every x_p advances else β·E(a,v)
            c = w + Σ_p D̂[p][x_p][π_p] / outdeg(p)
            keep c in rows[π] if cheaper

    # (H) coverage — within-row, iterated to the §4.1 fixed point
    for v' in Bpred(v):
        for π in keys(D̂[a][v']):
            keep α·E(a,v) + D̂[a][v'][π] in rows[π] if cheaper

    if a ∈ S: rows = { π ∪ {a: v} : ... }      # overwrite: v is a's run end so far
    rows = { π \ postdom_drop[a] : ... }       # §1.3 discharge, merging rows that collide
    D̂[a][v] = rows
```

Four points the pseudocode makes that prose blurs:

* **The min is joint, never per-parent.** Taking `min_x D̂[p][x]` per parent and then testing the
  winners is a different and much weaker algorithm — the one `cell_dag_extraction.md` §6.2 shows
  fails.
* **`a ↦ v` is overwritten, not accumulated.** The entry means "the cell this split's run currently
  ends on"; a coverage step extends the run, so the entry moves. Downstream consumers attach at the
  run end, so they read exactly the cell V3 binds — consistency on `a`'s entry **is** V3.
* **Discharge merges rows.** Two profiles differing only in a dropped key collide; keep the cheaper.
  This is where the state space actually contracts, and it is a `min` — the elimination step, not a
  forget. Dropping a key without taking the min over its values is a silent corruption.
* **Profiles are written in the same step as costs**, including inside the (H) relaxation, exactly as
  `bpD` is today. A profile that lags its value is a silent corruption.

A source: `D̂ = {π₀: E}` with `π₀ = {a ↦ v}` if `a ∈ S` else `{}`.

**Properties.** The phantom is blocked at construction — the `1/outdeg` fractions can no longer
combine two cells of one split, because the profiles carrying them disagree. `min_π D̂[a][v][π] = D*`,
the true min over consistent configurations, so it remains admissible (`≤ C(M)`) on a strictly smaller
feasible set than `D` — tighter, never looser.

### 2.1 `D` is the minimum over `Dp` — and the phantom is not in either

`D` minimises over *all* upstream configurations, `D̂` over the *consistent* ones only, and consistent
⊂ all, so in theory `D[a][v] ≤ min_π D̂[a][v][π]`. **Measured, they are equal on every cell:**

| edge | cells | V3 violations | `D == min_π D̂` | `D < min_π D̂` |
|---|---|---|---|---|
| 102752 | 993 | **2** | **993** | 0 |
| 100042 | 708 | 0 | 708 | 0 |
| 100341 | 1081 | 0 | 1081 | 0 |
| 100350 | 822 | **3** | **822** | 0 |

Equal on all 3 604 cells — *including the two edges that are V3-invalid*. So the profiled table does
not make any individual value smaller, and this is not an accident of the data:

**A single cell's `D` cannot be a phantom.** `D[a][v]` minimises over `a`'s upstream cone. A phantom
needs two branches disagreeing about one split, so it can only arise inside one cone when a merge's
arms share a split ancestor. On the hourglass the in-side is tree-shaped — the arms come from disjoint
in-stubs — so no single `D` is ever wrong.

**The phantom lives in the combination.** Re-read `cell_dag_extraction.md` §6.1 with this in mind:
`D[c₁][u] = 0.5` is *correct* for `c₁`'s cone, and `D[c₂][d] = 0.5` is *correct* for `c₂`'s. Neither
value is a phantom. Their **sum**, `1.0`, is — because the two were minimised independently and
happened to choose different cells of `J`.

> So what the profile buys is **not a tighter `D`. It is a *keyed* `D`** — each value labelled with
> what it assumed about the splits, so values can be combined consistently instead of blindly. That is
> also why `check_forward_v3` reports violations while every individual `D` is perfectly correct, and
> why the payoff shows up in the **extraction** (§6) rather than in the table's numbers.

Reproduce with `report/probe_D_vs_Dp.py`.

**Costs stay under-counted.** The `1/outdeg` fraction is §4.1's approximation for a shared upstream
point and this design does not change it. See §6.3 — a profile identifies the shared prefix exactly,
so it *could* be charged once, but that is a separate change.

---

## 3. Worked Case

The four-vertex split phantom (`cell_dag_extraction.md` §6.1), where today's forward table claims
`1.0` against a true optimum of `6`:

```
A:   J ──→ c₁        B:   v₁ ──→ u      v₁ ──→ d′      E(J,v₁) = E(J,v₂) = 1
     └───→ c₂             v₂ ──→ u′     v₂ ──→ d       E(c₁,u)  = 0   E(c₁,u′) = 5
                                                       E(c₂,d)  = 0   E(c₂,d′) = 5
```

| cell | cost | `π` |
|---|---|---|
| `(c₁,u)` | 0.5 | `{J: v₁}` |
| `(c₁,u′)` | 5.5 | `{J: v₂}` |
| `(c₂,d)` | 0.5 | `{J: v₂}` |
| `(c₂,d′)` | 5.5 | `{J: v₁}` |

The phantom combination `(c₁,u) + (c₂,d)` is rejected: `v₁ ≠ v₂`. The survivors are `(u, d′)` and
`(u′, d)`, both at **6** — correct.

---

## 4. What It Buys

| | effect |
|---|---|
| **A smaller coupling key** | `pending` keys on merge cells, this keys on split cells. On the hourglass there are **3 merges but only 2 splits**, and the waist post-dominates the whole in-side — 36–70× fewer rows (§5.2) |
| **Biggest win on the worst edge** | `100350`, the all-coupled tail no earlier fix helped, is where this gives the most: `extract_cell` **687.7 s / 783 MB**, profiled **0.44 s / 16 MB** (§5.2) |
| **Answers cases the current engine refuses** | contracting per **profile** leaves the judge fallbacks that contracting per **pending signature** destroys: **168/900** cyclic-B cases answered where `extract_cell` raises a spurious *"no valid root row"* (§5.3) |
| **V3 as an invariant** | `check_forward_v3` empty by construction, promoting a diagnostic to a guarantee |

---

## 5. Measurement

### 5.0 The gap being closed

`check_forward_v3` reads the forward table on its own and reports where its `bpD` trace places a split
on two cells. `check_split_exits` (§4.1a's own invariant) passes everywhere, so the coupling is doing
exactly what it promises — the violations are the part feasibility pruning cannot reach:

| input | splits | `check_split_exits` | `check_forward_v3` |
|---|---|---|---|
| 10 synthetic cases (chain, diamond, btree; both modes) | 0–15 | 0 bad | **0** |
| line 100042 | 2 | 0 bad | 0 |
| line 100341 | 2 | 0 bad | 0 |
| **line 102752** | 2 | 0 bad | **2** |
| **line 100350** | 2 | 0 bad | **3** |

**The two invalid edges are the two slow ones** — `102752` (~15 s) and `100350` (the all-coupled tail,
`∏ = 77 000`). Every violation has the same shape: an `outdeg = 2` split whose 44–66 exits are
feasible for *both* children, so the intersection removes nothing and each child's row independently
links its own cheapest exit. Choosing among *possible* exits requires pricing the children jointly,
which is what §2 does.

### 5.1 Synthetic families — discharge is the whole game

`scripts/extract_cell_dag.py` families, all profiles kept, max multiplicity per cell:

| case | `\|S\|` | cells | max mult, **no** discharge | max mult, **with** | width | entries | peak |
|---|---|---|---|---|---|---|---|
| dense_chain(50) point | 0 | 536 | 1 | 1 | 0 | 0 | 0.3 MB |
| diamond_chain(4) point | 4 | 163 | 3 100 | **9** | 1 | 367 | 0.20 MB |
| diamond_chain(4) segment | 4 | 222 | 4 640 | **11** | 1 | 979 | 0.43 MB |
| diamond_chain(10) point | 10 | 421 | — | **9** | 1 | 1 021 | 0.49 MB |
| diamond_chain(10) segment | 10 | 594 | — | **11** | 1 | 2 869 | 1.18 MB |
| btree(3) point | 7 | 284 | — | 40 | 3 | 13 157 | 2.11 MB |
| btree(4) point | 15 | 1 036 | 202 | 202 | 4 | 312 923 | 40.4 MB |

* **`S = ∅` ⇒ no-op**, confirmed on the chain.
* **Discharge turns exponential into linear** on reconvergent graphs: diamond_chain 4→10 more than
  doubles `|A|` while max multiplicity stays flat at 9/11 and width collapses to 1.
* **`btree` is untouched** — no merges, nothing post-dominates, nothing discharges. This is the honest
  mirror of the §8 wall: `pending` blows up on trees with multiple **merges** (no common ancestor);
  profiles blow up on trees with multiple **splits** (no post-dominator). Same wall, opposite junction.
  A pure out-tree source is this design's bad case.

### 5.2 The real hourglass — the four §8.5 slow edges

`LA` built exactly as `mapconflation.match.direction.match_task` does. The `pending` column reproduces
`cell_dag_extraction.md` §8.5 to the digit, which validates the construction:

| edge | `pending` ∏ (parts) | profile max mult | width | mult=1 | entries | peak | time |
|---|---|---|---|---|---|---|---|
| 102752 | 28 350 (45,45,14) | **140** | 2 | 57.7% | 113 892 | 22.9 MB | 0.94 s |
| 100042 | 7 888 (29,17,16) | **32** | 2 | 38.6% | 24 339 | 5.2 MB | 0.25 s |
| 100341 | 7 220 (20,19,19) | **38** | 2 | 66.0% | 24 370 | 5.6 MB | 0.15 s |
| **100350** | 77 000 (56,55,25) | **194** | 2 | 80.7% | 55 683 | 14.1 MB | 0.36 s |

Every hourglass line-graph here has **3 merges and 2 splits**, `|LA|` 21–29, and **width 2** — the
waist post-dominates the entire in-side, so in-side splits discharge there.

**Honest ratio.** `cell_dag_extraction.md` §8.6 records that `extract_cell` already carries only
**5 087** of `102752`'s 28 350 (infeasible pairs are dropped at `PathCost = ∞`), so the fair
comparison is `5 087 : 140` ≈ **36×**, not the raw `202×`. Applying the same ~18% survival to
`100350` gives ≈ `13 900 : 194` ≈ **70×**.

**Caveats.** This measures the profile *state space*, not a working implementation; four edges from
one AOI; `peak` is the probe holding Python `frozenset`s, where an implementation stores two ints and
a float per profile. Exactness is unproven until §7's cost parity.

---

## 6. The Extraction

### 6.0 What the table hands over

After the profiled forward pass, every cell holds `D̂[a][v][π]` — the exact cost of a **valid**
matching of `a`'s upstream cone with `a`'s run ending at `v` and the live splits placed per `π` — plus
`bpD[a][v][π]`, the parent cells and their profiles that achieved it.

At a **sink** `t`, `a`'s upstream cone is everything above `t`. The sinks' cones cover `A`. So the
extraction is no longer a search for a matching — it is a **join**: choose one `π`, let each sink pick
its own best cell under it, and add up.

### 6.1 The join

Let `π|ₜ` be `π` restricted to `S ∩ ancestors(t)`. Sinks whose split-ancestries are disjoint impose no
constraint on each other; where they share a split they must agree, which is exactly what a shared `π`
enforces.

$$
C^{*} \;=\; \min_{\pi}\ \sum_{t \in \text{sinks}} \ \min_{v \in \mathrm{cand}(t)} \widehat{D}[t][v][\pi|_t]
$$

```
best = ∞
for π in global_keys:                       # profiles live at the sinks
    total, pick = 0.0, {}
    for t in sinks:
        rows = [(D̂[t][v][π|t], v) for v in cand(t) if π|t in D̂[t][v]]
        if not rows: total = ∞; break       # π unreachable at this sink
        total_t, v_t = min(rows)
        total += total_t; pick[t] = (v_t, π|t)
    if total < best: best, best_pick = total, pick
```

**For a fixed `π` the sinks are independent** — that is the whole point of the key. The join is
`O(|global_keys| × |sinks| × |cand|)`, with no product over sinks.

> **Implementation gap — this is where `btree` dies.** `extract_profiled` folds the sinks one at a
> time, merging profiles. Two sinks whose split-ancestries are **disjoint** never conflict, so every
> pair merges successfully and the fold builds their **cross product** instead of minimising them
> separately. On the hourglass (2 sinks sharing both splits) that is one component and costs nothing;
> on `btree(4)` (16 sinks with largely disjoint ancestries) it is a 16-way product and exhausts memory
> — while the forward table it reads is only 4.9 MB.
>
> The fix is the same factoring as `cell_dag_extraction.md` §8.1: build a graph on sinks where an edge
> means *shares a profile key*, take connected components, minimise each independently, and **sum**.
> Disjoint sinks must never be crossed. Not yet implemented.

**Measured size** (same four edges, §5.2):

| edge | sinks | profiles per sink | distinct `π` at sinks | splits live at sinks |
|---|---|---|---|---|
| 102752 | 2 | 140, 140 | **140** | 2 |
| 100042 | 2 | 32, 32 | **32** | 2 |
| 100341 | 2 | 38, 38 | **38** | 2 |
| 100350 | 2 | 234, 225 | **234** | 2 |

So the entire extraction is a min over ≤234 keys across 2 sinks — against `pending`'s ~5 087 carried
signatures on `102752` and the backward sweep that consumes them. Note the keys are far below the raw
`|cand|²` (45² = 2 025 on `102752`): joint reachability has already pruned them during the forward
pass.

### 6.2 Why summing the sinks is exact

The `1/outdeg` split fractions of `dag_dtw_matching.md` §4.1 are **not** an approximation once V3
holds — they are exactly the weights that make a sink-sum count each vertex once.

*Argument.* A vertex `b`'s emission enters a descendant's `D` scaled by `∏ 1/outdeg` along the
connecting path. Summing over every path from `b` to a sink, those factors sum to **1** — unit mass
leaving `b`, split equally at each branch point, conserved to the leaves. A merge sums its arms
without dividing, so mass arriving by two arms recombines to 1 as well.

*Check on `cell_dag_extraction.md` §6.1*, with the split forced onto one cell (`J@v₁`):

```
D[J][v₁]   = 1
D[c₁][u]   = 0 + 1/2 = 0.5          D[c₂][d′] = 5 + 1/2 = 5.5
sink sum   = 0.5 + 5.5 = 6.0        true cost = 1 + 0 + 5 = 6   ✓
```

The `1.0` that same section reports as the failure is what happens when the two branches place `J` on
**different** cells: `E(J,v₁)/2 + E(J,v₂)/2` — half of each, the phantom. **The under-count is the V3
violation, not the fractions.** Blocking the phantom therefore fixes the arithmetic at the same time,
and no separate double-counting correction is needed.

> Verify, don't assume: this is the claim most likely to be wrong in a corner (β-stalls, α-coverage
> runs, cyclic B). The parity test in §7 is what settles it, and a mismatch here is the first thing to
> suspect.

### 6.3 Reconstruction

Already written. `_reconstruct_from_sinks(A, sink_labels)` (`dag_dtw.py:518`) floods `bpD` from pinned
sink cells and expands cover chains into run cells; `_one_sided` (`:1396`) is the same walk seeded at
each sink's argmin. Both are diagnostic-only today **because the table they read is V3-invalid** —
which is precisely what this design fixes. Promoting them needs one change: follow `bpD[a][v][π]` for
the chosen `π` instead of the single stored history.

### 6.4 What still needs the judge

`π` enforces V3 and the forward recurrence enforces V2 (a merge's cell is finite only if every arm
reaches it). **V1 is not covered** — on a cyclic `B` a cheap row can reuse a B-vertex, the known
contraction-eviction family (`dag_dtw_matching.md` §4.1a "Open", `scripts/repro_contraction_eviction/`).
So keep the terminal `check_rules` judge and the cheapest-first retry over joined rows, exactly as
`extract_cell` §5.3 does today. Drop the judge only if the parity suite says V1 never bites — it
currently does, in ~2% of adversarial cyclic cases.

### 6.5 Fallback — the contained migration

`extract_cell` keeps its structure: backward sweep over the cell DAG, the consumed-once rule
(`cell_dag_extraction.md` §3.3), the inbox-push freeing lifecycle (§4), the terminal judge and
cheapest-valid-root selection (§5.3). Only **what a row is keyed by** changes.

| | today | new |
|---|---|---|
| row key | merge-cell commitments (`pending`) | the profile `π` |
| key space, `102752` | 5 087 carried signatures | ≤ 140 |
| key space, `100350` | ≈ 13 900 | ≤ 194 |
| discharge point | first common ancestor of a merge's arms (§3.5) | post-dominator of a split (§1.3) |

Everything downstream reads a key it cannot distinguish. `_pend_union` becomes profile merge; the §3.5
discharge machinery is reused with the post-dominator table in place of the common-ancestor table.

This keeps the whole backward sweep and is strictly more work than §6.1. Its only use is as a safety
net: if §6.2's exactness claim fails the parity suite, this path still benefits from the smaller key
without depending on the sink-sum being exact.

---

## 7. Gate

As every prior integration (`cell_dag_extraction.md` §8.4): cost parity and refusal parity on the
164-case suite, the structured envelope 384/384 with `C(cell) ≤ C(join)`, benchmark parity against
`extract_cell_vertex`. The design is **exact**, so parity must hold to the digit — any divergence is a
bug in the design, not a tolerance.

`D̂` values differ from `D` by construction (they are tighter), so parity is asserted on the **final
matching cost**, never on `D`. Ship behind a flag (`profiles=False`) until green. A per-cell profile
cap with refusal — the `max_rows` pattern — bounds memory to a diagnosable error rather than an OOM.

---

## 8. Relation to Existing Machinery

| existing | relation |
|---|---|
| §4.1a forbid-and-rebuild | **retained as a prerequisite, not replaced** (§1.0). It answers "which exits can no child use?"; this answers "which exit should all children take?". `forward()` keeps owning the `forbidden` flags and the feasibility error; the profiled pass reads them and prices what survives. Its standing rule — *a feasibility intersection, never an optimality one* — is also why costs are held per profile rather than per cell (§1). |
| `pending` (`cell_dag_extraction.md` §2–3) | the same job keyed on merges instead of splits. §6.1 is a key swap; §5.2 measures the key spaces. |
| §3.5 early discharge | the backward mirror of §1.3. Same idea, opposite direction: first common ancestor ↔ post-dominator. |
| §8.6 inner-merge elimination | the same min-sum elimination, applied to one merge in the extraction. This applies it to every split in the forward pass. Composable; neither subsumes the other. |
| `check_forward_v3` (`dag_dtw.py:1419`) | today a diagnostic. Under this design it must return **empty** on every input — promote it to an invariant. |
