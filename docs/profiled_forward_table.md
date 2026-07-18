# The Profiled Forward Table

**Status:** implemented as `network_matching/profiled.py`. Gates all green (§6). **Not wired into
`match_dag`** — nothing calls it yet; adoption is a separate decision.

| | |
|---|---|
| what it fixes | the forward table places a split on two cells at once — measured on the two slow hourglass edges (§1) |
| what it costs | a cost per *profile* instead of one per cell: 3–14 MB per edge |
| what it buys | `100350` **687.7 s → 0.44 s**; and **168/900** cyclic-B cases answered where `extract_cell` refuses |
| where it is worse | pure out-trees — slower than `extract_cell`, though correct and bounded |

---

## 1. The Problem

`dag_dtw_matching.md` §4.1's forward table computes `D[a][v]` = the cheapest way to match `a`'s
upstream cone with `a`'s run ending at `v`. It minimises over **all** upstream configurations,
including invalid ones.

The invalid ones are **phantoms**: a vertex placed on two cells at once. They arise at a **split**
(`outdeg ≥ 2`), because each child's row is filled independently and each picks the split-cell that
suits it alone. §4.1a's coupling prunes exits *no child can use*, but where several exits are usable
by every child — the normal case — it keeps them all and the children still disagree.

**This is not hypothetical.** `check_forward_v3` reads the forward table on its own and reports where
its trace breaks V3:

| input | splits | `check_split_exits` | `check_forward_v3` |
|---|---|---|---|
| 10 synthetic cases | 0–15 | 0 bad | 0 |
| line 100042 / 100341 | 2 | 0 bad | 0 |
| **line 102752** | 2 | 0 bad | **2** |
| **line 100350** | 2 | 0 bad | **3** |

The two invalid edges are the two slow ones. Every violation has the same shape: an `outdeg = 2`
split whose 41–66 exits are feasible for **both** children, so the coupling has nothing to forbid and
each child links its own cheapest.

**Where the phantom actually lives.** Measured on all 3 604 hourglass cells, `D[a][v]` equals the
minimum over the profiled table — *including on the two invalid edges*. No single `D` is ever wrong: a
phantom needs two branches disagreeing about one split, which inside one cone needs a merge whose arms
share a split ancestor, and the hourglass in-side is tree-shaped. The phantom appears only when cells
are **combined** — `cell_dag_extraction.md` §6.1's `D[c₁][u] = 0.5` and `D[c₂][d] = 0.5` are each
correct; their *sum* is the phantom.

So this design does not make any value smaller. **It labels each value with what it assumed**, so
values can be combined consistently. The payoff is in the extraction, not the table.

---

## 2. Definitions

### 2.1 The profiled set `S`

$$S \;=\; \{\, s \in V(A) \;:\; \operatorname{outdeg}(s) \ge 2 \,\} \qquad\text{— the splits.}$$

A vertex's cell can only be disputed if **two downstream branches carry it**, and a branch point is
exactly `outdeg ≥ 2`. A vertex with `outdeg = 1` is seen by one branch and can never be contradicted.

The branches need **not** rejoin: a matching assigns every vertex one run *globally*, so two sinks
with disjoint descendants still conflict at assembly. `S = ∅` ⇒ the design is a no-op.

*Why splits and not sources.* Source cells were the original idea and cannot work: V3 binds the cell a
**split's run ends on**, and measured on all four hourglass edges **no source is ever a split**, so
every split lies strictly downstream of its source ancestors and both children of a split inherit the
*identical* source-cell assignment. The conflict is invisible to it. Splits are also no wider — equal
on three edges, and half the width on `100350` (2 vs 4).

### 2.2 A profile

> **A profile is a set of cells — one cell per live split.**

Each element `(J, v)` is a cell in the ordinary sense: an A-vertex paired with a B-vertex. It reads
*"split `J`'s run ends on B-vertex `v`"*. Concretely a `frozenset` of `(A_split, B_vertex)` pairs.

Using the house convention — `a, b, c, p, m, J, X` are A-vertices, `u, v, w, x` are B-vertices:

```python
frozenset({ (J1, u),      # split J1's run ends on B-vertex u
            (J2, w) })    # split J2's run ends on B-vertex w
```

| property | consequence |
|---|---|
| **set, unordered** | two rows agreeing on the same placements collide correctly during contraction |
| **at most one cell per split** | a vertex has one run; `{(J1,u), (J1,w)}` is the contradiction §2.3 rejects |
| **`frozenset`** | hashable, so it can key a dict |
| **`frozenset()` is legal** | no live splits upstream, or all discharged |

**Width** = `|π|`, the number of live keys. It is *structural* — a property of `A`'s topology, not of
any hyperparameter. `r`, `k_min`, `α`, `β` change `|cand|`, which drives *multiplicity* (how many
profiles a cell holds), not width.

*Segment mode.* On a line graph every vertex name is itself a `(u, v)` tuple, so both halves become
tuples: `frozenset({(('c','d'), ('u','v'))})`. Same structure, different names.

### 2.3 Consistency

Two profiles are **consistent** iff they agree on every key both name. A vertex has one run, so
disagreement means no matching realises the combination. On split cells, this test **is** V3.

### 2.4 Discharge

> Drop `s` from the profile at vertex `a` **iff `a` post-dominates `s`** — every path from `s` to any
> sink passes through `a`.

Below `a` only one branch continues, so nothing can ever contradict `s` again and the key is dead
weight. Computed once from a post-dominator tree (`nx.immediate_dominators` on the reversed graph,
rooted at a virtual super-sink).

**Discharge is a minimisation, not a deletion.** Rows that differed only in the dropped key become
identical; keep the cheapest:

```
{J@u}: c1 ┐
{J@v}: c2 ├─ drop J ─→  {} : min(c1, c2, c3)
{J@w}: c3 ┘
```

Dropping a key without taking that min is silent corruption.

This is the forward mirror of `cell_dag_extraction.md` §3.5's early discharge — *first common ancestor
going backward* is *post-dominator going forward*.

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

`J`'s key exists **only between `J` and `m`**. That lifetime is what keeps the table small, and it is
the substantive difference from a source-cell scheme, where a key is fixed at birth and never dies.

---

## 3. Phase 1 — Building the Table

### 3.1 Where it lives

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

`bp` is a list of `(vertex, cell, profile)` triples — one per predecessor, or a single same-vertex
triple for a coverage step, mirroring `bpD`'s convention.

**`Dp` does not replace `D`.** The profiled path never reads `D`; it reads `E`, `forbidden` and `Dp`.
`D` survives because `forward()` must run first to produce the `forbidden` flags, and computing `D` is
how it gets them.

### 3.2 The recurrence

Entry cells for parent `p` into `(a,v)` are `x ∈ ({v} ∪ Bpred(v)) ∩ cand(p)` — a **stall** at `v` or
an **advance** from a B-predecessor. That set is bounded by B's local in-degree (1–3 in a road
network), **not** by `|cand(p)|`, which is what makes a joint minimisation affordable.

```
fill(a, v):

  1. FOLD the predecessors, one at a time, keyed by (profile, any-stall):
       for each parent p:
         options = (entry cell x, is_stall, parent profile pi_p, cost/outdeg(p))
         for each running combo and each option:
             pi = merge(combo.pi, pi_p)        <- consistency test; skip if None
             keep the cheapest per (pi, stall)

  2. PRICE the emission — beta*E if any parent stalled, else E

  3. COVERAGE (H): relax within the row to a fixed point,
       cost(a,w) <- cost(a,v) + alpha*E(a,w)   for each B-arc v->w, per profile

  4. OWN KEY: if a in S, overwrite the pair (a, v)

  5. DISCHARGE: remove every key in drop[a]; rows that now collide keep the cheapest
```

Four points the steps make that prose blurs:

* **The fold is joint, never per-parent.** Minimising each parent separately and then testing the
  winners is the algorithm `cell_dag_extraction.md` §6.2 shows fails.
* **Folding, not enumerating.** The running table is bounded by the number of *consistent* profiles,
  not by the product of the parents'.
* **`a ↦ v` is overwritten, not accumulated** — the entry means "the cell this split's run currently
  ends on", and a coverage step moves it. Consumers attach at the run end, so this is exactly V3.
* **Step 5 is a min** (§2.4).

A source: `Dp = {pi0: (E, [])}`, with `pi0 = {(a,v)}` if `a ∈ S` else `frozenset()`.

### 3.3 Memory

A passthrough vertex — neither split nor discharge point — does not change the profile, so its rows
keep the **parent's frozenset object** rather than a rebuilt equal one. That is the common case and
cut per-row cost from 814 to **279 bytes**.

Real forward tables: **3–14 MB** per hourglass edge, everything retained. No freeing lifecycle is
warranted — and freeing would buy little anyway, since reconstruction needs either `bp` or the costs.

---

## 4. Phase 2 — The Extraction

At a **sink** `t`, the upstream cone is everything above `t`, and the sinks' cones cover `A`. So the
extraction is not a search: it is a **join over profiles**.

### 4.1 Sinks as factors

Each sink contributes a factor over its live splits:

$$f_t(\pi) \;=\; \min_{v \in \mathrm{cand}(t)} \widehat{D}[t][v][\pi|_t]$$

and the answer is

$$C^{*} \;=\; \min_{\pi}\ \sum_{t \in \text{sinks}} f_t(\pi|_t).$$

### 4.2 Why the sinks cannot simply be folded together

Combining sinks pairwise keys the running table by the **union** of their keys. Two sinks sharing no
key always merge successfully, so a pairwise fold enumerates their **cross product**. On `btree(4)` —
16 sinks — that exhausted memory while the forward table it read was only 4.9 MB.

> **A rejected fix, recorded because it looks right.** "Group sinks that share a key, minimise each
> group, sum" does nothing on an out-tree: every `btree` sink descends from the root split, which is
> never discharged, so all sinks share that key and the grouping gives one component.

### 4.3 The algorithm — eliminate keys, deepest first

```
  1. FACTORS      one per sink: its best cell for each profile it can carry

  2. ELIMINATE    for each split J, deepest first:
                     collect the factors mentioning J
                     combine them        (consistent pairs, costs summed)
                     minimise J out      -> one factor over the remaining keys

  3. COMBINE      join whatever factors remain (all keyless now) -> candidate list

  4. JUDGE        cheapest-first, take the first candidate that
                     covers every vertex and passes check_rules

  5. RECONSTRUCT  flood bp from the winning (sink, cell, profile) picks;
                  cover chains expand into run cells
```

**Deepest-first is what bounds it.** When `J` is eliminated, the only key its factors still share is
`J`'s **parent** split, so intermediate factors are bounded by the tree's **depth**, not its total
split count — `btree(4)` has 15 splits and depth 4.

| | pairwise fold | key elimination |
|---|---|---|
| `btree(3)` extraction | 0.763 s · 57.5 MB | **0.003 s · 0.27 MB** |
| `btree(4)` extraction | **MemoryError** (>4 GB) | **0.278 s · 26.9 MB** |

### 4.4 Why the judge is still needed

`π` enforces V3 and the recurrence enforces V2, but **V1 is not covered**: on a cyclic `B` a run can
revisit a B-vertex. So the elimination keeps the `keep` cheapest rows per key rather than only the
minimum, giving the judge fallbacks — the *"top-K contraction"* that
`scripts/repro_contraction_eviction/README.md` asks for.

Swept over 600 cyclic-B cases:

| `keep` | cost parity | cases answered where `extract_cell` raises |
|---|---|---|
| 4 | 487/487 | 94 |
| 8 | 487/487 | 107 |
| **32** *(default)* | 487/487 | **112** |
| 128 | 487/487 | 112 |

**Parity is unaffected at every value** — `keep` never changes correctness, only how many cases the
old engine refuses that this one still answers, and that saturates at 32. Treat it as a measured
plateau, not a proven bound; the failure mode is a refusal, never a wrong answer.

### 4.5 Why summing the sinks is exact

The `1/outdeg` split fractions are **not** an approximation once V3 holds — they are exactly the
weights that make a sink-sum count each vertex once. A vertex's emission enters a descendant's cost
scaled by `∏ 1/outdeg` along the connecting path; summed over every path from it to a sink those
factors total **1**. A merge sums its arms without dividing, so mass arriving by two arms recombines
to 1 as well.

Checked on `cell_dag_extraction.md` §6.1 with the split forced onto one cell: `0.5 + 5.5 = 6.0`, the
true cost. The `1.0` that section reports as the failure is `E(J,v1)/2 + E(J,v2)/2` — half of each
cell. **The under-count *is* the V3 violation**, so blocking the phantom fixes the arithmetic too.

Verified empirically on **384/384** envelope cases across `α ∈ {1, 0.7, 0.5}`, `β ∈ {1, 1.5}`, and on
all four hourglass edges: the sink-sum equals the recomputed matching cost every time.

---

## 5. Worked Example

The smallest case that shows a profile doing anything — one split, one chain:

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
ended on `w`*. Same pairing, two prices, because the upstream differs. The `min` of each list is
exactly what `D` holds.

**Why every row is kept.** If the sibling branch `d` can only be matched with `J` on `w`, then
`(c, w)` cannot use its cheapest row at 9.891 — it must use **11.953**, and the optimum is whichever
total is smallest once *both* branches are priced under the *same* placement of `J`. Collapsing this
list to its minimum throws away the row the optimum needs. That is why costs are held per profile,
and it is the same reason `pending` exists in the engine this replaces.

**Width > 1.** With a second split downstream (`a → J1 → {c,d}`, `c → J2 → {e,f}`), a cell below both
carries one pair per split:

```
cell (e, x)   live splits: J1, J2      width 2
     cost  14.814   when   J1 ends on v,  J2 ends on x
     cost  16.829   when   J1 ends on w,  J2 ends on x
     cost  17.586   when   J1 ends on v,  J2 ends on w
```

The rows enumerate the combinations — which is why each extra live split *multiplies* the row count
rather than adding to it. That graph is also the out-tree failure in miniature: neither split's
branches rejoin, so no key is ever discharged.

---

## 6. Results

### 6.1 Gate

| arm | result |
|---|---|
| unit suite (`tests/`) | **198 passed** |
| structured envelope, 384 cases | **384/384** cost parity vs `extract_cell` |
| sink-sum exactness (§4.5) | **384/384** |
| cyclic-B, 900 cases | **731/731** parity, **0** invalid |
| cyclic-B capability | **168** answered where `extract_cell` raises |
| benchmark families | parity on dense-chain, diamonds, btree |

### 6.2 Real hourglass edges

| edge | `forward` + `extract_cell` | profiled (both phases) | V3 |
|---|---|---|---|
| 100042 | 4.8 s · 63 MB | **0.26 s** | 0 → 0 |
| 102752 | 30.0 s · 248 MB | **0.98 s · 14 MB** | **2 → 0** |
| 100341 | 33.4 s · 215 MB | **0.19 s** | 0 → 0 |
| **100350** | **687.7 s · 783 MB** | **0.44 s · 16 MB** | **3 → 0** |

Profile state space is 36–70× smaller than the `pending` it replaces, and the sink join is a min over
32–234 keys across 2 sinks.

### 6.3 Synthetic families — discharge is the whole game

| case | `|S|` | max profiles/cell, **no** discharge | **with** | width |
|---|---|---|---|---|
| dense_chain(50) | 0 | 1 | 1 | 0 |
| diamond_chain(4) | 4 | 3 100 | **9** | 1 |
| diamond_chain(10) | 10 | — | **9** | 1 |
| btree(4) | 15 | 202 | 202 | 4 |

Discharge turns exponential into linear where branches rejoin: `diamond_chain` 4 → 10 more than
doubles `|A|` while max multiplicity stays flat. `btree` is untouched — no merges, so nothing
post-dominates and nothing discharges.

### 6.4 The capability gain

The 168 are not a speedup. `extract_cell` contracts to one row per **pending signature**, so when that
row is V1-invalid the valid alternative is already gone and it raises a spurious *"no valid root
row"*. This contracts per **profile**, so the judge still has every other profile to fall back on —
resolving the open defect in `scripts/repro_contraction_eviction/`.

---

## 7. Limits

| | |
|---|---|
| **out-trees** | nothing post-dominates, so no key discharges and width grows with depth. `btree(4)`: 2.3 s against `extract_cell`'s 14 ms. Correct and bounded, but slower — the mirror of `pending`'s tree-of-merges wall |
| **`keep` is a plateau** | measured, not proven (§4.4) |
| **no global budget** | `max_profiles` bounds one cell, `max_rows` one factor; neither bounds the aggregate |
| **not adopted** | nothing calls `profiled.py`; wiring into `match_dag` is a separate decision |

---

## 8. Relation to Existing Machinery

| existing | relation |
|---|---|
| §4.1a forbid-and-rebuild | **prerequisite, not replaced.** It answers *"which exits can no child use?"* using transition existence and no DP values; this answers *"which exit should all children take?"* by pricing them jointly. `forward()` keeps owning the `forbidden` flags and the feasibility error |
| `pending` (`cell_dag_extraction.md` §2–3) | the same job keyed on merges instead of splits; §6.2 compares the key spaces |
| §3.5 early discharge | the backward mirror of §2.4 — first common ancestor ↔ post-dominator |
| §8.6 inner-merge elimination | the same min-sum elimination, applied to one merge in the extraction rather than every split |
| `check_forward_v3` | today a diagnostic; under this design it must return **empty** — an invariant |

---

## 9. Reproducing

Everything in §6 is reproducible from `report/`:

| probe | what |
|---|---|
| `probe_v3.py`, `probe_v3_detail.py` | §1's V3 baseline and the dissection of each violation |
| `probe_D_vs_Dp.py` | §1's `D = min Dp` measurement |
| `probe_profile_list.py`, `probe_profile_life.py` | §5's dumps and §2.5's trace |
| `probe_sources_vs_splits.py` | §2.1's source-vs-split comparison |
| `probe_profiled_hourglass.py` | §6.2's per-phase timings and memory |
| `gate_profiled.py`, `gate_profiled_cyclicB.py`, `gate_profiled_bench.py` | §6.1's three gate arms |

The map-conflation ones need `PYTHONPATH=/home/kaveh/projects/map-conflation/src` and the
`osm-dra-conflation` venv.
