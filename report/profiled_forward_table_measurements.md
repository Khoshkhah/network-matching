# Measurements — the Profiled Forward Table

Raw results backing `docs/profiled_forward_table.md`. The probes in this folder reproduce every
number below; nothing here is hand-copied from a run that cannot be repeated.

**Date:** 2026-07-19 · **Engine:** `network_matching.dag_dtw` + `profiled` at `08c1cb8`

§1–§5 were measured against the unmodified engine at `ddba616`, before the profiled
engine existed; they are the *motivation*. §6–§7 are the final state.

**Hourglass numbers in §1, §2 and §4 predate map-conflation `d72c09b`** (near-reversal stubs
flipped into the junction), so their edge costs and split/merge counts describe the earlier
source geometry. They are kept as the motivating measurement; §6's table is the live baseline.

| probe | measures | needs `map-conflation` |
|---|---|---|
| `probe_v3.py` | V3 violation rate of today's forward table (§1) | optional — synthetic half runs alone |
| `probe_v3_detail.py` | dissection of each violation (§2) | yes |
| `probe_profiles.py` | profile multiplicity/width on synthetic families (§3) | no |
| `probe_hourglass.py` | profile state space + sink join on the real edges (§4) | yes |
| `probe_bplinks.py` | the refuted `bpD`-routing experiment (§5) | no |

Run the map-conflation ones as:

```bash
PYTHONPATH=/home/kaveh/projects/map-conflation/src \
/home/kaveh/projects/osm-dra-conflation/.venv/bin/python report/probe_hourglass.py
```

`probe_bplinks.py` additionally needs a `bp_links` flag on `extract_cell` — see §5, which records the
patch. That flag is **not** in the engine (the experiment was refuted and reverted).

---

## 1. V3 validity of today's forward table

`check_forward_v3` reads the forward table alone (seed each sink at its arg-min `D`, follow `bpD`) and
reports where the trace places a split on two cells. `check_split_exits` is §4.1a's own invariant.

| input | splits | `check_split_exits` | `check_forward_v3` | `extract_cell` cost |
|---|---|---|---|---|
| dense_chain(50) point / segment | 0 / 0 | 0 bad | 0 / 0 | 348.139 / 341.024 |
| diamond_chain(4) point / segment | 4 / 4 | 0 bad | 0 / 0 | 8.400 / 9.600 |
| diamond_chain(10) point / segment | 10 / 10 | 0 bad | 0 / 0 | 20.400 / 24.000 |
| btree(3) point / segment | 7 / 6 | 0 bad | 0 / 0 | 11.600 / 11.200 |
| btree(4) point / segment | 15 / 14 | 0 bad | 0 / 0 | 24.400 / 24.000 |
| line 100042 | 2 | 0 bad | 0 | 401.911 |
| line 100341 | 2 | 0 bad | 0 | 410.399 |
| **line 102752** | 2 | 0 bad | **2** | 481.201 |
| **line 100350** | 2 | 0 bad | **3** | 308.924 |

The two invalid edges are the two slow ones. `check_split_exits` passes everywhere, so §4.1a delivers
its stated invariant — the violations are outside what feasibility pruning can reach.

## 2. What each violation is

All five have one shape. Representative, from `102752`:

```
split ('i',3,4)@(1,485708.71,5453571.35)     outdeg=2  indeg=1
  child ('i',1,1):  feasible_exits=46   bpD-linked=22   chose (46, 7)
  child ('i',4,1):  feasible_exits=46   bpD-linked=20   chose (7, 110)
  §4.1a allowed = intersection of children's feasible exits = 46 cells
```

Both children can use **all 46** exits, so the intersection removes nothing and `forbidden` stays
empty — correctly. Each child then fills its row independently and links its own cheapest exit; they
differ, and the split lands on two cells. The other four violations (one more on `102752`, three on
`100350`) are identical in shape, with 41–66 feasible exits.

## 3. Profile multiplicity — synthetic families

All profiles kept (the measurement; the design keeps costs per profile). "no discharge" retains every
split key to the sinks; "with" applies the post-dominator rule.

| case | `S` | cells | max mult, no discharge | max mult, with | width | entries | peak |
|---|---|---|---|---|---|---|---|
| dense_chain(50) point | 0 | 536 | 1 | 1 | 0 | 0 | 0.27 MB |
| dense_chain(50) segment | 0 | 529 | 1 | 1 | 0 | 0 | 0.26 MB |
| diamond_chain(4) point | 4 | 163 | 3 100 | **9** | 1 | 367 | 0.20 MB |
| diamond_chain(4) segment | 4 | 222 | 4 640 | **11** | 1 | 979 | 0.43 MB |
| diamond_chain(10) point | 10 | 421 | — | **9** | 1 | 1 021 | 0.49 MB |
| diamond_chain(10) segment | 10 | 594 | — | **11** | 1 | 2 869 | 1.18 MB |
| btree(3) point | 7 | 284 | — | 40 | 3 | 13 157 | 2.11 MB |
| btree(3) segment | 6 | 284 | — | 20 | 2 | 4 503 | 0.95 MB |
| btree(4) point | 15 | 1 036 | 202 | 202 | 4 | 312 923 | 40.4 MB |

Discharge turns exponential into linear where branches rejoin: diamond_chain 4 → 10 more than doubles
`|A|` while max multiplicity stays flat at 9/11. `btree` has no merges, so nothing post-dominates and
nothing discharges — the design's bad case, and the mirror of `pending`'s (trees with multiple
merges).

## 4. The real hourglass — state space and sink join

`LA` built exactly as `mapconflation.match.direction.match_task` does (`tree_to_digraph` → `line_digraph`
→ bounce removal → `road_id`/`seq` copy). `hp` from `config/hyperparams.vancouver_city.json`:
`step=15.0, rladder=(41,60,90), k_min=4, buf=105.0, alpha=0.97, beta=1.19, bearing_weight=1.39`.
All four resolved at `r=41`.

| edge | `\|LA\|` | merges | splits | `pending` ∏ (parts) | profile max mult | width | mult=1 | entries | peak | time |
|---|---|---|---|---|---|---|---|---|---|---|
| 102752 | 29 | 3 | 2 | 28 350 (45,45,14) | **140** | 2 | 57.7% | 113 892 | 22.9 MB | 0.94 s |
| 100042 | 26 | 3 | 2 | 7 888 (29,17,16) | **32** | 2 | 38.6% | 24 339 | 5.2 MB | 0.25 s |
| 100341 | 29 | 3 | 2 | 7 220 (20,19,19) | **38** | 2 | 66.0% | 24 370 | 5.6 MB | 0.15 s |
| 100350 | 21 | 3 | 2 | 77 000 (56,55,25) | **194** | 2 | 80.7% | 55 683 | 14.1 MB | 0.36 s |

The `pending ∏` column reproduces `cell_dag_extraction.md` §8.5 to the digit (28 350 / 7 888 / 7 220 /
77 000), which validates the graph construction.

**Sink join** — for a fixed profile the sinks are independent, so this is the whole extraction:

| edge | sinks | profiles per sink | distinct keys | splits live at sinks |
|---|---|---|---|---|
| 102752 | 2 | 140, 140 | **140** | 2 |
| 100042 | 2 | 32, 32 | **32** | 2 |
| 100341 | 2 | 38, 38 | **38** | 2 |
| 100350 | 2 | 234, 225 | **234** | 2 |

**Honest ratio.** `cell_dag_extraction.md` §8.6 records that `extract_cell` already carries only 5 087
of `102752`'s 28 350 (infeasible pairs drop at `PathCost = ∞`), so the fair comparison is `5 087 : 140`
≈ **36×**, not the raw 202×. At ~18% survival, `100350` gives ≈ `13 900 : 194` ≈ **70×**.

`peak` is the probe holding Python `frozenset`s. Packed (two int pairs + a float per row, ~57 k rows)
is ~2–3 MB.

## 5. Refuted — routing the extraction by `bpD`

**Hypothesis.** `extract_cell` ignores the forward table's back-pointers entirely (verified: zero
`bpD` references in its body; it reads only `cand[v]["D"] < INF` at `:766`/`:913` and the `forbidden`
flags). Restricting its inbox routing to the parent cells `bpD` actually links (`_links`) instead of
every feasible transition (`_feasible_links`) would cut the branching for free.

**Patch used** (three edits, applied then reverted):

- `extract_cell(..., bp_links: bool = False)`
- in the merge-interface loop: `if bp_links and (X, u) not in A.nodes[c]["cand"][ce]["bpD"]: continue`
- at the inbox push: intersect the target cells `u` with `{x for (q, x) in cand[e]["bpD"] if q == absorbed_by[X]}`

**Result — refuted.** 25 cases (the 16-case parity suite + 10 family/mode combinations):

| | count |
|---|---|
| identical cost | 23 |
| **cost divergence** | **2** |
| both refuse (two-cycle) | 1 |

- `polytree2`: 40.8223 → 50.0199 (**+22.5%**)
- `polytree7`: 88.0488 → 95.5383 (**+8.5%**)

Both returned **valid but costlier** matchings — no error raised, the silent failure mode of
`cell_dag_extraction.md` §6.2. It was genuinely faster (diamond_chain(10) segment `34 ms → 7 ms`), so
the pruning is real; it just prunes away optima.

**Why.** `bpD` stores the parent cell that minimised the phantom-contaminated relaxation, blind to
downstream V3 and run structure. The optimum sometimes enters a cell from a parent the forward arg-min
did not pick.

**Retry only after** the forward table is V3-valid — the same experiment becomes sound once the table
it reads stops lying, and the ~4× speedup is then worth reclaiming.


---

## 6. Gate — final, at `7af3113`

Run twice: engines called directly, and again **through `match_dag(engine="auto")`** — the second
form exercises the shared `extract_by_engine` dispatch that both the library and DuckDB APIs use.

**Direct**

| gate | result |
|---|---|
| unit suite (`tests/`) | **198 passed** |
| structured envelope, 384 cases | **384/384** valid · **384/384** cost parity · **384/384** sink-sum identity |
| cyclic-B, 900 cases | **731/731** parity · **0** invalid · **166** answered where `extract_cell` raises (judge fallbacks at the last elimination step, docs §5.4) |

**Through `match_dag(engine="auto")`**

| gate | cases | parity | invalid | lost | bonus | width histogram |
|---|---|---|---|---|---|---|
| envelope | 384 | **384/384** | 0 | 0 | 0 | `{0:192, 1:96, 2:96}` |
| cyclic-B | 600 | **487/487** | 0 | 0 | **107** | `{1:411, 2:183, 3:6}` |

The width histograms confirm the dispatch is exercised rather than falling through: the **6 cyclic-B
cases at width 3** were routed to `"cell"`, everything else to `"profiled"`. Identical numbers before
and after the dispatch was deduplicated into `extract_by_engine`.

Reproducer: `report/gate_auto_dispatch.py`. It also reports merge pressure, and that exposed a
coverage gap — the generated population is **`Mo = 0` for all 600 cases**, so this gate never routes
to `"rebase"` (which needs `Mo >= W`, §9), and the other two gates take the cone branch. The re-based
path therefore had **no automated coverage at all**.

**Rebase gate** — `report/gate_rebase.py`, over `braid(k, j)` (`W ≈ k`, `Mo ≈ j` independently):

| cases | shapes routed to rebase | parity vs `extract_cell` | invalid |
|---|---|---|---|
| 66 (`k=3..6`, `j=0..k`, × 3 weightings) | **4** | **63/63** | 0 |

It asserts its own coverage — `rebase_cases > 0` is part of the GREEN condition — because a gate that
silently stops exercising its target is worse than no gate. `braid(6, 6)` is the case worth watching:
`extract_cell` **refuses** it and re-basing answers `16.800000`, which is the whole reason the engine
exists.

**Real hourglass edges, end-to-end through `match_dag`** — all four routed to `profiled` at width 2.
*Pre-`d72c09b` geometry; §6 has the live baseline.*

| edge | `extract_cell` | profiled | V3 |
|---|---|---|---|
| 100042 | 4.8 s · 63 MB | **0.10 s · 3.8 MB** | 0 → 0 |
| 102752 | 30.0 s · 248 MB | **0.83 s · 16.8 MB** | **2 → 0** |
| 100341 | 33.4 s · 215 MB | **0.10 s · 4.4 MB** | 0 → 0 |
| **100350** | **687.7 s · 783 MB** | **0.48 s · 14.9 MB** | **3 → 0** |

### Batch — 150 random `vancouver_city` edges

The population the library actually runs on, rather than hand-picked hard edges. Reproducer:
`report/probe_batch.py`.

| | |
|---|---|
| solved | **149 / 150** |
| radius used | `{41: 146, 60: 3}` |
| engine chosen | `profiled` x150 — **zero** routed to `rebase` |
| `(W, Mo)` shapes | `{(0,1): 3, (1,0): 3, (1,1): 142, (2,3): 2}` |
| total time | `auto` **1.5 s** vs `extract_cell` **22.9 s** — **15x** |
| cost disagreements · auto-only · cell-only | 0 · 0 · 0 |
| unmatched at every radius | `2490258` (solves at `r=160`, above `rladder`'s top of 90) |

Two things this settles.

**Real edges are overwhelmingly trivial.** 142 of 150 are `W=1, Mo=1`; nothing sampled exceeded
`W=2`. Line `100935` is an outlier, not the tip of a distribution — so the re-basing work buys
robustness on rare junctions, not throughput. It is also why `report/gate_rebase.py` had to be
written: real data would never have exercised that path, let alone caught a regression in it.

**The ladder must be run to measure refusals.** `mapconflation.match.direction:201` escalates
`hp.rladder` whenever a radius yields no result — including when *extraction* fails, not only
`forward`. An earlier version of this probe broke out of the ladder as soon as `forward` succeeded
and gave up after one extraction attempt; it reported **4** refusals instead of 1, and that was
wrongly written up (commit `f1a556e`) as a defect in `match_task`. There is no such defect. Three of
those four resolve at the ladder's own next rung, which the `{41: 146, 60: 3}` row shows directly.
The 15x figure supersedes the 22x in that commit message for the same reason — the earlier run had
`auto` doing less work than `extract_cell`.

**Regression check through `match_dag(engine="auto")`** — reproducer `report/gate_hourglass.py`,
which also cross-validates every cost against `extract_cell`:

| edge | `\|LA\|` | `W` | `Mo` | engine | cost | V3 in forward table | `auto` | `extract_cell` |
|---|---|---|---|---|---|---|---|---|
| **102752** | 29 | 2 | 3 | `profiled` | 496.2937 | **2** | **0.07 s** | 26.54 s |
| 100042 | 26 | 3 | 2 | `cell` | 420.8484 | 1 | 0.03 s | 0.02 s |
| 100341 | 29 | 3 | 2 | `cell` | 454.4490 | 1 | 0.23 s | 0.23 s |
| 100350 | 21 | 1 | 1 | `profiled` | 304.2849 | 1 | 0.02 s | 0.02 s |
| 100935 | 33 | 1 | 1 | `profiled` | 524.8200 | **2** | 0.21 s | 1.15 s |

**5/5 at baseline, 5/5 agreeing with `extract_cell`.** The V3 column is why this engine exists: the
forward table still places a split on two cells on 102752 and 100935, and the profiled path resolves
it — `V1/V2/V3 = 0/0/0` on all five.

> **The sources are an external input, and they move.** `local_dag.build_hourglass` in map-conflation
> constructs them, so a change there silently changes what this library is handed. Commit `d72c09b`
> ("flip near-reversal stubs into the junction, `TURN_MAX = 160°`") reshaped all five: a stub doubling
> back on the corridor is the opposing carriageway of a divided road, so flipping it makes the
> junction read as merge-then-continue rather than a U-turn. The table above is pinned to `d173727`.
>
> The effect on this library is large and favourable. Line `100935` — the source that motivated the
> whole extraction rewrite — went from `W=5, Mo=5` needing `rebase` at 4.21 s to `W=1, Mo=1` on
> `profiled` at 0.21 s. Most of its pathology was near-reversal stubs. The re-basing work still
> stands on `braid` (`report/gate_rebase.py`), which is now its only coverage.
---

## 7. Which engine wins on which shape

Timed via `match_dag`, medians. `report/probe_engine_choice.py` reproduces the ladder rows.

| shape | width | `profiled` | `cell` | `rebase` | best |
|---|---|---|---|---|---|
| dense_chain(400) | 0 | 0.045 s | 0.060 s | 0.035 s | ~tie |
| diamond_chain(40) | 1 | **0.013 s** | 0.026 s | 0.018 s | **profiled** |
| diamond_chain(120) | 1 | **0.053 s** | 0.125 s | 0.113 s | **profiled** |
| diamond_chain(400) | 1 | **0.256 s** | 0.886 s | 0.411 s | **profiled** |
| hourglass ×4 | 2 | **0.10–0.83 s** | 4.8–687.7 s | — | **profiled** |
| btree(4) | 4 | 0.289 s | **0.008 s** | 0.014 s | **cell** |
| btree(5) | 5 | 33.9 s | **0.048 s** | 0.066 s | **cell** |
| ladder(3…9) | 3–9 | 0.026 s → FAIL | **0.003–0.006 s** | FAIL | **cell** |

**`rebase` is never first.** The ladder family was built specifically to find its niche — deep nested
splits *and* a wide merge — and it fails there while `cell` runs in 5 ms. Two assumptions behind that
prediction were wrong:

* *"a wide merge hurts `cell`"* — it does not. `pending` pays per **concurrently-open** merge; one
  merge is one factor whatever its in-degree. The hourglass wall was **three** merges at 45×45×14.
* *"re-basing holds width 1 under nesting"* — only when a merge's arms share a last split. A fan-in
  merge gives each arm a **different** last split, so its frontier is as wide as the fan-in.

Hence `auto`'s two-way rule — `profiled` at width ≤ 2, else `cell` — with `rebase` opt-in and its
niche unproven.
