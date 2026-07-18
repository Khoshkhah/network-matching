# Measurements — the Profiled Forward Table

Raw results backing `docs/profiled_forward_table.md`. The probes in this folder reproduce every
number below; nothing here is hand-copied from a run that cannot be repeated.

**Date:** 2026-07-18 · **Engine:** `network_matching.dag_dtw` at `ddba616` (unmodified)

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
