# Improvement Plan

**Status:** proposed · **Date:** 2026-07-13
**Origin:** competitive comparison against [Hootenanny](https://github.com/ngageoint/hootenanny) (NGA's conflation platform), followed by a grounded multi-agent design + critique pass over this codebase.
**Plain-language version:** [improvement_plan_simple.md](improvement_plan_simple.md)

---

## 1. Strategic position

network-matching's edge over Hootenanny is the *local* matcher: continuous projection-based
directed graph-DTW (drift in meters, coverage, bearing), native directedness (`is_reverse` twins,
U-turn-impossible junctions), and Mode 3's exact, validity-checked global matching. Hootenanny's
edge is everything *around* the matcher: global consistency (its "Conflicts" score-propagation),
semantics (name/class scoring), match/miss/**review** triage, attribute merging, and a decade of
platform maturity.

**The strategy is NOT to become Hootenanny.** The library stays an accurate, embeddable,
no-training-data matcher. The plan closes exactly the gaps that block the two business goals —
the DAG-DTW paper and the BC/Canada conflation-as-a-service pitch:

1. **Global consistency** — a network-wide reconciliation pass over Mode 1 (Hootenanny's one
   genuinely good idea, rebuilt on our better local cost).
2. **Confidence + review triage** — the "how much manual review is left?" number that prices
   conflation work.
3. **Attribute transfer** — the deliverable a conflation customer actually buys.
4. **A defensible evaluation** — ground truth, metrics, baselines, and (timeboxed) a Hootenanny
   head-to-head.
5. **Semantics** — opt-in name/class terms for parallel-road disambiguation.
6. **Cyclic exactness** — the gated research track behind the paper.

Aggregate demand of everything designed is ~20–24 developer-weeks — not executable as written for
one developer. §3 commits a **~11–12-week core** and §5 defers the rest behind named gates.

---

## 2. Decision records (cross-cutting, decided now)

These resolve every conflict found between the six area designs. Any future work must follow them.

- **D1 — REVIEW lives in a `triage` column, never in `match_type`.** Three downstream sites
  treat non-`NO_MATCH` as matched (`matcher.py:524`, `thresholds.py:421`, `thresholds.py:506`);
  `match_type` values are frozen. Confidence-triage owns the `triage` column
  (`MATCH`/`REVIEW`/`NO_MATCH`) and its thresholds. Reconciliation emits
  `reconcile_action`/`reconcile_margin_m`/residual-conflict signals *into* triage instead of its
  own tier. Benchmark keeps Hootenanny's review state in `hoot_state`, never `match_type`.
- **D2 — Canonical pipeline:** `match_routes → resolve_network (optional) → triage_routes →
  apply_triage → transfer_attributes`. `resolve_routes` becomes the documented legacy path
  (kept working, no new behavior). One edit to `docs/graph_dtw_pipeline.md` states this; areas
  do not ship competing diagrams.
- **D3 — One label store, one campaign.** Ground-truth labels live in
  `ground_truth/match_labels.schema.yaml` (benchmark's format) with a `partition: eval | train`
  field. One ~400-label Sundbyberg campaign, split up front: ~250 **eval, frozen, never trained
  on**; ~150 train for the (deferred) calibrated model. One verdict vocabulary
  (`confirmed/corrected/rejected` + `AMBIGUOUS` + `partial`). One review/label UI: extend
  `graph_dtw_validation_map.py --label-queue` with a `--triage` mode — no second HTML flow.
- **D4 — One `network_matching/issues.py`, owned by confidence-triage**
  (`load_issues/merge_issues/save_issues/emit_match_labels`). Semantic-cost contributes
  `detect_oneway_issues` as a detector on top. One `conflation_issues` schema bump to v3 in a
  single change (adds `oneway_mistagged`, `low_confidence_match`, `route_conflict`).
- **D5 — One data-contract re-export.** `docs/data_contract.md` defines what the sample data
  carries; `scripts/export_to_csv.py` is widened **once** (name, derived `highway`,
  `functional_road_class`, `oneway`, `maxspeed_kmh`, `original_edge_id` — all verified present
  upstream in fetching-sweden-data `main.vehicle_edges`) and `data/sweden_edges.csv` regenerated
  **once**, in Phase 0. Semantic-cost, benchmark, and attribute-transfer consume it; none edits
  the script again.
- **D6 — Interval columns are `edge_b_from_m` / `edge_b_to_m`** (arc-length span of the
  traversed part of each B-edge), computed by reconciliation M1, consumed by attribute-transfer
  (exact A→B overlap) and benchmark (`interval_correctness`). Exported as constants from a shared
  schema module so probes can't drift.
- **D7 — `routes_long`/`routes_summary` schema is governed.** A schema-registry section in
  `docs/data_contract.md` lists every planned additive column and its owner. **Alternatives
  never enter `routes_long`** — top-k routes live in a separate `routes_alts` frame;
  `runner_up_margin_m` and `n_alternatives_considered` are persisted into `routes_summary` at
  resolve time (so triage's feature survives reconciliation's cleanup). Regression tests assert
  column-*subset* equality on pre-existing columns, never whole-frame goldens.
- **D8 — BC data comes from `~/projects/osm-dra-conflation`.** It already holds DRA extracts, an
  export pipeline, and a QA map. No area acquires fresh BC data; `DRA_CLASS_RANKS` is validated
  against that repo's actual extract, not a remembered data dictionary. Licensing: OGL-BC
  attribution; CRS: EPSG:3005 handled at export.
- **D9 — Paper-1 realignment is an owned, scheduled milestone** (Phase 1), not a footnote. The
  draft (`docs/paper/dag_dtw_map_matching.html`) cites functions that no longer exist
  (`match_dag_to_bgraph`, `check_sequence_rules`) and describes a superseded engine. Every week
  of new engine work widens the drift. It absorbs cyclic-exactness M1 (limitation inventory).
- **D10 — Reconciliation solver is exhaustive + greedy only.** Conflict components are
  typically 2–6 edges; the MILP rung is cut (and its published linearization was wrong anyway —
  support bonuses need `y ≤ x_ir, y ≤ x_js`, only conflict penalties use
  `y ≥ x_ir + x_js − 1`; recorded here in case a solver is ever revisited).
- **D11 — Oneway detector direction is B/NVDB-side first.** All 9 curated `oneway_mistagged`
  issues are NVDB-side ("tagged one-way in NVDB but actually two-way"). The detector infers
  from twin structure (an `original_edge_id` with one directed twin while both OSM directions
  match it cleanly); the A/OSM-side mirror check is secondary. Acceptance test: recover the
  curated Sundbyberg issues.
- **D12 — Same-vocabulary class comparison for OSM↔NVDB.** Upstream `vehicle_edges` already
  derives an OSM-vocabulary `highway` column for NVDB; export both it and
  `functional_road_class`. Default `class_compat` compares same-vocabulary ranks; pluggable rank
  maps stay for DRA/NRN.

---

## 3. Committed core (~11–12 developer-weeks)

Each phase is independently shippable; docs are written/updated **before** code in every phase.

### Phase 0 — Hygiene + foundations (~1 week) — *do first, everything depends on it*

Release hygiene (~2–3 days):
- Convert `tests/test_matching.py` to real pytest asserts (currently assert-free — failures exit 0),
  add `tests/conftest.py` + markers (`bench`), and a minimal GitHub Actions workflow running
  `pytest tests/`. Every later regression claim leans on this.
- `LICENSE` file (MIT classifier already claims it), `CHANGELOG.md`, tag `v0.1.0`, demote
  `geopandas` to an extra (restores the lazy-heavy-deps convention and the "pip-installable"
  pitch), decide PyPI publication (needed for the paper's reproducibility claim).
- Fix README's phantom `data/sensors.csv` (Mode 4 example references a file that doesn't exist).
- Cheap perf win: stop returning the full `LocalBGraph` + `warping_path` per A-edge through
  joblib (`graph_dtw.py:1010-1016` → discarded at `matcher.py:894-933`) — every parallel run
  pays pickling/IPC for objects nobody uses.

Shared foundations (~2–3 days):
- Write the decision records above into the repo: `docs/data_contract.md` (D5 + D7 schema
  registry), the D1/D2 pipeline statement in `docs/graph_dtw_pipeline.md`, D3's partition
  protocol note.
- The single `conflation_issues` schema v3 change (D4).
- The one `export_to_csv.py` widening + `data/sweden_edges.csv` regeneration (D5; needs the
  fetching-sweden-data DuckDB).

### Phase 1 — Paper-1 realignment (~2–3 weeks) — *owned work, not a prerequisite footnote*

Rewrite `docs/paper/dag_dtw_map_matching.html` to the shipped engine
(`extract_cell` / V1–V4 / forbid-and-rebuild), absorbing cyclic-exactness M1:
- Assumption inventory + honest cyclic-sources limitation section (`docs/cyclic_sources.md`
  §1–5: V1 2-cycle degeneracy, fixed-point unsoundness lemma, seam-conditioning sketch with the
  run-boundary pin refinement — seam = (run-tail of u, run-head of v)).
- Fix the stale `__main__` demo (`dag_dtw.py:1343-1349`, expects a diamond to raise `NotADAG`).
- Evaluation section placeholders that Phase 5's harness fills.

### Phase 2 — Reconciliation M1: sub-edge intervals (~3 days) — *shared infrastructure*

`LocalBGraph.vert_pos_m`; emit `edge_b_from_m`/`edge_b_to_m` in `route_edges`
(`graph_dtw.py:792-805`, where the needed B-points are in scope) and the `ROUTES_LONG` schema.
Cheapest highest-leverage change: unlocks attribute-transfer's exact A→B overlap, benchmark's
interval correctness, and reconciliation's own conflict detection. Run `validate_b_geometry`'s
gap tool on the **A** side here too (the 0.75 m snap default is a B-side convention).

### Phase 3 — Attribute transfer M1–M3 (~1.5 weeks) — *the sellable deliverable*

`network_matching/transfer.py` + `DuckDBMapMatcher.transfer_attributes()`:
- M1: B→A transfer with `wmean/wmedian/wmin/wmax/dominant/list` rules, provenance (contributing
  ids + length shares + confidence join), conflict/tie flags, NO_MATCH coverage. Weighting by
  `edge_matched_len` (not the capped pct); `groupby`-sum handles cyclic re-entry.
- M2: `ranked_name` (imports `normalize_name` from semantics.py once Phase 7 lands — one
  normalizer, D4-style single ownership), A→B direction with over-use flag + UNUSED rows,
  `is_reverse` twin provenance, physical-road collapse.
- M3: exports (GPKG via DuckDB GDAL `COPY` — verified working in this env; CSV; Parquet) +
  `scripts/transfer_demo.py`.
- Guard (D7): transfer raises if multiple alt ranks are present in `routes_long`. Policy:
  `triage == REVIEW` rows excluded from transfer by default, `include_review=True` to override.
  Transfer runs on **final** (post-reconciliation, post-triage) frames.

### Phase 4 — Confidence-triage M1–M2 (~1.5 weeks) — *establishes the one REVIEW convention*

- M1: `bands` output from `suggest_thresholds`' existing estimator spread (band width 0
  reproduces `resolve_routes` exactly — state the boundary rule: `x == cut` passes, matching the
  strict `>` in `matcher.py:1116-1122`, with exact-boundary test rows);
  `network_matching/confidence.py` (`featurize`, `band_confidence`, `triage`, `triage_report`);
  `triage_routes`/`apply_triage` on the matcher. Zero labels needed. Deliverable: the
  `review_rate` number on Sundbyberg.
- M2: `issues.py` (D4 owner) + the review artifact — extend the validation map with the ranked
  review queue (D3: one HTML flow), `--emit-issues` round-trip.

### Phase 5 — Benchmark M1–M4 (~3.5 weeks) — *evidence for paper and pitch*

- M1 (~4 days): `network_matching/evaluation.py` — pair P/R/F1 with dest-dedup for cycle
  re-entry, strict/lenient edge accuracy, length-weighted correctness/completeness, Wilson CIs,
  `review_rate_curve` (pluggable score column — consumes triage's confidence), McNemar.
- M2 (~1.5 weeks incl. labelling): the **unified** D3 label campaign — schema, stratified
  `seed_labels` sampler (n=400, power-analysis backed), `--label-queue` labelling sessions,
  partition field written at sampling time.
- M3 (~5 days): `scripts/benchmark.py` (run/eval/compare/report, `perf_counter` + `tracemalloc`,
  results JSON + history), bench-marker CI smoke on a mini fixture, **and the scale gate**: one
  Stockholm-county-scale intrinsic run (data exists in duckOSM) with timing/memory frozen into
  the doc. If it fails, a tiling design doc (spatial partition + halo edges, per-tile DuckDB
  files per the duckOSM convention) becomes a committed follow-up before any BC promise.
- M4 (~5 days): baselines — Mode 2 `one_to_one` as ours-edge-dtw, a Walter & Fritsch-style
  buffer-growing baseline, the paper's Saalfeld-family baseline. Our own numbers with CIs are
  the paper's evaluation section; no Hootenanny needed yet.

### Phase 6 — BC pilot integration (~1 week) — *converts the plan into a sellable demo*

On `osm-dra-conflation`'s existing extracts (D8): run match → triage → transfer → benchmark-eval
on one DRA tile. Define the pilot package in `docs/bc_pilot.md` **first**: enriched GPKG
(attribute-transfer) + review HTML + `review_rate` number (triage) + correspondence CSV +
metrics table (benchmark), with OGL-BC attribution and EPSG:3005 handling. ~200 labels on the
tile only if a customer conversation needs area-specific numbers (else defer).

### Phase 7 — Semantic-cost M1–M2 (~1 week) — *post-hoc columns and gates only*

- M1: `network_matching/semantics.py` (`normalize_name`, `name_similarity`, `class_compat`,
  `semantic_penalty`; Swedish/English abbreviation tables; rapidfuzz optional/lazy with
  pure-Python fallback).
- M2: `configure_attributes()`, post-hoc `name_sim`/`class_compat` (+ `_len_pct`) columns,
  `resolve_routes` gates, `suggest_thresholds` registration (coordinate the `_RESOLVE_KW` merge
  with triage's `bands` addition — one merge order). Columns **absent unless configured** (the
  pinned contract). Measures `name_sim_len_pct` — the gate for deferred M3/M4.
- Quick add once issues.py exists: the D11 oneway detector (SC M5, ~2 days) with the
  recover-the-curated-issues acceptance test.

### Phase 8 — Reconciliation M2: conflict/support report (~3–4 days) — *the probe*

`network_matching/reconcile.py`: A-adjacency (reuse the endpoint-snap pattern from
`bgraph_prep`/DuckDB `ST_DWithin` on A endpoints), B-claims from intervals, `find_conflicts` /
`find_supports`; `resolve_network(action="report")` adds `support_count`/`conflict_len_m` to
`routes_summary`; validation-map overlay switches to true intervals. **This report decides the
deferred tail:** if conflicts are mostly *selection* errors (a better alternative existed),
GR M3/M4 re-enter; if they're *candidate* errors, top-k + solver would be wasted and the fix is
upstream (radius, semantics).

---

## 4. Deferred, with named re-entry gates

| Deferred work | Re-enter when |
|---|---|
| **Reconciliation M3–M4** — top-k alternatives (`routes_alts` frame, D7) + exact component resolution (`reconcile_action` → triage, D1; exhaustive+greedy, D10). Budget honestly: the post-DP assembly tail (~230 lines, `graph_dtw.py:693-921`) must first be refactored into a per-path `_assemble_route(...)` with a regression pin — 6–8 days for M3, 6–8 for M4. | Phase 8's report shows ≥ a meaningful share of conflicts are selection errors on real data. |
| **Confidence-triage M3–M5** — labels→features, calibrated logistic model (JSON artifact, numpy predict), second-area validation. Train only on the D3 `train` partition. | The BC pilot or the paper needs a calibrated number the unsupervised band can't provide — and the band's review_rate has been tried on a real customer conversation first. |
| **Semantic-cost M3–M4** — in-emission `name_weight`/`class_weight` in the two DP modes + DAG mode. | Phase 7's measured `name_sim_len_pct` ≥ ~20% on Sundbyberg matched length (the design's own invalidation floor), or a DRA/NRN dataset shows dense names. |
| **Benchmark M5** — Hootenanny head-to-head. Timebox 2 weeks. Day-1 spike: pinned image boots, `score-matches`/REF options exist, `nm:*` tags survive a 10-way round-trip through `hoot conflate` — if the spike fails, ship the fallback (ours + baselines table, Hootenanny row "blocked") and stop. | After Phase 1 (metric definitions frozen) AND the pipeline is final (post any GR M4 / CT M4), so the comparison runs **once**. |
| **Benchmark M6** — BC second-area numbers (200 labels). | A real BC customer conversation needs them (merged into Phase 6's scope if so). |
| **Cyclic-exactness M2–M3** — SCC survey (GO/NO-GO gate; point it at osm-dra-conflation's extract per D8, decidable on Sundbyberg + Stockholm county alone) + FAS-relative validity semantics + bounded-unroll oracle tests. ~2 weeks, shippable even on NO-GO ("DAG paper first, cyclic as future work" becomes citable). | After Phase 1 ships. |
| **Cyclic-exactness M4–M6** — the conditioning engine + integration + paper experiments. | Paper 1 submitted AND the M2 survey gate passes. Default outcome is that this never runs — that's fine. |
| **Reconciliation M5, tiling implementation, PyPI automation, RF model option** | Explicit demand only. |

---

## 5. What is deliberately NOT planned

- **Geometry merging/snapping** — that's Hootenanny's game and its maintenance moat. Attribute
  transfer captures most customer value at ~10% of the cost. If a customer needs merged
  geometry, that's a Hootenanny-integration conversation, not a library feature.
- **A review UI/platform** — the review artifact stays static HTML + YAML round-trip.
- **Country-scale claims** — until the Phase 5 county-scale gate passes and a tiling design
  exists, all claims are municipality-to-county scale.
- **Building/POI/river conflation** — linear road networks (+ points-on-roads) only.

---

## 6. Risks

- **One developer, ~12 committed weeks, competing with sonocount + the Stockholm pipeline.**
  The phases are sequenced so that stopping after ANY phase leaves the library strictly better
  (hygiene → paper honest → intervals → sellable GPKG → review number → defensible numbers →
  BC demo). If the BC opportunity accelerates, run Phases 0→2→3→4→6 (≈ 5.5 weeks) and defer
  the rest.
- **Labelling time is the scarcest resource** (~2–3 days of human verdicts in Phase 5). D3
  exists precisely to spend it once. Do not start any labelling before the partition protocol
  is written.
- **Hootenanny interop is the riskiest single item** (three compounding unknowns in tag
  survival / REF options / merged-output extraction) — hence timeboxed with a day-1 spike and a
  defined fallback deliverable.
- **Schema drift across areas** was the biggest coherence hazard found in review — D1/D6/D7
  are the guardrails; enforce them in code review against the registry in
  `docs/data_contract.md`.

---

## 7. Provenance

Produced from: (a) Hootenanny algorithm docs (`RoadConflation.asciidoc`,
`NetworkConflation.asciidoc`, `References.asciidoc`); (b) five grounded code-reading passes over
this repo (Mode 1 internals, DAG-DTW internals, quality assets, data/semantics, test tooling);
(c) six area designs; (d) three adversarial critique passes (feasibility — verified against
code; coherence — cross-area contracts; completeness — business/scale/maintenance). All
file:line references above were verified against the working tree on 2026-07-13.
