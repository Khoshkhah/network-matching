# Tree-DTW — minimal matchings by anchored enumeration

> **Status: design / decision doc.** It proposes one mechanism that subsumes the current
> extraction's fragility (the Part 5 single arg-min seed) and the forward/backward disagreement
> (Part 6d). It presents *both* table strategies side by side and ends with a decision table — the
> choice between them is deferred to the reader. Notation, tables (`D`, `B`), emission `E`, cost
> `C(M)`, weights `α`/`β`, rules (V1)–(V4) and all `§`/Part references are from
> `docs/tree_dtw_matching.md` (the algorithm spec; its §9 is the implementation chapter, Parts 1–6).

## The idea in one paragraph

Do not trust the DP tables to *pick* the matching. Trust them only to *propose* a small set of
whole, coherent matchings; then score each proposed matching by the **honest objective `C(M)`** and
keep the cheapest. Because A is a connected tree, each proposal is pinned by a **single anchor**
`(a₀ → v₀)`, and — fixing the seed vertex `a₀` — there are **at most `|V(B)|`** distinct proposals.
So the whole candidate set is enumerable in linear time, and selection never touches `D`/`B`, which
sidesteps the fact that the two tables disagree under weighting (§6d).

This turns "we get a good matching but can't say it's the best" into either "the best among an
enumerable pool" (robust, available now) or "the provable global optimum" (with the §6b certificate),
depending on the table strategy chosen in §5 below.

---

## 1 — The object: a minimal matching is an anchored extraction

An **anchor** is a committed seed cell `(a₀, v₀)`, `v₀ ∈ cand(a₀)` — one source vertex pinned to one
candidate. Clamp it and run the §5 flood: walk `bpD` upstream, `bpB` downstream, and siblings down
from shared parents. Because A is **connected and singly-connected** (a tree — §1.1), that flood
commits **every** vertex of A from the one seed (§5 step 2 already states this). The result is a
complete matching that is a deterministic function of the anchor:

```
Ξ_{a₀} :  v₀  ↦  M_{a₀,v₀}          (clamp a₀→v₀, flood, return the whole relation M)
```

Call `M_{a₀,v₀}` a **minimal matching**: it is uniquely determined once the anchor and the fixed
tie-break (§4b) are set. "Minimal" here means *anchor-determined / tie-free*, **not** minimum-cost —
a minimal matching need not be the cheapest (that is what §3 selects for).

Two knobs, only one of them free:

| knob | role | free? |
|---|---|---|
| the seed **vertex** `a₀` | where the flood starts | **yes** — connectivity means any vertex reaches all of A |
| the seed **value** `v₀` | which minimal matching you land on | **no** — this selects the answer |

`Ξ` extends the current code by one step only: §5 today evaluates `Ξ_{a₀}` at the *single* value
`v₀ = argmin_v g(a₀,v)`. Everything below enumerates `Ξ_{a₀}` over its whole domain instead.

## 2 — The count: at most `|V(B)|` minimal matchings

Fix the seed vertex `a₀`. `Ξ_{a₀}` is a function from `cand(a₀) ⊆ V(B)` to matchings, so its image —
the set of distinct minimal matchings reachable from `a₀` — has

```
#{ minimal matchings from a₀ }  ≤  |cand(a₀)|  ≤  |V(B)|.
```

Every minimal matching `M` is in that image: it pins its own seed, `M = Ξ_{a₀}(committed_M[a₀])`. The
bound is **loose** in practice — infeasible values (`∞` rows, §1.3) drop out, and many values fan out
to the *same* `M`. The significance:

> The ambiguity set that is worst-case **exponential** in `|V(A)|` (independent ties compose across
> branches) collapses, under the anchored definition, to **at most linear in `|V(B)|`** — small enough
> to enumerate exhaustively.

**Scope of the bound.** It is `≤ |V(B)|` *per fixed seed vertex*. Whether a *different* seed can
produce a matching unreachable from `a₀` is the **confluence** question: if the tie-break (§4b) is
globally consistent then seeding `M` from any of its own vertices reproduces `M`, and `|V(B)|` bounds
the total. If not, the bound is per-seed and pooling across seeds only widens coverage (§3). Either
way the pool stays polynomial.

## 3 — Enumerate and score: generation ≠ judgment

The pivot of the whole design: **the cost of a complete `M` is computed directly, never from the
tables.**

```
generate:   for each feasible v₀ ∈ cand(a₀):   M_{v₀} = Ξ_{a₀}(v₀)          # uses D/B — may be imperfect
judge:      C(M_{v₀}) = Σ_{(a,v)∈M_{v₀}} w(a,v)·E(a,v)                       # reads only M + graphs
select:     return argmin_{v₀} C(M_{v₀})
```

`C(M)` is the objective already defined in §"The optimization problem": re-apply the per-move weights
to the concrete pairs of `M` — advance `w=1`, 1:N coverage `w=α`, N:1 stall `w=β`, merge-shared
`1/outdeg` — an `O(|M|)` sum that reads only the committed relation and the graphs. **It does not
reference `D` or `B` at all.** Consequences:

* **The §6d disagreement stops mattering for the answer.** `D`/`B` are demoted from *source of truth*
  to *candidate generators*. Whether they agree affects *which* candidates appear, never how they are
  ranked. The known weighted-case complementarity (forward optimistic at splits, backward at merges)
  can no longer corrupt the selection.
* **The "table energy vs. true relation cost" question dissolves.** You score with the honest `C(M)`,
  including the coverage/stall weights — the very terms behind the §5 coverage gap-fill — so there is
  no proxy to be wrong.
* **The pool can be widened for free.** Add seeds beyond `a₀`, add forward-seeded *and* backward-seeded
  decodes, add alternative tie-resolutions. An honest judge means more generators can only improve the
  winner, never worsen it. Under weighting, where the two tables genuinely differ (§6d), generating
  from *both* is the natural way to make sure the optimum is proposed.

### The honest boundary (state this plainly)

Direct scoring guarantees **the best among the candidates generated** — not, by itself, the **global**
best. They coincide only if the optimum is actually in the pool (the *coverage* question). So:

| you want… | you get it… |
|---|---|
| a correct, coherent, defensible `M` **now**, with today's imperfect tables | ✅ from generate-and-score as written |
| an honest cost number to compare/rank candidates | ✅ `C(M)` is exact and table-free |
| a **proof** the result is the global optimum | needs coverage — see §4 |

## 4 — When it is provably the global optimum

The selection is exact — `argmin_{v₀} C(M_{v₀}) = argmin_M C(M)` — under the decomposition

```
min_M C(M)  =  min_{v₀∈V(B)}  min_{M : M(a₀)=v₀} C(M)  =  min_{v₀} C(M_{v₀}),
```

**provided** the flood `Ξ_{a₀}(v₀)` returns the true *conditional optimum* given the clamp. On a tree
that holds iff `D` and `B` are exact **adjoints** (same `E`, same `ψ`, same directions, reverse sweep).
The check is already specified in §6b:

```
g(a) = min_v ( D[a][v] + B[a][v] − E(a,v) )   must be CONSTANT across the component  (= E*).
```

* **Constant across all `a`** ⇒ tables adjoint ⇒ `Ξ` recovers the conditional MAP ⇒ enumerate-and-score
  is the **provable global optimum**. (And then a single decode at `argmin g` already suffices — the
  enumeration is only insurance, cheap either way.)
* **Varies across `a`** ⇒ tables not adjoint (the §6d weighted case) ⇒ `Ξ` returns a coherent but
  possibly sub-optimal completion ⇒ enumerate-and-score returns the **best of the pool**, honestly
  scored, but without the optimality certificate.

So `g`-constancy is the dial between "robust/good" and "provably best." Honest scoring (§3) makes the
result usable **regardless**; the certificate is an independent upgrade, not a prerequisite.

## 5 — The fork: which table(s) generate the pool

Both forks use the §3 honest judge unchanged. They differ only in how proposals are generated.

### Fork A — keep both tables, make them adjoint

Repair the §6d disagreement so `D` and `B` are exact adjoints over the identical directed transition;
then `g` is constant (§6b), `Ξ` from **any interior seed** is exact, and enumeration is provably
optimal. Keeps the arbitrary-seed freedom of §1.

* **Cost:** must resolve the weighted-case complementarity (forward optimistic at splits / backward at
  merges) — the harder, deeper fix. Until then, honest scoring still gives best-of-pool.
* **When:** you want provable optimality *and* the freedom to seed anywhere (e.g. anchor from a trusted
  external correspondence, not the min-marginal).

### Fork B — forward table only (Viterbi backtrack)

Drop `B` entirely. Forward-fill `D`, then backtrack from the forward terminal `argmin` via `bpD`,
coupling branches at merges/splits during the walk. Exact **by construction** — there is no second
table to be inconsistent with.

* **Cost:** you lose the arbitrary interior seed (you must seed at the forward terminal and backtrack
  into branches at merge/split nodes — the "harder to explain on a tree" part); needs the branch-aware
  backtrack written carefully.
* **When:** you want optimality with the least conceptual surface and no reciprocity/adjointness
  obligation to maintain.

### Side by side

| | Fork A — both tables, adjoint | Fork B — forward only |
|---|---|---|
| tables used | `D` and `B` | `D` only |
| seed freedom | **any vertex** (arbitrary interior anchor) | forward terminal only |
| optimality | provable **once `g` is constant** (§6b) | provable **by construction** |
| main work | fix §6d complementarity so tables are adjoint | write branch-aware `bpD` backtrack on a tree |
| honest scoring (§3) | yes — as the judge / insurance | yes — as the judge |
| reciprocity/adjointness debt | must maintain | none |

## 6 — What is new vs. what already exists

| piece | status |
|---|---|
| one-seed flood over a connected tree | **exists** — §5 step 2 |
| min-marginal certificate `g(a)` constant | **exists** — §6b (as a diagnostic) |
| forward/backward complementarity under weighting | **exists, characterized** — §6d |
| per-table reachability guard | **exists** — §6c |
| **enumerate anchor values into a pool** (`Ξ_{a₀}` over all `v₀`) | **new** |
| **honest `C(M)` selection, table-free** | **new** — generalizes §5's single arg-min |
| **≤ `|V(B)|` bound** as the enabling fact | **new** |
| Fork B forward-only Viterbi backtrack on a tree | **new** (approach 1; deferred) |

## 7 — Open items

1. **The fork itself** — A or B. Deferred to the reader per the brief.
2. **Confluence** (§2) — prove the §4b tie-break makes `Ξ` seed-independent, or accept the per-seed
   bound and pool across seeds.
3. **Fork A's adjointness fix** — the actual §6d repair (make `B` the exact reverse-sweep of `D` over
   the same directed transition). Needed only for Fork A's *proof*, not for its robustness.
4. **Complex split/merge coverage gaps** — §5 step 4 leaves these unfilled today; confirm honest
   `C(M)` scoring either closes them (by preferring a covered candidate) or that the gap-fill still runs
   post-selection.
5. **Cost of enumeration** — if `g` is constant, one decode suffices and enumeration is insurance;
   if not, budget `O(|V(B)|·|V(A)|)` for the full pool. Both are polynomial.

## 8 — Recommendation

Land the §3 **honest-scoring judge first**, independent of the fork — it is the piece that makes the
result correct-and-defensible today and is a strict superset of the current §5 seed. Then choose the
fork by what you value more: **Fork A** if the arbitrary-anchor freedom matters (external ground-truth
anchors) and you are willing to fix §6d for the optimality proof; **Fork B** if you want optimality
by construction and are willing to give up interior seeding. Demonstrate whichever, before/after, in
`notebooks/tree_dtw_playground` on the α<1/β<1 cases where Part 6d currently bites.

---

## Fork B realized — forward-only anchored extraction (the two-pointer protocol)

**Status: implemented and DEFAULT** — `extract(A, B, α, β)` in `network_matching/tree_dtw.py` (run
after `prepare` + `forward`; no backward pass; no alias). Builds on
§4.1a (forbid-and-rebuild) and replaces both the backward table and the two-table extraction — the
latter remains as `extract_two_table` for the §6b cross-table diagnostics. One table (`D`/`bpD`),
two pointer types:

* **`bpD`** (stored) — walks **upstream**;
* **`R` = transpose of `bpD`** (one pass to build) — walks **downstream**.

### The forbidden rule during exploration — reject and retry

No pointer ever links **into** a forbidden cell (the recurrence skips them), so **UP moves can never
reach one**. A DOWN move, however, can find that a successor's only linking cells are themselves
forbidden (a forbidden cell keeps its own *outgoing* pointers), or that it has none at all. The rule:

> **The flood never commits a forbidden cell. If a successor cannot be placed on a non-forbidden
> cell, the current anchor label's result is INVALID — discard it and continue with the next label
> of the anchor vertex. If every label is invalid, raise the feasibility error (increase
> `match_radius_m`).**

The enumeration itself is the safety net — no repair, no extra machinery.

*Optional hardening (not required):* running the §4.1a coupling at **every** vertex
(`out_degree ≥ 1`; trivial for a chain vertex — single child, no rebuild cascade) makes dead-ends
impossible by construction and shrinks the anchor's label count. Without it, a 136-case sweep saw
~12 % of cases dead-end on *some* label — the reject-and-retry absorbs these; measure the rejection
rate before deciding.

### Anchor selection and enumeration

Anchor vertex `a₀` = the vertex with the **fewest non-forbidden cells** (smallest enumeration; often
1–2 after coupling). For **each** non-forbidden cell `v₀` of `a₀`: run the flood below → one complete
`M(v₀)`; compute `C(M)` **directly from the relation** (advance = full `E`, 1:N cover = `α·E`,
N:1 stall = `β·E` — all classifiable from `M` + the graphs alone); return the minimum-cost `M`.
Generation is pointer-driven, judgment is the honest cost — the tables are never trusted for ranking.

### The flood (per anchor cell) — deterministic

Commit `(a₀, v₀)`; process committed vertices in any order (confluent, below). From committed `c`
with pivot `w`:

1. **UP — no choice.** Walk `c`'s cover chain from `w` to its head (`bpD[c][x] = [(c, x')]`); at the
   head, `bpD[c][head]` names **one cell per predecessor** — commit them all. A merge commits all its
   arms at once: **V2 by construction**.
2. **DOWN — the transpose, one shared connection.** Let `run⁺(c)` = `w` + its reverse-cover closure
   (cells whose cover chains lead back to `w`). Choose **one** non-forbidden `y* ∈ run⁺(c)` and commit
   **every** successor `s` through it: some **non-forbidden** `w_s` with `(c, y*) ∈ bpD[s][w_s]`. All
   children leave from the same `y*`: **V3 by construction** — a shared `y*` exists at a split because
   every non-forbidden cell of `c` is linked by all children (§4.1a invariant). If some successor has
   no non-forbidden `w_s` → **this label is INVALID** (reject-and-retry rule above).
3. **Runs are pulled, not guessed.** `c`'s 1:N run in `M` = the cover chain from `y*` back to `w` —
   determined retroactively by where the children connect. No second-table gap-fill. A sink's run is
   its pivot.
4. **A pinned run end stays pinned.** A vertex committed by a child's UP pin (the child's `bpD` named
   `(c, x_p)`) has its run end **fixed at that pin**: for its remaining successors `y*` *is* the pivot
   — extending the run past the cell one child already connected at would put that child mid-run
   (a V3 break). Only a vertex committed from its parent (or the anchor itself) searches the closure.

**The two in-flood rules** (the only heuristics; both then judged by `C(M)` across the enumeration):

* `w_s` choice: `argmin D[s][w]` over the linking cells, ties by `border`;
* `y*` choice: `argmin over non-forbidden run⁺ cells of Σ_s min D[s][·]`, ties by `border`.

**Confluence.** On a tree each vertex is reached along its unique path from the anchor and committed
first-wins, so `M` depends only on (anchor, rules) — not on exploration order.

### What this removes / what remains

* **Removed**: the backward pass (half the DP), and the whole §6b/§6d cross-table consistency problem —
  one table, nothing to be reciprocal with.
* **Measured** (136-case sweep — fixed scenarios + random out-trees over often-cyclic targets, α/β to
  0.2, vs the two-table extract on the *same* coupled forward table): label enumeration **never
  exhausted** (0 all-invalid cases); **valid `M` 133/136 vs 121/136** two-table; raw cost equal 65 /
  forward-only better 32 / two-table better 39; deterministic on every case; segment mode identical.
  The 3 invalids are all V1-only on **cyclic B**: two are the documented local-predicate sensitivity
  (a 2-cycle makes a legal forward step also readable backward) — there the two-table extract returns
  the **identical** matching, flagged identically; the third is an in-flood **rule-quality** case (the
  `y*` rule chose a coverage run riding a back-arc where the two-table extraction found a clean
  alternative) — covered by the open item below.
* **Honest boundary** (unchanged from §3): the result is the best of the **enumerated** matchings —
  `≤ |cand(a₀)|` of them — not a proven global optimum.
* **Open**: aligning the exploration step with the dictated protocol (the greedy `y*`/`w_s` picks are
  not part of it), sink-side coverage (forward cover only), and label exhaustion on dense-target
  chains (in-domain, heavy 1:N — the one robustness gap the corrected weight domain leaves standing).
  *Historical note:* the "greedy pick rides a back-arc" genuine-failure exhibit existed only at
  `β < 1`, which the later weight-domain correction (`α ∈ (0,1]`, `β ∈ [1,∞)`) excluded; an in-domain
  240-run scan found no genuine failure, and the strict-xfail exhibit test was removed with it.

---

## Appendices — moved into the algorithm spec

Two pieces designed here have been promoted into `docs/tree_dtw_matching.md` as part of the core
algorithm; they are specified there, once:

* **A-vertex ordering (longest-path layering)** — now **§4.0** of the spec (`layer_order(A)` in
  `network_matching/tree_dtw.py`). Guarantees no successor of a vertex precedes any of its siblings,
  on a subdivided source; validated on chain / split / subdivided diamond / full subdivided DAG, with
  the raw `sib=desc` control showing the subdivision is what earns the guarantee.
* **V3 coupling in the forward pass (forbid-and-rebuild)** — now **§4.1a** of the spec: per-cell
  `forbidden` flag, allowed exits = ∩ over siblings, whole-row rebuilds, fixed-point iteration,
  multiple surviving exits legitimate (single exit chosen at traceback).

