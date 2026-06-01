# Symmetric (Two-Way) Map Matching & Split-Aware Reconciliation

> **Status: design specification (proposed extension).**
> The library today ships a **directed** pipeline — `match()` (Source A → Destination B) and
> the `resolve()` decision layer. This document specifies a **symmetric** reconciliation built
> *on top of* those primitives: run the directed matcher both ways, then combine the two results
> into a single undirected matching that correctly preserves **split roads** (1:N) and
> **merges** (N:1). It is not yet implemented in `match()`; the proposed API is in §7.

---

## 1. Motivation

The directed matcher answers a one-sided question: *"for each segment `a` in A, which segment
`b` in B does it best align with?"* That is exactly right for asymmetric problems (e.g. assigning
sensor points to roads), but it has two weaknesses for **network-to-network conflation**:

1. **It is forced and one-sided.** Every `a` must pick *some* `b` within `max_distance`, even when
   `a` has no true counterpart — so a nearby-but-unrelated road becomes a false match.
2. **A single direction cannot represent a split.** If one long road `a` in A corresponds to three
   short pieces `b1, b2, b3` in B, then `a → B` (best per source) keeps only **one** of them. The
   split is lost.

Running the matcher in **both** directions and reconciling the two results fixes both problems:
the disagreement between directions exposes the false matches, and the B→A direction recovers the
split pieces (each small `bi` votes for the big `a`).

The naive way to combine two directions — keep only pairs both directions agree on (mutual /
reciprocal best) — gives very clean **1:1** matches but **drops every split** (the two directions
can never agree on a single pair for a split). This document specifies a combination that keeps
splits, using the two directional **overlap** values as the discriminating signal.

---

## 2. Notation & Prerequisites

For an ordered pair `(a, b)` with `a ∈ A`, `b ∈ B`, the directed Tier-2 evaluation produces:

| Symbol | Meaning | Symmetric? |
|--------|---------|------------|
| `dtw(a,b)` | average geometric drift in meters along the DTW warping path | ~yes |
| `bearing_diff(a,b)` | direction difference over the **matched span** (first→last warping-path point on each side), `0–180°` | yes |
| `ov_ab` | **integer % of `a`'s length** covered by the aligned section with `b` | **no** |
| `ov_ba` | **integer % of `b`'s length** covered by the aligned section with `a` | **no** |

`dtw` and `bearing_diff` are (to a good approximation) direction-independent. **Overlap is the
asymmetric quantity** — it is always measured *relative to the source* — and it is the heart of
this algorithm. See [`dtw_matching.md`](dtw_matching.md) and [`algorithm.md`](algorithm.md) for how
each metric is computed.

**Key fact — candidate generation is symmetric.** Tier 1 uses `ST_DWithin(a, b, max_distance)`,
which is symmetric: `a` is within `d` of `b` ⟺ `b` is within `d` of `a`. Therefore the *set of
candidate pairs* is identical in both directions, and **every candidate pair `{a,b}` is evaluated
in both runs** — so for every pair we can obtain *both* `ov_ab` and `ov_ba`.

---

## 3. Pipeline Overview

```
        ┌───────────────┐        ┌───────────────┐
        │  match A → B  │        │  match B → A  │     (two directed runs;
        │  (Tier 1+2)   │        │  (Tier 1+2)   │      identical candidate set)
        └───────┬───────┘        └───────┬───────┘
                │ E_ab(a,b,dtw,bear,ov_ab)│ E_ba(b,a,dtw,bear,ov_ba)
                └────────────┬────────────┘
                             v
                 ┌───────────────────────┐
                 │ STEP A: JOIN on {a,b}  │  -> U(a, b, dtw, bearing_diff, ov_ab, ov_ba)
                 └───────────┬───────────┘
                             v
                 ┌───────────────────────┐
                 │ STEP B: FEASIBILITY    │  drop if dtw too big OR bearing too off
                 │   (the "score" gate)   │  -> every survivor is a plausible alignment
                 └───────────┬───────────┘
                             v
                 ┌───────────────────────┐
                 │ STEP C: OVERLAP RULE   │  use (ov_ab, ov_ba) to keep/drop & classify
                 │  (matched-length floor)│  -> keeps 1:1 AND splits, drops incidental
                 └───────────┬───────────┘
                             v
                 ┌───────────────────────┐
                 │ STEP D: CARDINALITY    │  label 1:1 / 1:N / N:1 / N:M per component
                 └───────────────────────┘
                             v
                    Symmetric match table
```

The decision is intentionally split into **two gates**:

- **Step B (feasibility / "score")** answers *"is this alignment geometrically any good?"* using
  `dtw` and `bearing_diff`. After this gate, every surviving pair is a *feasible* alignment.
- **Step C (overlap)** answers *"do these two roads actually correspond, and how?"* using the two
  overlaps. This is what separates a real correspondence (including a split) from two roads that
  merely **cross** or **briefly run parallel**.

This ordering matters: `dtw` alone cannot reject a crossing (see §5.2), but overlap can.

---

## 4. Step A — Join the two directions

Run the directed evaluation both ways and join on the unordered pair. In the B→A run the columns
are swapped (`id_a` holds the `b` id, `id_b` holds the `a` id), so the join key is crossed:

```
U = E_ab  JOIN  E_ba  ON  E_ab.id_a = E_ba.id_b  AND  E_ab.id_b = E_ba.id_a
```

producing one undirected row per candidate pair:

| a | b | dtw | bearing_diff | ov_ab | ov_ba |
|---|---|-----|--------------|-------|-------|

Define the combined feasibility score with the **more favorable** direction (a true match looks
good from at least one side):

$$\text{dtw} = \min\big(\text{dtw}(a,b),\ \text{dtw}(b,a)\big), \qquad
  \text{bearing\_diff} = \text{bearing\_diff}(a,b)$$

> If you used `best_per_source` first and joined only the selected rows, a one-directional pair
> would be missing one of the two overlaps. **Join the full Tier-2 tables instead** — because
> candidate generation is symmetric, both overlaps exist for every pair.

---

## 5. Step B — Feasibility gate (the "score" filter)

Drop any pair that is not a plausible alignment:

$$\text{keep}_\text{feasible} \iff \big(\text{dtw} \le \text{max\_dtw}\big)\ \wedge\ \big(\text{bearing\_diff} \le \text{max\_angle}\big)$$

- `max_dtw` — maximum average drift in meters. Defaults to `max_distance` (the candidate
  radius) so it adds no hidden, tighter distance filter; lower it (e.g. `10–15 m`) only to
  deliberately demand tighter alignment. If two networks are offset by ~20 m on some streets,
  a `max_dtw` below that offset will wrongly reject otherwise-perfect matches.
- `max_angle` — maximum travel-direction difference (e.g. `45°`).

After this gate, **every surviving pair is feasible**; what remains is to decide which feasible
pairs are *real correspondences* and of what kind.

### 5.2 Why a `dtw`-only filter is not enough

Two roads that **cross at an intersection** are geometrically very close *right at the crossing*,
so on that tiny shared stretch their `dtw` is small — they pass a `dtw`-only filter. But each road
covers only a few percent of the other, so both overlaps are tiny. Only the overlap test (Step C)
can reject them. This is the whole reason overlap is a *separate, second* gate.

---

## 6. Step C — Matched-length floor

This is the core of the algorithm. **Do NOT drop on the overlap *percentage*.** A raw `%` threshold
conflates *"are these the same road?"* (a quality question — already answered by `dtw` + `bearing`)
with *"how much do they coincide?"*, and it punishes legitimate matches whenever the two networks
**segment a road differently**: a 23 m OSM roundabout arc that shares a real 7 m stretch with a 34 m
Sweden segment has `containment` only ~32 % — a perfect partial match that a `%` gate wrongly kills.

Instead, derive an **absolute length** from the overlap and gate on that:

$$\text{matched\_len\_m} = \max\big(\tfrac{\text{ov\_ab}}{100}\cdot\text{len}_a,\ \tfrac{\text{ov\_ba}}{100}\cdot\text{len}_b\big)
  \quad\text{(meters of shared co-linear geometry)}$$

**Keep rule:**

$$\boxed{\ \text{keep} \iff \text{matched\_len\_m} \ge \texttt{min\_overlap\_m}\ }$$

- A genuine partial/split match shares **meters** of co-linear geometry (the roundabout arc shares ~7 m) → **kept**.
- A trivial crossing / point-touch shares ~0–2 m → **dropped**.

This keeps split and partial matches that a percentage gate would discard, while still removing
incidental touches — and it never penalises a pair just because one segment happens to be long.

### 6.1 Overlap-% is for *classification*, not dropping

The `(ov_ab, ov_ba)` pair is still a useful **2-D signal of the relationship type** — but only to
*label* a kept edge, never to drop it:

| `ov_ab` (a covered) | `ov_ba` (b covered) | Label |
|:---:|:---:|---|
| **high** | **high** | **1:1** (same road) |
| **low** | **high** | **split** (b is a piece of a) |
| **high** | **low** | **merge** (a is a piece of b) |

With `symmetry = min(ov_ab, ov_ba)`: `symmetry ≥ sym_overlap` → `1:1`, else `split`/`merge` by which
side is the more contained. (`containment = max(ov_ab, ov_ba)` and `symmetry` remain in the output as
metadata.)

---

## 7. Step D — Cardinality classification

`symmetry = min(ov_ab, ov_ba)` classifies each kept edge for free:

- `symmetry ≥ T`  → **1:1** (both well covered).
- `symmetry < T` and `ov_ba ≥ T`  → **1:N split** (the B side is the contained piece).
- `symmetry < T` and `ov_ab ≥ T`  → **N:1 merge** (the A side is the contained piece).

For a network-wide label, treat the kept edges as an undirected bipartite graph and inspect each
connected component by the degrees `deg(a)`, `deg(b)`:

| Component shape | Label |
|---|---|
| one `a`, one `b` | `1:1` |
| one `a`, many `b` | `1:N_SPLIT` |
| many `a`, one `b` | `N:1_MERGE` |
| many `a`, many `b` | `N:M_COMPLEX` |

---

## 8. Worked Example — a split

A has one long road `a`. B splits it into `b1, b2, b3` (consecutive thirds), all lying ~3 m beside `a`.

**Directed runs (per-source best is shown only to illustrate the asymmetry):**

- `A → B`: `a` aligns with all three, but its single *best* is the middle piece `b2`.
- `B → A`: each of `b1, b2, b3` aligns with `a` (their only neighbour).

**Step A — join (full Tier-2, all candidate pairs):**

| a | b | dtw | bearing | ov_ab | ov_ba |
|---|----|-----|---------|-------|-------|
| a | b1 | 3.0 | 4° | 33 | 98 |
| a | b2 | 2.5 | 3° | 34 | 99 |
| a | b3 | 3.1 | 5° | 33 | 97 |

**Step B — feasibility** (`max_dtw=12`, `max_angle=45`): all three pass.

**Step C — matched-length** (`min_overlap_m = 5`): each `bi` is ~fully covered, so the shared length
is the full piece length (tens of meters) → all `≥ 5 m` → **all kept**. `symmetry = min(ov_ab, ov_ba)`
is `33, 34, 33` → all `< sym_overlap` → each labelled **split**.

**Step D — cardinality:** `a` has degree 3, each `bi` degree 1 → component is **`1:N_SPLIT`**.

Contrast with the broken final step ("each `a` keeps its single best `b`"): that keeps only `a–b2`
and discards `b1, b3` — the split is destroyed. The per-edge matched-length rule is what avoids this.

---

## 9. Proposed API

```python
def reconcile_symmetric(eval_ab, eval_ba,
                        max_dtw=25.0, max_angle=45.0,
                        min_overlap_m=5.0, sym_overlap=70):
    # NB: the high-level match_symmetric() defaults max_dtw to max_distance (the
    # candidate radius), so the feasibility gate adds no hidden, tighter distance filter.
    """
    Combine the two directed Tier-2 evaluation tables (A->B and B->A) into a single
    SYMMETRIC, split-aware match table.

    Parameters
    ----------
    eval_ab, eval_ba : DataFrames from compute_dtw_metrics() in each direction
        (eval_ba has A and B roles swapped; each carries overlap_pct AND matched_len).
    max_dtw, max_angle : feasibility gate (Step B).
    min_overlap_m      : min shared length in METERS to keep an edge (Step C). Derived
                         from overlap, but absolute — NOT a raw overlap-% threshold.
    sym_overlap        : min symmetry = min(ov_ab, ov_ba) overlap-% to LABEL an edge 1:1
                         (Step D; classification only, never drops a match).

    Returns
    -------
    DataFrame: [a_id, b_id, dtw, bearing_diff, ov_ab, ov_ba, matched_len_m,
                containment, symmetry, relation, cardinality]
        relation    in {1:1, split, merge}
        cardinality in {1:1, 1:N_SPLIT, N:1_MERGE, N:M_COMPLEX}  (per component)
    """
```

Sketch of the body, in terms of existing primitives:

```python
# 1. Two directed evaluations (symmetric candidate set => both overlaps exist)
cand          = matcher.generate_candidate_pairs()
eval_ab       = matcher.compute_dtw_metrics(cand)                 # id_a=a, id_b=b, overlap_pct=ov_ab
eval_ba       = matcher.compute_dtw_metrics(swap_roles(cand))     # id_a=b, id_b=a, overlap_pct=ov_ba

# 2. Step A: join on the unordered pair
U = eval_ab.merge(eval_ba, left_on=['id_a','id_b'], right_on=['id_b','id_a'],
                  suffixes=('_ab','_ba'))
U['dtw']      = U[['dtw_distance_ab','dtw_distance_ba']].min(axis=1)
U['ov_ab'], U['ov_ba'] = U['overlap_pct_ab'], U['overlap_pct_ba']

# 3. Step B: feasibility gate
U = U[(U['dtw'] <= max_dtw) & (U['bearing_diff_ab'] <= max_angle)]

# 4. Step C: matched-length floor (overlap -> absolute meters, NOT a % threshold)
U['containment']   = U[['ov_ab','ov_ba']].max(axis=1)     # kept as metadata / label input
U['symmetry']      = U[['ov_ab','ov_ba']].min(axis=1)
U['matched_len_m'] = U[['matched_len_ab','matched_len_ba']].max(axis=1)
U = U[U['matched_len_m'] >= min_overlap_m]

# 5. Step D: relation + per-component cardinality
U['relation'] = np.where(U['symmetry'] >= sym_overlap, '1:1',
                  np.where(U['ov_ba'] >= U['ov_ab'], 'split', 'merge'))
U['cardinality'] = label_components(U)     # bipartite connected-component degrees
```

---

## 10. Caveats & Tuning

- **`min_overlap_m` is the knob for short features.** Lowering it keeps shorter shared stretches
  (short roundabout arcs, stubs); raising it demands more co-linear geometry before accepting a match.
  Because it's an absolute length, it does **not** penalise a pair just because one segment is long —
  which is exactly the failure mode of a raw overlap-% gate.
- **`N:M_COMPLEX` tangles.** Per-edge keeping can leave clusters where several `a`s and several `b`s
  all interconnect. That is often the honest answer (the data really is many-to-many there). If you
  need them resolved, add a per-component rule (e.g. within a component drop edges dominated by a
  much stronger competing edge, or run an assignment on the component).
- **Units:** `dtw`/`max_dtw` and `matched_len_m`/`min_overlap_m` in meters, `bearing`/`max_angle`
  in degrees `0–180`. `ov_ab`/`ov_ba` are integer percent `0–100` used only for the relation label.
- **Reuses, does not replace, the directed core.** Everything above is built from
  `generate_candidate_pairs()` + `compute_dtw_metrics()`; the directed `match()`/`resolve()` API is
  unchanged.

---

## 11. Relationship to `resolve()` strategies

| Goal | Use |
|------|-----|
| Asymmetric assignment (each A → one B; B reusable) | `resolve(..., "best_per_source")` |
| Asymmetric assignment (each B → one A) | `resolve(..., "best_per_dest")` |
| Strict global unique 1:1 | `resolve(..., "one_to_one")` |
| **Symmetric conflation that preserves splits/merges** | **`reconcile_symmetric()` (this document)** |

`resolve()` decides cardinality *within one direction*. `reconcile_symmetric()` decides it
*across both directions*, which is what makes splits and merges first-class.
