# Segment-to-Segment Emission for Graph-DTW

A design proposal to generalize the graph-DTW **local cost** from *point-to-point* (a single
point-to-vertex distance) to *segment-to-segment* — using an **endpoint-average distance** that is
already direction-aware, with an **optional** bearing term that is **off by default**. Builds on the
algorithm in [`graph_dtw_matching.md`](graph_dtw_matching.md); the cost lives in `emit(i)` inside
[`network_matching/graph_dtw.py`](../network_matching/graph_dtw.py).

Status: **proposed** (design). Default behaviour is a strict improvement over point-to-point with no
new hyperparameter; the optional bearing term defaults to `λ = 0`, so it is fully backward-compatible.

---

## 1. Motivation

The current local cost is pure point-to-point drift:

```
emit(i)[v] = dist(aᵢ, v)          # one A-point to one B-vertex
```

Distance between *points* has no notion of **direction**, so it can't separate cases that are close
in space but opposite (or divergent) in heading: divided carriageways, reverse twins, and forks.

## 2. The idea — compare directed segments, not points

Match the **directed micro-segment** of A (the piece `a_{i-1} → a_i` between consecutive pool
points) against the **directed micro-segment** of B (`u → v` along a forward arc). The natural,
cheap distance between two directed segments is the **average of their endpoint distances**:

```
d(A_seg, B_seg) = ½( ‖a_{i-1} − u‖ + ‖aᵢ − v‖ )        # start↔start, end↔end (directional)
```

This single metric — in metres, no weight to tune — already encodes direction:

- **Reversal** (dual carriageway / reverse twin / opposing traffic): the endpoints effectively swap,
  so the distance jumps to ≈ the segment length. Killed hard — the case that matters most.
- **Heading error** `θ` over a segment of length `L`: the endpoints diverge by ≈ `L · sin(θ/2)`.
  At `step_meters = 10`, a 30° fork ⇒ ~2.6 m — plenty to beat a ~1–2 m correct match.
- **Accumulation:** consecutive micro-segments *share* endpoints, so a heading disagreement piles up
  as lateral drift along the chain — the DTW total captures direction at the *edge* scale even when a
  single short segment's contribution is small.

## 3. How it fits the DP — a transition cost; point-to-point is the sub-case

The endpoint distance needs the segment's *start*, which depends on the predecessor — so the cost
attaches to a **transition** rather than a node. The recurrence shape is unchanged (monotonic in A,
forward-arc walk in B, the three moves, per-row Dijkstra); only *where the cost is added* moves from
node to edge. On the moves where **one** side advances, the metric **degenerates automatically**:

| move | A advances? | B advances? | cost becomes |
|------|:-----------:|:-----------:|--------------|
| **Diagonal** | yes | yes | full segment-to-segment `½(‖a_{i-1}−u‖ + ‖aᵢ−v‖)` |
| **Vertical** | yes | no  | `½(‖a_{i-1}−v‖ + ‖aᵢ−v‖)` — A-segment vs the B-point (≈ point-to-point) |
| **Horizontal** | no | yes | `½(‖aᵢ−u‖ + ‖aᵢ−v‖)` — B-segment vs the A-point (≈ point-to-point) |

So **point-to-point is literally the degenerate case** of the segment metric whenever a side holds
still — one cost function spans both, no separate "mode." Projected points (the continuous-DTW
enrichment) keep working: each is an endpoint like any other.

## 4. Optional refinement — a length-independent bearing term (default off)

The endpoint distance's *own* directional signal scales with `L` (`≈ L·sin(θ/2)`), so on **very short**
segments it weakens. If needed, add a separate, length-**independent** heading penalty:

```
cost = d(A_seg, B_seg)  +  λ · d_circ(bearing(A_seg), bearing(B_seg))     # λ = 0 by default
```

- `d_circ` is the circular heading difference in `[0°, 180°]`.
- **Default `λ = 0`** — identical to the endpoint-only metric, so results don't change until opted in.
- Turn it on **only if** validation shows residual wrong-direction / ambiguous-parallel mismatches,
  and set `λ` from a measured lateral-equivalent rather than by feel.

This keeps the design simple by default and reversible: re-adding direction sensitivity is a one-line,
backward-compatible change, not a redesign.

## 5. Micro ⇒ macro consistency

Grouping consecutive matched micro-segments by their B-edge (the existing `groups` / `vert_edge`
step) collapses the directed micro-matches into the directed **edge chain** — so segment-to-segment
matching at the micro level *is* the edge-to-edge crosswalk at the macro level.

## 6. Scope & limits

- **Endpoints only ⇒ ignores mid-segment shape.** Fine for **short** micro-segments (~`step_meters`,
  effectively straight); the DTW aggregation recovers curvature at the edge level. **Do not** apply the
  bare endpoint metric to long edges — resample first.
- **Requires corresponding endpoints**, which the resampling + projection pools already provide.
- **Degenerate / very short** segments give noisy endpoints — smooth or skip, as with any tangent.

## 7. Config & compatibility

| preset | cost | note |
|--------|------|------|
| `point` | `dist(aᵢ, v)` | today's behaviour |
| `segment` *(opt-in; see §9)* | `½(Δstart + Δend)` | direction-aware, no weight to tune |
| `segment+bearing` | `½(Δstart + Δend) + λ·Δbearing` | opt-in; `λ = 0` unless enabled |

## 8. Validation plan

`bearing_diff` and the match metrics are already logged, so this is a clean before/after:

1. Ship `segment` (endpoint-only) and compare to `point`: overall match rate, median drift, and
   especially **wrong-direction / dual-carriageway mismatches eliminated**.
2. Inspect any residual direction errors; **only then** enable `segment+bearing` with a measured `λ`.

---

## Summary

Replace the point-to-point drift with an **endpoint-average segment distance**: one metric in metres,
no weight, that already handles the direction cases that matter (reversal outright; heading errors via
endpoint divergence and chain accumulation). It slots into the DP as a **transition cost** with
point-to-point as its degenerate sub-case. A separate `λ·diff_bearing` term stays available for very
short / ambiguous-parallel cases but is **off by default** — added only if the data asks for it.

---

## 9. Validation findings (2026-07-06, Sundbyberg OSM↔NVDB)

Implemented as `emission="point"|"segment"` + `bearing_weight` (λ) in `graph_dtw.py`; `point` is
byte-identical to the previous behaviour (all tests pass).

**segment vs point — a wash.** Same match count (3466/3948) and median drift (1.83 m). 641 routes
(16%) differ, but the differences are knife-edge ties (drift gaps ≲ 0.05 m, usually one extra
end-edge) and **vanish under any perturbation**: shifting the A-edge laterally (4–40 m sweeps on
edges 1377, 947, 417, 1671, 2401) produces identical routes and drift for both emissions at every
step. `point` therefore stays the default; `segment` is a safe opt-in with no demonstrated upside
on this data.

**Shift robustness (both emissions).** Drift grows ≈ shift (ideal lateral response); the route
holds against *fragment* parallels (417: no jump even sitting on a 12-m-away side street — the
504 m shape context anchors it) and against *opposite-direction* parallels (1671: at 12 m shift the
edge lies exactly on the one-way opposite carriageway `605`, which the forward-arcs-only digraph
structurally refuses — the classic wrong-carriageway error is impossible). Tolerance is set by the
reject threshold and candidate radius, not by matcher instability.

**bearing (λ) — design flaw found.** The heading penalty is charged **only on diagonal (advance)
moves**, so vertical stalls are penalty-free: on edges whose corners transiently disagree with all
candidates (~45–90°, i.e. ~22–45 m at λ=0.5), the DP escapes by stalling and the route **collapses
onto the single direction-compatible edge** (2401 shifted left → `[597]`, 20–27% coverage; shifted
right → `[577]`, 20% — mirror images of the same failure). λ=2 additionally prunes ~20% of all
matches. Until the penalty also constrains stalls/horizontal moves (or is capped at corners),
`bearing_weight` should be treated as **experimental**; its collapses are at least easy to reject
via `overlap_pct`. Note the direction *calculation* is correct (verified against travel bearings of
the directed twins); the flaw is where the penalty applies in the recurrence.

---

## 10. Correction — true segment-to-segment states (implemented 2026-07-06)

> **Superseded by §11 (2026-07-06).** The (A-segment, B-arc) state DP described here is retained,
> but its endpoint-average emission `½(‖aᵢ−u‖ + ‖aᵢ₊₁−v‖)` was replaced by the middle-to-middle
> distance, and `emission="segment"` now denotes that final form. The state/move machinery below
> is unchanged; only the per-state cost formula differs. There are now exactly **two** emission
> modes: `"point"` and `"segment"`.

§9's bearing flaw is not a tuning problem; it is structural, and the diagnosis is simple:

> What was shipped as `emission="segment"` is a segment-shaped **cost** on a **point-state** DP.
> The states are still (A-point i, B-vertex v) and the warping path still pairs *points*. On a
> vertical (stall) move several A points map to **one B point** — there is no B-segment there, so
> `Δbearing` is undefined and was silently charged as zero. The DP therefore escapes the heading
> penalty by stalling, which is exactly the route-collapse observed in §9.

True segment-to-segment matching aligns **two sequences of segments**, never points:

**States.** A is the segment sequence `sᵢ = aᵢ → aᵢ₊₁` (i = 0…N−2); B is the set of directed
**arcs** `e = u → v` of the local digraph. A state is a pair `(i, e)`: *A-segment i rides arc e*.

**Emission — paid by every state, no exceptions:**

    E(i, e) = ½( ‖aᵢ − u‖ + ‖aᵢ₊₁ − v‖ )  +  λ · Δbearing(sᵢ, e)

Both terms are always defined, because both sides of every pairing are segments. There is no move
that evades the heading term; overhang now pairs the surplus A-segments with the terminal arcs and
pays their (honest) mismatch instead of hiding at a vertex.

**Moves** (replacing vertical/diagonal/horizontal):

| move | meaning |
|------|---------|
| `(i−1, e) → (i, e)` | next A-segment stays on the same arc — *N A-segments : 1 arc* |
| `(i, e′) → (i, e)`, `e′→e` adjacent | same A-segment spans consecutive arcs — *1 A-segment : N arcs* (within-row Dijkstra; **each arc pays its own emission**) |
| `(i−1, e′) → (i, e)`, `e′→e` adjacent | both advance |

Free entry on row 0, termination after the last A-segment, backtrack → arc path → route (an arc
carries its owning edge, as vertices do today). Complexity is unchanged in order: N × |arcs|
states with the same Dijkstra-per-row structure; |arcs| ≈ |vertices| in the pooled graph.

**Notes.**
- Arcs are short (the projection pool densifies B to roughly the A sampling step), so the
  endpoint-average against a whole arc stays honest even in N:1 pairings.
- `point` mode is untouched — it remains the vertex-state DP and the default.
- The §9 experiments (collapse on 2401 left/right shifts) become the regression tests: with
  per-state bearing the collapse must not occur, because stalling on one arc now costs that arc's
  full distance + bearing for every additional A-segment.

**Implementation outcome.** Implemented as `_segment_dp_pairs()` in `graph_dtw.py`; the arc-state
path is emitted as vertex-level pairs, so all grouping/metrics/route post-processing is shared with
point mode, which is byte-untouched (13/13 tests).

One rule was needed beyond the design above: **stitch arcs are not segments.** The first cut merely
exempted sub-half-metre junction-snap connectors from the bearing term — and the DP immediately
parked A-segments on them (the §9 collapse reborn through the exemption). The structural rule: an
A-segment may never pair with a stitch; stitches are traversable only *within* a row (B passing
through instantly), and carry no (A-segment : arc) state.

Results (Sundbyberg, `snap=0.5, step=10`):

| mode | matched | median drift | median coverage | runtime |
|------|---------|--------------|-----------------|---------|
| point | 3466/3948 | 1.83 m | 99% | 18 s |
| segment (λ=0) | **3611**/3948 | 2.26 m | 99% | 21 s |
| segment + λ=0.5 | **3611**/3948 | 2.73 m | 98% | 21 s |

Segment mode always traverses ≥ 1 real arc, so the point-mode "zero-traversal touch" NO_MATCH
cannot occur — hence +145 matches (to be quality-validated before changing any default).
Regression (2401 shifted ±20 m, λ=0.5): **no collapse** — routes are directional subsets or the
full corridor; on the unshifted edge, bearing now *improves* the match (full 6-edge corridor at
100% coverage vs point's 89%). `point` remains the default.

## 11. Final form — `emission="segment"` is middle-to-middle (2026-07-06)

The library ships **two** emission modes: `"point"` (the default) and `"segment"`. This section
defines the final `"segment"`, developed while debugging the segment correspondence in the
playground notebook; it keeps §10's (A-segment, B-arc) states and moves and changes only the
per-state cost. Three refinements over §10's endpoint-average:

1. **The local cost is ONE distance between the two segment MIDDLES**,
   `E(i, e) = |mid(aᵢ, aᵢ₊₁) − mid(u, v)| + λ·Δbearing(sᵢ, e)` — not the endpoint average.
   Trade-off: a middle-to-middle distance is **blind to a segment rotating about its own middle**,
   so pair it with `bearing_weight` (λ ≈ 1–5) when heading matters — this is the recommended
   working config.
2. **Stitch arcs are free.** A zero-length junction connector is connectivity, not a segment;
   crossing it carries no cost. (Stitches still cannot host a state, so no route collapse — the
   §10 structural rule stands.)
3. **The reported distances ARE the state costs.** `average`/`max`/`min` (overall and per route
   edge) are statistics of the middle-to-middle distances over the matched states — what the
   playground's segment view draws is literally what is scored.

Supporting pool changes (via `min_pool_gap_m`, default `step_meters/2` for `"segment"`;
`"point"` unchanged at `0`): gap-fill points are spread **evenly** over each gap (spacing in
`(step/2, step]`, no leftover slivers), and added (non-node) pool points closer than
`min_pool_gap_m` to a kept neighbour are dropped — so every DP state pairs two genuine segments,
never a centimeter sliver whose position/heading is noise.

`emission="midpoint"` is accepted as a **deprecated alias** for `"segment"` (it named this mode
during development). `"point"` remains the shipped default and is byte-for-byte unchanged.
