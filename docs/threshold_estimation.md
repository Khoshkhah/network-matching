# Threshold estimation for `resolve_routes`

How to pick the quality cutoffs that [`resolve_routes`](graph_dtw_pipeline.md#filter-by-quality-resolve_routes)
uses — from the data, instead of by eye. Implementation:
[`network_matching/thresholds.py`](../network_matching/thresholds.py).

---

## 1. The problem

`match_routes` returns the best route for **every** A-edge; `resolve_routes` then drops the poor
ones by upper/lower bounds on a few quality signals:

| metric (`routes_summary` column) | meaning | direction | `resolve_routes` kwarg |
|----------------------------------|---------|-----------|------------------------|
| `dtw_distance`     | average match distance (m) | lower better | `max_match_dist` |
| `max_dtw_distance` | max match distance (m)     | lower better | `max_max_dist` |
| `bearing_diff`     | route bearing difference (°)| lower better | `max_bearing_diff` |
| `overlap_pct`      | A-edge coverage (%)        | higher better| `min_overlap_pct` |

Hand-picking values like `25 / 45 / 95` is fragile — the right cut depends on the data. On a real
network each metric forms a tight **"good" cluster** (distances/bearing near 0, overlap near 100)
plus a **tail** of wrong/poor matches. `suggest_thresholds` finds the cut **between the cluster and
the tail**, per metric, and returns values that drop straight into `resolve_routes`.

```python
from network_matching import suggest_thresholds
sugg = suggest_thresholds(routes_summary, report=True,
                          plot_path="output/threshold_suggestions.png")
routes_summary, routes_long = m.resolve_routes(routes_summary, routes_long, **sugg["recommended"])
```

`NO_MATCH` rows (NaN / `pd.NA` metrics) are excluded automatically.

---

## 2. The estimators

For each metric the tool runs **every** estimator below side by side (so you can compare, not just
trust one). All are oriented as an upper-tail problem (`overlap_pct` is mirrored — §4).

### Robust outlier fences — "where do the outliers start"
Right when the data is **unimodal** ("all good, no distinct bad cluster").

- **Tukey IQR fence** — `Q3 + k·IQR` (`k_iqr`, default 1.5). Fallback chain when IQR collapses
  (≥50 % identical values, common with many perfect 0/100 matches): MAD → high percentile.
- **MAD fence** — `median + k·1.4826·MAD` (`k_mad`, default 3.0); `1.4826` makes MAD a
  normal-consistent σ. Robust to the heavy tail. Zero-MAD → std → median.
- **Percentile** — a high percentile (`percentile`, default 97.5). Distribution-free, conservative.

### Two-population (stochastic) separators — "the trough between good and bad"
Right when there genuinely **is** a second mode.

- **GMM crossover** — a 2-component 1-D Gaussian mixture fit by **EM** (pure numpy: deterministic
  quantile-split init, variance floor against singularities, log-sum-exp E-step). Components are
  ordered good (small μ) / bad (large μ); the threshold is the **posterior crossover**
  `P(bad|x)=0.5`, else the density valley between the means. Reports a standardized separation
  `sep = |μ_bad−μ_good| / √(½(σ²_good+σ²_bad))` and the bad-component weight `pi_bad`.
- **KDE valley** — a Gaussian KDE (Silverman bandwidth, pure-numpy sum-of-gaussians on a grid); the
  **antimode** (interior trough) between the main mode and a prominent tail mode. Returns `nan`
  when the density is unimodal — that "no valley" signal feeds the recommendation rule.

### IsolationForest — tree-based outlier cut (needs scikit-learn, `[ml]` extra)
Per metric, a 1-D `IsolationForest` (deterministic `random_state=0`); the threshold is the **onset
of the flagged-outlier region above the median**. *Caveat:* in 1-D it flags the low-density upper
tail of the good cluster, so it can cut a little early — IsolationForest's real strength is the
**multivariate** flag (§6), not single-axis thresholds.

### Corroborators (reported/plotted, never drive the recommendation)
- **Otsu** — the 1-D histogram split maximizing between-class variance; always returns a value even
  on unimodal data, so it is shown but not trusted alone.
- **Kneedle** — the knee of the sorted-value-vs-rank curve; useful but tail-noisy.

---

## 3. The recommendation rule

One value per metric, chosen from the candidates:

```
bimodal := (GMM converged and sep >= sep_min and pi_bounds[0] <= pi_bad <= pi_bounds[1])
           or (KDE valley exists)

if n < min_n:            recommended = p95                         # too little data
elif bimodal:            recommended = median(GMM, KDE valley, IsolationForest)   # good/bad cut
else (unimodal):         recommended = min(max(MAD, IQR), p99)     # robust fence, no bad cluster
recommended = clamp(recommended, p50, max)   # never reject the good cluster, never exceed the data
```

Defaults: `sep_min=2.0`, `pi_bounds=(0.01, 0.45)`, `min_n=20` — all keyword args. The `rationale`
string in the output names the trigger that actually fired (e.g. `"bimodal (KDE valley (GMM
sep=1.6<2.0)): good/bad crossover"` vs `"unimodal (...): robust MAD/IQR fence"`).

---

## 4. `overlap_pct` (higher is better)

Mirrored: run the identical upper-tail machinery on `deficit = 100 − overlap`, then convert every
result back with `100 − value` (clipped to `[0, 100]`) so the recommendation reads as a **minimum
overlap** in native %.

---

## 5. Output

`suggest_thresholds(...) -> dict`:

```python
{
  "metrics": {
    "dtw_distance": {
      "n": 3344, "direction": "lower",
      "summary": {"p50":.., "p90":.., "p95":.., "p99":.., "max":.., "mean":.., "std":..},
      "methods": {"iqr_fence":.., "mad_fence":.., "percentile":.., "otsu":..,
                  "kde_valley":.., "gmm_crossover":.., "kneedle":.., "iforest":..},
      "gmm": {"mu_good":.., "mu_bad":.., "sep":.., "pi_bad":.., "converged": True},
      "recommended": 7.418,
      "rationale": "bimodal (...): good/bad crossover",
    },
    "max_dtw_distance": {...}, "bearing_diff": {...}, "overlap_pct": {...},
  },
  "recommended": {                       # maps DIRECTLY to resolve_routes kwargs
    "max_match_dist": 7.418, "max_max_dist": 12.874,
    "max_bearing_diff": 4.734, "min_overlap_pct": 97.0,
  },
  "params": {...}, "n_total": 3948, "n_nomatch": 604, "plot_path": "output/threshold_suggestions.png",
}
```

`report=True` prints a `validate_b_geometry`-style block (per-metric method values + chosen cut +
rationale, then a copy-paste `resolve_routes(...)` line). Any metric with no data → `recommended`
`None`, and that kwarg is simply omitted.

### Diagnostic plot
`plot_path` (needs `matplotlib`, `[viz]`) writes a 2×2 grid — one panel per metric: histogram + KDE
curve + a thin line per candidate method + the **bold** recommended line, with the rejected region
shaded (right for lower-better, left for `overlap_pct`).

---

## 6. Multivariate review — `isolation_forest_flags`

Per-metric cuts treat each axis independently. `isolation_forest_flags(routes_summary)` fits **one**
IsolationForest on **all** quality signals jointly (z-scored so meters / degrees / percent are
comparable; `overlap_pct` flipped to a deficit so "high = worse" everywhere) and flags matches
anomalous **in combination** — a route can look acceptable on each axis yet sit in a sparse region
of the joint space.

```python
from network_matching import isolation_forest_flags
flagged = isolation_forest_flags(routes_summary, report=True)   # adds if_outlier (bool), if_score (lower=worse)
```

It is a **complement** to the thresholds, not a replacement. On the Sundbyberg data:

```
                  iforest_outlier=False  True
threshold_rejected
False                            1919       0
True                              575     850
```

IsolationForest flags 850 jointly-anomalous matches — **all** also rejected by the per-metric
thresholds — while the thresholds reject a broader 1425 (575 bad on a single axis but not jointly
extreme). So: **thresholds = the broad, interpretable, directly-applicable filter**;
**IsolationForest = a conservative joint-review tool** that surfaces correlated weirdness to inspect.

---

## 7. Run it

- **Library:** `suggest_thresholds(routes_summary)` / `isolation_forest_flags(routes_summary)`.
- **CLI:** `python scripts/suggest_thresholds.py` — rebuilds matches (or `--summary-csv` a saved
  table), prints the report, writes `output/threshold_suggestions.png`, and prints the
  `resolve_routes(...)` kwargs. Tuning flags: `--k-iqr --k-mad --percentile --sep-min`.
- **Notebook:** [`notebooks/threshold_estimation.ipynb`](../notebooks/threshold_estimation.ipynb) —
  the per-metric comparison, the apply-vs-hand-picked table, and the IsolationForest cross-tab.

---

## 8. Dependencies & caveats

- The numpy estimators (IQR, MAD, percentile, Otsu, GMM-EM, KDE, Kneedle) need **no extra deps**.
- **IsolationForest** needs `scikit-learn` (`[ml]` extra) — guarded import; the `iforest` method and
  `isolation_forest_flags` are skipped/raise clearly if it is absent.
- The **plot** needs `matplotlib` (`[viz]`) — skipped with a warning if absent.
- The recommended values are **data-driven starting points**, not gospel: widen if you over-reject,
  tighten if wrong matches survive, and re-run `suggest_thresholds` on the resolved set to confirm
  the tail is gone. All gates (`sep_min`, `k_iqr`, `k_mad`, `percentile`, `pi_bounds`, `min_n`) are
  keyword args.
- Determinism: GMM uses a deterministic quantile-split init; IsolationForest uses
  `random_state=0`. Same input → same thresholds.
