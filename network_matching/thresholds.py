"""
Data-driven threshold suggestion for :meth:`DuckDBMapMatcher.resolve_routes`.

``resolve_routes`` filters graph-DTW route matches by upper/lower bounds on a few quality
signals — average match distance, max match distance, bearing difference, and A-coverage. Picking
those bounds by eye is fragile: the right cut depends on the data, which on a real network forms a
tight **"good" cluster** (distances/bearing near 0, overlap near 100) plus a **tail** of wrong /
poor matches. :func:`suggest_thresholds` estimates the cut that separates the good cluster from the
tail, per metric, and returns values that drop straight into ``resolve_routes``.

It runs several estimators side by side and picks one:

- **Robust outlier fences** — Tukey IQR fence, MAD fence, a high percentile. These answer "where do
  the outliers start" and are the right choice when the data is unimodal ("all good, no bad
  cluster").
- **Two-population (stochastic) separators** — a 2-component Gaussian mixture (EM) and a Gaussian
  KDE valley (antimode). These find the trough *between* a good and a bad population and are the
  right choice when there genuinely is a second mode.
- **IsolationForest** (``[ml]`` extra, scikit-learn) — a tree-based outlier detector; per metric it
  contributes the onset of the flagged-outlier region. Its real strength is *multivariate*:
  :func:`isolation_forest_flags` fits one forest on all quality signals jointly and flags matches
  that are anomalous *in combination* (OK on each axis alone, weird together).

Otsu and Kneedle are reported as corroborators but never drive the recommendation.

The numpy estimators need no extra deps; IsolationForest needs ``scikit-learn`` (``[ml]``) and the
diagnostic plot needs ``matplotlib`` (``[viz]``) — both are guarded and skipped if unavailable.

Mirrors the report-style of :func:`network_matching.bgraph_prep.validate_b_geometry`:
``suggest_thresholds(routes_summary, report=True) -> dict``.
"""

import logging
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:                                                  # optional [ml] extra
    from sklearn.ensemble import IsolationForest
except Exception:                                     # pragma: no cover - depends on optional dep
    IsolationForest = None

logger = logging.getLogger("network_matching.thresholds")

# metric -> "lower" (lower is better, upper-tail outliers) or "higher" (higher is better)
_DIRECTION = {
    "dtw_distance": "lower",
    "bearing_diff": "lower",
    "overlap_pct": "higher",
}
# metric -> the resolve_routes keyword its recommended value maps to
_RESOLVE_KW = {
    "dtw_distance": "max_match_dist",
    "bearing_diff": "max_bearing_diff",
    "overlap_pct": "min_overlap_pct",
}
_DEFAULT_METRICS = ("dtw_distance", "bearing_diff", "overlap_pct")


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def _clean(series: pd.Series) -> np.ndarray:
    """Coerce a routes_summary column to a finite float array (drops NaN / pd.NA / inf)."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return x[np.isfinite(x)]


def _cap(x: np.ndarray, q: float = 99.5) -> np.ndarray:
    """Clip to the q-th percentile so a heavy tail doesn't starve histogram/KDE resolution near
    the good cluster (the far outliers fold into the top of the range)."""
    if x.size == 0:
        return x
    hi = float(np.percentile(x, q))
    lo = float(x.min())
    if hi <= lo:
        return x
    return np.clip(x, lo, hi)


# --------------------------------------------------------------------------------------
# Upper-tail estimators (pure numpy). Each takes a 1-D float array, returns a float (or nan).
# --------------------------------------------------------------------------------------
def _iqr_fence(x: np.ndarray, k: float = 1.5) -> float:
    """Tukey upper fence Q3 + k*IQR, with a fallback chain when IQR is 0 (many identical values)."""
    if x.size == 0:
        return float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        return float(q3 + k * iqr)
    # >=50% identical -> IQR collapses; fall back to a MAD fence, then a high percentile.
    mad = _mad_fence(x, k=3.0)
    if math.isfinite(mad) and mad > q3:
        return mad
    return float(np.percentile(x, 97.5))


def _mad_fence(x: np.ndarray, k: float = 3.0) -> float:
    """Robust fence median + k*1.4826*MAD (1.4826 -> normal-consistent sigma)."""
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        return med + k * 1.4826 * mad
    std = float(x.std(ddof=0))
    if std > 0:
        return med + k * std
    return med  # constant column: degenerate but finite


def _percentile(x: np.ndarray, p: float = 97.5) -> float:
    return float(np.percentile(x, p)) if x.size else float("nan")


def _otsu(x: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold: the histogram split maximizing between-class variance. Always returns a
    value (even for unimodal data), so it is reported/plotted but not used as a recommender."""
    xc = _cap(x)
    if xc.size < 2 or xc.min() == xc.max():
        return float("nan")
    counts, edges = np.histogram(xc, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = counts.astype("float64")
    tot = p.sum()
    if tot <= 0:
        return float("nan")
    p /= tot
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = w0 * w1
    valid = denom > 1e-12
    sigma_b2 = np.where(valid, (mu_t * w0 - mu) ** 2 / np.where(valid, denom, 1.0), -1.0)
    t = int(np.argmax(sigma_b2))
    return float(centers[t])


def _kde_grid(x: np.ndarray, n_grid: int = 512) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pure-numpy Gaussian KDE (Silverman bandwidth) evaluated on a grid over [min, p99.5].
    Returns (grid, density, bandwidth). bandwidth 0 (constant column) -> empty grid."""
    xc = _cap(x)
    n = xc.size
    if n < 2 or xc.min() == xc.max():
        return np.empty(0), np.empty(0), 0.0
    sd = float(xc.std(ddof=1))
    q1, q3 = np.percentile(xc, [25, 75])
    iqr = q3 - q1
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    if a <= 0:
        a = sd
    h = 0.9 * a * n ** (-1.0 / 5.0)
    if h <= 0:
        return np.empty(0), np.empty(0), 0.0
    grid = np.linspace(xc.min(), xc.max(), n_grid)
    # density = (1/(n h sqrt(2pi))) sum_i exp(-0.5 ((g-x_i)/h)^2); block over x to bound memory.
    dens = np.zeros(n_grid)
    norm = 1.0 / (n * h * math.sqrt(2.0 * math.pi))
    block = max(1, 1_000_000 // max(1, n_grid))
    for s in range(0, n, block):
        xb = xc[s:s + block]
        z = (grid[:, None] - xb[None, :]) / h
        dens += np.exp(-0.5 * z * z).sum(axis=1)
    return grid, dens * norm, h


def _kde_valley(grid: np.ndarray, dens: np.ndarray, rho: float = 1.3) -> float:
    """Interior antimode (trough) to the right of the main mode, only if a prominent later mode
    exists (tail peak >= rho * trough). Returns nan when the density is unimodal."""
    if grid.size < 3:
        return float("nan")
    mode = int(np.argmax(dens))
    if mode >= grid.size - 2:
        return float("nan")
    # local minima strictly right of the main mode
    for i in range(mode + 1, grid.size - 1):
        if dens[i] < dens[i - 1] and dens[i] <= dens[i + 1]:
            tail_peak = float(dens[i + 1:].max())
            if tail_peak >= rho * dens[i]:
                return float(grid[i])
    return float("nan")


def _gmm_em(x: np.ndarray, max_iter: int = 200, tol: float = 1e-6,
            var_floor_frac: float = 1e-3) -> Dict[str, Any]:
    """2-component 1-D Gaussian mixture via EM (deterministic quantile-split init, logsumexp
    E-step, variance floor against singularities). Components are ordered good (small mu) / bad."""
    xc = _cap(x)
    n = xc.size
    out = {"mu_good": float("nan"), "mu_bad": float("nan"), "var_good": float("nan"),
           "var_bad": float("nan"), "pi_bad": float("nan"), "sep": float("nan"),
           "converged": False}
    if n < 10 or xc.min() == xc.max():
        return out
    var_floor = max(var_floor_frac * float(xc.var(ddof=0)), 1e-12)
    m = float(np.median(xc))
    lo, hi = xc[xc <= m], xc[xc > m]
    if lo.size < 2 or hi.size < 2:
        return out
    mu = np.array([lo.mean(), hi.mean()], dtype="float64")
    var = np.array([max(lo.var(ddof=0), var_floor), max(hi.var(ddof=0), var_floor)])
    pi = np.array([0.5, 0.5])
    prev_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        # E-step in log space
        logp = np.empty((2, n))
        for k in range(2):
            logp[k] = (np.log(pi[k] + 1e-300) - 0.5 * np.log(2 * np.pi * var[k])
                       - 0.5 * (xc - mu[k]) ** 2 / var[k])
        mx = logp.max(axis=0)
        lse = mx + np.log(np.exp(logp - mx).sum(axis=0))
        ll = float(lse.sum())
        r = np.exp(logp - lse)  # responsibilities, shape (2, n)
        # M-step
        nk = r.sum(axis=1)
        if (nk < 1e-6).any():
            break
        pi = nk / n
        mu = (r * xc).sum(axis=1) / nk
        var = np.array([max((r[k] * (xc - mu[k]) ** 2).sum() / nk[k], var_floor) for k in range(2)])
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll
    order = np.argsort(mu)  # good = smaller mean
    g, b = int(order[0]), int(order[1])
    pooled = math.sqrt(0.5 * (var[g] + var[b]))
    sep = float(abs(mu[b] - mu[g]) / pooled) if pooled > 0 else float("nan")
    out.update(mu_good=float(mu[g]), mu_bad=float(mu[b]), var_good=float(var[g]),
               var_bad=float(var[b]), pi_bad=float(pi[b]), sep=sep, converged=converged)
    return out


def _gmm_threshold(gmm: Dict[str, Any], grid: np.ndarray) -> float:
    """Posterior crossover P(bad|x)=0.5 (smallest x>mu_good), else the between-means density min."""
    mg, mb = gmm["mu_good"], gmm["mu_bad"]
    vg, vb, pib = gmm["var_good"], gmm["var_bad"], gmm["pi_bad"]
    if not all(math.isfinite(v) for v in (mg, mb, vg, vb, pib)) or grid.size == 0:
        return float("nan")
    pig = 1.0 - pib

    def comp(val, mu, var, w):
        return w / math.sqrt(2 * math.pi * var) * np.exp(-0.5 * (val - mu) ** 2 / var)

    g = grid[grid > mg]
    if g.size:
        good = comp(g, mg, vg, pig)
        bad = comp(g, mb, vb, pib)
        post_bad = bad / (good + bad + 1e-300)
        idx = np.where(post_bad >= 0.5)[0]
        if idx.size:
            return float(g[idx[0]])
    # fallback: density minimum between the means
    between = grid[(grid >= mg) & (grid <= mb)]
    if between.size >= 3:
        tot = comp(between, mg, vg, pig) + comp(between, mb, vb, pib)
        return float(between[int(np.argmin(tot))])
    return float("nan")


def _kneedle(x: np.ndarray) -> float:
    """Knee of the sorted-value-vs-rank curve (max deviation from the chord). Tail-noisy; a
    corroborator only."""
    xs = np.sort(_cap(x))
    span = float(np.ptp(xs))
    if xs.size < 3 or span == 0:
        return float("nan")
    xn = (xs - xs.min()) / span
    yn = np.linspace(0.0, 1.0, xs.size)
    d = yn - xn
    return float(xs[int(np.argmax(d))])


def _iforest(x: np.ndarray, contamination="auto", random_state: int = 0,
             min_n: int = 50) -> float:
    """IsolationForest upper-tail cut: fit a 1-D isolation forest, then take the **onset of the
    flagged-outlier region above the median** (smallest value the forest calls anomalous on the
    upper side). Returns nan if sklearn is missing, n is too small, or no upper outliers found.

    Note: a tree-based, distribution-free outlier detector. Used per-metric here for comparison; its
    real strength is the *multivariate* joint outlier flag exposed by :func:`isolation_forest_flags`."""
    if IsolationForest is None or x.size < min_n:
        return float("nan")
    med = float(np.median(x))
    clf = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    labels = clf.fit_predict(x.reshape(-1, 1))         # -1 = outlier, 1 = inlier
    upper_out = x[(labels == -1) & (x > med)]
    return float(upper_out.min()) if upper_out.size else float("nan")


# --------------------------------------------------------------------------------------
# Per-metric orchestration
# --------------------------------------------------------------------------------------
def _suggest_one_tail(x: np.ndarray, *, k_iqr: float, k_mad: float, percentile: float,
                      sep_min: float, pi_bounds: Tuple[float, float], min_n: int) -> Dict[str, Any]:
    """Run every estimator on an upper-tail series and apply the recommendation rule. ``x`` must
    already be oriented lower-is-better (overlap is mirrored by the caller)."""
    n = int(x.size)
    res: Dict[str, Any] = {"n": n, "methods": {}, "gmm": {}, "recommended": None, "rationale": ""}
    if n == 0:
        res["summary"] = {}
        res["rationale"] = "no data"
        return res

    pcts = np.percentile(x, [50, 90, 95, 99])
    res["summary"] = {"min": float(x.min()), "p50": float(pcts[0]), "p90": float(pcts[1]),
                      "p95": float(pcts[2]), "p99": float(pcts[3]), "max": float(x.max()),
                      "mean": float(x.mean()), "std": float(x.std(ddof=0))}

    iqr = _iqr_fence(x, k_iqr)
    mad = _mad_fence(x, k_mad)
    pct = _percentile(x, percentile)
    otsu = _otsu(x)
    grid, dens, _h = _kde_grid(x)
    kde_v = _kde_valley(grid, dens)
    gmm = _gmm_em(x)
    gmm_x = _gmm_threshold(gmm, grid)
    knee = _kneedle(x)
    iforest = _iforest(x)

    def _f(v):
        return None if (v is None or not math.isfinite(v)) else round(float(v), 3)

    res["methods"] = {"iqr_fence": _f(iqr), "mad_fence": _f(mad), "percentile": _f(pct),
                      "otsu": _f(otsu), "kde_valley": _f(kde_v), "gmm_crossover": _f(gmm_x),
                      "kneedle": _f(knee), "iforest": _f(iforest)}
    res["gmm"] = {k: (round(v, 3) if isinstance(v, float) and math.isfinite(v) else v)
                  for k, v in gmm.items()}
    # arrays for the diagnostic plot (upper-tail orientation); stripped from the public result
    res["_x"] = x
    res["_grid"] = grid
    res["_dens"] = dens

    p50, p99, xmax = res["summary"]["p50"], res["summary"]["p99"], res["summary"]["max"]

    if n < min_n:
        rec = _percentile(x, 95.0)
        rationale = f"n<{min_n}: insufficient data, using p95"
    else:
        gmm_ok = (gmm["converged"] and math.isfinite(gmm["sep"]) and gmm["sep"] >= sep_min
                  and pi_bounds[0] <= gmm["pi_bad"] <= pi_bounds[1])
        kde_ok = math.isfinite(kde_v)
        sep = gmm["sep"]
        sep_s = f"{sep:.1f}" if math.isfinite(sep) else "n/a"
        if gmm_ok or kde_ok:
            # average the separators (GMM crossover, KDE valley) with the IsolationForest cut
            cands = [v for v in (gmm_x, kde_v, iforest) if math.isfinite(v)]
            rec = float(np.median(cands)) if cands else max(mad, iqr)
            # name the trigger that actually fired
            if gmm_ok and kde_ok:
                trig = f"GMM sep={sep_s} + KDE valley"
            elif gmm_ok:
                pib = gmm["pi_bad"]
                pib_s = f"{pib:.0%}" if math.isfinite(pib) else "n/a"
                trig = f"GMM sep={sep_s}, P(bad)={pib_s}"
            else:
                trig = f"KDE valley (GMM sep={sep_s}<{sep_min})"
            rationale = f"bimodal ({trig}): good/bad crossover"
        else:
            rec = min(max(mad, iqr), p99)
            rationale = f"unimodal (sep={sep_s}<{sep_min}, no KDE valley): robust MAD/IQR fence"

    # clamp so the cut never rejects the good cluster (below p50) nor exceeds observed data
    rec = float(min(max(rec, p50), xmax))
    res["recommended"] = round(rec, 3)
    res["rationale"] = rationale
    return res


def suggest_thresholds(
    routes_summary: pd.DataFrame,
    *,
    report: bool = True,
    metrics: Sequence[str] = _DEFAULT_METRICS,
    k_iqr: float = 1.5,
    k_mad: float = 3.0,
    percentile: float = 97.5,
    sep_min: float = 2.0,
    pi_bounds: Tuple[float, float] = (0.01, 0.45),
    min_n: int = 20,
    plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimate good :meth:`DuckDBMapMatcher.resolve_routes` thresholds from a ``routes_summary``.

    For each quality metric it runs several outlier / two-population estimators and recommends one
    cut (a bimodal separator when a genuine bad cluster exists, else a robust fence). The top-level
    ``"recommended"`` dict maps directly to ``resolve_routes`` keywords.

    Parameters
    ----------
    routes_summary:
        Output of :meth:`DuckDBMapMatcher.match_routes` (one row per A-edge). ``NO_MATCH`` rows are
        excluded automatically.
    report:
        Print a readable per-metric report (mirrors :func:`validate_b_geometry`).
    metrics:
        Which columns to analyze. Default: ``dtw_distance`` (avg), ``max_dtw_distance``,
        ``bearing_diff`` (all lower-is-better) and ``overlap_pct`` (higher-is-better, mirrored).
    k_iqr, k_mad, percentile:
        Robust-fence tuning (IQR multiplier, MAD multiplier, percentile cut).
    sep_min, pi_bounds:
        Bimodality gate: a GMM split counts only if the standardized component separation is
        ``>= sep_min`` and the bad-component weight is within ``pi_bounds``.
    min_n:
        Below this many non-NaN values an metric falls back to p95.
    plot_path:
        If set (and matplotlib is installed), save a 2x2 diagnostic plot there.

    Returns
    -------
    dict with ``metrics`` (per-metric breakdown), ``recommended`` (``resolve_routes`` kwargs),
    ``params``, and ``plot_path``.
    """
    if "match_type" in routes_summary.columns:
        rs = routes_summary[routes_summary["match_type"] != "NO_MATCH"]
    else:
        rs = routes_summary
    n_total = len(routes_summary)
    n_nomatch = n_total - len(rs)

    per_metric: Dict[str, Any] = {}
    recommended: Dict[str, Optional[float]] = {}
    for metric in metrics:
        if metric not in rs.columns:
            logger.warning("suggest_thresholds: column %r not in routes_summary; skipping", metric)
            continue
        direction = _DIRECTION.get(metric, "lower")
        x = _clean(rs[metric])
        if direction == "higher":
            # mirror to an upper-tail problem on the deficit, then convert recommendations back
            xt = 100.0 - x
            sub = _suggest_one_tail(xt, k_iqr=k_iqr, k_mad=k_mad, percentile=percentile,
                                    sep_min=sep_min, pi_bounds=pi_bounds, min_n=min_n)
            sub = _mirror_back(sub, x)
        else:
            sub = _suggest_one_tail(x, k_iqr=k_iqr, k_mad=k_mad, percentile=percentile,
                                    sep_min=sep_min, pi_bounds=pi_bounds, min_n=min_n)
        sub["direction"] = direction
        per_metric[metric] = sub
        recommended[_RESOLVE_KW[metric]] = sub["recommended"]

    params = {"k_iqr": k_iqr, "k_mad": k_mad, "percentile": percentile, "sep_min": sep_min,
              "pi_bounds": list(pi_bounds), "min_n": min_n}
    result = {"metrics": per_metric, "recommended": recommended, "params": params,
              "n_total": n_total, "n_nomatch": n_nomatch, "plot_path": None}

    # Pull the plotting arrays out into native units, then strip the private keys so the returned
    # dict stays clean/serializable.
    plot_data: Dict[str, Any] = {}
    for metric, sub in per_metric.items():
        x = sub.pop("_x", np.empty(0))
        grid = sub.pop("_grid", np.empty(0))
        dens = sub.pop("_dens", np.empty(0))
        if sub.get("direction") == "higher":
            x = 100.0 - x
            grid = 100.0 - grid
        plot_data[metric] = (x, grid, dens)

    if plot_path:
        result["plot_path"] = _plot(per_metric, plot_data, plot_path)

    if report:
        _print_report(result)

    logger.info("suggest_thresholds: %d rows (%d NO_MATCH excluded); recommended %s",
                n_total, n_nomatch, recommended)
    return result


def isolation_forest_flags(
    routes_summary: pd.DataFrame,
    *,
    metrics: Sequence[str] = _DEFAULT_METRICS,
    contamination="auto",
    random_state: int = 0,
    n_estimators: int = 300,
    report: bool = True,
) -> pd.DataFrame:
    """**Multivariate** outlier flag via IsolationForest — the natural use of the algorithm.

    Where :func:`suggest_thresholds` cuts each metric independently, this fits ONE isolation forest
    on all quality signals **jointly** (z-scored so meters / degrees / percent are comparable, and
    ``overlap_pct`` flipped to a deficit so "high = worse" on every axis). It flags matches that are
    anomalous *in combination* — a route can be acceptable on each axis alone yet land in a sparse
    region of the joint space. This catches correlated weirdness a per-axis threshold misses.

    Returns a copy of ``routes_summary`` with two added columns on the matched rows:
    ``if_outlier`` (bool; ``True`` = anomaly) and ``if_score`` (forest decision function, lower =
    more anomalous). ``NO_MATCH`` rows get ``if_outlier=False`` / ``if_score=NaN``.

    Requires scikit-learn (the ``[ml]`` extra); raises ``ImportError`` if unavailable.
    """
    if IsolationForest is None:
        raise ImportError("isolation_forest_flags needs scikit-learn (install the [ml] extra)")

    out = routes_summary.copy()
    out["if_outlier"] = False
    out["if_score"] = np.nan

    matched = out["match_type"] != "NO_MATCH" if "match_type" in out.columns else pd.Series(
        True, index=out.index)
    feats = []
    for metric in metrics:
        if metric not in out.columns:
            continue
        col = pd.to_numeric(out[metric], errors="coerce").to_numpy("float64")
        if _DIRECTION.get(metric, "lower") == "higher":
            col = 100.0 - col            # flip so higher = worse everywhere
        feats.append(col)
    if not feats:
        return out
    X = np.column_stack(feats)
    valid = matched.to_numpy() & np.isfinite(X).all(axis=1)
    Xv = X[valid]
    if Xv.shape[0] < 10:
        return out

    # z-score per column so no metric dominates by scale (guard zero-variance columns)
    mu = Xv.mean(axis=0)
    sd = Xv.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (Xv - mu) / sd

    clf = IsolationForest(contamination=contamination, random_state=random_state,
                          n_estimators=n_estimators)
    labels = clf.fit_predict(Xs)                 # -1 outlier / 1 inlier
    scores = clf.decision_function(Xs)           # lower = more anomalous

    idx = out.index[valid]
    out.loc[idx, "if_outlier"] = labels == -1
    out.loc[idx, "if_score"] = scores

    n_flag = int((labels == -1).sum())
    if report:
        print("=" * 56)
        print("   MULTIVARIATE OUTLIERS (IsolationForest, joint)")
        print("=" * 56)
        print(f"matched rows scored : {Xv.shape[0]}   features: {[m for m in metrics if m in out.columns]}")
        print(f"flagged as outliers : {n_flag}  ({100.0 * n_flag / max(1, Xv.shape[0]):.1f}%)")
        flagged = out.loc[idx][labels == -1]
        if n_flag:
            cols = [c for c in metrics if c in out.columns]
            print("worst (lowest if_score) examples:")
            worst = flagged.nsmallest(min(5, n_flag), "if_score")
            for _, r in worst.iterrows():
                vals = "  ".join(f"{c}={r[c]}" for c in cols)
                print(f"   src {r.get('source_id', '?')}: {vals}  score={r['if_score']:.3f}")
        print("=" * 56)
    logger.info("isolation_forest_flags: %d/%d matched rows flagged", n_flag, Xv.shape[0])
    return out


def _mirror_back(sub: Dict[str, Any], x_native: np.ndarray) -> Dict[str, Any]:
    """Convert an upper-tail result computed on (100-overlap) back into native overlap units, so a
    fence becomes a *minimum* overlap and the numbers read in %. ``x_native`` is the raw overlap
    series, used to recompute the summary with correct (monotonic) percentiles."""
    def back(v):
        return None if v is None else round(float(np.clip(100.0 - v, 0.0, 100.0)), 3)

    sub = dict(sub)
    sub["methods"] = {k: back(v) for k, v in sub["methods"].items()}
    sub["recommended"] = back(sub["recommended"])
    if x_native.size:
        p = np.percentile(x_native, [50, 90, 95, 99])
        sub["summary"] = {"min": float(x_native.min()), "p50": float(p[0]), "p90": float(p[1]),
                          "p95": float(p[2]), "p99": float(p[3]), "max": float(x_native.max()),
                          "mean": float(x_native.mean()), "std": float(x_native.std(ddof=0))}
    return sub


# --------------------------------------------------------------------------------------
# Report + plot
# --------------------------------------------------------------------------------------
def _print_report(result: Dict[str, Any]) -> None:
    print("=" * 56)
    print("      QUALITY THRESHOLD SUGGESTIONS (resolve_routes)")
    print("=" * 56)
    print(f"rows total             : {result['n_total']}   "
          f"(NO_MATCH excluded: {result['n_nomatch']})")
    for metric, sub in result["metrics"].items():
        d = sub.get("direction", "lower")
        better = "lower" if d == "lower" else "higher"
        print(f"\n-- {metric} ({better} better, n={sub['n']}) --")
        if sub["n"] == 0:
            print("   no data")
            continue
        s = sub["summary"]
        print(f"   summary   : p50={s['p50']:.2f}  p90={s['p90']:.2f}  p95={s['p95']:.2f}  "
              f"p99={s['p99']:.2f}  max={s['max']:.2f}")
        m = sub["methods"]
        def sval(v):
            return "  n/a" if v is None else f"{v:.2f}"
        print(f"   iqr={sval(m['iqr_fence'])}  mad={sval(m['mad_fence'])}  "
              f"p{result['params']['percentile']:g}={sval(m['percentile'])}  "
              f"otsu={sval(m['otsu'])}")
        g = sub.get("gmm", {})
        sep = g.get("sep")
        sep_s = f"{sep:.1f}" if isinstance(sep, (int, float)) and math.isfinite(sep) else "n/a"
        print(f"   kde_valley={sval(m['kde_valley'])}  gmm={sval(m['gmm_crossover'])} (sep={sep_s})"
              f"  kneedle={sval(m['kneedle'])}  iforest={sval(m.get('iforest'))}")
        kw = _RESOLVE_KW[metric]
        print(f"   -> {kw} = {sub['recommended']}   [{sub['rationale']}]")
    print("\n" + "-" * 56)
    kwargs = ", ".join(f"{k}={v}" for k, v in result["recommended"].items() if v is not None)
    print("resolve_routes kwargs:")
    print(f"  {kwargs}")
    if result.get("plot_path"):
        print(f"plot saved -> {result['plot_path']}")
    print("Tip: data-driven starting points. Widen if you over-reject, tighten if wrong")
    print("matches survive; re-run suggest_thresholds after resolve_routes to confirm.")
    print("=" * 56)


def _plot(per_metric: Dict[str, Any], plot_data: Dict[str, Any], plot_path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional [viz] extra
        logger.warning("suggest_thresholds: matplotlib unavailable (%s); skipping plot", exc)
        return None

    metrics = list(per_metric.keys())
    ncols = 2
    nrows = max(1, math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = {"iqr_fence": "#1f77b4", "mad_fence": "#2ca02c", "percentile": "#9467bd",
              "otsu": "#8c564b", "kde_valley": "#ff7f0e", "gmm_crossover": "#d62728",
              "kneedle": "#7f7f7f", "iforest": "#e377c2"}

    for ax, metric in zip(axes, metrics):
        sub = per_metric[metric]
        x, grid, dens = plot_data.get(metric, (np.empty(0), np.empty(0), np.empty(0)))
        ax.set_title(f"{metric}\n{sub['rationale']}", fontsize=9)
        if sub["n"] == 0 or x.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        # histogram of the (native-unit) series, clipped at p99.5 for readability
        hi = float(np.percentile(x, 99.5))
        lo = float(x.min())
        xc = np.clip(x, lo, hi) if hi > lo else x
        ax.hist(xc, bins=60, density=True, color="#cfd8dc", edgecolor="white", linewidth=0.3)
        if grid.size and dens.size:
            ax.plot(grid, dens, color="#37474f", lw=1.3, label="KDE")
        for name, val in sub["methods"].items():
            if val is None:
                continue
            ax.axvline(val, color=colors.get(name, "gray"), lw=1.0, alpha=0.85, label=name)
        rec = sub["recommended"]
        if rec is not None:
            ax.axvline(rec, color="black", lw=2.6, label=f"recommended={rec}")
            # shade the rejected region: right of the cut for lower-better, left for higher-better
            if sub.get("direction") == "higher":
                ax.axvspan(ax.get_xlim()[0], rec, color="red", alpha=0.06)
            else:
                ax.axvspan(rec, ax.get_xlim()[1], color="red", alpha=0.06)
        ax.legend(fontsize=7, loc="best")
        ax.set_ylabel("density")
        ax.set_xlabel(metric)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return plot_path
