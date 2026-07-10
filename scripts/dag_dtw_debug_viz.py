"""DAG-DTW debugging visualization (Mode 3) — standalone HTML written to output/.

Runs the full pipeline (prepare -> forward -> both extraction engines) on a synthetic case and
renders two linked views:

  1. CORRESPONDENCE — source A (black, real position), target B (grey, lifted), orange match links;
     a dropdown switches between the two engines (cell join / vertex join), the title
     showing each engine's decision cost and validity.
  2. CELL TABLE — every candidate cell (A-vertex row x B-cell column) colored by state:
     green = alive, orange X = forbidden (the §4.1a coupling), grey = removed by the sink-search
     pre-pass, hollow = D = inf; cells in the CELL engine's matching are ring-highlighted.
     Hover shows E, D and the stored back-pointer.

Also prints a text diagnostic: per-vertex cell counts, the engine comparison table, and the
cross-validation verdict (C(cell) <= C(join) -- an exactness invariant, docs §10.2).

Run:
    python scripts/dag_dtw_debug_viz.py --case y_split
    python scripts/dag_dtw_debug_viz.py --case diamond                 # reconvergent source
    python scripts/dag_dtw_debug_viz.py --case dense_chain --alpha .5  # coverage regime
    python scripts/dag_dtw_debug_viz.py --case wsplit --shift 2        # forbidden cells visible
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from network_matching.dag_dtw import (digraph, line_digraph, prepare, forward,
                                       extract_join, extract_cell, check_rules, _cost_of,
                                       _cell_reachable, _b_order, INF)

CASES = {
    "chain":       (({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)]),
                    ({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)},
                     [("b0", "b1"), ("b1", "b2")])),
    "y_split":     (({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)}, [(0, 1), (1, 2), (1, 3)]),
                    ({"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
                     [("s", "j"), ("j", "u"), ("j", "d")])),
    "merge":       (({0: (0, 6), 1: (0, -6), 2: (10, 0), 3: (20, 0)}, [(0, 2), (1, 2), (2, 3)]),
                    ({"a": (0, 6.5), "b": (0, -5.5), "m": (10, .5), "o": (20, .5)},
                     [("a", "m"), ("b", "m"), ("m", "o")])),
    "dense_chain": (({0: (0, 0), 1: (9, 0), 2: (18, 0)}, [(0, 1), (1, 2)]),      # B finer: 1:N runs
                    ({f"b{i}": (3 * i, .4) for i in range(7)},
                     [(f"b{i}", f"b{i + 1}") for i in range(6)])),
    "wsplit":      (({0: (6.73, 18.65), 1: (28.37, 0.46), 2: (25.41, 14.29)}, [(1, 0), (1, 2)]),
                    ({"b0": (10.71, 2.92), "b1": (30.12, 17.11), "b2": (-1.67, 12.48)},
                     [("b1", "b2")])),                                            # coupling forbids
    "diamond":     (({0: (0, 0), 1: (10, 3), 2: (10, -3), 3: (20, 0)},           # DAG (--dag)
                     [(0, 1), (0, 2), (1, 3), (2, 3)]),
                    ({"p": (0, .5), "q": (10, 3.5), "r": (10, -2.5), "t": (20, .5)},
                     [("p", "q"), ("p", "r"), ("q", "t"), ("r", "t")])),
}
_COL = dict(B="#9aa0a6", Bn="#5f6368", A="#111111", M="#ff7f0e",
            alive="#2e9e5b", forb="#e8710a", removed="#b6bcc4", inf="#dfe3e8")


def _xy(G, n):
    return G.nodes[n]["x"], G.nodes[n]["y"]


def _dy(A, B):
    ys = [_xy(g, n)[1] for g in (A, B) for n in g.nodes]
    xs = [_xy(g, n)[0] for g in (A, B) for n in g.nodes]
    return (max(ys) - min(ys)) + 0.35 * (max(xs) - min(xs)) + 3.0


def _edges_trace(G, dy, color, width, vis):
    ex, ey = [], []
    for u, v in G.edges:
        (x0, y0), (x1, y1) = _xy(G, u), _xy(G, v)
        ex += [x0, x1, None]
        ey += [y0 + dy, y1 + dy, None]
    return go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=width),
                      hoverinfo="skip", visible=vis, showlegend=False)


def _arrows_trace(G, dy, color, vis):
    xs, ys, ang = [], [], []
    for u, v in G.edges:
        (x0, y0), (x1, y1) = _xy(G, u), _xy(G, v)
        xs.append(x0 + .62 * (x1 - x0))
        ys.append(y0 + dy + .62 * (y1 - y0))
        ang.append(float(np.degrees(np.arctan2(x1 - x0, y1 - y0))))
    return go.Scatter(x=xs, y=ys, mode="markers",
                      marker=dict(symbol="arrow", size=14, angle=ang, color=color,
                                  line=dict(color="white", width=1.2)),
                      hoverinfo="skip", visible=vis, showlegend=False)


def _nodes_trace(G, dy, color, prefix, vis):
    return go.Scatter(x=[_xy(G, n)[0] for n in G.nodes], y=[_xy(G, n)[1] + dy for n in G.nodes],
                      mode="markers", marker=dict(color=color, size=9),
                      text=[f"{prefix} · {n}" for n in G.nodes],
                      hovertemplate="%{text}<extra></extra>", visible=vis, showlegend=False)


def correspondence_figure(bgA, bgB, srcG, tgtG, engines):
    """One figure, engine dropdown. `engines` = {name: (M, cost, valid)}; positions come from
    srcG/tgtG (the graphs the matching lives on -- L(A)/L(B) in segment mode)."""
    dy = _dy(bgA, bgB)
    traces, groups = [], []
    for name, (M, cost, valid) in engines.items():
        start = len(traces)
        vis = len(groups) == 0
        traces += [_edges_trace(bgB, dy, _COL["B"], 7, vis), _arrows_trace(bgB, dy, _COL["Bn"], vis),
                   _nodes_trace(bgB, dy, _COL["Bn"], "B", vis),
                   _edges_trace(bgA, 0, _COL["A"], 3, vis), _arrows_trace(bgA, 0, _COL["A"], vis),
                   _nodes_trace(bgA, 0, _COL["A"], "A", vis)]
        mx, my, hx, hy, ht = [], [], [], [], []
        if M is not None:
            for (a, v) in sorted(M, key=str):
                ax, ay = _xy(srcG, a)
                bx, by = _xy(tgtG, v)
                mx += [ax, bx, None]
                my += [ay, by + dy, None]
                hx.append((ax + bx) / 2)
                hy.append((ay + by + dy) / 2)
                ht.append(f"{a} → {v}")
        traces.append(go.Scatter(x=mx, y=my, mode="lines",
                                 line=dict(color=_COL["M"], width=2.2), hoverinfo="skip",
                                 visible=vis, showlegend=False))
        traces.append(go.Scatter(x=hx, y=hy, mode="markers",
                                 marker=dict(color=_COL["M"], size=8, symbol="diamond"),
                                 text=ht, hovertemplate="match %{text}<extra></extra>",
                                 visible=vis, showlegend=False))
        groups.append((name, start, len(traces) - start, cost, valid))
    fig = go.Figure(traces)
    n = len(traces)

    def title(g):
        name, _s, _c, cost, valid = g
        c = "infeasible" if cost is None else f"C(M) = {cost:.3f}"
        return f"engine: {name} — {c} — {'VALID' if valid else 'invalid/failed'}"

    buttons = []
    for gi, g in enumerate(groups):
        mask = [False] * n
        s, cnt = g[1], g[2]
        for k in range(s, s + cnt):
            mask[k] = True
        buttons.append(dict(label=g[0], method="update",
                            args=[{"visible": mask}, {"title.text": title(g)}]))
    fig.update_layout(title=dict(text=title(groups[0])), height=560, width=760,
                      plot_bgcolor="white", xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
                      margin=dict(l=10, r=10, t=90, b=10),
                      updatemenus=[dict(active=0, buttons=buttons, x=0, xanchor="left",
                                        y=1.14, yanchor="top", showactive=True)])
    return fig


def cell_table_figure(A, B, seen, M_cell):
    """Every candidate cell as a marker: state color + hover (E, D, bp); cell-engine M ringed."""
    border = _b_order(B)
    a_order = [str(a) for a in nx.topological_sort(A)]
    b_order = sorted({v for a in A.nodes for v in A.nodes[a]["cand"]}, key=lambda t: border[t])
    b_pos = {v: i for i, v in enumerate(b_order)}
    pts = {k: ([], [], []) for k in ("alive", "forb", "removed", "inf")}
    ringx, ringy = [], []
    for ai, a in enumerate(nx.topological_sort(A)):
        for v, c in A.nodes[a]["cand"].items():
            if c.get("forbidden"):
                k = "forb"
            elif (a, v) not in seen:
                k = "removed"
            elif c["D"] >= INF:
                k = "inf"
            else:
                k = "alive"
            d = "inf" if c["D"] >= INF else f"{c['D']:.2f}"
            pts[k][0].append(b_pos[v])
            pts[k][1].append(ai)
            pts[k][2].append(f"({a}, {v})  E={c['E']:.2f}  D={d}<br>bpD={c['bpD']}")
            if M_cell is not None and (a, v) in M_cell:
                ringx.append(b_pos[v])
                ringy.append(ai)
    fig = go.Figure()
    style = {"alive": dict(color=_COL["alive"], symbol="circle", size=13),
             "forb": dict(color=_COL["forb"], symbol="x", size=12),
             "removed": dict(color=_COL["removed"], symbol="circle", size=10),
             "inf": dict(color=_COL["inf"], symbol="circle-open", size=11)}
    label = {"alive": "alive", "forb": "forbidden (§4.1a coupling)",
             "removed": "removed (sink-search pre-pass)", "inf": "D = ∞ (upstream-infeasible)"}
    for k in ("alive", "inf", "removed", "forb"):
        x, y, t = pts[k]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=style[k], name=label[k],
                                 text=t, hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ringx, y=ringy, mode="markers", name="in M (cell engine)",
                             marker=dict(symbol="circle-open", size=21, color=_COL["M"],
                                         line=dict(width=3)), hoverinfo="skip"))
    fig.update_layout(title="candidate cells — state and the cell engine's matching",
                      height=max(360, 40 * len(a_order) + 180), width=max(700, 46 * len(b_order) + 260),
                      plot_bgcolor="white",
                      xaxis=dict(tickvals=list(range(len(b_order))),
                                 ticktext=[str(v) for v in b_order], title="B cells"),
                      yaxis=dict(tickvals=list(range(len(a_order))), ticktext=a_order,
                                 autorange="reversed", title="A vertices (topological)"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="y_split", choices=sorted(CASES))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--mode", default="point", choices=["point", "segment"])
    ap.add_argument("--shift", type=float, default=0.0, help="shift the source north by this many meters")
    ap.add_argument("--out", default=None, help="output HTML (default output/dag_dtw_debug_<case>.html)")
    args = ap.parse_args()
    (an, ae), (bn, be) = CASES[args.case]
    A0 = digraph({k: (x, y + args.shift) for k, (x, y) in an.items()}, ae)
    B0 = digraph(bn, be)
    if args.mode == "segment":
        src, tgt = line_digraph(A0), line_digraph(B0)
    else:
        src, tgt = A0, B0
    prepare(src, tgt, r=args.r)
    forward(src, tgt, alpha=args.alpha, beta=args.beta)
    seen = _cell_reachable(src, tgt)

    print(f"case={args.case}  mode={args.mode}  alpha={args.alpha} beta={args.beta}  r={args.r}")
    print(f"{'vertex':>14} {'cand':>5} {'alive':>6} {'forbidden':>10} {'removed':>8} {'D=inf':>6}")
    for a in nx.topological_sort(src):
        cand = src.nodes[a]["cand"]
        nf = sum(1 for c in cand.values() if c.get("forbidden"))
        nr = sum(1 for v in cand if (a, v) not in seen and not cand[v].get("forbidden"))
        ni = sum(1 for v, c in cand.items() if c["D"] >= INF and not c.get("forbidden")
                 and (a, v) in seen)
        na = len(cand) - nf - nr - ni
        print(f"{str(a):>14} {len(cand):>5} {na:>6} {nf:>10} {nr:>8} {ni:>6}")

    engines = {}
    for name, fn in (("cell join", extract_cell), ("vertex join", extract_join)):
        t0 = time.perf_counter()
        try:
            M, _ = fn(src, tgt, args.alpha, args.beta)
            cost = _cost_of(src, tgt, M, args.alpha, args.beta)
            valid = not any(check_rules(M, src, tgt))
        except ValueError as e:
            M, cost, valid = None, None, f"infeasible: {e}"
        engines[name] = (M, cost, valid)
        dt = 1e3 * (time.perf_counter() - t0)
        c = "infeasible" if cost is None else f"{cost:10.3f}"
        print(f"{name:>14}: C(M) = {c}   valid = {valid}   ({dt:.1f} ms)")
    cc = engines["cell join"][1]
    if cc is not None:
        ok = all(o[1] is None or cc <= o[1] + 1e-6 for o in engines.values())
        print(f"cross-validation: C(cell) <= C(vertex join)  ->  {'OK' if ok else '*** VIOLATED (exactness bug) ***'}")

    fig1 = correspondence_figure(A0, B0, src, tgt, engines)
    fig2 = cell_table_figure(src, tgt, seen, engines["cell join"][0])
    out = args.out or os.path.join("output", f"dag_dtw_debug_{args.case}"
                                             f"{'_seg' if args.mode == 'segment' else ''}.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("<html><head><meta charset='utf-8'><title>DAG-DTW debug — "
                f"{args.case}</title></head><body>\n")
        f.write(fig1.to_html(full_html=False, include_plotlyjs="inline"))
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
