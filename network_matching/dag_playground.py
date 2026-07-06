"""
Correspondence visualization + interactive playground for DAG-DTW
(`notebooks/dag_dtw_playground.ipynb`), the DAG analog of
:mod:`network_matching.playground`.

Shows only **who matched whom**: the source DAG (several A-edges, drawn dark with direction
arrows and junction rings), the target B-network (matched B-edges coloured, candidates grey), and
one link per A-vertex to its matched B-vertex `φ(a)`, coloured by the B-edge it landed on. No
analysis panels.
"""

import numpy as np
from shapely.geometry import LineString

from .dag_dtw import match_dag_to_bgraph
from .dag_synthetic import DAG_SCENARIOS, get_dag

DAG_PLAYGROUND_VERSION = "v1 -- point-to-point DAG-DTW correspondence view"


def _edges_to_ls(a_edges):
    return [(eid, g if isinstance(g, LineString) else LineString(g)) for eid, g in a_edges]


def perturb_dag(a_edges, shift=0.0, rotate=0.0, noise=0.0, seed=0, bearing_deg=90.0):
    """Rigidly move the WHOLE DAG (so junctions stay stitched): rotate about the DAG centroid,
    translate `shift` m toward `bearing_deg`, then add per-unique-vertex Gaussian `noise`
    (shared junction points get the SAME jitter, so the topology is preserved). Returns new
    ``(id, LineString)`` edges."""
    a_edges = _edges_to_ls(a_edges)
    pts = np.array([p for _e, g in a_edges for p in g.coords], float)
    if len(pts) == 0:
        return a_edges
    c = pts.mean(axis=0)
    th = np.radians(rotate)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = np.radians(bearing_deg)
    trans = shift * np.array([np.sin(a), np.cos(a)])

    def xf(p):
        q = (np.asarray(p, float) - c) @ R.T + c + trans
        return q

    rng = np.random.default_rng(seed)
    jitter = {}                                      # shared coords -> one noise sample

    def noisy(p):
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        if key not in jitter:
            jitter[key] = rng.normal(0.0, noise, 2) if noise else np.zeros(2)
        return jitter[key]

    out = []
    for eid, g in a_edges:
        new = [tuple(xf(p) + noisy(p)) for p in g.coords]
        out.append((eid, LineString(new)))
    return out


def draw_dag_match(res, a_edges, b_edges, ax=None):
    """Draw the DAG correspondence (see module docstring)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a_edges, b_edges = _edges_to_ls(a_edges), _edges_to_ls(b_edges)
    ga, gb = res["GA"], res["GB"]
    routes = res["routes"]
    matched_b = [b for seq in routes.values() for b in seq]
    palette = [plt.cm.tab10(i) for i in (0, 1, 2, 4, 5, 6, 8, 9, 3, 7)]
    seen, bcolor = [], {}
    for b in matched_b:                              # stable color per matched B-edge
        if b not in bcolor:
            bcolor[b] = palette[len(seen) % len(palette)]
            seen.append(b)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.5))

    # target B-network: matched edges coloured, candidates grey
    for eid, g in b_edges:
        xy = np.asarray(g.coords, float)
        c = bcolor.get(eid)
        if c is not None:
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=7, alpha=0.22, zorder=1, solid_capstyle="round")
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=2.2, zorder=2)
        else:
            ax.plot(xy[:, 0], xy[:, 1], color="0.75", lw=1.6, zorder=2)
        ax.annotate(str(eid), xy[len(xy) // 2], textcoords="offset points", xytext=(4, -11),
                    fontsize=8, color=c if c is not None else "0.5",
                    fontweight="bold" if c is not None else "normal")

    # correspondence links: each A-vertex -> φ(a), coloured by the matched B-edge
    for (axx, ayy, v, beid, d) in res["a_vertex_match"]:
        c = bcolor.get(beid, "0.55")
        bxx, byy = float(gb.vx[v]), float(gb.vy[v])
        ax.plot([axx, bxx], [ayy, byy], color=c, lw=1.0, alpha=0.8, zorder=4)
        ax.plot(bxx, byy, "o", ms=4.5, color=c, zorder=5)

    # source DAG: each A-edge dark with a direction arrow; junction rings on branch/merge vertices
    for eid, g in a_edges:
        xy = np.asarray(g.coords, float)
        ax.plot(xy[:, 0], xy[:, 1], color="#111111", lw=2.4, zorder=6)
        k = max(1, len(xy) // 2)
        ax.annotate("", xy=xy[k], xytext=xy[k - 1], zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.6))
        ax.annotate(str(eid), xy[0], textcoords="offset points", xytext=(3, 6),
                    fontsize=7.5, color="#374151")
    for a in range(ga.n_vertices):
        od, idg = len(ga.succ_arcs[a]), len(ga.pred_arcs[a])
        if od > 1 or idg > 1:                        # a branch or merge junction
            ax.plot(ga.vx[a], ga.vy[a], marker="o", ms=13, mfc="none",
                    mec="#dc2626" if od > 1 else "#2563eb", mew=1.8, zorder=8)

    handles = [Line2D([], [], color="#111111", lw=2.4, marker=">", ms=7, label="source A-edge (DAG)")]
    handles += [Line2D([], [], color=bcolor[b], lw=3, label=f"matched → {b}") for b in seen]
    handles += [Line2D([], [], marker="o", mfc="none", mec="#dc2626", lw=0, ms=10, label="branch junction"),
                Line2D([], [], marker="o", mfc="none", mec="#2563eb", lw=0, ms=10, label="merge junction")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, framealpha=0.95)

    na, nb = len(a_edges), len(seen)
    ax.set_title(f"{na} A-edges → {nb} B-edges   ·   avg drift {res['avg_drift']:.2f} m"
                 f"   ·   junction-consistent", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=8)
    return ax


def match_and_draw_dag(a_edges, b_edges, ax=None, snap_tolerance_m=0.5, step_meters=2.0):
    """Match a source DAG to a B-network and draw the correspondence. Returns the result dict."""
    import matplotlib.pyplot as plt
    a_edges, b_edges = _edges_to_ls(a_edges), _edges_to_ls(b_edges)
    res = match_dag_to_bgraph(a_edges, b_edges, snap_tolerance_m=snap_tolerance_m,
                              step_meters=step_meters, debug=True)
    draw_dag_match(res, a_edges, b_edges, ax=ax)
    if ax is None:
        plt.show()
    return res


def dag_playground(a_edges=None, b_edges=None):
    """Interactive panel: move the whole DAG with sliders and watch the joint match.

    No args -> a dropdown over the built-in scenarios (chain / y_split / merge / diamond).
    Pass your **own** DAG -- ``a_edges=[(id, [(x, y), ...]), ...]`` and the target
    ``b_edges=[...]`` the same way -- to drive the sliders on it (like the graph-DTW
    ``playground(my_a, my_b_edges)``). The perturbations move the whole source rigidly so its
    junctions stay stitched.
    """
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display

    custom = a_edges is not None
    if custom:
        a_edges, b_edges = _edges_to_ls(a_edges), _edges_to_ls(b_edges)

    sl = dict(continuous_update=False, style={"description_width": "90px"},
              layout=w.Layout(width="340px"))
    k = {}
    if not custom:
        k["case"] = w.Dropdown(options=sorted(DAG_SCENARIOS), value="y_split", description="DAG",
                               style=sl["style"], layout=sl["layout"])
    k["shift"] = w.FloatSlider(0, min=-8, max=8, step=0.5, description="shift m", **sl)
    k["bearing"] = w.FloatSlider(90, min=0, max=360, step=15, description="direction °", **sl)
    k["rotate"] = w.FloatSlider(0, min=-30, max=30, step=1, description="rotate °", **sl)
    k["noise"] = w.FloatSlider(0, min=0, max=3, step=0.25, description="noise σ m", **sl)
    k["seed"] = w.IntSlider(0, min=0, max=20, description="noise seed", **sl)
    k["step"] = w.FloatSlider(2.0, min=1, max=8, step=0.5, description="sample m", **sl)

    def update(**v):
        if custom:
            a0, b0 = a_edges, b_edges
        else:
            sc = get_dag(v["case"])
            a0, b0 = sc["a_edges"], sc["b_edges"]
        a = perturb_dag(a0, shift=v["shift"], rotate=v["rotate"], noise=v["noise"],
                        seed=v["seed"], bearing_deg=v["bearing"])
        fig, ax = plt.subplots(figsize=(11, 7.5))
        match_and_draw_dag(a, b0, ax=ax, step_meters=v["step"])
        plt.show()

    out = w.interactive_output(update, k)
    rows = [w.HBox([k[n] for n in names if n in k]) for names in
            (["case", "step"], ["shift", "bearing", "rotate"], ["noise", "seed"])]
    display(w.VBox(rows), out)
