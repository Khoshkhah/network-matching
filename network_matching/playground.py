"""
Interactive matching playground helpers (used by ``notebooks/graph_dtw_playground.ipynb``).

Pure who-matched-whom visualization and slider panel for graph-DTW -- no analysis panels:

- :func:`draw_match` -- the correspondence view. ``emission="point"`` draws one link per matched
  point pair; ``emission="segment"`` (match with ``debug=True``) draws one link per matched
  (A-segment, B-segment) state, from the MIDDLE of the A-segment to the MIDDLE of its matched
  B-segment. The plot title states which view was drawn.
- :func:`perturb_edge` -- apply slider-style perturbations (shift / rotate / noise / ...) in the
  same fixed order the CLI scripts use.
- :func:`playground` -- the ipywidgets panel: built-in case chooser or your own network.
- :func:`to_edges` -- ``[(id, [(x, y), ...]), ...]`` -> ``[(id, LineString), ...]``.

Living in the library (instead of notebook cells) means an open notebook only holds thin
imports: with ``%autoreload`` the kernel picks up code changes, and the printed
:data:`PLAYGROUND_VERSION` shows at a glance which code a session is running.
"""

import numpy as np
from shapely.geometry import LineString

from .graph_dtw import match_edge_to_bgraph
from .synthetic import SCENARIOS, apply_perturbation, as_array, get_scenario, reverse

PLAYGROUND_VERSION = ("v7 -- emission='midpoint': one middle-to-middle distance per segment "
                      "pair, no sliver states, evenly spread sample points")

PERTURB_ORDER = ["crop", "stretch", "rotate", "translate", "shift", "longitudinal", "noise"]


def to_edges(edge_list):
    """``[(id, [(x, y), ...]), ...]`` -> ``[(id, LineString), ...]`` (LineStrings pass through)."""
    return [(eid, g if isinstance(g, LineString) else LineString(g)) for eid, g in edge_list]


def perturb_edge(coords_a, seed=0, reverse_dir=False, translate_bearing=90.0, **mags):
    """Apply perturbations (keywords = family names, values = magnitudes) in the same order as
    the CLI scripts, then optionally reverse. Returns the moved copy of the edge."""
    P = as_array(coords_a)
    for fam in PERTURB_ORDER:
        m = mags.get(fam, 0)
        if m:
            kw = {"bearing": translate_bearing} if fam == "translate" else {}
            P = apply_perturbation(P, fam, m, seed=seed, **kw)
    return reverse(P) if reverse_dir else P


def draw_match(coords_a, b_edges, res, original_a=None, ax=None):
    """Show WHICH part of A matched WHICH B edge -- nothing else (see module docstring)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = res["metrics"]
    wp = res["warping_path"]                      # [((ax, ay), (bx, by)), ...]
    step_edge = M["warp_edge"]                    # matched B-edge id per pair
    route = [re["dest_id"] for re in M["route_edges"]]
    palette = [plt.cm.tab10(i) for i in (0, 1, 2, 4, 5, 6, 8, 9, 3, 7)]
    color = {eid: palette[i % len(palette)] for i, eid in enumerate(route)}

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.5))

    # B network: matched edges in their own color, other candidates light gray
    for eid, geom in b_edges:
        xy = np.asarray(geom.coords if hasattr(geom, "coords") else geom, float)
        c = color.get(eid)
        if c is not None:
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=7, alpha=0.25, zorder=1,
                    solid_capstyle="round")
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=2.0, zorder=2)
        else:
            ax.plot(xy[:, 0], xy[:, 1], color="0.72", lw=1.6, zorder=2)
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=3.5, lw=0,
                color=c if c is not None else "0.6", zorder=3)
        mid = xy[len(xy) // 2]
        ax.annotate(str(eid), mid, textcoords="offset points", xytext=(5, -12), fontsize=9,
                    color=c if c is not None else "0.45",
                    fontweight="bold" if c is not None else "normal")

    if original_a is not None:                    # the edge before perturbation, for reference
        O, A = np.asarray(original_a, float), np.asarray(coords_a, float)
        if O.shape != A.shape or not np.allclose(O, A):
            ax.plot(O[:, 0], O[:, 1], color="0.6", lw=1.4, ls="--", zorder=3)

    # the correspondence itself -- drawn the way the chosen emission actually matches:
    dbg = res.get("debug")
    emission = (dbg or {}).get("params", {}).get("emission")
    seg_mode = bool(dbg) and emission in ("segment", "midpoint") and "arc_path" in dbg
    if seg_mode:
        # SEGMENT-TO-SEGMENT: the matched POINTS are drawn exactly like the point view (dots at
        # their true locations); only the CONNECTION LINES differ -- one per DP state
        # (A-segment, B-arc), running from the middle of the A-segment to the middle of its
        # matched B-segment. No markers at the midpoints.
        for k, (pa, pb) in enumerate(wp):
            c = color.get(step_edge[k], "0.55")
            ax.plot(pb[0], pb[1], "o", ms=5, color=c, zorder=5)
            ax.plot(pa[0], pa[1], "o", ms=7, color=c, mec="white", mew=1.0, zorder=7)
        gb = res["graph"]
        ap = dbg["a_pool"]
        arcs = dbg["arcs"]
        lo, hi = dbg.get("kept_span", (0, len(dbg["pairs_all"]) - 1))
        ridable = dbg["ridable"]
        for t, (i, k, _mv) in enumerate(dbg["arc_path"]):
            if not (lo <= t + 1 <= hi):               # state t produced alignment pair t+1
                continue                              # -> outside = trimmed overhang
            u, w_ = arcs[k]
            eid = gb.edge_ids[gb.vert_edge[u]]
            c = color.get(eid, "0.55")
            a0, a1 = ap[i][:2], ap[i + 1][:2]
            b0, b1 = (gb.vx[u], gb.vy[u]), (gb.vx[w_], gb.vy[w_])
            ma = ((a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2)   # A-segment midpoint
            mb = ((b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2)   # B-segment midpoint
            if ridable[k]:
                ax.plot([ma[0], mb[0]], [ma[1], mb[1]], color=c, lw=1.2, alpha=0.8, zorder=4)
            elif lb := float(np.hypot(b1[0] - b0[0], b1[1] - b0[1])):
                # pass-through over a non-ridable arc (junction stitch / sliver): NOT a segment
                # match -- the DP only crosses it. Drawn dotted gray; zero-length connectors
                # (lb == 0) are pure bookkeeping and not drawn at all.
                ax.plot([ma[0], mb[0]], [ma[1], mb[1]], color="0.5", lw=1.0, ls=":",
                        alpha=0.9, zorder=4)
    else:
        # POINT-TO-POINT: one link per matched pair, A point painted in its B-edge color
        for k, (pa, pb) in enumerate(wp):
            c = color.get(step_edge[k], "0.55")
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=c, lw=1.2, alpha=0.8, zorder=4)
            ax.plot(pb[0], pb[1], "o", ms=5, color=c, zorder=5)
            ax.plot(pa[0], pa[1], "o", ms=7, color=c, mec="white", mew=1.0, zorder=7)

    A = np.asarray(coords_a, float)
    ax.plot(A[:, 0], A[:, 1], color="#111111", lw=2.2, zorder=6)
    ax.plot(A[0, 0], A[0, 1], marker="^", ms=11, color="#111111", zorder=8)

    handles = [Line2D([], [], color="#111111", lw=2.2, marker="^", ms=8,
                      label="edge A (▲ = start)")]
    handles += [Line2D([], [], color=color[eid], lw=3, label=f"matched → {eid}")
                for eid in route]
    if seg_mode:
        handles.append(Line2D([], [], color="0.55", lw=1.2,
                              label="link: segment middle ↔ segment middle"))
        handles.append(Line2D([], [], color="0.5", lw=1.0, ls=":",
                              label="junction/sliver pass-through (not a segment match)"))
    if any(eid not in color for eid, _ in b_edges):
        handles.append(Line2D([], [], color="0.72", lw=2, label="candidate (unmatched)"))
    if original_a is not None:
        handles.append(Line2D([], [], color="0.6", lw=1.4, ls="--",
                              label="A before perturbation"))
    # legend OUTSIDE the axes (right side) so it never covers the drawing; the inline/widget
    # backend saves with bbox_inches="tight", so the extra width is kept, not clipped
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, framealpha=0.95)

    view = (f"segment ↔ segment ({'middle dist' if emission == 'midpoint' else 'endpoint avg'})"
            if seg_mode else "point ↔ point")
    ttl = ("NO MATCH" if not np.isfinite(res["avg_distance"]) else
           " → ".join(str(e) for e in route)
           + f"   ·   avg distance {res['avg_distance']:.2f} m"
           + f"   ·   {view} view")
    ax.set_title(ttl, fontsize=11)
    ax.set_aspect("equal")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(x1 - x0, y1 - y0)
    if (y1 - y0) < 0.35 * span:
        pad = (0.35 * span - (y1 - y0)) / 2
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=8)
    return ax


def match_and_draw(coords_a, b_edges, original_a=None, ax=None, **match_kwargs):
    """One call: match + draw, always requesting the debug payload so the segment view works.

    Prefer this over calling ``match_edge_to_bgraph`` + ``draw_match`` yourself -- with
    ``emission="segment"`` the segment-to-segment view NEEDS ``debug=True``, and forgetting it
    silently falls back to the point view. Returns the match result.
    """
    import matplotlib.pyplot as plt

    coords = coords_a.tolist() if hasattr(coords_a, "tolist") else list(coords_a)
    b = to_edges(b_edges)
    match_kwargs.pop("debug", None)
    res = match_edge_to_bgraph(coords, b, debug=True, **match_kwargs)
    draw_match(coords_a, b, res, original_a=original_a, ax=ax)
    if ax is None:
        plt.show()
    return res


def playground(coords_a=None, b_edges=None, snap=0.5, step=2.0):
    """Interactive matcher panel. No args -> built-in case chooser; or pass your own
    ``coords_a=[(x, y), ...]`` and ``b_edges=[(id, [(x, y), ...]), ...]``."""
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display

    custom = coords_a is not None
    if custom:
        b_edges = to_edges(b_edges)

    sl = dict(continuous_update=False, style={"description_width": "90px"},
              layout=w.Layout(width="340px"))
    k = {}
    if not custom:
        k["case"] = w.Dropdown(options=sorted(SCENARIOS), value="split", description="case",
                               style=sl["style"], layout=sl["layout"])
    k["emission"] = w.Dropdown(options=["point", "segment", "midpoint"], description="emission",
                               style=sl["style"], layout=sl["layout"])
    k["shift"] = w.FloatSlider(0, min=-16, max=16, step=0.5, description="shift m", **sl)
    k["longitudinal"] = w.FloatSlider(0, min=-16, max=16, step=0.5, description="along m", **sl)
    k["translate"] = w.FloatSlider(0, min=0, max=16, step=0.5, description="translate m", **sl)
    k["translate_bearing"] = w.FloatSlider(90, min=0, max=360, step=15,
                                           description="direction °", **sl)
    k["rotate"] = w.FloatSlider(0, min=-45, max=45, step=1, description="rotate °", **sl)
    k["noise"] = w.FloatSlider(0, min=0, max=5, step=0.25, description="noise σ m", **sl)
    k["seed"] = w.IntSlider(0, min=0, max=20, description="noise seed", **sl)
    k["crop"] = w.FloatSlider(0, min=0, max=80, step=5, description="crop %", **sl)
    k["stretch"] = w.FloatSlider(0, min=0, max=15, step=1, description="stretch m", **sl)
    k["reverse_dir"] = w.Checkbox(False, description="reverse direction")
    k["snap"] = w.FloatSlider(snap, min=0.1, max=3, step=0.1, description="snap tol m", **sl)
    k["step"] = w.FloatSlider(step, min=1, max=20, step=1, description="sample m", **sl)
    k["bearing_w"] = w.FloatSlider(0, min=0, max=1, step=0.05, description="bearing λ", **sl)

    def update(**v):
        if custom:
            A0, B = coords_a, b_edges
        else:
            sc = get_scenario(v["case"])
            A0, B = sc["coords_a"], sc["b_edges"]
        A = perturb_edge(A0, seed=v["seed"], reverse_dir=v["reverse_dir"],
                         translate_bearing=v["translate_bearing"],
                         crop=v["crop"], stretch=v["stretch"], rotate=v["rotate"],
                         translate=v["translate"], shift=v["shift"],
                         longitudinal=v["longitudinal"], noise=v["noise"])
        res = match_edge_to_bgraph(A.tolist(), B, snap_tolerance_m=v["snap"],
                                   step_meters=v["step"], emission=v["emission"],
                                   bearing_weight=v["bearing_w"], debug=True)
        fig, ax = plt.subplots(figsize=(11, 7.5))
        draw_match(A, B, res, original_a=A0, ax=ax)
        plt.show()

    out = w.interactive_output(update, k)
    rows = [w.HBox([k[n] for n in names if n in k]) for names in
            (["case", "emission", "reverse_dir"], ["shift", "longitudinal", "rotate"],
             ["translate", "translate_bearing", "noise"], ["seed", "crop", "stretch"],
             ["snap", "step", "bearing_w"])]
    display(w.VBox(rows), out)
