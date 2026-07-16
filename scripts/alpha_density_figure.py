"""Generate the paper's Figure 2 (§9.5.2) as inline SVG, straight from the real matchings.

Shows what α actually changes: the matching RELATION. Nothing is hand-drawn — every dot, every
run and every point of the anchor-error curve comes from running the shipped engine.

    python scripts/alpha_density_figure.py       # writes scripts/_alpha_fig.svg
"""
from pathlib import Path

from alpha_density_demo import A_STEP, ALPHAS, B_STEP, N_A, N_B, case, solve

W, H = 720, 470
X0, X1 = 66, 690
MIN_M, MAX_M = -1.5, 26.5


def mx(m):
    return X0 + (m - MIN_M) * (X1 - X0) / (MAX_M - MIN_M)


def relation_panel(alpha, top, title, note):
    """Source row over target row; each source vertex's coverage run drawn as a bracket + fan."""
    A, B = case()
    ay, by = top + 40, top + 92
    s = solve(alpha)
    covered = {j for js in s["runs"].values() for j in js}

    out = [f'<text class="ft" x="{X0}" y="{top + 4}">{title}</text>',
           f'<text class="fn" x="{X0}" y="{top + 20}">{note}</text>']
    for a, js in s["runs"].items():
        ax = mx(A_STEP * a)
        for j in js:                                        # the fan: this vertex's covered cells
            out.append(f'<line class="arun" x1="{ax:.1f}" y1="{ay}" '
                       f'x2="{mx(B_STEP * j):.1f}" y2="{by}"/>')
        out.append(f'<line class="abr" x1="{mx(B_STEP * js[0]):.1f}" y1="{by + 10}" '
                   f'x2="{mx(B_STEP * js[-1]):.1f}" y2="{by + 10}"/>')
        # the ANCHOR: the entry of the run -- the one cell this vertex is really pinned to
        out.append(f'<line class="aanch" x1="{ax:.1f}" y1="{ay}" '
                   f'x2="{mx(B_STEP * js[0]):.1f}" y2="{by}"/>')
        out.append(f'<circle class="ad-anch" cx="{mx(B_STEP * js[0]):.1f}" cy="{by}" r="4.6"/>')
        err = abs(B_STEP * js[0] - A_STEP * a)
        if err > 1e-9:                                      # ... and how far it is from the vertex
            out.append(f'<line class="aerr" x1="{ax:.1f}" y1="{by + 20}" '
                       f'x2="{mx(B_STEP * js[0]):.1f}" y2="{by + 20}"/>')
            out.append(f'<text class="ake" x="{(ax + mx(B_STEP * js[0])) / 2:.1f}" '
                       f'y="{by + 33}" text-anchor="middle">{err:.0f} m off</text>')
    for a in range(N_A):
        out.append(f'<circle class="ad-a" cx="{mx(A_STEP * a):.1f}" cy="{ay}" r="4"/>')
        out.append(f'<text class="ak" x="{mx(A_STEP * a):.1f}" y="{ay - 10}" '
                   f'text-anchor="middle">A{a}</text>')
    for j in range(N_B):
        out.append(f'<circle class="{"ad-b" if j in covered else "ad-idle"}" '
                   f'cx="{mx(B_STEP * j):.1f}" cy="{by}" r="2.8"/>')
    out.append(f'<text class="ak" x="{X0 - 10}" y="{ay + 4}" text-anchor="end">A</text>')
    out.append(f'<text class="ak" x="{X0 - 10}" y="{by + 4}" text-anchor="end">B</text>')
    return "\n".join(out)


def error_panel(top):
    """Worst anchor error against alpha -- the relation getting right, not the cost getting small."""
    pts = [(a, solve(a)["anchor_err"]) for a in sorted(set(ALPHAS) | {0.4, 0.6, 0.8, 0.9})]
    py0, py1 = top + 34, top + 118
    def px(a): return X0 + a * (X1 - X0)
    def py(e): return py1 - (e / 5.0) * (py1 - py0)

    s = [f'<text class="ft" x="{X0}" y="{top + 4}">What α buys — the anchor, not the cost</text>',
         f'<text class="fn" x="{X0}" y="{top + 20}">how far a source vertex sits from the cell it '
         f'is pinned to, against α</text>',
         f'<line class="aax" x1="{X0}" y1="{py1}" x2="{X1}" y2="{py1}"/>',
         f'<line class="aax" x1="{X0}" y1="{py0}" x2="{X0}" y2="{py1}"/>']
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.append(f'<line class="aax" x1="{px(a):.1f}" y1="{py1}" x2="{px(a):.1f}" y2="{py1 + 4}"/>')
        s.append(f'<text class="ak" x="{px(a):.1f}" y="{py1 + 16}" text-anchor="middle">{a:g}</text>')
    for e in (0, 2, 4):
        s.append(f'<text class="ak" x="{X0 - 8}" y="{py(e) + 3.5:.1f}" text-anchor="end">{e} m</text>')
    poly = " ".join(f"{px(a):.1f},{py(e):.1f}" for a, e in pts)
    s.append(f'<polyline class="aerrline" points="{poly}"/>')
    for a, e in pts:
        s.append(f'<circle class="ad-e" cx="{px(a):.1f}" cy="{py(e):.1f}" r="3"/>')
    s.append(f'<text class="ak4" x="{px(0.03):.1f}" y="{py(2.3):.1f}">α → 0: every vertex on its '
             f'own cell — 0 m</text>')
    s.append(f'<text class="ak3" x="{X1}" y="{py(4) - 10:.1f}" text-anchor="end">'
             f'α = 1: pinned 4 m from where it is</text>')
    s.append(f'<text class="ak" x="{(X0 + X1) / 2:.0f}" y="{py1 + 34}" text-anchor="middle">'
             f'coverage discount α</text>')
    return "\n".join(s)


CSS = """<style>
    .alphafig{width:100%;height:auto;font-family:var(--sans);--warn:#b0492c}
    .alphafig .ft{font-size:12.5px;font-weight:700;fill:var(--ink)}
    .alphafig .fn{font-size:11px;fill:var(--muted)}
    .alphafig .ak{font-size:10.5px;fill:var(--muted)}
    .alphafig .ake{font-size:9.5px;font-weight:700;fill:var(--warn)}
    .alphafig .ak3{font-size:10.5px;font-weight:700;fill:var(--warn)}
    .alphafig .ak4{font-size:10.5px;font-weight:700;fill:var(--accent)}
    .alphafig .arun{stroke:var(--muted);stroke-width:1;opacity:.4}
    .alphafig .aanch{stroke:var(--accent);stroke-width:1.8;opacity:.9}
    .alphafig .abr{stroke:var(--accent);stroke-width:2.6;opacity:.3;stroke-linecap:round}
    .alphafig .aerr{stroke:var(--warn);stroke-width:1.6;opacity:.9}
    .alphafig .ad-a{fill:var(--ink2)}
    .alphafig .ad-b{fill:var(--muted)}
    .alphafig .ad-idle{fill:none;stroke:var(--muted);stroke-width:1;opacity:.45}
    .alphafig .ad-anch{fill:var(--accent)}
    .alphafig .ad-e{fill:var(--ink)}
    .alphafig .aax{stroke:var(--hair);stroke-width:1}
    .alphafig .aerrline{fill:none;stroke:var(--ink2);stroke-width:2}
  </style>"""


def svg():
    hi, lo = solve(1.0), solve(0.0)
    n1 = (f'the whole relation is priced, so covering is dear: the matcher buys only cells '
          f'{hi["span"][0]}–{hi["span"][1]} of 0–{N_B - 1}, runs {tuple(hi["lens"])}')
    n2 = (f'only the anchors are priced: every vertex takes the cell beneath it, runs '
          f'{tuple(lo["lens"])}, gaps filled free')
    alt = (f"Two relation panels and an error curve. At alpha = 1 the matcher covers only cells "
           f"{hi['span'][0]} to {hi['span'][1]} and anchors the first source vertex 4 m from where "
           f"it is. At alpha = 0 every source vertex anchors on the cell directly beneath it, with "
           f"even runs of five. The third panel plots the worst anchor error against alpha: 4 m at "
           f"alpha = 1, falling to 0.")
    return f"""<svg class="alphafig" viewBox="0 0 {W} {H}" role="img" aria-label="{alt}">
  {CSS}
{relation_panel(1.0, 14, 'α = 1 — the whole relation is charged', n1)}
{relation_panel(0.0, 170, 'α → 0 — only the anchors are charged', n2)}
{error_panel(326)}
</svg>"""


if __name__ == "__main__":
    out = Path(__file__).with_name("_alpha_fig.svg")
    out.write_text(svg())
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
