"""Generate the paper's Figure 1 (§9.5) as inline SVG, straight from the real matchings.

Nothing here is drawn by hand: every dot, every matching line and every cost point is computed by
running the shipped engine (prepare -> forward -> extract_cell) on the same geometry the worked
example describes, so the picture cannot drift from the table beside it.

    python scripts/beta_shift_figure.py            # writes scripts/_beta_fig.svg and prints it
"""
import math
from pathlib import Path

from beta_shift_demo import LAT, N, R, SHIFT, STEP, case, crossover, overrun, solve

from network_matching.dag_dtw import digraph, extract_cell, forward, prepare, _cost_of

W, H = 720, 470
X0, X1 = 58, 690                      # plot box in px
MIN_M, MAX_M = -1.0, 30.0             # metres spanned by the road panels


def mx(m):
    """metres along the road -> px"""
    return X0 + (m - MIN_M) * (X1 - X0) / (MAX_M - MIN_M)


def road_panel(beta, top, title, note):
    """One panel: source vertices (upper row), target cells (lower row), matched pairs as lines."""
    A, B = case()
    ay, by = top + 34, top + 92
    M = solve(beta)["M"]
    # a target cell claimed by more than one source vertex is an N:1 stall -- draw those in accent
    claims = {}
    for a, v in M:
        claims.setdefault(v, []).append(a)

    stall_free = all(len(c) == 1 for c in claims.values())
    s = [f'<text class="ft" x="{X0}" y="{top + 4}">{title}</text>',
         f'<text class="fn" x="{X0}" y="{top + 20}">{note}</text>']
    for a, v in sorted(M):
        piled = len(claims[v]) > 1
        cls = "fl-stall" if piled else ("fl-11" if stall_free else "fl")
        s.append(f'<line class="{cls}" '
                 f'x1="{mx(A.nodes[a]["x"]):.1f}" y1="{ay}" '
                 f'x2="{mx(B.nodes[v]["x"]):.1f}" y2="{by}"/>')
    for a in A.nodes:
        s.append(f'<circle class="fd-a" cx="{mx(A.nodes[a]["x"]):.1f}" cy="{ay}" r="3.4"/>')
    for v in B.nodes:
        n = len(claims.get(v, []))
        cls = "fd-b-hot" if n > 1 else ("fd-b" if n == 1 else "fd-b-idle")   # hollow = never matched
        s.append(f'<circle class="{cls}" cx="{mx(B.nodes[v]["x"]):.1f}" cy="{by}" '
                 f'r="{4.6 if n > 1 else 3.4}"/>')
    idle = sum(1 for v in B.nodes if v not in claims)
    if idle:
        s.append(f'<text class="fk" x="{X1}" y="{by + 24}" text-anchor="end">'
                 f'{idle} target cells left unmatched (hollow)</text>')
    s.append(f'<text class="fk" x="{X0 - 8}" y="{ay + 4}" text-anchor="end">A</text>')
    s.append(f'<text class="fk" x="{X0 - 8}" y="{by + 4}" text-anchor="end">B</text>')
    return "\n".join(s)


def cost_panel(top):
    """C(M) against beta for the two competing readings. Both lines are computed by re-pricing a
    FIXED matching through the ledger (_cost_of), so the crossing is the real one: the 1:1 reading
    has no stall cell and is flat; the collapse that survives to compete is affine and rising."""
    A, B = case()
    prepare(A, B, r=R)
    one2one = {(i, f"b{i}") for i in range(N)}
    collapse = solve(5.0)["M"]                  # the 4-stall pile -- the reading that competes at beta*
    bstar = crossover()

    def c_flat(b): return _cost_of(A, B, one2one, 1.0, b)    # no stall, no cover -> weight-free
    def c_stall(b): return _cost_of(A, B, collapse, 1.0, b)  # affine: slope = sum of stall E's

    py0, py1 = top + 34, top + 150
    cy0, cy1 = 20.0, 135.0
    def px(b): return X0 + (b - 1.0) * (X1 - X0) / 7.0
    def py(c): return py1 - (c - cy0) * (py1 - py0) / (cy1 - cy0)

    s = [f'<text class="ft" x="{X0}" y="{top + 4}">Why it flips — a flat line and a rising one</text>',
         f'<text class="fn" x="{X0}" y="{top + 20}">cost C(M) of each reading, re-priced through '
         f'the ledger at every β</text>',
         f'<line class="fax" x1="{X0}" y1="{py1}" x2="{X1}" y2="{py1}"/>',
         f'<line class="fax" x1="{X0}" y1="{py0}" x2="{X0}" y2="{py1}"/>']
    for b in (1, 2, 3, 4, 5, 6, 7, 8):
        s.append(f'<line class="fax" x1="{px(b):.1f}" y1="{py1}" x2="{px(b):.1f}" y2="{py1 + 4}"/>')
        s.append(f'<text class="fk" x="{px(b):.1f}" y="{py1 + 16}" text-anchor="middle">{b}</text>')
    for c in (40, 60, 80, 100, 120):
        s.append(f'<text class="fk" x="{X0 - 8}" y="{py(c) + 3.5:.1f}" text-anchor="end">{c}</text>')
    s.append(f'<text class="fk" x="{X0 - 40}" y="{(py0 + py1) / 2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 {X0 - 40} {(py0 + py1) / 2:.0f})">C(M)</text>')

    # the collapsed reading: solid where it is the optimum, dashed once it has been beaten
    s.append(f'<line class="fc-stall" x1="{px(1.0):.1f}" y1="{py(c_stall(1.0)):.1f}" '
             f'x2="{px(bstar):.1f}" y2="{py(c_stall(bstar)):.1f}"/>')
    s.append(f'<line class="fc-stall-out" x1="{px(bstar):.1f}" y1="{py(c_stall(bstar)):.1f}" '
             f'x2="{px(8.0):.1f}" y2="{py(c_stall(8.0)):.1f}"/>')
    # the 1:1 reading: dashed while it is still too expensive, solid once it wins
    s.append(f'<line class="fc-flat-out" x1="{X0}" y1="{py(c_flat(1.0)):.1f}" '
             f'x2="{px(bstar):.1f}" y2="{py(c_flat(1.0)):.1f}"/>')
    s.append(f'<line class="fc-flat" x1="{px(bstar):.1f}" y1="{py(c_flat(1.0)):.1f}" '
             f'x2="{X1}" y2="{py(c_flat(1.0)):.1f}"/>')
    # the crossing
    s.append(f'<line class="fx" x1="{px(bstar):.1f}" y1="{py0}" x2="{px(bstar):.1f}" y2="{py1}"/>')
    s.append(f'<circle class="fd-x" cx="{px(bstar):.1f}" cy="{py(c_flat(1.0)):.1f}" r="4.5"/>')
    s.append(f'<text class="fk2" x="{px(bstar) + 9:.1f}" y="{py0 + 11:.1f}">β* = {bstar:.4f}</text>')
    s.append(f'<text class="fk2" x="{X1}" y="{py(c_flat(1.0)) - 14:.1f}" text-anchor="end">'
             f'1:1 — no stall cell, flat at {c_flat(1.0):.2f}</text>')
    s.append(f'<text class="fk3" x="{px(3.35):.1f}" y="{py(46):.1f}">'
             f'collapsed — stalls re-priced βE, so it rises</text>')
    s.append(f'<text class="fk" x="{(X0 + X1) / 2:.0f}" y="{py1 + 34}" text-anchor="middle">'
             f'stall penalty β</text>')
    return "\n".join(s)


# One colour per READING, used consistently in all three panels: warm = the collapse (the N:1
# pile), accent-blue = the shifted 1:1 reading. --warn is local to the figure; it does not touch
# the paper's :root palette.
CSS = """<style>
    .betafig{width:100%;height:auto;font-family:var(--sans);--warn:#b0492c}
    .betafig .ft{font-size:12.5px;font-weight:700;fill:var(--ink)}
    .betafig .fn{font-size:11px;fill:var(--muted)}
    .betafig .fk{font-size:10.5px;fill:var(--muted)}
    .betafig .fk2{font-size:10.5px;font-weight:700;fill:var(--accent)}
    .betafig .fk3{font-size:10.5px;font-weight:700;fill:var(--warn)}
    .betafig .fl{stroke:var(--muted);stroke-width:1.3;opacity:.5}
    .betafig .fl-stall{stroke:var(--warn);stroke-width:1.7;opacity:.9}
    .betafig .fl-11{stroke:var(--accent);stroke-width:1.5;opacity:.75}
    .betafig .fd-a{fill:var(--ink2)}
    .betafig .fd-b{fill:var(--muted)}
    .betafig .fd-b-hot{fill:var(--warn)}
    .betafig .fd-b-idle{fill:none;stroke:var(--muted);stroke-width:1.2;opacity:.5}
    .betafig .fax{stroke:var(--hair);stroke-width:1}
    .betafig .fc-stall{fill:none;stroke:var(--warn);stroke-width:2.2}
    .betafig .fc-stall-out{stroke:var(--warn);stroke-width:1.4;stroke-dasharray:4 4;opacity:.4}
    .betafig .fc-flat{stroke:var(--accent);stroke-width:2.2}
    .betafig .fc-flat-out{stroke:var(--accent);stroke-width:1.4;stroke-dasharray:4 4;opacity:.45}
    .betafig .fx{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3;opacity:.5}
    .betafig .fd-x{fill:var(--ink)}
  </style>"""


def svg():
    lo, hi = solve(1.0), solve(8.0)
    note_lo = (f'{lo["pile"]} source vertices pile onto B’s first cell · '
               f'{lo["correct"]}/{N} pairs correct · drift {lo["drift"]:.2f} m')
    note_hi = (f'1:1, no stall anywhere · {hi["correct"]}/{N} pairs correct · '
               f'drift {hi["drift"]:.2f} m (= the shift)')
    alt = (f"Three panels. At beta = 1, {lo['pile']} source vertices collapse onto the target's "
           f"first cell and only {lo['correct']} of {N} pairs are correct. At beta = 8 the same "
           f"source matches the target one-to-one, {hi['correct']} of {N} correct. The third "
           f"panel plots the cost of both readings against beta: the one-to-one reading is flat, "
           f"the collapsed reading rises, and they cross at beta* = {crossover():.2f}.")
    return f"""<svg class="betafig" viewBox="0 0 {W} {H}" role="img" aria-label="{alt}">
  {CSS}
{road_panel(1.0, 14, 'β = 1 — the collapse', note_lo)}
{road_panel(8.0, 168, 'β = 8 — the shifted reading', note_hi)}
{cost_panel(300)}
</svg>"""


if __name__ == "__main__":
    out = Path(__file__).with_name("_beta_fig.svg")
    out.write_text(svg())
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
