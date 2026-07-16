# Open defect: validity-blind cheapest-per-signature contraction in `extract_cell`

Found 2026-07-13 by adversarial differential review of the §4.1a role-aware forbidding change.

**The defect** (pre-existing; affects role-blind AND role-aware enforcement): `extract_cell` keeps
only the CHEAPEST row per pending-signature (per_sig contraction, root fold) before the terminal
`check_rules` judge. A cheap-but-invalid row (e.g. one whose run occupies a flagged/crossing cell
on a cyclic target and trips V1) can EVICT the valid-but-costlier row that shares its signature;
the judge then rejects the sole survivor and `extract_cell` raises a spurious
"no valid root row" — or returns a displaced optimum — on inputs where a valid matching exists.

Measured on 900 random tree-onto-cyclic-B cases (generator in `regress_hunt.py`, seeds 500-799,
three weightings): role-aware arm 18 spurious raises + 1 displaced optimum; role-blind arm
22 raises + 3 displaced. The role change is a net improvement but introduced 2/800
old-ok -> new-raise regressions (generator seeds 21 and 86; dissect21.py / judge86.py are the
standalone dissections). Requires a B-cycle — which real road networks always have.

**Fix direction**: make the contraction validity-aware, or keep top-K rows per signature so the
judge has fallbacks (K=2-3 likely suffices; measure).
