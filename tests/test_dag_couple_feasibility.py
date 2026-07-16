"""§4.1a regression — the split coupling must forbid by FEASIBILITY, never by OPTIMALITY.

Distilled from a real OSM↔DRA conflation failure (Expo Blvd / Griffiths Way, Vancouver): a split
whose children each own a cheap *private* entry cell, while the single cell valid for BOTH children
is slightly more expensive. The old coupling read ``links(child)`` off the child's optimal row
(best back-pointers only), so the first child forbade the shared cell; the forbidden set is
monotone, the shared cell could never return, the exits emptied out, and ``forward()`` raised
``no surviving V3 exit`` — although a valid V3 warping exists (spec §4.1a claimed that raise means
none exists; this fixture is the counterexample). The fix forbids an exit only when a child has NO
legal transition out of it (stall or advance) — soundness the extraction's pruning contract needs.
"""
import pytest

from network_matching.dag_dtw import (check_rules, check_split_exits, digraph, forward,
                                      match_dag, prepare)

# NOTE the companion capacity change in test_dag_dtw.py::_full_space_brute: the feasibility
# coupling forbids fewer cells, so the brute-force ground-truth space it enumerates grew past the
# old 200k guard on the pinned divergence case (234,256). The cap is a suite-runtime guard, not a
# correctness bound.


def _private_entry_trap():
    """Split S -> {b1, d1}. cand(S) = {Q1, Q2, X}: Q1 is b1's cheap private entry (Q1->w1 only),
    Q2 is d1's (Q2->w2 only), X — slightly dearer — is the ONLY cell serving both (X->w1, X->w2)."""
    A = digraph({"a0": (0, 0), "a1": (10, 0), "S": (20, 0), "b1": (30, 5), "d1": (30, -6)},
                [("a0", "a1"), ("a1", "S"), ("S", "b1"), ("S", "d1")])
    B = digraph({"m0": (0, 1), "m1": (10, 1), "Q1": (20, 1), "Q2": (20, -1), "X": (20, 4),
                 "w1": (30, 5), "w2": (30, -6)},
                [("m0", "m1"), ("m1", "Q1"), ("m1", "Q2"), ("m1", "X"),
                 ("Q1", "w1"), ("X", "w1"), ("X", "w2"), ("Q2", "w2")])
    return A, B


def test_forward_keeps_the_shared_feasible_exit():
    """forward() must succeed (a valid warping via X exists) and the fixed point must keep exactly
    the cells every child can use: X. The private cells are genuinely unusable by the sibling
    (Q1 cannot reach w2, Q2 cannot reach w1) — forbidding them is sound; forbidding X was the bug."""
    A, B = _private_entry_trap()
    prepare(A, B, r=10.0)
    forward(A, B)                                   # the old coupling raised here
    surv = {v for v, c in A.nodes["S"]["cand"].items() if not c["forbidden"]}
    assert surv == {"X"}
    assert check_split_exits(A, B) == []


def test_match_dag_extracts_through_the_shared_exit():
    """End to end: the extraction threads the split through X and both children continue —
    a legal warping by the §3 rules."""
    A, B = _private_entry_trap()
    M, committed = match_dag(A, B, r=10.0, mode="point", engine="cell")
    assert ("S", "X") in M and ("b1", "w1") in M and ("d1", "w2") in M
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)


def test_truly_disjoint_children_still_raise():
    """The feasibility raise keeps its teeth: children forced onto disjoint target chains leave no
    exit ANY child can use -> ValueError (this is test_forward_raises_when_no_shared_exit's case,
    re-pinned here so the semantics change cannot silently widen feasibility)."""
    A = digraph({"J": (0, 0), "b1": (5, 1), "b2": (5, -1)}, [("J", "b1"), ("J", "b2")])
    B = digraph({"p0": (0, 1), "p1": (5, 1), "q0": (0, -1), "q1": (5, -1)},
                [("p0", "p1"), ("q0", "q1")])
    prepare(A, B, r=1.5)
    with pytest.raises(ValueError, match="no surviving V3 exit"):
        forward(A, B)
