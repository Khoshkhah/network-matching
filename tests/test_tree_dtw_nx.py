"""Tests for the networkx tree-DTW rebuild (`tree_dtw_nx`): §6b cross-table agreement
(`check_reciprocity`) and §6c per-table source<->sink reachability (`check_reachability`)."""
import math

import networkx as nx
import pytest

from network_matching.tree_dtw_nx import (digraph, line_digraph, prepare, forward, backward,
                                          extract, check_reciprocity, check_reachability,
                                          check_forward_v3, check_backward_v2, check_rules, _advance_anchor)


def make(name):
    """Fresh (A, B) DiGraphs per call -- prepare/forward/backward mutate them in place."""
    if name == "chain":
        return (digraph({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)]),
                digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")]))
    if name == "split":
        return (digraph({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)}, [(0, 1), (1, 2), (1, 3)]),
                digraph({"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
                        [("s", "j"), ("j", "u"), ("j", "d")]))
    if name == "merge":
        return (digraph({0: (0, 6), 1: (0, -6), 2: (10, 0), 3: (20, 0)}, [(0, 2), (1, 2), (2, 3)]),
                digraph({"a": (0, 6.5), "b": (0, -5.5), "m": (10, .5), "o": (20, .5)},
                        [("a", "m"), ("b", "m"), ("m", "o")]))
    raise KeyError(name)


def _match(A, B, r=20.0, alpha=1.0, beta=1.0):
    prepare(A, B, r=r)
    forward(A, B, alpha=alpha, beta=beta)
    backward(A, B, alpha=alpha, beta=beta)
    return extract(A, B)


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5)])
def test_reciprocity_holds_point(name, alpha, beta):
    """On the extracted matching the two tables agree (docs §6b) -- for every scenario and weighting."""
    A, B = make(name)
    M, committed = _match(A, B, alpha=alpha, beta=beta)
    assert check_reciprocity(A, committed) == [], f"{name} a={alpha} b={beta}: tables disagree"
    # sanity: the matching itself is a legal warping
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("bw", [0.0, 1.0, 3.0])
def test_reciprocity_holds_segment(name, bw):
    """Same agreement in segment mode -- the identical check on the line-graph tables."""
    A, B = make(name)
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0, bearing_weight=bw)
    forward(LA, LB); backward(LA, LB)
    M, committed = extract(LA, LB)
    assert check_reciprocity(LA, committed) == [], f"{name} bw={bw}: segment tables disagree"


def test_reciprocity_has_teeth():
    """Severing one backward advance pointer at a split must be CAUGHT (negative control)."""
    A, B = make("split")
    M, committed = _match(A, B)
    assert check_reciprocity(A, committed) == []
    t1 = _advance_anchor(A, 1, committed[1], "bpB")       # vertex 1 is the split; it feeds successors 2, 3
    A.nodes[1]["cand"][t1]["bpB"] = [(s, w) for (s, w) in A.nodes[1]["cand"][t1]["bpB"] if s != 2]
    bad = check_reciprocity(A, committed)
    assert bad and any(edge[:2] == (1, 2) for edge in bad), bad


def test_reciprocity_on_coverage_run():
    """A genuine 1:N COVER run: predecessors connect at the run START, successors at the run END --
    a fabricated chain (coverage never fires on tiny clean geometry) exercises the anchor walk."""
    A = nx.DiGraph(); A.add_edge(0, 1)
    A.nodes[0]["cand"] = {                                  # source 0 covers the run b0 -> b1 -> b2
        "b0": {"E": 0, "D": 0, "bpD": [],           "B": 2, "bpB": [(0, "b1")]},
        "b1": {"E": 0, "D": 1, "bpD": [(0, "b0")],  "B": 1, "bpB": [(0, "b2")]},
        "b2": {"E": 0, "D": 2, "bpD": [(0, "b1")],  "B": 0, "bpB": [(1, "c9")]},   # run end feeds succ 1
    }
    A.nodes[1]["cand"] = {"c9": {"E": 0, "D": 9, "bpD": [(0, "b2")], "B": 0, "bpB": []}}
    committed = {0: "b0", 1: "c9"}                          # 0 pinned at the run START
    assert _advance_anchor(A, 0, "b0", "bpD") == "b0"       # fwd anchor = run start
    assert _advance_anchor(A, 0, "b0", "bpB") == "b2"       # bwd anchor = run end
    assert check_reciprocity(A, committed) == []            # 0->1 threaded at the run-end b2, reciprocally
    A.nodes[0]["cand"]["b2"]["bpB"] = []                    # sever 0's continuation into 1
    assert check_reciprocity(A, committed)                  # ... and it is caught


# ---------------------------------------------------------------------------------------------------
# §6c per-table reachability: each table's back-pointers reconstruct the tree's source<->sink structure
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.2, 0.2)])
def test_reachability_holds_point(name, alpha, beta):
    """Every finite sink cell's bpD reaches exactly its ancestor sources, and every finite source cell's
    bpB reaches exactly its descendant sinks -- for every scenario and weighting (docs §6c)."""
    A, B = make(name)
    prepare(A, B, r=20.0); forward(A, B, alpha=alpha, beta=beta); backward(A, B, alpha=alpha, beta=beta)
    assert check_reachability(A, "D") == [], f"{name} a={alpha} b={beta}: forward reachability broken"
    assert check_reachability(A, "B") == [], f"{name} a={alpha} b={beta}: backward reachability broken"


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("bw", [0.0, 3.0])
def test_reachability_holds_segment(name, bw):
    """Same reachability soundness on the line-graph tables (segment mode)."""
    A, B = make(name)
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0, bearing_weight=bw); forward(LA, LB); backward(LA, LB)
    assert check_reachability(LA, "D") == [] and check_reachability(LA, "B") == []


def test_reachability_has_teeth():
    """A severed back-pointer (a None cell reference) on a finite endpoint cell must be caught, both
    directions -- a sink whose bpD can no longer reach its sources / a source whose bpB can't reach its
    sinks."""
    A, B = make("merge")                                    # sources 0,1 ; sink 3 ; merge at 2
    prepare(A, B, r=20.0); forward(A, B); backward(A, B)
    assert check_reachability(A, "D") == [] and check_reachability(A, "B") == []

    for v, c in A.nodes[3]["cand"].items():                # forward: break a finite sink cell's bpD
        if not math.isinf(c["D"]) and c["bpD"]:
            c["bpD"] = [(c["bpD"][0][0], None)]; break
    assert check_reachability(A, "D"), "severed forward path not caught"

    A2, B2 = make("merge")
    prepare(A2, B2, r=20.0); forward(A2, B2); backward(A2, B2)
    for v, c in A2.nodes[0]["cand"].items():               # backward: break a finite source cell's bpB
        if not math.isinf(c["B"]) and c["bpB"]:
            c["bpB"] = [(c["bpB"][0][0], None)]; break
    assert check_reachability(A2, "B"), "severed backward path not caught"


def test_extract_raises_feasibility_not_keyerror():
    """A merge whose branches can't co-reach within r must raise the feasibility ValueError -- not crash
    with KeyError on a None back-pointer (the coupled-infeasibility guard in extract)."""
    A = digraph({0: (26.65, 8.0), 1: (17.66, 17.2), 2: (6.86, 10.31)}, [(0, 1), (2, 1)])   # merge at 1
    B = digraph({"b0": (19.95, 17.41), "b1": (1.14, 15.82), "b2": (12.09, 1.99),
                 "b3": (3.67, -1.07), "b4": (5.08, 4.52), "b5": (9.15, -1.29)},
                [("b0", "b1"), ("b0", "b3"), ("b1", "b2"), ("b1", "b3"), ("b3", "b4"), ("b3", "b5"), ("b4", "b5")])
    prepare(A, B, r=5.0); forward(A, B); backward(A, B)
    with pytest.raises(ValueError):
        extract(A, B)


def test_coverage_gap_fill_backward_run():
    """A 1:N run recorded on the BACKWARD cover chain must be materialised (§8.6 gap-fill), not dropped:
    a chain 0->1->2 at alpha=0.5 where node 2 covers b3->b5 -- b3 is recorded backward. extract() now
    fills the gap (2, b3) instead of leaving a V2/V3 hole."""
    import random
    A = digraph({0: (1.34, 12.48), 1: (16.92, 1.96), 2: (0.9, 18.8)}, [(0, 1), (1, 2)])
    r = random.Random(127 + 90001)
    Bn = {f"b{i}": (round(r.uniform(-3, 33), 2), round(r.uniform(-3, 23), 2)) for i in range(7)}
    B = digraph(Bn, [("b0", "b1"), ("b1", "b2"), ("b1", "b3"), ("b3", "b5"), ("b4", "b6")])
    prepare(A, B, r=40.0); forward(A, B, alpha=0.5, beta=1.0); backward(A, B, alpha=0.5, beta=1.0)
    M, _ = extract(A, B)
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3), f"dropped-coverage gap not filled: V2={v2} V3={v3}"
    assert (2, "b3") in M                                  # the previously-dropped cell is now covered


def test_argmin_tie_break_deterministic():
    """§4b: advance-argmin ties are broken by a fixed B-vertex order -- the smaller-border cell wins,
    and the forward/backward tables are invariant to B's dict/insertion order."""
    A = digraph({0: (0, 0), 1: (0, 0), 2: (5, 0)}, [(0, 2), (1, 2)])          # merge at 2, preds coincident
    B = digraph({"b0": (0, 0), "b1": (0, 0), "m": (5, 0)}, [("b0", "m"), ("b1", "m")])   # b0,b1 tie as pred cells
    prepare(A, B, r=10.0); forward(A, B); backward(A, B)
    assert all(x == "b0" for (_p, x) in A.nodes[2]["cand"]["m"]["bpD"])       # smaller-border cell (b0) wins

    A2 = digraph({0: (0, 0), 1: (0, 0), 2: (5, 0)}, [(0, 2), (1, 2)])
    B2 = digraph({"m": (5, 0), "b1": (0, 0), "b0": (0, 0)}, [("b0", "m"), ("b1", "m")])   # reversed insertion
    prepare(A2, B2, r=10.0); forward(A2, B2); backward(A2, B2)
    for a in A.nodes:
        for v in A.nodes[a]["cand"]:
            assert A.nodes[a]["cand"][v]["bpD"] == A2.nodes[a]["cand"][v]["bpD"]
            assert A.nodes[a]["cand"][v]["bpB"] == A2.nodes[a]["cand"][v]["bpB"]


def test_forward_v3_backward_v2_coupling():
    """§6d: the forward table couples merges (V2), not splits -- read alone it CAN violate V3 (and the
    backward table mirror-violates V2). Clean α=β=1 inputs don't; a weighted split does."""
    A = digraph({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)])                # chain, no split/merge
    B = digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")])
    prepare(A, B, r=20.0); forward(A, B); backward(A, B)
    assert check_forward_v3(A, B) == [] and check_backward_v2(A, B) == []

    A2 = digraph({0: (6.73, 18.65), 1: (28.37, 0.46), 2: (25.41, 14.29)}, [(1, 0), (1, 2)])   # split at 1
    B2 = digraph({"b0": (10.71, 2.92), "b1": (30.12, 17.11), "b2": (-1.67, 12.48)}, [("b1", "b2")])
    prepare(A2, B2, r=40.0); forward(A2, B2, alpha=0.2, beta=0.2); backward(A2, B2, alpha=0.2, beta=0.2)
    assert check_forward_v3(A2, B2)                        # split vertex 1 lands on two cells -> V3 violation
