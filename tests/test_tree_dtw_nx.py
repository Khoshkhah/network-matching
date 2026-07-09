"""Tests for the networkx tree-DTW rebuild (`tree_dtw_nx`), focused on §6b cross-table agreement:
the forward table `D` and backward table `B` must agree on the one optimum -- every source edge the
forward back-pointers thread, the backward ones thread back (`check_reciprocity`)."""
import networkx as nx
import pytest

from network_matching.tree_dtw_nx import (digraph, line_digraph, prepare, forward, backward,
                                          extract, check_reciprocity, check_rules, _advance_anchor)


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
