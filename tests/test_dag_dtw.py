"""Tests for DAG-DTW (`dag_dtw`): §6b cross-table agreement (`check_reciprocity`)
and §6c per-table source<->sink reachability (`check_reachability`)."""
import math
from collections import Counter

import networkx as nx
import pytest

from network_matching.dag_dtw import (digraph, line_digraph, prepare, forward, backward, match_dag, NotADAG,
                                       extract_join, extract_cell, extract_two_table,
                                       check_reciprocity, check_reachability, check_forward_v3,
                                       check_backward_v2, check_rules, check_split_exits,
                                       layer_order, _advance_anchor, _cost_of,
                                       _reconstruct_from_sinks, INF)


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
    return extract_two_table(A, B)


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
    M, committed = extract_two_table(LA, LB)
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
    with KeyError on a None back-pointer (the coupled-infeasibility guard in the two-table traceback)."""
    A = digraph({0: (26.65, 8.0), 1: (17.66, 17.2), 2: (6.86, 10.31)}, [(0, 1), (2, 1)])   # merge at 1
    B = digraph({"b0": (19.95, 17.41), "b1": (1.14, 15.82), "b2": (12.09, 1.99),
                 "b3": (3.67, -1.07), "b4": (5.08, 4.52), "b5": (9.15, -1.29)},
                [("b0", "b1"), ("b0", "b3"), ("b1", "b2"), ("b1", "b3"), ("b3", "b4"), ("b3", "b5"), ("b4", "b5")])
    prepare(A, B, r=5.0); forward(A, B); backward(A, B)
    with pytest.raises(ValueError):
        extract_two_table(A, B)


def test_coverage_gap_fill_backward_run():
    """A 1:N run recorded on the BACKWARD cover chain must be materialised (the §6b gap-fill), not
    dropped: a chain 0->1->2 at alpha=0.5 where node 2 covers b3->b5 -- b3 is recorded backward.
    extract_two_table() fills the gap (2, b3) instead of leaving a V2/V3 hole."""
    import random
    A = digraph({0: (1.34, 12.48), 1: (16.92, 1.96), 2: (0.9, 18.8)}, [(0, 1), (1, 2)])
    r = random.Random(127 + 90001)
    Bn = {f"b{i}": (round(r.uniform(-3, 33), 2), round(r.uniform(-3, 23), 2)) for i in range(7)}
    B = digraph(Bn, [("b0", "b1"), ("b1", "b2"), ("b1", "b3"), ("b3", "b5"), ("b4", "b6")])
    prepare(A, B, r=40.0); forward(A, B, alpha=0.5, beta=1.0); backward(A, B, alpha=0.5, beta=1.0)
    M, _ = extract_two_table(A, B)
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
    """§6d diagnostics: on a clean chain both per-table reads are consistent; on the weighted split the
    per-sink forward read still diverges -- across the split's MULTIPLE surviving options, which is
    legitimate (§4.1a) and exactly why check_forward_v3 is a diagnostic, not an invariant."""
    A = digraph({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)])                # chain, no split/merge
    B = digraph({"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")])
    prepare(A, B, r=20.0); forward(A, B); backward(A, B)
    assert check_forward_v3(A, B) == [] and check_backward_v2(A, B) == []

    A2 = digraph({0: (6.73, 18.65), 1: (28.37, 0.46), 2: (25.41, 14.29)}, [(1, 0), (1, 2)])   # split at 1
    B2 = digraph({"b0": (10.71, 2.92), "b1": (30.12, 17.11), "b2": (-1.67, 12.48)}, [("b1", "b2")])
    prepare(A2, B2, r=40.0); forward(A2, B2, alpha=0.2, beta=0.2); backward(A2, B2, alpha=0.2, beta=0.2)
    assert check_forward_v3(A2, B2)            # per-sink reads land on two (both-valid) surviving exits


# ---------------------------------------------------------------------------------------------------
# §4.0 longest-path layer order + §4.1a forward V3 coupling (forbid-and-rebuild) -- check_split_exits
# ---------------------------------------------------------------------------------------------------
def _weighted_split():
    """The §6d case where the PLAIN forward table violates V3 at α=β=0.2 (split at vertex 1)."""
    A = digraph({0: (6.73, 18.65), 1: (28.37, 0.46), 2: (25.41, 14.29)}, [(1, 0), (1, 2)])
    B = digraph({"b0": (10.71, 2.92), "b1": (30.12, 17.11), "b2": (-1.67, 12.48)}, [("b1", "b2")])
    return A, B


def test_layer_order_longest_path_layering():
    """§4.0: L(v) = longest source->v depth; the order is topological; a split's children share one
    layer ahead of all their successors; a merge lands after ALL its branches (the doc's example)."""
    A = nx.DiGraph()
    A.add_edges_from([("S", "a1"), ("a1", "J"), ("J", "b1"), ("J", "b2"), ("b2", "c1"),
                      ("b1", "M"), ("c1", "M"), ("M", "d1"), ("d1", "T")])
    order, L = layer_order(A)
    assert L == {"S": 0, "a1": 1, "J": 2, "b1": 3, "b2": 3, "c1": 4, "M": 5, "d1": 6, "T": 7}
    pos = {v: i for i, v in enumerate(order)}
    assert all(pos[u] < pos[v] for u, v in A.edges)             # a valid topological order
    assert L["b1"] == L["b2"]                                   # the split's children share a layer
    assert max(pos["b1"], pos["b2"]) < min(pos["c1"], pos["M"])  # ... ahead of every successor


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.2, 0.2)])
def test_split_exits_invariant_point(name, alpha, beta):
    """§4.1a: after the forward pass, every surviving split exit is USABLE by all children (feasibility,
    not the rows' optimal links) and the survivor set is non-empty -- for every scenario and weighting."""
    A, B = make(name)
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    assert check_split_exits(A, B) == [], f"{name} a={alpha} b={beta}"


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.2, 0.2)])
def test_split_exits_invariant_segment(alpha, beta):
    """The identical algorithm on the line-graph (segment mode) satisfies the same invariant."""
    A, B = make("split")
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    assert check_split_exits(LA, LB) == []


def test_split_exits_invariant_and_teeth():
    """The §4.1a invariant holds after the forward pass -- here every cell of the split is usable by
    both children (each can β-stall on any of them), so the feasibility coupling forbids NOTHING and
    all three cells survive (multiple options are legitimate; the extraction chooses). Negative
    control: on the private-entry-trap fixture the coupling forbids the children's mutually-unusable
    private cells -- wiping the ``forbidden`` flags (= pretending the coupling never ran) must make
    the invariant fire, since those cells then "survive" while a sibling cannot use them."""
    A, B = _weighted_split()
    prepare(A, B, r=40.0); forward(A, B, alpha=0.2, beta=0.2)
    assert check_split_exits(A, B) == []
    surv = {v for v, c in A.nodes[1]["cand"].items() if not c["forbidden"]}
    assert surv == {"b0", "b1", "b2"}                           # all usable (stall) -> all survive

    A2 = digraph({"a0": (0, 0), "a1": (10, 0), "S": (20, 0), "b1": (30, 5), "d1": (30, -6)},
                 [("a0", "a1"), ("a1", "S"), ("S", "b1"), ("S", "d1")])
    B2 = digraph({"m0": (0, 1), "m1": (10, 1), "Q1": (20, 1), "Q2": (20, -1), "X": (20, 4),
                  "w1": (30, 5), "w2": (30, -6)},
                 [("m0", "m1"), ("m1", "Q1"), ("m1", "Q2"), ("m1", "X"),
                  ("Q1", "w1"), ("X", "w1"), ("X", "w2"), ("Q2", "w2")])
    prepare(A2, B2, r=10.0); forward(A2, B2)
    assert check_split_exits(A2, B2) == []
    for c in A2.nodes["S"]["cand"].values():
        c["forbidden"] = False                                  # undo the coupling's verdict
    assert check_split_exits(A2, B2), "invariant must fire once the coupling flags are wiped"



def test_forward_raises_when_no_shared_exit():
    """§4.1a feasibility: children forced onto disjoint target chains leave the split with no exit
    every child can use -> ValueError (increase match_radius_m), never a silently-broken table."""
    A = digraph({"J": (0, 0), "b1": (5, 1), "b2": (5, -1)}, [("J", "b1"), ("J", "b2")])
    B = digraph({"p0": (0, 1), "p1": (5, 1), "q0": (0, -1), "q1": (5, -1)},
                [("p0", "p1"), ("q0", "q1")])
    prepare(A, B, r=1.5)
    with pytest.raises(ValueError, match="no surviving V3 exit"):
        forward(A, B)


def test_forward_requires_subdivision():
    """§4.0: a split whose children span layers (one child is also fed by a deeper branch) has no
    grouped sibling order -- rejected with the add-an-interior-point message."""
    A = digraph({"s": (0, 0), "c1": (1, 1), "c2": (5, -1),
                 "q0": (1, -3), "q1": (2, -3), "q2": (3, -3), "q3": (4, -3)},
                [("s", "c1"), ("s", "c2"), ("q0", "q1"), ("q1", "q2"), ("q2", "q3"), ("q3", "c2")])
    B = digraph({"x": (0, 0)}, [])
    prepare(A, B, r=100.0)
    with pytest.raises(ValueError, match="not subdivided"):
        forward(A, B)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_forward_pipeline_with_two_table_extraction(alpha, beta):
    """the forward pass composes with the unchanged backward + two-table traceback: the committed
    matching is a legal warping and the split commits to a surviving (non-forbidden) cell."""
    A, B = make("split")
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    backward(A, B, alpha=alpha, beta=beta)
    M, committed = extract_two_table(A, B)
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)
    assert not A.nodes[1]["cand"][committed[1]]["forbidden"]    # split pinned on a surviving exit


# ---------------------------------------------------------------------------------------------------
# Cross-table validation (forward -> backward under the flags -> extract_two_table):
# §6b reciprocity and §6c reachability must survive the V3 coupling.
# ---------------------------------------------------------------------------------------------------
def _rand_tree_case(seed):
    """Random out-tree A (splits, no merges — every split's children share a layer) over a random,
    possibly CYCLIC target B. Deterministic per seed."""
    import random
    rng = random.Random(seed)
    na, nb = rng.randint(3, 7), rng.randint(5, 10)
    A = nx.DiGraph()
    for i in range(na):
        A.add_node(i, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(1, na):
        A.add_edge(rng.randrange(i), i)
    B = nx.DiGraph()
    vs = [f"v{i}" for i in range(nb)]
    for v in vs:
        B.add_node(v, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(nb - 1):
        B.add_edge(vs[i], vs[i + 1])
    for _ in range(rng.randint(1, nb // 2)):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return A, B


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5)])
def test_v3_pipeline_cross_table_agreement(name, alpha, beta):
    """The v3 pipeline satisfies the SAME cross-table guarantees the plain pipeline is tested for:
    §6b reciprocity on the committed matching, §6c reachability of both tables, a legal final M, and
    the §4.1a split invariant — for every scenario and suite weighting."""
    A, B = make(name)
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    backward(A, B, alpha=alpha, beta=beta)                      # canonical order: backward under the flags
    M, committed = extract_two_table(A, B)
    assert check_reciprocity(A, committed) == [], f"{name} a={alpha} b={beta}: tables disagree"
    assert check_reachability(A, "D") == [] and check_reachability(A, "B") == []
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)
    assert check_split_exits(A, B) == []


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.2, 0.2)])
def test_v3_reachability_and_invariant_random_sweep(alpha, beta):
    """§6c reachability is UNAFFECTED by the V3 coupling (0 failures over the probe's 170-case sweep):
    both tables' back-pointers still reconstruct exactly the tree's source<->sink structure, and the
    split invariant holds — on random out-trees over random (cyclic) targets, incl. harsh weights."""
    for seed in range(20):
        A, B = _rand_tree_case(seed)
        prepare(A, B, r=40.0)
        forward(A, B, alpha=alpha, beta=beta)
        backward(A, B, alpha=alpha, beta=beta)
        assert check_reachability(A, "D") == [], f"seed {seed}: forward reachability broken"
        assert check_reachability(A, "B") == [], f"seed {seed}: backward reachability broken"
        assert check_split_exits(A, B) == [], f"seed {seed}: split invariant broken"



# ---------------------------------------------------------------------------------------------------
# The judge on adversarial targets + located feasibility (docs §5.3, §7)
# ---------------------------------------------------------------------------------------------------
def test_two_cycle_judge_prefers_valid():
    """The SMALLEST adversarial input: a 2-vertex chain over a 2-cycle target (|A|=2, |B|=2). On p⇄q
    the local V1 predicate cannot orient a step, so EVERY matching separating the two vertices —
    including the geometrically nicest {(0,p),(1,q)} — is flagged (docs §7). NO engine ever returns
    an invalid matching: the cell join's root contraction leaves only the flagged row, so it REFUSES
    (loud ValueError); the vertex join returns a VALID stall (both vertices on one cell); the
    cross-validating engine='all' therefore still succeeds."""
    A = digraph({0: (0, 0), 1: (10, 0)}, [(0, 1)])
    B = digraph({"p": (0, 1), "q": (10, 1)}, [("p", "q"), ("q", "p")])
    prepare(A, B, r=20.0)
    forward(A, B)
    with pytest.raises(ValueError):                             # refusal, never an invalid return
        extract_cell(A, B)
    M, _ = extract_join(A, B)
    assert not any(check_rules(M, A, B))                        # the valid stall
    M2, _ = match_dag(A, B, r=20.0, engine="all")               # cross-validation saves the case
    assert not any(check_rules(M2, A, B))
    assert check_rules({(0, "p"), (1, "q")}, A, B)[0]           # the separating matching is flagged ...
    assert check_rules({(0, "q"), (1, "p")}, A, B)[0]           # ... and so is its mirror: 2-cycle limit


def test_extract_cell_raises_when_infeasible():
    """A vertex whose row is all-infinite (gate severed between two far-apart target chains) has no
    surviving cell -> the located feasibility ValueError (docs §5.2), never a broken matching."""
    A = digraph({0: (0, 0), 1: (100, 0)}, [(0, 1)])
    B = digraph({"c0": (0, 1), "c1": (1, 1), "d0": (100, 1), "d1": (101, 1)},
                [("c0", "c1"), ("d0", "d1")])
    prepare(A, B, r=5.0)
    forward(A, B)
    with pytest.raises(ValueError):
        extract_cell(A, B)


# ---------------------------------------------------------------------------------------------------
# The vertex-level junction join (docs/dag_dtw_matching.md §10) -- the cross-validation engine
# ---------------------------------------------------------------------------------------------------
IN_DOMAIN = [(1.0, 1.0), (0.5, 1.0), (0.3, 1.5), (1.0, 2.0)]


def _merge_shape():
    """The canonical shape U -> x -> m <- z <- V with sinks below m and on both other branches."""
    A = digraph({"sU": (0, 10), "U": (6, 10), "x": (12, 8), "sV": (0, -10), "V": (6, -10),
                 "z": (12, -8), "m": (18, 0), "d1": (24, 0), "T": (30, 0),
                 "y1": (12, 16), "T2": (18, 16), "w1": (12, -16), "T3": (18, -16)},
                [("sU", "U"), ("U", "x"), ("x", "m"), ("sV", "V"), ("V", "z"), ("z", "m"),
                 ("m", "d1"), ("d1", "T"), ("U", "y1"), ("y1", "T2"), ("V", "w1"), ("w1", "T3")])
    B = digraph({"BsU": (0, 10.5), "BU": (6, 10.5), "Bx": (12, 8.5), "BsV": (0, -9.5),
                 "BV": (6, -9.5), "Bz": (12, -7.5), "Bm": (18, .5), "Bd": (24, .5),
                 "BT": (30, .5), "By": (12, 16.5), "BT2": (18, 16.5), "Bw": (12, -15.5),
                 "BT3": (18, -15.5)},
                [("BsU", "BU"), ("BU", "Bx"), ("Bx", "Bm"), ("BsV", "BV"), ("BV", "Bz"),
                 ("Bz", "Bm"), ("Bm", "Bd"), ("Bd", "BT"), ("BU", "By"), ("By", "BT2"),
                 ("BV", "Bw"), ("Bw", "BT3")])
    return A, B


def _rand_polytree_case(seed):
    """Random undirected tree -> random orientation -> subdivide every edge: a subdivided polytree
    with natural splits AND merges, over a random (often cyclic) target."""
    import random
    rng = random.Random(seed)
    n = rng.randint(4, 7)
    und = [(rng.randrange(i), i) for i in range(1, n)]
    pos = {i: (rng.uniform(0, 30), rng.uniform(0, 30)) for i in range(n)}
    A = nx.DiGraph()
    for i in range(n):
        A.add_node(i, x=pos[i][0], y=pos[i][1])
    for k, (a, b) in enumerate(und):
        if rng.random() < 0.5:
            a, b = b, a
        mid = f"m{k}"
        A.add_node(mid, x=(pos[a][0] + pos[b][0]) / 2, y=(pos[a][1] + pos[b][1]) / 2)
        A.add_edge(a, mid)
        A.add_edge(mid, b)
    B = nx.DiGraph()
    nb = rng.randint(6, 10)
    vs = [f"v{i}" for i in range(nb)]
    for v in vs:
        B.add_node(v, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(nb - 1):
        B.add_edge(vs[i], vs[i + 1])
    for _ in range(rng.randint(1, nb // 2)):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return A, B


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", IN_DOMAIN)
def test_extract_join_scenarios_valid(name, alpha, beta):
    """extract_join returns a legal, fully-covering, deterministic matching on every scenario and
    in-domain weighting."""
    A, B = make(name)
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    M, committed = extract_join(A, B, alpha=alpha, beta=beta)
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)
    assert set(committed) == set(A.nodes)
    M2, c2 = extract_join(A, B, alpha=alpha, beta=beta)
    assert M2 == M and c2 == committed


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.3, 1.5)])
def test_extract_join_segment_mode(alpha, beta):
    """The identical join on the line graphs (segment mode)."""
    A, B = make("split")
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    M, committed = extract_join(LA, LB, alpha=alpha, beta=beta)
    v1, v2, v3 = check_rules(M, LA, LB)
    assert not (v1 or v2 or v3)
    assert set(committed) == set(LA.nodes)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0)])
def test_extract_join_exact_on_merge_shape(alpha, beta):
    """The consumed-once merge arithmetic: on the canonical U->x->m<-z<-V shape the join equals the
    brute-force optimum over all sink-label combinations (docs/dag_dtw_matching.md §10.1)."""
    import itertools
    A, B = _merge_shape()
    prepare(A, B, r=40.0)
    forward(A, B, alpha=alpha, beta=beta)
    Mj, _ = extract_join(A, B, alpha=alpha, beta=beta)
    sinks = [n for n in A.nodes if A.out_degree(n) == 0]
    options = [[v for v, c in A.nodes[s]["cand"].items()
                if not c.get("forbidden") and c["D"] < INF] for s in sinks]
    best = None
    for combo in itertools.product(*options):
        got = _reconstruct_from_sinks(A, dict(zip(sinks, combo)))
        if got is None:
            continue
        M, _pin = got
        if {a for a, _ in M} != set(A.nodes) or any(check_rules(M, A, B)):
            continue
        cost = _cost_of(A, B, M, alpha, beta)
        if best is None or cost < best - 1e-12:
            best = cost
    assert best is not None
    assert abs(_cost_of(A, B, Mj, alpha, beta) - best) < 1e-6


# ---------------------------------------------------------------------------------------------------
# The cell-level join -- THE extraction (docs §5) + the standing cross-validation (docs §10.2)
# ---------------------------------------------------------------------------------------------------
def _full_space_brute(A, B, alpha, beta, run_cap=3, cap=400_000):
    # cap = suite-runtime guard on the enumeration size, not a correctness bound. Raised from 200k:
    # the §4.1a feasibility coupling forbids fewer cells than the old optimality rule, so the pinned
    # divergence case's full space grew to 234,256 options (see test_dag_couple_feasibility.py).
    """Ground truth over the FULL space: every (entry, run) combination per vertex, judged by
    check_rules, costed by C(M). Tiny cases only."""
    import itertools
    def runs(a, e):
        cand = A.nodes[a]["cand"]
        out, stack = [], [(e,)]
        while stack:
            path = stack.pop()
            out.append(path)
            if len(path) > run_cap:
                continue
            for w in B.successors(path[-1]):
                if w in cand and not cand[w].get("forbidden") and w not in path:
                    stack.append(path + (w,))
        return out
    per_vertex = []
    for a in A.nodes:
        opts = [r for e, c in A.nodes[a]["cand"].items()
                if not c.get("forbidden") and c["D"] < INF for r in runs(a, e)]
        if not opts:
            return None
        per_vertex.append(opts)
    n = 1
    for o in per_vertex:
        n *= len(o)
        assert n <= cap, "brute too big for a suite test"
    best, verts = None, list(A.nodes)
    for combo in itertools.product(*per_vertex):
        M = {(a, v) for a, run in zip(verts, combo) for v in run}
        if any(check_rules(M, A, B)):
            continue
        cost = _cost_of(A, B, M, alpha, beta)
        if best is None or cost < best - 1e-12:
            best = cost
    return best


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", IN_DOMAIN)
def test_extract_cell_scenarios_valid(name, alpha, beta):
    """extract_cell returns a legal, fully-covering, deterministic matching on every scenario and
    in-domain weighting."""
    A, B = make(name)
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    M, committed = extract_cell(A, B, alpha=alpha, beta=beta)
    assert not any(check_rules(M, A, B))
    assert set(committed) == set(A.nodes)
    M2, c2 = extract_cell(A, B, alpha=alpha, beta=beta)
    assert M2 == M and c2 == committed


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.3, 1.5)])
def test_extract_cell_segment_mode(alpha, beta):
    A, B = make("split")
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    M, committed = extract_cell(LA, LB, alpha=alpha, beta=beta)
    assert not any(check_rules(M, LA, LB))
    assert set(committed) == set(LA.nodes)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (0.3, 1.5)])
def test_extract_cell_equals_full_space_brute(alpha, beta):
    """On a tiny dense-B chain (the coverage regime), extract_cell equals the FULL-SPACE brute-force
    optimum -- all entry+run combinations, not just the sink-label family."""
    A = digraph({0: (0, 0), 1: (9, 0), 2: (18, 0)}, [(0, 1), (1, 2)])
    B = digraph({f"b{i}": (3 * i, .4) for i in range(7)},
                [(f"b{i}", f"b{i+1}") for i in range(6)])
    prepare(A, B, r=25.0)
    forward(A, B, alpha=alpha, beta=beta)
    M, _ = extract_cell(A, B, alpha=alpha, beta=beta)
    bf = _full_space_brute(A, B, alpha, beta, run_cap=3)
    assert bf is not None
    assert abs(_cost_of(A, B, M, alpha, beta) - bf) < 1e-6


def test_extract_cell_beats_vertex_join_on_divergence():
    """The pinned divergence case (docs §10.2): on this dense-B split, the cell join is strictly
    cheaper than the vertex-level join, equals the full-space optimum, and stays valid."""
    A = digraph({0: (0, 0), 1: (7, 0.371), 2: (14, 4.368), 3: (14, -3.301)},
                [(0, 1), (1, 2), (1, 3)])
    B = digraph({"b0": (0.0, 0.743), "b1": (2.2, 0.766), "b2": (4.4, 1.091), "b3": (6.6, 0.726),
                 "b4": (8.8, -0.0), "b5": (11.0, -0.187), "b6": (13.2, 1.053)},
                [(f"b{i}", f"b{i+1}") for i in range(6)])
    prepare(A, B, r=20.0)
    forward(A, B, alpha=0.5, beta=1.0)
    Mc, _ = extract_cell(A, B, alpha=0.5, beta=1.0)
    Mv, _ = extract_join(A, B, alpha=0.5, beta=1.0)
    cc = _cost_of(A, B, Mc, 0.5, 1.0)
    cv = _cost_of(A, B, Mv, 0.5, 1.0)
    assert cc < cv - 1e-6                                       # strictly cheaper than the vertex join
    bf = _full_space_brute(A, B, 0.5, 1.0, run_cap=3)
    assert abs(cc - bf) < 1e-6                                  # and equal to the full-space optimum
    assert not any(check_rules(Mc, A, B))


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_cross_validation(alpha, beta):
    """The standing harness (docs §10.2): whenever an engine succeeds it is valid, and the cell join
    is never costlier than the vertex join -- no coverage exception (it is exact over the full
    space). Random subdivided polytrees over often-cyclic targets."""
    for seed in range(14):
        A, B = _rand_polytree_case(seed)
        prepare(A, B, r=40.0)
        forward(A, B, alpha=alpha, beta=beta)
        results = {}
        for tag, fn in (("cell", extract_cell), ("vtx", extract_join)):
            try:
                M, _ = fn(A, B, alpha, beta)
                assert not any(check_rules(M, A, B)), f"seed {seed}: {tag} returned invalid M"
                results[tag] = _cost_of(A, B, M, alpha, beta)
            except ValueError:
                results[tag] = None
        if results["cell"] is not None and results["vtx"] is not None:
            assert results["cell"] <= results["vtx"] + 1e-6, \
                f"seed {seed}: cell join costlier than the vertex join -- exactness bug"


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_cross_validation_segment(name, alpha, beta):
    """SEGMENT mode: the same harness on the line graphs -- every returning engine is valid on
    L(A)/L(B), and the cell join is never costlier than the vertex join."""
    A, B = make(name)
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0, bearing_weight=2.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    results = {}
    for tag, fn in (("cell", extract_cell), ("vtx", extract_join)):
        try:
            M, _ = fn(LA, LB, alpha, beta)
            assert not any(check_rules(M, LA, LB)), f"{name} {tag}: invalid segment M"
            assert {a for a, _ in M} == set(LA.nodes)
            results[tag] = _cost_of(LA, LB, M, alpha, beta)
        except ValueError:
            results[tag] = None
    assert results["cell"] is not None, f"{name}: cell join infeasible in segment mode"
    if results["vtx"] is not None:
        assert results["cell"] <= results["vtx"] + 1e-6, \
            f"{name}: segment cell join costlier than the vertex join -- exactness bug"


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (0.3, 1.5)])
def test_extract_cell_segment_equals_full_space_brute(alpha, beta):
    """SEGMENT mode ground truth: on a dense-B chain lifted to the line graphs (bearing term
    active), extract_cell equals the FULL-SPACE brute force over all arc entry+run combinations."""
    A = digraph({0: (0, 0), 1: (9, 0), 2: (18, 0)}, [(0, 1), (1, 2)])
    B = digraph({f"b{i}": (3 * i, .4) for i in range(7)},
                [(f"b{i}", f"b{i+1}") for i in range(6)])
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=25.0, bearing_weight=2.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    M, _ = extract_cell(LA, LB, alpha=alpha, beta=beta)
    bf = _full_space_brute(LA, LB, alpha, beta, run_cap=3)
    assert bf is not None
    assert abs(_cost_of(LA, LB, M, alpha, beta) - bf) < 1e-6


@pytest.mark.parametrize("engine", ["cell", "join", "all"])
def test_match_dag_pipeline_wrapper(engine):
    """The one-call pipeline entry: prepare -> forward -> extraction, every engine, both modes."""
    A, B = make("split")
    M, com = match_dag(A, B, r=20.0, engine=engine)
    assert not any(check_rules(M, A, B))
    assert set(com) == set(A.nodes)

    A2, B2 = make("split")
    M_seg, com_seg = match_dag(A2, B2, r=20.0, mode="segment", engine=engine, bearing_weight=2.0)
    assert all(isinstance(a, tuple) and isinstance(v, tuple) for a, v in M_seg)   # arcs = edge tuples
    assert len(com_seg) == A2.number_of_edges()


# ---------------------------------------------------------------------------------------------------
# DAG sources (accepted by default): the cell engine is exact on subdivided reconvergent DAGs (docs §7, §10.2)
# ---------------------------------------------------------------------------------------------------
def _diamond_case(shift=0.4, jitter=0.0, seed=0):
    """Subdivided diamond: S→s1→J→{x,z}→m→t1→T (J splits, m reconverges) over a congruent
    (optionally jittered) target."""
    import random
    rng = random.Random(seed)
    An = {"S": (0, 0), "s1": (4, 0), "J": (8, 0), "x": (12, 3), "z": (12, -3),
          "m": (16, 0), "t1": (20, 0), "T": (24, 0)}
    Ae = [("S", "s1"), ("s1", "J"), ("J", "x"), ("J", "z"), ("x", "m"), ("z", "m"),
          ("m", "t1"), ("t1", "T")]
    Bn = {k + "'": (v[0] + (rng.uniform(-jitter, jitter) if jitter else 0.0),
                    v[1] + shift + (rng.uniform(-jitter, jitter) if jitter else 0.0))
          for k, v in An.items()}
    return digraph(An, Ae), digraph(Bn, [(a + "'", b + "'") for a, b in Ae])


def test_dag_source_gate():
    """A reconvergent (diamond) source is accepted by DEFAULT -- DAG-DTW's source is any subdivided
    DAG; only a directed cycle is rejected (NotADAG)."""
    A, B = _diamond_case()
    prepare(A, B, r=6.0)                                        # diamond: no raise
    C = nx.DiGraph()
    for n, (x, y) in {0: (0, 0), 1: (1, 0)}.items():
        C.add_node(n, x=x, y=y)
    C.add_edge(0, 1); C.add_edge(1, 0)                          # directed cycle
    with pytest.raises(NotADAG):
        prepare(C, B, r=6.0)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_extract_cell_exact_on_dag_diamonds(alpha, beta):
    """Jittered diamonds: extract_cell equals the FULL-SPACE brute-force optimum on reconvergent
    sources -- the tree property is not needed by the cell engine."""
    for seed in range(6):
        A, B = _diamond_case(jitter=1.2, seed=seed)
        prepare(A, B, r=4.0)
        forward(A, B, alpha=alpha, beta=beta)
        M, com = extract_cell(A, B, alpha=alpha, beta=beta)
        assert not any(check_rules(M, A, B))
        assert set(com) == set(A.nodes)
        bf = _full_space_brute(A, B, alpha, beta, run_cap=1)
        assert bf is not None
        assert abs(_cost_of(A, B, M, alpha, beta) - bf) < 1e-6, f"seed {seed}"


def test_extract_cell_exact_on_random_reconvergent_dags():
    """Random subdivided polytrees + one reconvergent (subdivided) edge, over target chains:
    extract_cell == full-space brute force, valid, every case."""
    import random
    ran = 0
    for seed in range(30):
        rng = random.Random(seed)
        n = 3
        und = [(rng.randrange(i), i) for i in range(1, n)]
        pos = {i: (rng.uniform(0, 14), rng.uniform(0, 14)) for i in range(n)}
        A = nx.DiGraph()
        for i in range(n):
            A.add_node(i, x=pos[i][0], y=pos[i][1])
        for k, (a, b) in enumerate(und):
            if rng.random() < 0.5:
                a, b = b, a
            mid = f"m{k}"
            A.add_node(mid, x=(pos[a][0] + pos[b][0]) / 2, y=(pos[a][1] + pos[b][1]) / 2)
            A.add_edge(a, mid); A.add_edge(mid, b)
        topo = list(nx.topological_sort(A))
        pairs = [(u, v) for i, u in enumerate(topo) for v in topo[i + 1:]
                 if nx.has_path(A, u, v) and not A.has_edge(u, v)]
        if not pairs:
            continue
        u, v = rng.choice(pairs)
        A.add_node("rc", x=(A.nodes[u]["x"] + A.nodes[v]["x"]) / 2,
                   y=(A.nodes[u]["y"] + A.nodes[v]["y"]) / 2)
        A.add_edge(u, "rc"); A.add_edge("rc", v)
        if not nx.is_directed_acyclic_graph(A) or nx.is_forest(A.to_undirected()):
            continue                                            # want a genuine reconvergence
        B = nx.DiGraph()
        vs = [f"v{i}" for i in range(4)]
        for w in vs:
            B.add_node(w, x=rng.uniform(0, 14), y=rng.uniform(0, 14))
        for i in range(3):
            B.add_edge(vs[i], vs[i + 1])
        try:
            prepare(A, B, r=20.0)
            forward(A, B, 0.5, 1.0)
        except ValueError:
            continue
        M, _ = extract_cell(A, B, 0.5, 1.0)
        assert not any(check_rules(M, A, B)), f"seed {seed}"
        bf = _full_space_brute(A, B, 0.5, 1.0, run_cap=1)
        assert bf is not None and abs(_cost_of(A, B, M, 0.5, 1.0) - bf) < 1e-6, f"seed {seed}"
        ran += 1
    assert ran >= 15, f"only {ran} reconvergent cases exercised -- generator drifted"


def test_match_dag_dag_pipeline():
    """The one-call pipeline on a DAG source: point mode (cell default), engine='all', and segment
    mode (the line graph of a diamond is itself reconvergent -- reconvergent line graphs accepted natively)."""
    A, B = _diamond_case()
    M, com = match_dag(A, B, r=6.0)
    assert not any(check_rules(M, A, B))
    assert set(com) == set(A.nodes)
    A2, B2 = _diamond_case()
    M2, _ = match_dag(A2, B2, r=6.0, engine="all")
    assert not any(check_rules(M2, A2, B2))
    A3, B3 = _diamond_case()
    Ms, cs = match_dag(A3, B3, r=6.0, mode="segment", bearing_weight=2.0)
    assert all(isinstance(a, tuple) and isinstance(v, tuple) for a, v in Ms)
    assert len(cs) == A3.number_of_edges()


def test_match_dag_duckdb_pipeline(tmp_path):
    """Mode-3 I/O parity: the DuckDBMapMatcher inputs (lon/lat WKT CSVs, utm_srid projection) feed
    match_dag directly -- conversion to networkx happens inside -- and the outputs are the
    Mode-1-style (long, summary) DataFrames. Source is a reconvergent DAG (diamond)."""
    import csv
    import math
    from network_matching import DuckDBMapMatcher
    LON0, LAT0 = 18.06, 59.33
    mx = 1.0 / (111320 * math.cos(math.radians(LAT0)))
    my = 1.0 / 111320.0

    def ls(pts):
        return "LINESTRING (" + ", ".join(f"{LON0 + x * mx:.8f} {LAT0 + y * my:.8f}"
                                          for x, y in pts) + ")"

    A_rows = [("a_stem", ls([(0, 0), (30, 0)])), ("a_up", ls([(30, 0), (55, 12)])),
              ("a_dn", ls([(30, 0), (55, -12)])), ("a_up2", ls([(55, 12), (80, 0)])),
              ("a_dn2", ls([(55, -12), (80, 0)])), ("a_out", ls([(80, 0), (110, 0)]))]
    B_rows = [("b_stem", ls([(0, 3), (28, 3)])), ("b_up", ls([(28, 3), (55, 15), (80, 3)])),
              ("b_dn", ls([(28, 3), (55, -9), (80, 3)])), ("b_out", ls([(80, 3), (110, 3)]))]
    for name, rows in (("a.csv", A_rows), ("b.csv", B_rows)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["edge_id", "geometry"])
            w.writerows(rows)
    m = DuckDBMapMatcher.from_wkt_csv(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
                                      id_a="edge_id", id_b="edge_id", utm_srid=3006,
                                      max_distance=15, id_cast=None)
    dag_long, dag_summary = m.match_dag(alpha=0.5, beta=1.5, engine="all", step_meters=5.0)
    assert set(dag_summary.columns) >= {"source_id", "dest_ids", "n_dest", "n_pairs",
                                        "avg_dist_m", "match_type"}
    assert set(dag_long.columns) >= {"source_id", "dest_id", "seq", "n_pairs", "avg_dist_m"}
    assert set(dag_summary.source_id) == {r[0] for r in A_rows}          # every A-edge matched
    by = dict(zip(dag_summary.source_id, dag_summary.dest_ids))
    assert by["a_stem"] == "b_stem" and by["a_out"] == "b_out"
    assert by["a_up"] == by["a_up2"] == "b_up"                           # different segmentation
    assert by["a_dn"] == by["a_dn2"] == "b_dn"
    assert (dag_summary.avg_dist_m < 6.0).all()                          # ~3 m shift + geometry


# ---------------------------------------------------------------------------------------
# The stall penalty beta on a rigidly-shifted source (paper 9.5; scripts/beta_shift_demo.py)
# ---------------------------------------------------------------------------------------
def _shift_case(shift=9.0, lat=0.5, n=11, step=2.0):
    """A chain, and the SAME chain rigidly shifted along the direction of travel."""
    A = digraph({i: (step * i, 0.0) for i in range(n)}, [(i, i + 1) for i in range(n - 1)])
    B = digraph({f"b{i}": (step * i + shift, lat) for i in range(n)},
                [(f"b{i}", f"b{i + 1}") for i in range(n - 1)])
    return A, B


def _solve_shift(beta, alpha=1.0, r=15.0):
    A, B = _shift_case()
    prepare(A, B, r=r)
    forward(A, B, alpha=alpha, beta=beta)
    M, _ = extract_cell(A, B, alpha, beta)
    per_b = Counter(v for (_a, v) in M)
    return M, sum(c - 1 for c in per_b.values() if c > 1), max(per_b.values())


def test_beta_converts_n_to_1_collapse_into_1_to_1_under_shift():
    """beta = 1: the cheapest-emission matching COLLAPSES the source's head onto B's first cell
    -- the plain nearest-cell reading -- and agrees with the rigid-shift correspondence on 1 pair
    of 11. Raising beta prices that stall until the stall-free 1:1 reading wins: 11/11, no pile.
    The pile's exact DEPTH is not pinned: at beta = 1 the six- and five-deep piles cost the same
    (a5 sits midway between b0 and b1), so the depth is an argmin tie-break, not a fact."""
    M1, stalls1, pile1 = _solve_shift(beta=1.0)
    assert stalls1 >= 4 and pile1 >= 5                       # the head collapsed onto one cell
    assert sum(1 for (a, v) in M1 if v == f"b{a}") <= 1      # ... and the correspondence is wrong

    M8, stalls8, pile8 = _solve_shift(beta=8.0)
    assert stalls8 == 0 and pile8 == 1                       # no N:1 anywhere
    assert M8 == {(i, f"b{i}") for i in range(11)}           # exactly the rigid-shift reading

    # the 1:1 matching has no stall cell AND no coverage cell, so neither weight can touch it
    A, B = _shift_case()
    prepare(A, B, r=15.0)                                    # _cost_of reads the E table
    flat = pytest.approx(11 * math.hypot(9.0, 0.5), abs=1e-6)
    assert _cost_of(A, B, M8, 1.0, 6.0) == flat
    assert _cost_of(A, B, M8, 1.0, 99.0) == flat
    assert _cost_of(A, B, M8, 0.5, 99.0) == flat


def test_beta_can_destroy_a_correct_overhang():
    """The warning the worked example carries (paper 9.5.1): beta does NOT dominate beta = 1.
    Where the source genuinely starts before its target -- and the target still has spare cells --
    the head stall is CORRECT, and a beta above the local crossover destroys it, pairing the
    source's head 1:1 onto cells it never reaches. Spatial overrun does not force a stall."""
    A = digraph({i: (2.0 * i, 0.0) for i in range(6)}, [(i, i + 1) for i in range(5)])
    B = digraph({f"b{j}": (6.0 + 2.0 * j, 0.5) for j in range(13)},
                [(f"b{j}", f"b{j + 1}") for j in range(12)])

    def solve(beta):
        A2 = digraph({i: (2.0 * i, 0.0) for i in range(6)}, [(i, i + 1) for i in range(5)])
        B2 = digraph({f"b{j}": (6.0 + 2.0 * j, 0.5) for j in range(13)},
                     [(f"b{j}", f"b{j + 1}") for j in range(12)])
        prepare(A2, B2, r=15.0)
        forward(A2, B2, alpha=1.0, beta=beta)
        M, _ = extract_cell(A2, B2, 1.0, beta)
        per_b = Counter(v for (_a, v) in M)
        return M, sum(c - 1 for c in per_b.values() if c > 1)

    M1, stalls1 = solve(1.0)
    assert stalls1 == 3                                      # the true head overhang, read right
    assert {v for (_a, v) in M1} == {"b0", "b1", "b2"}       # A only reaches B's first cells

    M8, stalls8 = solve(8.0)
    assert stalls8 == 0                                      # the overhang is GONE ...
    assert M8 == {(i, f"b{i}") for i in range(6)}            # ... replaced by a spurious 1:1


def test_beta_cannot_beat_the_pigeonhole():
    """What genuinely forces a stall is CARDINALITY, not extent: with more source vertices than
    reachable target cells, no beta removes the pile (paper 9.5.1)."""
    def solve(beta):
        A = digraph({i: (2.0 * i, 0.0) for i in range(11)}, [(i, i + 1) for i in range(10)])
        B = digraph({f"b{j}": (2.0 * j + 9.0, 0.5) for j in range(5)},      # only 5 cells for 11
                    [(f"b{j}", f"b{j + 1}") for j in range(4)])
        prepare(A, B, r=1e6)                                 # radius cannot be the binding constraint
        forward(A, B, alpha=1.0, beta=beta)
        M, _ = extract_cell(A, B, 1.0, beta)
        per_b = Counter(v for (_a, v) in M)
        return sum(c - 1 for c in per_b.values() if c > 1)

    assert solve(1.0) >= 6 and solve(50.0) >= 6 and solve(1000.0) >= 6


def test_beta_stall_count_is_monotone_and_crosses_over_once():
    """Stalls never increase with beta, and the switch to the 1:1 reading happens once."""
    seen = [_solve_shift(beta=b)[1] for b in (1.0, 2.0, 4.0, 5.0, 5.5, 8.0)]
    assert seen == sorted(seen, reverse=True)                # monotone non-increasing
    assert seen[0] > 0 and seen[-1] == 0                     # collapse -> stall-free


# ---------------------------------------------------------------------------------------
# The coverage discount alpha changes the matching RELATION (paper 9.5.2;
# scripts/alpha_density_demo.py)
# ---------------------------------------------------------------------------------------
def _finer_case(n_a=6, a_step=5.0, b_step=1.0, lat=0.5):
    """ONE road, sampled 5x finer on the B side. 1:N coverage is FORCED by the density -- it
    happens at every alpha. What alpha changes is WHICH cells each source vertex takes."""
    n_b = int(a_step * (n_a - 1) / b_step) + 1
    A = digraph({i: (a_step * i, 0.0) for i in range(n_a)}, [(i, i + 1) for i in range(n_a - 1)])
    B = digraph({f"b{j}": (b_step * j, lat) for j in range(n_b)},
                [(f"b{j}", f"b{j + 1}") for j in range(n_b - 1)])
    return A, B


def _runs_at(alpha, r=8.0):
    A, B = _finer_case()
    prepare(A, B, r=r)
    forward(A, B, alpha=alpha, beta=1.0)
    M, _ = extract_cell(A, B, alpha, 1.0)
    runs = {}
    for a, v in M:
        runs.setdefault(a, []).append(int(v[1:]))
    return A, B, M, {a: sorted(js) for a, js in runs.items()}


def test_coverage_is_forced_by_density_at_every_alpha():
    """The premise: B being finer than A forces 1:N coverage. alpha does not switch it on."""
    for alpha in (1.0, 0.5, 0.0):
        _A, _B, _M, runs = _runs_at(alpha)
        assert max(len(js) for js in runs.values()) > 1, f"no coverage at alpha={alpha}"


def test_alpha_changes_the_matching_relation_not_just_its_cost():
    """alpha = 1 prices the WHOLE relation, so covering is dear: the matcher shrinks the covered
    span and drags each source vertex's anchor (its run's entry) away from where the vertex is.
    alpha -> 0 prices only the anchors, so every vertex takes the cell directly beneath it. The
    two RELATIONS differ -- this is not a cost-only effect."""
    _A, _B, M1, runs1 = _runs_at(1.0)
    _A0, _B0, M0, runs0 = _runs_at(0.0)
    assert M1 != M0                                          # the relation itself changed

    # alpha = 1: the covered span is truncated at both ends, and anchors are off by up to 4 m
    covered1 = sorted({j for js in runs1.values() for j in js})
    assert covered1[0] > 0 and covered1[-1] < 25             # B's ends are not bought
    assert [len(js) for js in (runs1[a] for a in sorted(runs1))] == [1, 3, 5, 5, 3, 1]
    assert max(abs(runs1[a][0] * 1.0 - a * 5.0) for a in runs1) == pytest.approx(4.0)

    # alpha -> 0: every vertex anchors on its own cell, runs are even, the whole road is covered
    assert [runs0[a][0] for a in sorted(runs0)] == [0, 5, 10, 15, 20, 25]
    assert [len(js) for js in (runs0[a] for a in sorted(runs0))] == [5, 5, 5, 5, 5, 1]
    assert max(abs(runs0[a][0] * 1.0 - a * 5.0) for a in runs0) == pytest.approx(0.0)


def test_alpha_zero_objective_is_entry_only():
    """At alpha = 0 the ledger keeps only the entries, so the cost is exactly the sum over source
    vertices of the cheapest reachable cell: one 1:1 anchor each, gaps filled at zero cost."""
    A, B, M, _runs = _runs_at(0.0)
    best = sum(min(math.dist((A.nodes[a]["x"], A.nodes[a]["y"]), (B.nodes[v]["x"], B.nodes[v]["y"]))
                   for v in A.nodes[a]["cand"]) for a in A.nodes)
    assert _cost_of(A, B, M, 0.0, 1.0) == pytest.approx(best, abs=1e-9)
    assert best == pytest.approx(6 * 0.5, abs=1e-9)          # 6 vertices, each 0.5 m off its cell


# ---------------------------------------------------------------------------------------
# The weight family has TWO degrees of freedom, not three (paper 8.3, eq. "gauge")
# ---------------------------------------------------------------------------------------
def _decompose(A, B, M):
    """Sums of E by the move that ENTERS each cell -- the same ledger _cost_of applies:
    the run's entry pays 1 (advance / free entry) or beta (stall); the rest pay alpha."""
    rows = {}
    for a, v in M:
        rows.setdefault(a, set()).add(v)
    s_adv = s_stall = s_cover = 0.0
    for a, cells in rows.items():
        cand = A.nodes[a]["cand"]
        entries = sorted((v for v in cells if not any(x in cells for x in B.predecessors(v))),
                         key=str)
        entry = entries[0] if entries else sorted(cells, key=str)[0]
        if any(entry in rows.get(p, ()) for p in A.predecessors(a)):
            s_stall += cand[entry]["E"]
        else:
            s_adv += cand[entry]["E"]
        s_cover += sum(cand[v]["E"] for v in cells if v != entry)
    return s_adv, s_stall, s_cover


@pytest.mark.parametrize("gamma,beta,alpha", [(2.0, 3.0, 0.5), (0.4, 1.2, 0.2), (7.0, 9.0, 3.0)])
def test_a_third_weight_on_the_advance_is_a_gauge_freedom(gamma, beta, alpha):
    """C(M; gamma, beta, alpha) == gamma * C(M; 1, beta/gamma, alpha/gamma) for every M, so a
    weight on the advance cannot change the argmin: the family has two degrees of freedom, not
    three, and fixing the advance at 1 is a normalisation (paper 8.3)."""
    A, B = _finer_case()
    prepare(A, B, r=8.0)
    for al, be in ((1.0, 1.0), (0.3, 1.0), (1.0, 5.0), (0.1, 2.0)):
        A2, B2 = _finer_case()
        prepare(A2, B2, r=8.0)
        forward(A2, B2, alpha=al, beta=be)
        M, _ = extract_cell(A2, B2, al, be)

        s_adv, s_stall, s_cover = _decompose(A, B, M)
        weighted = gamma * s_adv + beta * s_stall + alpha * s_cover        # the hypothetical 3-weight cost
        normalised = gamma * _cost_of(A, B, M, alpha / gamma, beta / gamma)
        assert weighted == pytest.approx(normalised, abs=1e-9)
