"""Tests for tree-DTW (`tree_dtw`): §6b cross-table agreement (`check_reciprocity`)
and §6c per-table source<->sink reachability (`check_reachability`)."""
import math

import networkx as nx
import pytest

from network_matching.tree_dtw import (digraph, line_digraph, prepare, forward, backward, match_tree, NotATree,
                                       extract, extract_join, extract_cell, extract_two_table,
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
    with KeyError on a None back-pointer (the coupled-infeasibility guard in extract)."""
    A = digraph({0: (26.65, 8.0), 1: (17.66, 17.2), 2: (6.86, 10.31)}, [(0, 1), (2, 1)])   # merge at 1
    B = digraph({"b0": (19.95, 17.41), "b1": (1.14, 15.82), "b2": (12.09, 1.99),
                 "b3": (3.67, -1.07), "b4": (5.08, 4.52), "b5": (9.15, -1.29)},
                [("b0", "b1"), ("b0", "b3"), ("b1", "b2"), ("b1", "b3"), ("b3", "b4"), ("b3", "b5"), ("b4", "b5")])
    prepare(A, B, r=5.0); forward(A, B); backward(A, B)
    with pytest.raises(ValueError):
        extract_two_table(A, B)


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
    """§4.1a: after the forward pass, every surviving split exit is linked by ALL children and the survivor
    set is non-empty -- for every scenario and weighting."""
    A, B = make(name)
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    assert check_split_exits(A) == [], f"{name} a={alpha} b={beta}"


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.2, 0.2)])
def test_split_exits_invariant_segment(alpha, beta):
    """The identical algorithm on the line-graph (segment mode) satisfies the same invariant."""
    A, B = make("split")
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    assert check_split_exits(LA) == []


def test_split_exits_invariant_and_teeth():
    """The §4.1a invariant holds after the forward pass -- here leaving TWO surviving exits, both
    linked by both children (multiple options are legitimate). Negative control: wiping the coupling's
    ``forbidden`` flags (= pretending the coupling never ran) must make the invariant fire, since the
    children link only their own winners while every cell then "survives"."""
    A, B = _weighted_split()
    prepare(A, B, r=40.0); forward(A, B, alpha=0.2, beta=0.2)
    assert check_split_exits(A) == []
    surv = {v for v, c in A.nodes[1]["cand"].items() if not c["forbidden"]}
    assert surv == {"b0", "b1"}                                 # multi-exit fixed point, all shared
    for c in A.nodes[1]["cand"].values():
        c["forbidden"] = False                                  # undo the coupling's verdict
    assert check_split_exits(A), "invariant must fire once the coupling flags are wiped"



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
    """the forward pass composes with the unchanged backward + extract: the committed matching is a legal
    warping and the split commits to a surviving (non-forbidden) cell."""
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
    assert check_split_exits(A) == []


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
        assert check_split_exits(A) == [], f"seed {seed}: split invariant broken"



# ---------------------------------------------------------------------------------------------------
# Forward-only anchored extraction (design doc "Fork B realized"): two pointer types, no backward
# table -- anchor = fewest usable cells, per-label flood, reject-and-retry, direct-cost selection.
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.2, 0.2)])
def test_extract_scenarios_valid(name, alpha, beta):
    """extract (forward table only -- no backward pass) returns a legal warping on every
    scenario and weighting; committed cells are never forbidden; the result is deterministic."""
    A, B = make(name)
    prepare(A, B, r=20.0)
    forward(A, B, alpha=alpha, beta=beta)
    M, committed = extract(A, B, alpha=alpha, beta=beta)
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3), f"{name} a={alpha} b={beta}: {v1} {v2} {v3}"
    assert set(committed) == set(A.nodes)                       # full coverage (V4)
    assert all(not A.nodes[a]["cand"][v].get("forbidden") for a, v in committed.items())
    M2, c2 = extract(A, B, alpha=alpha, beta=beta)
    assert M2 == M and c2 == committed                          # deterministic


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.2, 0.2)])
def test_extract_segment_mode(alpha, beta):
    """The identical extraction on the line-graphs (segment mode) -- same code, arc index set."""
    A, B = make("split")
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    M, committed = extract(LA, LB, alpha=alpha, beta=beta)
    v1, v2, v3 = check_rules(M, LA, LB)
    assert not (v1 or v2 or v3)
    assert set(committed) == set(LA.nodes)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.2, 0.2)])
def test_extract_random_sweep(alpha, beta):
    """Random out-trees over random (often cyclic) targets: the label enumeration NEVER exhausts
    (reject-and-retry always finds a workable label), the result NEVER violates V2/V3 (the couplings
    are by construction), and any V1 flag occurs only on a CYCLIC B -- the documented local-predicate
    sensitivity (a 2-cycle makes a legal forward step also readable backward), which the two-table
    extract shares on the same inputs."""
    for seed in range(30):
        A, B = _rand_tree_case(seed)
        prepare(A, B, r=40.0)
        forward(A, B, alpha=alpha, beta=beta)
        M, committed = extract(A, B, alpha=alpha, beta=beta)   # must not raise
        v1, v2, v3 = check_rules(M, A, B)
        assert not v2 and not v3, f"seed {seed}: V2/V3 must be impossible by construction"
        if v1:
            assert not nx.is_directed_acyclic_graph(B), \
                f"seed {seed}: V1 on an ACYCLIC target is a real bug"
        assert set(committed) == set(A.nodes)


def test_two_cycle_judge_prefers_valid():
    """The SMALLEST adversarial input: a 2-vertex chain over a 2-cycle target (|A|=2, |B|=2). On p⇄q
    the local V1 predicate cannot orient a step, so EVERY matching separating the two vertices —
    including the geometrically nicest {(0,p),(1,q)} — is flagged. The judge discards them (validity
    IS the definition of a matching) and returns a VALID candidate instead — here a stall, both
    vertices on one cell. The separating alternative stays flagged, documenting the validator's
    2-cycle limit; the extraction no longer *returns* anything invalid."""
    A = digraph({0: (0, 0), 1: (10, 0)}, [(0, 1)])
    B = digraph({"p": (0, 1), "q": (10, 1)}, [("p", "q"), ("q", "p")])
    prepare(A, B, r=20.0)
    forward(A, B)
    M, _ = extract(A, B)
    v1, v2, v3 = check_rules(M, A, B)
    assert not (v1 or v2 or v3)                                 # the judge only returns VALID matchings
    assert check_rules({(0, "p"), (1, "q")}, A, B)[0]           # the separating matching is flagged ...
    assert check_rules({(0, "q"), (1, "p")}, A, B)[0]           # ... and so is its mirror: 2-cycle limit



def test_extract_raises_when_infeasible_forward_only():
    """A vertex whose row is all-infinite (gate severed between two far-apart target chains) leaves
    the anchor without a usable cell -> ValueError, never a broken matching."""
    A = digraph({0: (0, 0), 1: (100, 0)}, [(0, 1)])
    B = digraph({"c0": (0, 1), "c1": (1, 1), "d0": (100, 1), "d1": (101, 1)},
                [("c0", "c1"), ("d0", "d1")])
    prepare(A, B, r=5.0)
    forward(A, B)
    with pytest.raises(ValueError):
        extract(A, B)


# ---------------------------------------------------------------------------------------------------
# The junction-join extraction (docs/junction_join_extraction.md) + cross-validation of both engines
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
    brute-force optimum over all sink-label combinations (docs/junction_join_extraction.md §5)."""
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


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_extractions_cross_validate(alpha, beta):
    """The two engines check each other: whenever both succeed, both matchings are valid, and the
    join can lose to the branching ONLY through coverage (intra-vertex run alternatives -- outside
    the join's stored-history family). Random subdivided polytrees over often-cyclic targets."""
    for seed in range(16):
        A, B = _rand_polytree_case(seed)
        prepare(A, B, r=40.0)
        forward(A, B, alpha=alpha, beta=beta)
        try:
            Mj, _ = extract_join(A, B, alpha=alpha, beta=beta)
        except ValueError:
            Mj = None
        try:
            Mb, _ = extract(A, B, alpha=alpha, beta=beta)
        except ValueError:
            Mb = None
        if Mj is not None:
            assert not any(check_rules(Mj, A, B)), f"seed {seed}: join returned invalid M"
        if Mb is not None:
            assert not any(check_rules(Mb, A, B)), f"seed {seed}: branching returned invalid M"
        if Mj is not None and Mb is not None:
            cj = _cost_of(A, B, Mj, alpha, beta)
            cb = _cost_of(A, B, Mb, alpha, beta)
            if cj > cb + 1e-6:
                # the join is exact over the STORED-HISTORY family (vertex resolution); branching
                # can only beat it through intra-vertex run alternatives -- so a divergence must
                # involve coverage (docs/junction_join_extraction.md, cell-resolution scope)
                runs = any(len({v for a2, v in M if a2 == a}) > 1
                           for M in (Mj, Mb) for a in {x for x, _ in M})
                assert runs, (f"seed {seed}: join ({cj:.3f}) > branching ({cb:.3f}) "
                              f"WITHOUT coverage -- a real exactness bug")


# ---------------------------------------------------------------------------------------------------
# The cell-level join (docs/junction_join_extraction.md §8) + the three-way cross-validation
# ---------------------------------------------------------------------------------------------------
def _full_space_brute(A, B, alpha, beta, run_cap=3, cap=200_000):
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
    M, _ = extract_cell(A, B, alpha=alpha, beta=beta, run_cap=3)
    bf = _full_space_brute(A, B, alpha, beta, run_cap=3)
    assert bf is not None
    assert abs(_cost_of(A, B, M, alpha, beta) - bf) < 1e-6


def test_extract_cell_beats_vertex_join_on_divergence():
    """The §6a/§8 closure, pinned: on this dense-B split (hunted divergence case), the cell join is
    strictly cheaper than the vertex-level join, equals the full-space optimum, and stays valid."""
    A = digraph({0: (0, 0), 1: (7, 0.371), 2: (14, 4.368), 3: (14, -3.301)},
                [(0, 1), (1, 2), (1, 3)])
    B = digraph({"b0": (0.0, 0.743), "b1": (2.2, 0.766), "b2": (4.4, 1.091), "b3": (6.6, 0.726),
                 "b4": (8.8, -0.0), "b5": (11.0, -0.187), "b6": (13.2, 1.053)},
                [(f"b{i}", f"b{i+1}") for i in range(6)])
    prepare(A, B, r=20.0)
    forward(A, B, alpha=0.5, beta=1.0)
    Mc, _ = extract_cell(A, B, alpha=0.5, beta=1.0, run_cap=3)
    Mv, _ = extract_join(A, B, alpha=0.5, beta=1.0)
    cc = _cost_of(A, B, Mc, 0.5, 1.0)
    cv = _cost_of(A, B, Mv, 0.5, 1.0)
    assert cc < cv - 1e-6                                       # strictly cheaper than the vertex join
    bf = _full_space_brute(A, B, 0.5, 1.0, run_cap=3)
    assert abs(cc - bf) < 1e-6                                  # and equal to the full-space optimum
    assert not any(check_rules(Mc, A, B))


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_three_way_cross_validation(alpha, beta):
    """The standing harness: whenever engines succeed they are valid, and the cell join is never
    costlier than EITHER other engine -- no coverage exception (it is exact over the full space)."""
    for seed in range(14):
        A, B = _rand_polytree_case(seed)
        prepare(A, B, r=40.0)
        forward(A, B, alpha=alpha, beta=beta)
        results = {}
        for tag, fn in (("cell", extract_cell), ("branch", extract), ("vtx", extract_join)):
            try:
                M, _ = fn(A, B, alpha, beta)
                assert not any(check_rules(M, A, B)), f"seed {seed}: {tag} returned invalid M"
                results[tag] = _cost_of(A, B, M, alpha, beta)
            except ValueError:
                results[tag] = None
        if results["cell"] is not None:
            for other in ("branch", "vtx"):
                if results[other] is not None:
                    assert results["cell"] <= results[other] + 1e-6, \
                        f"seed {seed}: cell join costlier than {other} -- exactness bug"


@pytest.mark.parametrize("name", ["chain", "split", "merge"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_three_way_cross_validation_segment(name, alpha, beta):
    """SEGMENT mode: the same three-way harness on the line graphs -- every returning engine is
    valid on L(A)/L(B), and the cell join is never costlier than either other engine."""
    A, B = make(name)
    LA, LB = line_digraph(A), line_digraph(B)
    prepare(LA, LB, r=20.0, bearing_weight=2.0)
    forward(LA, LB, alpha=alpha, beta=beta)
    results = {}
    for tag, fn in (("cell", extract_cell), ("branch", extract), ("vtx", extract_join)):
        try:
            M, _ = fn(LA, LB, alpha, beta)
            assert not any(check_rules(M, LA, LB)), f"{name} {tag}: invalid segment M"
            assert {a for a, _ in M} == set(LA.nodes)
            results[tag] = _cost_of(LA, LB, M, alpha, beta)
        except ValueError:
            results[tag] = None
    assert results["cell"] is not None, f"{name}: cell join infeasible in segment mode"
    for other in ("branch", "vtx"):
        if results[other] is not None:
            assert results["cell"] <= results[other] + 1e-6, \
                f"{name}: segment cell join costlier than {other} -- exactness bug"


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
    M, _ = extract_cell(LA, LB, alpha=alpha, beta=beta, run_cap=3)
    bf = _full_space_brute(LA, LB, alpha, beta, run_cap=3)
    assert bf is not None
    assert abs(_cost_of(LA, LB, M, alpha, beta) - bf) < 1e-6


@pytest.mark.parametrize("engine", ["cell", "branch", "join", "all"])
def test_match_tree_pipeline_wrapper(engine):
    """The one-call pipeline entry: prepare -> forward -> extraction, every engine, both modes."""
    A, B = make("split")
    M, com = match_tree(A, B, r=20.0, engine=engine)
    assert not any(check_rules(M, A, B))
    assert set(com) == set(A.nodes)

    A2, B2 = make("split")
    M_seg, com_seg = match_tree(A2, B2, r=20.0, mode="segment", engine=engine, bearing_weight=2.0)
    assert all(isinstance(a, tuple) and isinstance(v, tuple) for a, v in M_seg)   # arcs = edge tuples
    assert len(com_seg) == A2.number_of_edges()


# ---------------------------------------------------------------------------------------------------
# DAG sources (allow_dag): the cell engine is exact on subdivided reconvergent DAGs (spec §8.6)
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


def test_allow_dag_gate():
    """A reconvergent source is rejected by default (NotATree, with the allow_dag hint) and
    accepted with allow_dag=True; a directed cycle stays rejected either way."""
    A, B = _diamond_case()
    with pytest.raises(NotATree):
        prepare(A, B, r=6.0)
    A2, B2 = _diamond_case()
    prepare(A2, B2, r=6.0, allow_dag=True)                      # no raise
    C = nx.DiGraph()
    for n, (x, y) in {0: (0, 0), 1: (1, 0)}.items():
        C.add_node(n, x=x, y=y)
    C.add_edge(0, 1); C.add_edge(1, 0)                          # directed cycle
    with pytest.raises(NotATree):
        prepare(C, B2, r=6.0, allow_dag=True)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0)])
def test_extract_cell_exact_on_dag_diamonds(alpha, beta):
    """Jittered diamonds: extract_cell equals the FULL-SPACE brute-force optimum on reconvergent
    sources -- the tree property is not needed by the cell engine."""
    for seed in range(6):
        A, B = _diamond_case(jitter=1.2, seed=seed)
        prepare(A, B, r=4.0, allow_dag=True)
        forward(A, B, alpha=alpha, beta=beta)
        M, com = extract_cell(A, B, alpha=alpha, beta=beta, run_cap=1)
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
            prepare(A, B, r=20.0, allow_dag=True)
            forward(A, B, 0.5, 1.0)
        except ValueError:
            continue
        M, _ = extract_cell(A, B, 0.5, 1.0, run_cap=1)
        assert not any(check_rules(M, A, B)), f"seed {seed}"
        bf = _full_space_brute(A, B, 0.5, 1.0, run_cap=1)
        assert bf is not None and abs(_cost_of(A, B, M, 0.5, 1.0) - bf) < 1e-6, f"seed {seed}"
        ran += 1
    assert ran >= 15, f"only {ran} reconvergent cases exercised -- generator drifted"


def test_match_tree_dag_pipeline():
    """The one-call pipeline on a DAG source: point mode (cell default), engine='all', and segment
    mode (the line graph of a diamond is itself reconvergent -- allow_dag flows through)."""
    A, B = _diamond_case()
    M, com = match_tree(A, B, r=6.0, allow_dag=True)
    assert not any(check_rules(M, A, B))
    assert set(com) == set(A.nodes)
    A2, B2 = _diamond_case()
    M2, _ = match_tree(A2, B2, r=6.0, allow_dag=True, engine="all")
    assert not any(check_rules(M2, A2, B2))
    A3, B3 = _diamond_case()
    Ms, cs = match_tree(A3, B3, r=6.0, allow_dag=True, mode="segment", bearing_weight=2.0)
    assert all(isinstance(a, tuple) and isinstance(v, tuple) for a, v in Ms)
    assert len(cs) == A3.number_of_edges()
