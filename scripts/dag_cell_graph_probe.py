"""Empirical probe of the CELL-MOVE GRAPH (docs/dag_dtw_matching.md §5.0).

Materializes the graph the extraction implicitly walks -- nodes = cells (a, v), edges = the three
moves (cover within a row, stall/advance across A-edges) -- and verifies its structural facts:

  1. GRADING    -- stall/advance edges strictly increase A's topological layer; cover edges stay
                   inside one row.  (=> any directed cycle is confined to a single row.)
  2. DAG-NESS   -- the cell graph is a DAG  <=>  every row-induced target subgraph B[cand(a)] is
                   acyclic: always on an acyclic B; on a cyclic B exactly when the radius gate
                   cuts every cycle; every cycle found projects to a directed B-cycle in one row.
  3. EMBEDDING  -- every matching returned by extract_cell IS an embedding of A into this graph:
                   each vertex's run is a directed cover path from its committed entry, and each
                   A-edge is realized by exactly one stall/advance edge from the parent's run END
                   to the child's entry.

Run:  python scripts/dag_cell_graph_probe.py
"""
from __future__ import annotations

import random

import networkx as nx

from network_matching.dag_dtw import (digraph, line_digraph, prepare, forward, extract_cell,
                                      extract_join, check_rules)


def cell_graph(A: nx.DiGraph, B: nx.DiGraph) -> nx.DiGraph:
    """The explicit cell-move graph (docs §5.0) over prepare()'d rows."""
    G = nx.DiGraph()
    for a in A.nodes:
        for v in A.nodes[a]["cand"]:
            G.add_node((a, v))
    for a in A.nodes:
        cand = A.nodes[a]["cand"]
        for v in cand:
            for w in B.successors(v):                           # COVER: stay in the row
                if w in cand:
                    G.add_edge((a, v), (a, w), kind="cover")
        for c in A.successors(a):
            cc = A.nodes[c]["cand"]
            for v in cand:
                if v in cc:                                     # STALL: child enters on the same cell
                    G.add_edge((a, v), (c, v), kind="stall")
                for w in B.successors(v):                       # ADVANCE: child enters one B-arc ahead
                    if w in cc:
                        G.add_edge((a, v), (c, w), kind="advance")
    return G


def check_grading(A: nx.DiGraph, G: nx.DiGraph) -> None:
    L = {a: i for i, a in enumerate(nx.topological_sort(A))}
    for (a, _v), (c, _w), d in G.edges(data=True):
        if d["kind"] == "cover":
            assert a == c, "cover edge left its row"
        else:
            assert L[c] > L[a], f"{d['kind']} edge does not advance A's order"


def check_cycles_confined(B: nx.DiGraph, G: nx.DiGraph) -> int:
    """Every directed cycle lies in ONE row and projects to a directed cycle of B."""
    n = 0
    for cyc in nx.simple_cycles(G):
        n += 1
        rows = {a for (a, _v) in cyc}
        assert len(rows) == 1, f"cycle spans rows {rows}"
        proj = [v for (_a, v) in cyc]
        for i, u in enumerate(proj):                            # projected edges are B-arcs
            assert B.has_edge(u, proj[(i + 1) % len(proj)]), "cycle does not project to a B-cycle"
    return n


def check_embedding(A: nx.DiGraph, B: nx.DiGraph, G: nx.DiGraph, M: set, committed: dict) -> None:
    """The returned matching is an embedding of A into the cell graph (docs §5.0)."""
    runs = {}
    for a in A.nodes:
        cells = {v for (x, v) in M if x == a}
        cur, path = committed[a], [committed[a]]
        rest = cells - {cur}
        while rest:                                             # the run is a directed cover path
            nxt = [w for w in B.successors(cur) if w in rest]
            assert nxt, f"run of {a!r} is not a cover path from its entry"
            cur = nxt[0]
            path.append(cur)
            rest.discard(cur)
        for x, w in zip(path, path[1:]):
            assert G.edges[(a, x), (a, w)]["kind"] == "cover"
        runs[a] = path
    for p, c in A.edges:                                        # each A-edge = one stall/advance edge
        end, entry = runs[p][-1], runs[c][0]
        assert G.has_edge((p, end), (c, entry)), f"A-edge {p}->{c} not realized in the cell graph"
        assert G.edges[(p, end), (c, entry)]["kind"] in ("stall", "advance")


def rand_polytree_cyclic_B(seed: int):
    """Random subdivided polytree over a random target WITH extra (often cycle-making) arcs."""
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
    for _ in range(rng.randint(2, nb)):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return A, B


SCENARIOS = {
    "chain":  ({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)],
               {"b0": (0, .5), "b1": (10, .5), "b2": (20, .5)}, [("b0", "b1"), ("b1", "b2")]),
    "split":  ({0: (0, 0), 1: (10, 0), 2: (20, 6), 3: (20, -6)}, [(0, 1), (1, 2), (1, 3)],
               {"s": (0, .5), "j": (10, .5), "u": (20, 6.5), "d": (20, -5.5)},
               [("s", "j"), ("j", "u"), ("j", "d")]),
    "merge":  ({0: (0, 6), 1: (0, -6), 2: (10, 0), 3: (20, 0)}, [(0, 2), (1, 2), (2, 3)],
               {"a": (0, 6.5), "b": (0, -5.5), "m": (10, .5), "o": (20, .5)},
               [("a", "m"), ("b", "m"), ("m", "o")]),
    "dense":  ({0: (0, 0), 1: (9, 0), 2: (18, 0)}, [(0, 1), (1, 2)],
               {f"b{i}": (3 * i, .4) for i in range(7)}, [(f"b{i}", f"b{i+1}") for i in range(6)]),
    "diamond": ({"S": (0, 0), "s1": (4, 0), "J": (8, 0), "x": (12, 3), "z": (12, -3),
                 "m": (16, 0), "t1": (20, 0), "T": (24, 0)},
                [("S", "s1"), ("s1", "J"), ("J", "x"), ("J", "z"), ("x", "m"), ("z", "m"),
                 ("m", "t1"), ("t1", "T")],
                {k + "'": (x, y + .4) for k, (x, y) in
                 {"S": (0, 0), "s1": (4, 0), "J": (8, 0), "x": (12, 3), "z": (12, -3),
                  "m": (16, 0), "t1": (20, 0), "T": (24, 0)}.items()},
                [(a + "'", b + "'") for a, b in
                 [("S", "s1"), ("s1", "J"), ("J", "x"), ("J", "z"), ("x", "m"), ("z", "m"),
                  ("m", "t1"), ("t1", "T")]]),
}


def main() -> None:
    print(f"{'case':<22} {'|cells|':>7} {'|edges|':>7} {'B DAG':>6} {'cell DAG':>9} "
          f"{'cycles':>7} {'confined':>9} {'embedding':>10}")

    def report(name, A, B, r=20.0, mode="point"):
        if mode == "segment":
            A, B = line_digraph(A), line_digraph(B)
        prepare(A, B, r=r)
        forward(A, B)
        G = cell_graph(A, B)
        check_grading(A, G)
        b_dag = nx.is_directed_acyclic_graph(B)
        g_dag = nx.is_directed_acyclic_graph(G)
        rows_acyclic = all(nx.is_directed_acyclic_graph(
            B.subgraph(A.nodes[a]["cand"]).copy()) for a in A.nodes)
        assert g_dag == rows_acyclic, "DAG <=> acyclic-rows equivalence broken"
        ncyc = 0 if g_dag else check_cycles_confined(B, G)
        emb = "-"
        try:
            M, com = extract_cell(A, B)
        except ValueError:
            try:
                M, com = extract_join(A, B)                     # the refusal case: cross-check engine
            except ValueError:
                M = None
        if M is not None:
            assert not any(check_rules(M, A, B))
            check_embedding(A, B, G, M, com)
            emb = "OK"
        print(f"{name:<22} {G.number_of_nodes():>7} {G.number_of_edges():>7} "
              f"{str(b_dag):>6} {str(g_dag):>9} {ncyc:>7} {'OK':>9} {emb:>10}")

    for name, (an, ae, bn, be) in SCENARIOS.items():
        report(name, digraph(an, ae), digraph(bn, be))
    report("split (segment)", digraph(*SCENARIOS["split"][:2]),
           digraph(*SCENARIOS["split"][2:]), mode="segment")

    # the §7 two-cycle: B cycles INSIDE the gate -> the cell graph cycles (confined to rows)
    report("two-cycle B", digraph({0: (0, 0), 1: (10, 0)}, [(0, 1)]),
           digraph({"p": (0, 1), "q": (10, 1)}, [("p", "q"), ("q", "p")]))

    # a cyclic B whose cycle the gate CUTS: every row sees only part of it -> cell graph is a DAG
    sq = {"c0": (0, 1), "c1": (40, 1), "c2": (40, 41), "c3": (0, 41)}
    report("gate-cut 4-cycle B", digraph({0: (0, 0), 1: (40, 0)}, [(0, 1)]),
           digraph(sq, [("c0", "c1"), ("c1", "c2"), ("c2", "c3"), ("c3", "c0")]), r=10.0)

    n_dag = n_cyc = 0
    for seed in range(20):                                      # random polytrees over cyclic-B targets
        A, B = rand_polytree_cyclic_B(seed)
        prepare(A, B, r=40.0)
        forward(A, B)
        G = cell_graph(A, B)
        check_grading(A, G)
        if nx.is_directed_acyclic_graph(G):
            n_dag += 1
        else:
            check_cycles_confined(B, G)
            n_cyc += 1
        assert nx.is_directed_acyclic_graph(G) == all(
            nx.is_directed_acyclic_graph(B.subgraph(A.nodes[a]["cand"]).copy()) for a in A.nodes)
        try:
            M, com = extract_cell(A, B)
            check_embedding(A, B, G, M, com)
        except ValueError:
            pass
    print(f"\nrandom polytrees over cyclic-B targets: {n_dag} cell-DAGs, {n_cyc} cyclic "
          f"(all cycles confined to single rows, DAG<=>acyclic-rows held 20/20, embeddings OK)")


if __name__ == "__main__":
    main()
