"""Standalone reproduction of seed 21 (alpha=0.6, beta=1.4, r=15) on PURE current code."""
import copy, random, itertools
import networkx as nx
import network_matching.dag_dtw as dd
from network_matching.dag_dtw import (INF, prepare, forward, extract_cell, check_rules,
                                      _cost_of, check_split_exits, _cell_reachable)

def rand_case(seed, nb_hi=10, extra_hi=None):
    rng = random.Random(seed)
    na, nb = rng.randint(4, 7), rng.randint(6, nb_hi)
    A = nx.DiGraph()
    for i in range(na):
        A.add_node(i, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(1, na):
        A.add_edge(rng.randrange(i), i)
    if max(d for _, d in A.out_degree()) < 2:
        leaf = max(A.nodes)
        A.add_node(na, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
        A.add_edge(next(iter(A.predecessors(leaf))), na)
    B = nx.DiGraph()
    vs = [f"v{i}" for i in range(nb)]
    for v in vs:
        B.add_node(v, x=rng.uniform(0, 30), y=rng.uniform(0, 30))
    for i in range(nb - 1):
        B.add_edge(vs[i], vs[i + 1])
    hi = extra_hi if extra_hi is not None else max(2, nb - 2)
    for _ in range(rng.randint(1, hi)):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return A, B

for seed in (21, 86):
    A, B = rand_case(seed)
    alpha, beta, r = 0.6, 1.4, 15.0
    prepare(A, B, r=r)
    forward(A, B, alpha=alpha, beta=beta)
    print(f"--- seed {seed} ---")
    print("A edges:", list(A.edges))
    print("B edges:", sorted(B.edges))
    for a in A.nodes:
        row = A.nodes[a]["cand"]
        print(f"  row {a}: " + ", ".join(
            f"{v}(D={c['D'] if c['D']==INF else round(c['D'],2)}"
            f"{',FORB' if c['forbidden'] else ''},bp={c['bpD']})" for v, c in sorted(row.items())))
    seen = _cell_reachable(A, B)
    print("  seen:", sorted(seen))
    print("  split_exits check:", check_split_exits(A, B))
    try:
        M, com = extract_cell(A, B, alpha, beta)
        print("  extract_cell OK cost", _cost_of(A, B, M, alpha, beta))
    except ValueError as e:
        print("  extract_cell RAISED:", e)
