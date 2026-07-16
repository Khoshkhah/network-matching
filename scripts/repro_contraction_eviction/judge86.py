import random, itertools
import networkx as nx
import network_matching.dag_dtw as dd
from network_matching.dag_dtw import INF, prepare, forward, extract_cell, check_rules, _cost_of

def rand_case(seed, nb_hi=10):
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
    for _ in range(rng.randint(1, max(2, nb - 2))):
        u, v = rng.choice(vs), rng.choice(vs)
        if u != v:
            B.add_edge(u, v)
    return A, B

orig = dd.check_rules
def logged(M, src, tgt):
    r = orig(M, src, tgt)
    if isinstance(src, nx.DiGraph) and src.number_of_nodes() and 0 in src.nodes:
        runs = {}
        for (a, v) in M:
            runs.setdefault(a, []).append(v)
        print("   judge:", {a: sorted(vs) for a, vs in sorted(runs.items())},
              "-> V1", r[0], "V2", r[1], "V3", r[2])
    return r
dd.check_rules = logged

for seed in (86, 21):
    print(f"=== seed {seed} ===")
    A, B = rand_case(seed)
    prepare(A, B, r=15.0)
    forward(A, B, alpha=0.6, beta=1.4)
    try:
        M, com = dd.extract_cell(A, B, 0.6, 1.4)
        print(" OK", _cost_of(A, B, M, 0.6, 1.4))
    except ValueError as e:
        print(" RAISED:", e)
