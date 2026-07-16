"""Do the seed-21/86 sole-surviving invalid rows actually USE cells only reachable via the
role-aware relaxation (flagged interiors / flagged entries)?"""
import random
import networkx as nx
import network_matching.dag_dtw as dd
from network_matching.dag_dtw import prepare, forward

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

# the sole judged rows from the instrumented run:
rows = {86: {0: ['v8', 'v9'], 1: ['v4', 'v5'], 2: ['v4'], 3: ['v6']},
        21: {0: ['v2'], 1: ['v0', 'v1'], 2: ['v2'], 3: ['v3'], 4: ['v1']}}
for seed in (86, 21):
    A, B = rand_case(seed)
    prepare(A, B, r=15.0)
    forward(A, B, alpha=0.6, beta=1.4)
    used_flagged = [(a, v) for a, run in rows[seed].items() for v in run
                    if A.nodes[a]["cand"][v].get("forbidden")]
    print(f"seed {seed}: flagged cells USED by the sole (invalid) surviving row: {used_flagged}")
