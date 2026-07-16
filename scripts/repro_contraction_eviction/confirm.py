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

# shipped brute, verbatim semantics
def flagless_brute(A, B, alpha, beta, run_cap=3, cap=300_000):
    def runs(a, e):
        out, frontier = [(e,)], [(e,)]
        for _ in range(run_cap - 1):
            nxt = []
            for r in frontier:
                for w in B.successors(r[-1]):
                    if w in A.nodes[a]["cand"] and w not in r:
                        nxt.append(r + (w,))
            out += nxt
            frontier = nxt
        return out
    per_vertex = []
    for a in A.nodes:
        opts = [r for e, c in A.nodes[a]["cand"].items() if c["D"] < INF for r in runs(a, e)]
        if not opts:
            return None, None
        per_vertex.append(opts)
    n = 1
    for o in per_vertex:
        n *= len(o)
        if n > cap:
            return "too-big", None
    best, bestM, verts = None, None, list(A.nodes)
    for combo in itertools.product(*per_vertex):
        M = {(a, v) for a, run in zip(verts, combo) for v in run}
        if any(check_rules(M, A, B)):
            continue
        cost = _cost_of(A, B, M, alpha, beta)
        if best is None or cost < best - 1e-12:
            best, bestM = cost, M
    return best, bestM

for seed in (86, 21):
    A, B = rand_case(seed)
    prepare(A, B, r=15.0)
    forward(A, B, alpha=0.6, beta=1.4)
    bf, bM = flagless_brute(A, B, 0.6, 1.4)
    print(f"seed {seed}: flag-free brute optimum = {bf}")
    if bM:
        runs = {}
        for (a, v) in bM:
            runs.setdefault(a, []).append(v)
        print("  brute-optimal M:", {a: sorted(v) for a, v in sorted(runs.items())},
              "check_rules:", check_rules(bM, A, B))
    try:
        M, _ = extract_cell(A, B, 0.6, 1.4)
        print("  extract_cell:", _cost_of(A, B, M, 0.6, 1.4))
    except ValueError as e:
        print("  extract_cell RAISED (spurious -- valid M shown above exists):", e)

# broader incidence sweep: new code vs shipped brute, cyclic-B-rich geometry
print("\n--- incidence sweep: extract_cell vs flag-free brute ---")
tot = feas = raises = displaced = 0
bad_seeds = []
for seed in range(500, 800):
    for (alpha, beta, r) in ((1.0, 1.0, 12.0), (0.6, 1.4, 15.0), (0.5, 1.0, 12.0)):
        A, B = rand_case(seed)
        prepare(A, B, r=r)
        try:
            forward(A, B, alpha=alpha, beta=beta)
        except ValueError:
            continue
        bf, _bM = flagless_brute(A, B, alpha, beta)
        if bf in (None, "too-big"):
            continue
        feas += 1
        try:
            M, _ = extract_cell(A, B, alpha, beta)
        except ValueError as e:
            raises += 1
            bad_seeds.append((seed, alpha, beta, "RAISE", round(bf, 2)))
            continue
        got = _cost_of(A, B, M, alpha, beta)
        if got > bf + 1e-9:
            displaced += 1
            bad_seeds.append((seed, alpha, beta, "DISPLACED", round(bf, 2), round(got, 2)))
print(f"feasible cases: {feas}; spurious raises: {raises}; displaced optima: {displaced}")
for b in bad_seeds[:20]:
    print("  ", b)
