"""Same 900-case sweep, but with role-BLIND enforcement (coverage + reachability flag checks
restored) -- everything else (feasibility coupling) identical. Counts its spurious raises."""
import inspect, random, itertools
import networkx as nx
import network_matching.dag_dtw as dd
from network_matching.dag_dtw import INF, check_rules, _cost_of

src = inspect.getsource(dd._fill_row)
ns = dict(dd.__dict__)
exec(src.replace("if dv == INF:", 'if dv == INF or cand[v].get("forbidden"):'), ns)
dd._fill_row = ns["_fill_row"]
src2 = inspect.getsource(dd._cell_reachable)
ns2 = dict(dd.__dict__)
exec(src2.replace("if u in cand and (X, u) not in seen:",
                  'if u in cand and not cand[u].get("forbidden") and (X, u) not in seen:'), ns2)
dd._cell_reachable = ns2["_cell_reachable"]

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
            return None
        per_vertex.append(opts)
    n = 1
    for o in per_vertex:
        n *= len(o)
        if n > cap:
            return "too-big"
    best = None
    for combo in itertools.product(*per_vertex):
        M = {(a, v) for a, run in zip(list(A.nodes), combo) for v in run}
        if any(check_rules(M, A, B)):
            continue
        c = _cost_of(A, B, M, alpha, beta)
        if best is None or c < best - 1e-12:
            best = c
    return best

feas = raises = displaced = 0
bad = []
for seed in range(500, 800):
    for (alpha, beta, r) in ((1.0, 1.0, 12.0), (0.6, 1.4, 15.0), (0.5, 1.0, 12.0)):
        A, B = rand_case(seed)
        dd.prepare(A, B, r=r)
        try:
            dd.forward(A, B, alpha=alpha, beta=beta)
        except ValueError:
            continue
        bf = flagless_brute(A, B, alpha, beta)
        if bf in (None, "too-big"):
            continue
        feas += 1
        try:
            M, _ = dd.extract_cell(A, B, alpha, beta)
        except ValueError:
            raises += 1
            bad.append((seed, alpha, beta, "RAISE"))
            continue
        got = _cost_of(A, B, M, alpha, beta)
        if got > bf + 1e-9:
            displaced += 1
            bad.append((seed, alpha, beta, "DISPLACED", round(bf, 2), round(got, 2)))
print(f"ROLE-BLIND arm: feasible {feas}; spurious raises {raises}; displaced {displaced}")
for b in bad[:20]:
    print("  ", b)
