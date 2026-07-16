"""Old(role-blind) vs new(role-aware) on ALL engines + reciprocity, random trees.
Both arms keep the vetted feasibility-coupling (_feasible_links); only the ROLE change toggles."""
import copy, inspect, random, itertools, math
import networkx as nx
import network_matching.dag_dtw as dd
from network_matching.dag_dtw import INF, check_rules, _cost_of

# --- build the role-blind variants by source patching (faithful to the pre-change lines) ---
src = inspect.getsource(dd._fill_row)
assert "if dv == INF:" in src
old_fill_src = src.replace("if dv == INF:", 'if dv == INF or cand[v].get("forbidden"):')
ns = dict(dd.__dict__)
exec(old_fill_src, ns)
old_fill_row = ns["_fill_row"]
new_fill_row = dd._fill_row

src2 = inspect.getsource(dd._cell_reachable)
tgt = "if u in cand and (X, u) not in seen:"
assert tgt in src2
old_reach_src = src2.replace(tgt, 'if u in cand and not cand[u].get("forbidden") and (X, u) not in seen:')
ns2 = dict(dd.__dict__)
exec(old_reach_src, ns2)
old_reach = ns2["_cell_reachable"]
new_reach = dd._cell_reachable

def set_mode(old):
    dd._fill_row = old_fill_row if old else new_fill_row
    dd._cell_reachable = old_reach if old else new_reach

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

def run_arm(A, B, old, alpha, beta, r):
    set_mode(old)
    A = copy.deepcopy(A)
    out = {}
    dd.prepare(A, B, r=r)
    try:
        dd.forward(A, B, alpha=alpha, beta=beta)
    except ValueError as e:
        return {"forward": ("raise", str(e)[:60])}
    out["forward"] = ("ok",)
    for name, fn in (("cell", dd.extract_cell), ("join", dd.extract_join)):
        try:
            M, com = fn(A, B, alpha, beta)
            out[name] = ("ok", _cost_of(A, B, M, alpha, beta))
        except ValueError as e:
            out[name] = ("raise", str(e)[:60])
    try:
        dd.backward(A, B, alpha=alpha, beta=beta)
        M2, c2 = dd.extract_two_table(A, B)
        rec = dd.check_reciprocity(A, c2)
        out["two"] = ("ok", _cost_of(A, B, M2, alpha, beta), len(rec))
    except ValueError as e:
        out["two"] = ("raise", str(e)[:60])
    return out

bad = []
stats = {"n": 0, "flags_fired": 0}
for seed in range(400):
    for (alpha, beta, r) in ((1.0, 1.0, 12.0), (0.6, 1.4, 15.0)):
        A, B = rand_case(seed)
        old = run_arm(A, B, True, alpha, beta, r)
        new = run_arm(A, B, False, alpha, beta, r)
        stats["n"] += 1
        for eng in ("cell", "join", "two"):
            o, n = old.get(eng), new.get(eng)
            if o is None or n is None:
                continue
            if o[0] == "ok" and n[0] == "raise":
                bad.append((seed, alpha, eng, "old ok -> new RAISE", o, n))
            elif o[0] == "ok" and n[0] == "ok" and n[1] > o[1] + 1e-9:
                bad.append((seed, alpha, eng, "new COSTLIER than old", o[1], n[1]))
            if eng == "two" and o and n and o[0] == "ok" and n[0] == "ok" and o[2] == 0 and n[2] > 0:
                bad.append((seed, alpha, eng, "reciprocity old-agree new-disagree", o, n))
        # documented invariant on NEW: both succeed => C(cell) <= C(join)
        if new.get("cell", ("x",))[0] == "ok" and new.get("join", ("x",))[0] == "ok":
            if new["cell"][1] > new["join"][1] + 1e-9:
                bad.append((seed, alpha, "inv", "C(cell) > C(join)", new["cell"][1], new["join"][1]))

print("cases:", stats["n"])
if bad:
    print(f"{len(bad)} DISCREPANCIES:")
    for b in bad[:25]:
        print(" ", b)
else:
    print("no old->new regression on cell/join/two-table/reciprocity; invariant holds")
