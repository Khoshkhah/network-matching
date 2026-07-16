"""Angle B/C probe: legacy engines on the canonical run-through trap."""
import copy, math, traceback
from network_matching.dag_dtw import (INF, digraph, prepare, forward, backward,
    extract_cell, extract_join, extract_two_table, check_rules, check_reciprocity,
    check_reachability, validate_tables, check_split_exits, _cost_of)

def trap():
    A = digraph({"a0": (0, 0), "p": (10, 0), "c1": (20, 4), "c2": (20, -4)},
                [("a0", "p"), ("p", "c1"), ("p", "c2")])
    B = digraph({"x0": (0, 1), "x": (8, 1), "u": (12, 1), "w1": (20, 5), "w2": (20, -3)},
                [("x0", "x"), ("x", "u"), ("u", "w1"), ("u", "w2")])
    return A, B

A, B = trap()
prepare(A, B, r=4.0)
forward(A, B)
print("flags p:", {v: c["forbidden"] for v, c in A.nodes["p"]["cand"].items()})
print("D row p:", {v: (c["D"], c["bpD"]) for v, c in A.nodes["p"]["cand"].items()})

# 1. THE extraction
M, com = extract_cell(A, B)
print("extract_cell OK:", sorted(M), "cost", _cost_of(A, B, M, 1, 1))

# 2. extract_join
try:
    Mj, cj = extract_join(A, B)
    print("extract_join OK:", sorted(Mj), "cost", _cost_of(A, B, Mj, 1, 1))
except ValueError as e:
    print("extract_join RAISED:", e)

# 3. backward + two-table + reciprocity
backward(A, B)
print("B row a0:", {v: c["B"] for v, c in A.nodes["a0"]["cand"].items()})
print("B row p:", {v: (c["B"], c["bpB"]) for v, c in A.nodes["p"]["cand"].items()})
try:
    M2, c2 = extract_two_table(A, B)
    print("extract_two_table OK:", sorted(M2))
    print("reciprocity:", check_reciprocity(A, c2))
except ValueError as e:
    print("extract_two_table RAISED:", e)
print("check_reachability B-table:", check_reachability(A, "B"))
print("validate_tables B:", validate_tables(A, B, "B")[1])
print("check_split_exits:", check_split_exits(A, B))
