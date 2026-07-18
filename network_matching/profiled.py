"""The profiled forward table (docs/profiled_forward_table.md).

Today's forward table is optimistic at splits: `_couple` (§4.1a) forbids only exits some child
*cannot* use, so where several exits are feasible for every child it keeps them all and each child's
row links its own cheapest. The trace can then place a split on two cells -- measured on the
map-conflation hourglass as 2 violations on line 102752 and 3 on 100350 (report/).

Here every cell carries a cost **per profile**: where the upstream splits are placed. Parents may only
combine when their profiles agree, so the split's children are priced jointly and the phantom is
blocked at construction. A profile key is discharged once the current vertex post-dominates it -- from
there one branch carries it and nothing can contradict it -- which is what keeps the table small.

Layered on top of :func:`network_matching.dag_dtw.forward`, which is left untouched: it still owns the
``forbidden`` flags and this pass reads them. Nothing here changes shipped behaviour.

Entry points:
    forward_profiled(A, B, alpha, beta)   -> fills cand[v]["Dp"] = {profile: (cost, bp)}
    extract_profiled(A, B, alpha, beta)   -> (M, committed) via the §6.1 sink join
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

import networkx as nx

from .dag_dtw import INF, _b_order, check_rules, layer_order

Profile = frozenset          # frozenset of (split_vertex, cell) pairs


# ---------------------------------------------------------------------------------------
# the profiled set and its discharge points (docs §1.1, §1.3)
# ---------------------------------------------------------------------------------------
def profiled_splits(A: nx.DiGraph) -> set:
    """``S`` = the splits. A vertex's cell can only be disagreed about if two distinct downstream
    branches carry it, and a branch point is exactly ``outdeg >= 2``. The branches need NOT rejoin:
    a matching assigns every vertex one run globally, so two sinks with disjoint descendants still
    conflict at assembly (docs §1.1)."""
    return {n for n in A.nodes if A.out_degree(n) >= 2}


def postdom_drop(A: nx.DiGraph, S: set) -> Dict[Hashable, set]:
    """``drop[a]`` = the splits ``a`` post-dominates, i.e. those whose every path to a sink passes
    through ``a``. From there one branch carries the key, so no later tuple can contradict it and it
    is dead weight (docs §1.3). Forward mirror of cell_dag_extraction.md §3.5's early discharge:
    *first common ancestor going backward* is *post-dominator going forward*."""
    R = A.reverse(copy=True)
    sink_root = ("__postdom_root__",)
    R.add_node(sink_root)
    for t in [n for n in A.nodes if A.out_degree(n) == 0]:
        R.add_edge(sink_root, t)
    idom = nx.immediate_dominators(R, sink_root)             # dominators on the reverse graph
    drop: Dict[Hashable, set] = {a: set() for a in A.nodes}
    for s in S:
        x = idom.get(s)
        while x is not None and x != sink_root and x != s:
            drop[x].add(s)
            nxt = idom.get(x)
            if nxt is None or nxt == x:
                break
            x = nxt
    return drop


_MERGE_CACHE: Dict[tuple, object] = {}                   # (p0, p1) -> merged profile or None
_ABSENT = object()                                       # cache miss marker: None is a real result


def _merge(p0: Profile, p1: Profile):
    """Union two profiles; ``None`` if they disagree on a shared split (docs §1.2). A vertex has one
    run in a matching, so disagreement means no matching realises the combination -- on split cells
    this test **is** V3.

    Memoised: the profile universe is tiny (width <= 4, a few hundred distinct values) while the fold
    calls this once per (combo, option) pair per cell, so the same pairs recur constantly."""
    if not p0:
        return p1
    if not p1:
        return p0
    key = (p0, p1)
    hit = _MERGE_CACHE.get(key, _ABSENT)
    if hit is not _ABSENT:                               # None is a real result -- a conflict
        return hit
    d = dict(p0)
    out = None
    for s, v in p1:
        got = d.get(s)
        if got is None:
            d[s] = v
        elif got != v:
            break
    else:
        out = frozenset(d.items())
    _MERGE_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------------------
# the recurrence (docs §2)
# ---------------------------------------------------------------------------------------
def _fill_row_profiled(A, B, a, S, drop_a, alpha, beta, border, deg, max_profiles) -> None:
    """One vertex's row: ``cand[v]["Dp"] = {profile: (cost, bp)}``.

    ``bp`` is a list of ``(vertex, cell, profile)`` -- the parents (advance/stall) that produced the
    value, or a single same-vertex triple for an alpha-coverage step, mirroring ``bpD``'s convention.
    """
    cand = A.nodes[a]["cand"]
    preds = list(A.predecessors(a))

    # ---- (D)/(V): fold the predecessors one at a time, contracting per profile as we go.
    # Folding (rather than enumerating the full tuple product) keeps the working set bounded by the
    # number of CONSISTENT profiles, not by the product of the parents' -- the whole economy (docs §2).
    base: Dict[Hashable, dict] = {}
    for v in cand:
        if not preds:                                        # source: free entry
            base[v] = {frozenset(): (cand[v]["E"], [])}
            continue
        combos: Dict[Tuple[Profile, bool], tuple] = {(frozenset(), False): (0.0, [])}
        dead = False
        entries = [v] + list(B.predecessors(v))
        for pi_p, p in enumerate(preds):
            pc = A.nodes[p]["cand"]
            dp = 1.0 / deg[p]
            opts = []                                        # (entry cell, is_stall, profile, cost)
            for x in entries:
                cx = pc.get(x)
                if cx is None or cx.get("forbidden"):
                    continue                                 # §4.1a: not a valid run END
                stall = x == v
                for pi, (cost, _bp) in cx.get("Dp", {}).items():
                    if cost < INF:
                        opts.append((x, stall, pi, cost * dp))
            if not opts:
                dead = True
                break
            nxt: Dict[Tuple[Profile, bool], tuple] = {}
            if pi_p == 0:
                # first predecessor: combos is the single empty seed, so the merge is the identity
                # and the whole product collapses to one pass over the options.
                for (x, is_stall, pip, cp) in opts:
                    key = (pip, is_stall)
                    cur = nxt.get(key)
                    if cur is None or cp < cur[0] - 1e-12:
                        nxt[key] = (cp, [(p, x, pip)])
            else:
                for (pi0, st0), (c0, bp0) in combos.items():
                    for (x, is_stall, pip, cp) in opts:
                        pi = _merge(pi0, pip)
                        if pi is None:
                            continue                         # ← the consistency test (docs §1.2)
                        key = (pi, st0 or is_stall)
                        c = c0 + cp
                        cur = nxt.get(key)
                        if cur is None or c < cur[0] - 1e-12:
                            nxt[key] = (c, bp0 + [(p, x, pip)])
            combos = nxt
            if len(combos) > max_profiles:                   # loud, never a silent truncation
                raise ValueError(f"profiled fold at {a!r} cell {v!r} exceeded {max_profiles} "
                                 f"profiles -- raise max_profiles")
            if not combos:
                dead = True
                break
        if dead:
            base[v] = {}
            continue
        Ev = cand[v]["E"]
        rows: Dict[Profile, tuple] = {}                      # price E only now: beta iff any stall
        for (pi, st), (c, bp) in combos.items():
            tot = (beta if st else 1.0) * Ev + c
            cur = rows.get(pi)
            if cur is None or tot < cur[0] - 1e-12:
                rows[pi] = (tot, bp)
        base[v] = rows

    # ---- (H) alpha-coverage: within-row, relaxed to the fixed point exactly as dag_dtw does.
    # B carries no order of its own and may be cyclic, so a single pass would leave cells un-relaxed.
    Dp = {v: dict(rows) for v, rows in base.items()}
    succ = {v: [w for w in B.successors(v) if w in cand] for v in cand}
    succ_set = {v: set(ws) for v, ws in succ.items()}
    aE = {w: alpha * cand[w]["E"] for w in cand}
    # Gauss-Seidel sweeps, as dag_dtw's own (H) relaxation does: in-place propagation within a pass
    # converges in very few passes. A worklist was tried and is SLOWER here (measured 2-3.7x on the
    # coverage-heavy rows) -- per-improvement bookkeeping costs more than the passes it saves.
    changed = True
    while changed:
        changed = False
        for v in cand:
            src = Dp[v]
            if not src:
                continue
            items = list(src.items()) if v in succ_set[v] else src.items()   # B self-loop safety
            for w in succ[v]:
                dw, e = Dp[w], aE[w]
                for pi, (val, _bp) in items:
                    nw = val + e
                    cur = dw.get(pi)
                    if cur is None or nw < cur[0] - 1e-12:
                        dw[pi] = (nw, [(a, v, pi)])          # same-vertex triple == COVER
                        changed = True
                if len(dw) > max_profiles:
                    raise ValueError(f"profiled coverage at {a!r} cell {w!r} exceeded "
                                     f"{max_profiles} profiles -- raise max_profiles")

    # ---- own split cell (overwrite: v is a's run end so far), then discharge (docs §1.3).
    # Discharge is a MIN, not a forget: rows differing only in a dropped key collide and the cheaper
    # survives. That is the variable-elimination step.
    in_S = a in S

    def remap(pi, cell):
        """The key a row lands on: own split cell overwritten, then discharged keys removed."""
        d = dict(pi)
        if in_S:
            d[a] = cell
        if drop_a:
            d = {s: c for s, c in d.items() if s not in drop_a}
        return frozenset(d.items())

    for v in cand:
        out: Dict[Profile, tuple] = {}
        for pi, (val, bp) in Dp[v].items():
            # A COVER back-pointer names a cell of THIS row, so its profile is an intermediate key
            # that remap() is about to rewrite. Rewrite the reference too, or the reconstruction
            # walk looks up a key that no longer exists, stops early, and leaves the rest of the
            # branch uncovered (V4). Advance/stall pointers are unaffected: they name a parent's
            # already-final keys.
            if len(bp) == 1 and bp[0][0] == a:
                _self, vsrc, pisrc = bp[0]
                bp = [(a, vsrc, remap(pisrc, vsrc))]
            npi = remap(pi, v)
            cur = out.get(npi)
            if cur is None or val < cur[0] - 1e-12:
                out[npi] = (val, bp)
        cand[v]["Dp"] = out


def forward_profiled(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0,
                     max_profiles: int = 50000) -> nx.DiGraph:
    """Fill ``cand[v]["Dp"]`` for every cell, in the §4.0 layer order.

    Requires :func:`prepare` + :func:`forward` to have run: this pass reads the ``forbidden`` flags
    that ``forward`` owns and does not modify them. Returns ``A`` (mutated in place).

    Raises ``ValueError`` when a cell exceeds ``max_profiles`` -- loud, never a silent truncation, and
    never an OOM. A pure out-tree source (no merges, so nothing post-dominates and no key ever
    discharges) is the shape that trips it; see docs §5.1."""
    order, _L = layer_order(A)
    border = _b_order(B)
    deg = {n: max(1, len(list(A.successors(n)))) for n in A.nodes}
    S = profiled_splits(A)
    drop = postdom_drop(A, S)
    _MERGE_CACHE.clear()                                     # profiles are graph-local
    for a in order:
        _fill_row_profiled(A, B, a, S, drop[a], alpha, beta, border, deg, max_profiles)
    return A


# ---------------------------------------------------------------------------------------
# the extraction -- the sink join (docs §6.1)
# ---------------------------------------------------------------------------------------
def _flood(A, seeds) -> Dict[Hashable, set]:
    """Walk ``Dp`` back-pointers from pinned ``(vertex, cell, profile)`` seeds, collecting each
    vertex's run cells. Cover triples (same vertex) and advance/stall triples are both cell->cell
    moves; a vertex accumulates every cell its walk touches, which is its run."""
    cells: Dict[Hashable, set] = {}
    stack, seen = list(seeds), set()
    while stack:
        a, v, pi = stack.pop()
        if (a, v, pi) in seen:
            continue
        seen.add((a, v, pi))
        cells.setdefault(a, set()).add(v)
        row = A.nodes[a]["cand"][v].get("Dp", {}).get(pi)
        if row is None:
            continue
        for (p, x, pip) in row[1]:
            stack.append((p, x, pip))
    return cells


def extract_profiled(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0):
    """``(M, committed)`` by the §6.1 join: pick one profile, let each sink take its own best cell
    under it, add up. For a fixed profile the sinks are **independent**, so this is a min over the
    profile keys -- no product over sinks.

    Requires :func:`forward_profiled`. Raises ``ValueError`` if no profile is jointly reachable."""
    sinks = [n for n in A.nodes if A.out_degree(n) == 0]
    if not sinks:
        raise ValueError("A has no sink")

    # best cell per (sink, profile)
    per_sink: List[Dict[Profile, tuple]] = []
    for t in sinks:
        best: Dict[Profile, tuple] = {}
        for v, c in A.nodes[t]["cand"].items():
            for pi, (val, _bp) in c.get("Dp", {}).items():
                cur = best.get(pi)
                if cur is None or val < cur[0] - 1e-12:
                    best[pi] = (val, v)
        per_sink.append(best)

    # fold the sinks, merging profiles (they must agree where their key sets overlap)
    joined: Dict[Profile, tuple] = {frozenset(): (0.0, [])}
    for t, best in zip(sinks, per_sink):
        nxt: Dict[Profile, tuple] = {}
        for pi0, (c0, picks0) in joined.items():
            for pi, (val, v) in best.items():
                m = _merge(pi0, pi)
                if m is None:
                    continue
                tot = c0 + val
                cur = nxt.get(m)
                if cur is None or tot < cur[0] - 1e-12:
                    nxt[m] = (tot, picks0 + [(t, v, pi)])
        joined = nxt
        if not joined:
            raise ValueError("profiled join: no jointly reachable profile across sinks")

    # Terminal judge (docs §6.4). The profile enforces V3 and the recurrence enforces V2, but V1 is
    # NOT covered: on a cyclic B a run can revisit a B-vertex, and the per-profile contraction keeps
    # only the cheapest row, so a cheap-but-V1-invalid row can hide a valid costlier one -- the same
    # validity-blind contraction as scripts/repro_contraction_eviction. Taking the single minimum
    # returned an invalid matching on 169/900 random trees over cyclic B. Trying candidates
    # cheapest-first gives the judge the other profiles as fallbacks.
    verts = set(A.nodes)
    for cost, picks in sorted(joined.values(), key=lambda r: r[0]):
        cells = _flood(A, picks)
        M = {(a, v) for a, run in cells.items() for v in run}
        if {a for a, _ in M} != verts:                       # V4: every vertex must be placed
            continue
        v1, v2, v3 = check_rules(M, A, B)
        if v1 or v2 or v3:
            continue
        committed = {a: sorted(run, key=str)[0] for a, run in cells.items()}
        return M, committed, cost
    raise ValueError("profiled join: no valid root row -- increase match_radius_m")
