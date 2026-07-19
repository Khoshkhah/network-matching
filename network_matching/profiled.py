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

import os
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from .dag_dtw import INF, _b_order, check_rules, layer_order

Profile = frozenset          # frozenset of (split_vertex, cell) pairs
REBASE = os.environ.get("PROFILED_REBASE") == "1"   # EXPERIMENT: cost SINCE THE LAST SPLIT
SEG: Dict[Hashable, dict] = {}                     # REBASE only: split -> {(parent_key, own_cell): cost}
START_SEMANTICS = os.environ.get("PROFILED_START") == "1"   # key = run START not END
SHIELD = os.environ.get("PROFILED_SHIELD") == "1"   # EXPERIMENT (docs §7.1): drop a key
                                                   # once a nearer split shields it


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


def shield_drop(A: nx.DiGraph, S: set) -> Dict[Hashable, set]:
    """REFUTED experiment, kept because the idea is compelling (docs §7.1).

    Premise: ``s`` is shielded at ``a`` when a nearer split ``s'`` lies on every path from ``s`` to
    ``a``, so everything below ``s'`` depends on ``s`` only through ``s'`` and the key can be dropped.
    That would make width 1 on a tree, and it does: profiles/cell go 14 -> 61 -> MemoryError at btree
    depth 3/4/5 without it, and stay FLAT at 5 with it (btree(5): MemoryError -> 82 ms).

    But it is WRONG, because ``Dp`` is cumulative over the whole upstream cone: cost(s -> a) is
    already inside every descendant's value, scaled by the 1/outdeg fractions. Dropping ``s`` at
    ``a``'s children lets them minimise it out INDEPENDENTLY, so they can pick different cells of
    ``s`` while each carries half of the shared cost -- the phantom, one level up.

    Measured on the 384-case envelope: 334/384 valid (was 384), parity 329/334, and the §6.2 sink-sum
    identity itself breaks (333/334). Divergences up to +33%. EVERY failure is on the `deep`
    structure, the only one with a split below a split; chain/ysplit/merge all pass, which is why a
    handful of hand-picked cases looked clean. btree hides it too: its geometry is congruent and
    symmetric, so independent minimisation happens to agree.

    Making it sound needs a different recurrence -- one where ``Dp`` means "cost since the nearest
    split" so each segment's cost belongs to exactly one factor -- not a change to the drop rule.
    Enable with PROFILED_SHIELD=1 to reproduce the failure."""
    out: Dict[Hashable, set] = {a: set() for a in A.nodes}
    for s in S:
        reach = nx.descendants(A, s) | {s}
        H = A.subgraph(reach)
        idom = nx.immediate_dominators(H, s)
        for a in reach:
            if a == s:
                continue
            x = idom.get(a)
            while x is not None and x != s:
                if x in S:
                    out[a].add(s)
                    break
                nxt = idom.get(x)
                if nxt is None or nxt == x:
                    break
                x = nxt
    return out


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

    # A vertex that is neither a split nor a discharge point does not change the profile at all, so
    # the row keeps its PARENT's frozenset by reference instead of rebuilding an identical copy.
    # That is the common case (most vertices are indeg-1, outdeg-1), and rebuilding it per cell per
    # profile was the bulk of the per-row memory.
    passthrough = not in_S and not drop_a
    _remap_cache: Dict[tuple, Profile] = {}

    def remap(pi, cell):
        """The key a row lands on: own split cell overwritten, then discharged keys removed."""
        if passthrough:
            return pi                                        # same object -- no allocation
        ck = (pi, cell) if in_S else (pi, None)
        hit = _remap_cache.get(ck, _ABSENT)
        if hit is not _ABSENT:
            return hit
        d = dict(pi)
        if in_S:
            if START_SEMANTICS:
                d.setdefault(a, cell)      # run START: written once, never overwritten
            else:
                d[a] = cell                # run END: overwritten as the run extends
        if drop_a:
            d = {s: c for s, c in d.items() if s not in drop_a}
        out = frozenset(d.items())
        _remap_cache[ck] = out
        return out

    for v in cand:
        out: Dict[Profile, tuple] = {}
        for pi, (val, bp) in Dp[v].items():
            if REBASE and in_S:
                # Bank the segment cost, then RESET the accumulator: nothing downstream carries the
                # cost of getting here, so branches below `a` share no quantity to disagree about.
                #
                # Bank the CHAIN with it. Walk the cover run now, while this row's pre-reset profile
                # keys are still valid, and store (cost, advance-bp, run cells). Reconstruction then
                # replays it verbatim -- no assignment matching, no profile lookup, nothing to get
                # wrong. Re-deriving the chain from the recovered assignment was tried twice and
                # failed on one corpus or the other each time.
                run, b, pcur = [v], bp, pi
                while len(b) == 1 and b[0][0] == a:          # cover step: same vertex, earlier cell
                    nxt = b[0][1]
                    if nxt in run:
                        break            # cyclic B lets a cover chain close on itself; stop rather
                                         # than loop (this is the WALK terminating, not a V1 test)
                    run.append(nxt)
                    r = Dp.get(nxt, {}).get(pcur)
                    if not r:
                        break
                    b = r[1]
                seg = SEG.setdefault(a, {})
                k = (pi, v)
                if k not in seg or val < seg[k][0] - 1e-12:
                    seg[k] = (val, tuple(b), tuple(run))      # b is now the advance into the PARENT
                out[frozenset({(a, v)})] = (0.0, [])
                continue
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
    SEG.clear()
    if SHIELD:
        sh = shield_drop(A, S)
        drop = {a: drop[a] | sh[a] for a in A.nodes}
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


def _join(f1: dict, f2: dict, keep: int, max_rows: int) -> dict:
    """Combine two factors: consistent profile pairs, costs summed, ``keep`` cheapest per key."""
    out: Dict[Profile, list] = {}
    for pi1, rows1 in f1.items():
        for pi2, rows2 in f2.items():
            m = _merge(pi1, pi2)
            if m is None:
                continue                                     # arms disagree -- not a matching
            bucket = out.setdefault(m, [])
            for c1, p1 in rows1:
                for c2, p2 in rows2:
                    bucket.append((c1 + c2, p1 + p2))
    for m, bucket in out.items():
        bucket.sort(key=lambda r: r[0])
        del bucket[keep:]
    if len(out) > max_rows:
        raise ValueError(f"profiled join factor exceeded {max_rows} rows -- raise max_rows")
    return out


def _eliminate(factors: List[dict], order: List[Hashable], keep: int, max_rows: int) -> List[dict]:
    """Min-sum variable elimination over the split keys (docs §6.1).

    The sinks are factors over their live splits, and the answer is ``min over pi of sum of factors``.
    Folding the sinks pairwise builds the CROSS PRODUCT of sinks that share no key -- on an out-tree
    that is one factor per sink and it explodes. Eliminating a key instead touches only the factors
    that mention it: combine those, minimise the key out, emit one factor over what is left.

    ``order`` must be deepest-split-first, which the DAG supplies: when a split is eliminated the only
    key its factors still share is its parent split, so intermediate factors stay narrow.
    """
    for J in order:
        idx = [i for i, f in enumerate(factors)
               if any(any(s == J for s, _ in pi) for pi in f)]
        if not idx:
            continue
        combined = factors[idx[0]]
        for i in idx[1:]:
            combined = _join(combined, factors[i], keep, max_rows)
        out: Dict[Profile, list] = {}                        # minimise J out
        for pi, rows in combined.items():
            npi = frozenset((s, v) for s, v in pi if s != J)
            bucket = out.setdefault(npi, [])
            bucket.extend(rows)                               # picks already carry their own chains
        for npi, bucket in out.items():
            bucket.sort(key=lambda r: r[0])
            del bucket[keep:]
        drop_idx = set(idx)
        factors = [f for i, f in enumerate(factors) if i not in drop_idx] + [out]
    return factors


def _flood_rebased(A, picks: list) -> Dict[Hashable, set]:
    """Reconstruct under re-basing. Every pick already carries what to do, so nothing is looked up:

        ("SEG", a, run, advance_bp)  -- a banked segment: take all of `a`'s run cells, then continue
                                        from the advance into its parent
        (a, v, profile)              -- an ordinary cell: continue from its back-pointers
    """
    cells: Dict[Hashable, set] = {}
    stack, seen = [], set()
    for q in picks:
        if q[0] == "SEG":
            _tag, a, run, adv = q
            cells.setdefault(a, set()).update(run)
            stack.extend(adv)
        else:
            stack.append(q)
    while stack:
        a, v, pi = stack.pop()
        if (a, v, pi) in seen:
            continue
        seen.add((a, v, pi))
        cells.setdefault(a, set()).add(v)
        row = A.nodes[a]["cand"][v].get("Dp", {}).get(pi)
        if row:
            stack.extend(row[1])
    return cells


def _extract_rebased(A, B, alpha, beta, keep, max_rows):
    """The §5.3 join, on SEGMENT factors: one per split (parent-key x own cell) plus one per sink."""
    sinks = [n for n in A.nodes if A.out_degree(n) == 0]
    factors: List[dict] = []
    for a, seg in SEG.items():
        f: Dict[Profile, list] = {}
        for (pi_par, v), (cost, adv, run) in seg.items():
            key = _merge(pi_par, frozenset({(a, v)}))
            if key is None:
                continue
            f.setdefault(key, []).append((cost, [("SEG", a, run, adv)]))
        for k, b in f.items():
            b.sort(key=lambda r: r[0]); del b[keep:]
        if f:
            factors.append(f)
    for t in sinks:
        f = {}
        for v, c in A.nodes[t]["cand"].items():
            for pi, (cost, _bp) in c.get("Dp", {}).items():
                f.setdefault(pi, []).append((cost, [(t, v, pi)]))
        if not f:
            raise ValueError(f"profiled join: sink {t!r} has no reachable profile")
        for k, b in f.items():
            b.sort(key=lambda r: r[0]); del b[keep:]
        factors.append(f)

    _o, L = layer_order(A)
    factors = _eliminate(factors, sorted(profiled_splits(A), key=lambda s: -L[s]), keep, max_rows)
    joined = factors[0]
    for f in factors[1:]:
        joined = _join(joined, f, keep, max_rows)
    if not joined:
        raise ValueError("profiled join: no jointly reachable profile across sinks")

    verts = set(A.nodes)
    for cost, picks in sorted((c, p) for rows in joined.values() for c, p in rows):
        cells = _flood_rebased(A, picks)
        M = {(a, v) for a, run in cells.items() for v in run}
        if {a for a, _ in M} != verts:
            continue
        v1, v2, v3 = check_rules(M, A, B)
        if v1 or v2 or v3:
            continue
        return M, {a: sorted(run, key=str)[0] for a, run in cells.items()}, cost
    raise ValueError("profiled join: no valid root row -- increase match_radius_m")


def extract_profiled(A: nx.DiGraph, B: nx.DiGraph, alpha: float = 1.0, beta: float = 1.0,
                     keep: int = None, max_rows: int = 50000):
    """``(M, committed, cost)`` by the §6.1 join: pick one profile, let each sink take its own best cell
    under it, add up. For a fixed profile the sinks are **independent**, so this is a min over the
    profile keys -- no product over sinks.

    Requires :func:`forward_profiled`. Raises ``ValueError`` if no profile is jointly reachable."""
    # `keep` defaults differ by path. The cone path truncates once per sink factor and SATURATES at
    # 32 (measured: 65 bonus answers at 32, 128 and 512 alike). Re-basing truncates at every SEG
    # factor, every sink factor AND every elimination step, so the same 32 loses candidates the judge
    # later needs for V1 on a cyclic B -- 56 bonus at 32, 64 at 128, 65 at 512. It is not free:
    # on line 100350, keep=512 costs 0.62s/40MB -> 1.11s/80MB. Parity is 354/354 at every setting on
    # both paths, so this only ever trades work for cases extract_cell refuses outright.
    if keep is None:
        keep = 512 if REBASE else 32
    if REBASE:
        return _extract_rebased(A, B, alpha, beta, keep, max_rows)

    sinks = [n for n in A.nodes if A.out_degree(n) == 0]
    if not sinks:
        raise ValueError("A has no sink")

    # one factor per sink: its best cell for each profile it can carry
    factors: List[dict] = []
    for t in sinks:
        best: Dict[Profile, tuple] = {}
        for v, c in A.nodes[t]["cand"].items():
            for pi, (val, _bp) in c.get("Dp", {}).items():
                cur = best.get(pi)
                if cur is None or val < cur[0] - 1e-12:
                    best[pi] = (val, [(t, v, pi)])
        if not best:
            raise ValueError(f"profiled join: sink {t!r} has no reachable profile")
        factors.append({pi: [(cost, picks)] for pi, (cost, picks) in best.items()})

    # eliminate the split keys, deepest first -- NOT a pairwise fold over sinks (docs §6.1)
    _ord, L = layer_order(A)
    elim = sorted(profiled_splits(A), key=lambda s: -L[s])
    factors = _eliminate(factors, elim, keep, max_rows)

    joined: Dict[Profile, list] = factors[0]
    for f in factors[1:]:
        joined = _join(joined, f, keep, max_rows)
    if not joined:
        raise ValueError("profiled join: no jointly reachable profile across sinks")
    candidates = sorted((c, picks) for rows in joined.values() for c, picks in rows)

    # Terminal judge (docs §6.4). The profile enforces V3 and the recurrence enforces V2, but V1 is
    # NOT covered. V1 is the NON-CROSSING rule (dag_dtw_matching.md §3): for all (a,v) in M, all
    # a- in Apred(a) and all v+ in Bsucc(v), (a-, v+) must not be in M -- the matching may not run
    # backwards. A cyclic B is what makes it bite, since Bsucc(v) can wrap around so a cell that
    # looks earlier is reachable as a successor. Nothing in the forward pass rules it out, so it is
    # only detectable once a complete matching exists, and the per-profile contraction keeps
    # only the cheapest row, so a cheap-but-V1-invalid row can hide a valid costlier one -- the same
    # validity-blind contraction as scripts/repro_contraction_eviction. Taking the single minimum
    # returned an invalid matching on 169/900 random trees over cyclic B. Trying candidates
    # cheapest-first gives the judge the other profiles as fallbacks.
    verts = set(A.nodes)
    for cost, picks in candidates:
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
