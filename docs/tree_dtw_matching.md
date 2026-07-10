# Tree-DTW: Matching a Directed Tree to a Directed Network

This document specifies **Tree-DTW**, an exact algorithm for aligning a **directed tree** — a road/edge structure that **branches and merges but never loops** — onto a **directed target network**. It generalizes Dynamic Time Warping (which aligns one *sequence* to another) so that the source may split and merge, returning a matching that is **exact and optimal** — computed with one forward pass, one backward pass, and a traceback, with **a single number per DP cell**. In the common point-to-point case the matching is single-valued; a single source point may also cover a **run** of target points (a 1:N match), priced by the coverage weight $\alpha$ of §4.1.

---

## 1. The Source: A Directed Tree

The source is a directed graph $G_A$ representing the path traveled:

* **Points & Arcs:** Points (nodes) and arcs (edges) are directed by the travel direction. We write $A_{\text{pred}}(a)$ for immediate predecessors of point $a$, and $A_{\text{succ}}(a)$ for its immediate successors.
* **Sources & Sinks:** A *source* is a point with no incoming edges ($A_{\text{pred}} = \emptyset$). A *sink* is a point with no outgoing edges ($A_{\text{succ}} = \emptyset$).
* **Splits & Merges:** A **split** (fork) is a point where a road branches into more than one successor ($|A_{\text{succ}}(a)| > 1$). A **merge** (join) is a point where two or more roads join into one ($|A_{\text{pred}}(a)| > 1$).
* **The Tree Property:** The graph contains **no loops**. Between any two points, there is **at most one** directed path. Roads can fork and join, but a road can *never* fork and later rejoin itself.

The target $G_B$ is any **directed network** and **may cycle** (e.g., roundabouts, city grids). We write $B_{\text{pred}}(v)$ / $B_{\text{succ}}(v)$ for the immediate in-/out-neighbours of a target point $v$.

### The Local Emission Cost
Every time we pair a source point $a$ to a target point $v$, we pay a penalty called the **emission cost**, denoted $E(a, v)$. A single source point may also cover a **run** of target points — the ordinary DTW "stay" move — priced along the run (see the coverage weight $\alpha$ in §4.1).

The library offers **two matching modes**, selected by the `emission` argument. They differ in **what a DP state is** — not merely in a cost formula:

* **`"point"` (default).** A state is a **point pair** $(a, v)$: one source point matched to one target point, scored by plain distance $E(a, v) = \lVert a - v\rVert$. This is the algorithm specified in §2–§7. Point distance is blind to heading, so under a lateral shift a fork's arm can collapse onto the *nearest* target point rather than the *corresponding* one — the **nearest-vs-corresponding** limit.

* **`"segment"` (specified in §8).** A state is a **segment pair** $(\text{A-arc}, \text{B-arc})$: a directed source *micro-segment* matched to a directed target *arc*, scored **middle-to-middle** with an optional heading term. Because the state is two real segments, the local cost — heading included — is defined on *every* move and no alignment can bypass it. This is **true segment-to-segment** matching (the warping path pairs segments, not points), and it is what resolves nearest-vs-corresponding. The point-state core of §2–§7 is reused wholesale, lifted from points to arcs (§8).

> A middle-to-middle *cost* on point states is **not** segment-to-segment: scoring $(a, v)$ by the midpoints of the segments each vertex happens to own leaves the state a point pair, so a stall (many source points → one target point) still has no target segment and the heading term is silently free (graph-DTW's `weighted_emission.md` §9–§11). What makes §8 segment-to-segment is that the **state itself is a segment pair**.

---

## 2. The Structural Core: Independence of Merges

The entire algorithm relies on a single geometric fact: **a merge's incoming branches are completely independent**.

```
   … ─► a ─┐
           ├─► c   (Merge Point)
   … ─► b ─┘
```

If you trace backward from a merge point $c$ through its predecessors $a$ and $b$, you find two upstream **subtrees**. Note these need not be single roads — each may itself branch and merge inside. In a true tree, **these two subtrees share absolutely no points**. If they did share an upstream point $s$, it would mean $s$ forks and later merges at $c$, creating a loop — which is strictly forbidden in a tree.

### Why this makes the math simple:
Because the subtrees are completely disjoint, you can optimize the path for the $a$-side and the path for the $b$-side **completely independently**, and simply add their costs together at the merge point:

$$\text{Cost of matching } c \text{ to } v = E(c, v) + (\text{best for the } a\text{-side subtree to reach } v) + (\text{best for the } b\text{-side subtree to reach } v)$$

Each bracketed term is the *entire* cost of one side — however much it branches inside — as a function only of where its top point ($a$ or $b$) lands. Those two terms are exactly the forward cells $D[a][\cdot]$ and $D[b][\cdot]$ of §4.1: one number per landing, already folding in the whole subtree behind it. **This is why an exact solution needs only a single number per DP cell.** (Had the two sides shared an ancestor $s$, "best for $a$" and "best for $b$" could place $s$ in *two* spots at once — a phantom saving that no matching realizes. A tree forbids exactly this.)

> **The split is the mirror — with a twist.** A split's branches lead into disjoint *downstream* subtrees (they never rejoin), **but they share the split point** as their common ancestor. So — exactly as a merge's approaches must agree on the merge point — a split's branches must **agree on the split point**, and that agreement is *not* enforced by the forward sum; it is the job of the **backward** pass (§4.2). The forward table is exact at merges but must be paired with the backward pass at splits.

---

## 3. The Objective: Minimum-Cost Valid Warping

A matching is a relation $M \subseteq \text{Points}(G_A) \times \text{Points}(G_B)$, where $(a,v) \in M$ means source point $a$ is matched to target point $v$. Write $M(a) = \{\, v : (a,v) \in M \,\}$ for the target points $a$ covers (possibly a run). Our goal is to solve the optimization problem

$$\text{minimize} \quad C(M) = \sum_{(a,v) \in M} E(a, v) \qquad \text{subject to } M \text{ a valid warping (V1)–(V4).}$$

**Point-to-point vs. coverage.** In the point-to-point case — the common one — each source point is matched to exactly one target point, so every $M(a)$ is a singleton and $M$ is effectively a one-arrow-per-point map. Under 1:N coverage (§4.1) a point covers a *run*, so $M(a)$ holds several points and $M$ is a genuine many-to-many relation. **Either way the cost is the sum over the matched pairs** — every member of $M$ contributes its own $E(a,v)$; nothing is subsampled to a single representative.

$M$ is a **valid warping** iff it satisfies all four constraints below, using only immediate neighbours and membership:

**(V1) Monotonicity (No Cross).**
```
∀ (a,v) ∈ M,  ∀ a⁻ ∈ Apred(a),  ∀ v⁺ ∈ Bsucc(v):    (a⁻, v⁺) ∉ M
```
If $a^- \to a$ and $v \to v^+$ are arcs, you may not match $a$ to the earlier $v$ while its predecessor $a^-$ sits on the later $v^+$ — that pair runs backward.

**(V2) Predecessor Rule (Every Cell is Fed).**
```
∀ (a,v) ∈ M :
    [ ∃ v⁻ ∈ Bpred(v) : (a, v⁻) ∈ M ]                                             (i)  continues a run at a
  ∨ [ ∀ a⁻ ∈ Apred(a) : ( (a⁻, v) ∈ M ) ∨ ( ∃ v⁻ ∈ Bpred(v) : (a⁻, v⁻) ∈ M ) ]   (ii) every predecessor feeds it
```
Either $(a,v)$ is **interior** to $a$'s run — case (i), a matched B-predecessor at the same $a$ — or it is the run's **entry**, and then case (ii) forces **every** predecessor $a^-$ to feed it (matched to $v$ itself, or one B-arc before $v$). The universal $\forall a^-$ is what bites at a **merge**: all incoming roads must arrive at the **same** $v$. A source ($A_{\text{pred}}(a)=\emptyset$) satisfies (ii) vacuously.

**(V3) Successor Rule (Every Cell Continues).** The exact mirror, over successors:
```
∀ (a,v) ∈ M :
    [ ∃ v⁺ ∈ Bsucc(v) : (a, v⁺) ∈ M ]                                             (i)  continues a run at a
  ∨ [ ∀ a⁺ ∈ Asucc(a) : ( (a⁺, v) ∈ M ) ∨ ( ∃ v⁺ ∈ Bsucc(v) : (a⁺, v⁺) ∈ M ) ]   (ii) every successor carries it on
```
Either $(a,v)$ continues in B, or it is the run's **exit** and **every** successor carries it on. The universal $\forall a^+$ bites at a **split**: all exits must leave from the **same** $v$. A sink ($A_{\text{succ}}(a)=\emptyset$) satisfies (ii) vacuously.

**(V4) Boundary & Full Coverage.** $M(a) \neq \emptyset$ for every point $a$ (nothing is left unmatched); a source's entry and a sink's exit are **free** (they may land mid-edge); every interior junction is **pinned**.

These are structural properties of the matching itself — no algorithm, no cost. (V1) forbids a backward step; **(V2)** keeps a **merge** from being *entered at two different points*; **(V3)** keeps a **split** from being *left at two different points*; (V4) forces full coverage. On a tree all four are simultaneously and exactly satisfiable — which is what the algorithm below produces.

---

## 4. The Execution Algorithm

Because the source is a tree, an exact optimum is reached by one **forward** cost pass, one **backward** cost pass, and one **traceback** (min-sum message passing on the tree). Every table cell holds a single number.

### 4.0 Vertex Ordering — Longest-Path Layering

Any topological order suffices for the plain cost recurrences (§4.1, §4.2). The forward V3 coupling (§4.1a) needs a **stronger** property: **every split's children are placed before any of their successors** — the split's sibling group is *complete and grouped* before anything downstream of it is filled, so the group can be reconciled (forbid-and-rebuild) while nothing has yet read it.

**Why a plain topological sort is not enough.** Take a split $J \to \{b_1, b_2\}$ with $b_1 \to x$. The order $J,\ b_1,\ x,\ b_2$ is perfectly topological — every edge points forward — yet $x$ (a successor of $b_1$) is filled **before** the sibling $b_2$. If $b_2$ then forbids an exit cell of $J$ that $b_1$ had used, $b_1$'s row is rebuilt — but $x$ has *already read* $b_1$'s old row. The layered order below makes that impossible: $x$ waits until the whole sibling group $\{b_1, b_2\}$ is settled.

**The algorithm.** Assign each vertex a depth $L(v)$ = the **longest path** (in edges) from any source to $v$:

1. sweep $A$ in **topological order** (a vertex is reached only after all its ancestors);
2. a **source** ($\text{in-degree}=0$) gets $L(s) = 0$;
3. every other vertex gets $L(v) = \max_{p \in A_{\text{pred}}(v)} L(p) + 1$;
4. **sort** all vertices by $L$ ascending (ties broken by id, for determinism). The result is the sweep order $\pi$.

$\pi$ is always a valid topological order ($L$ strictly increases along every edge), computed in $O(|A|)$.

**Worked example** — a subdivided source with a split, unequal branch lengths, and a merge (interior points $a_1, b_1, b_2, c_1, d_1$ on the real edges):

```text
S → a₁ → J ─→ b₁ ────────→ M → d₁ → T          L:  S=0  a₁=1  J=2
             └→ b₂ → c₁ ──↗                        b₁=3  b₂=3   ← the split's children share a layer
                                                   c₁=4  M=5  d₁=6  T=7
π = S, a₁, J, b₁, b₂, c₁, M, d₁, T
```

The split's children $b_1, b_2$ sit **together in layer 3**, before everything downstream ($c_1, M, \dots$); the merge $M$ takes $\max(L(b_1), L(c_1)) + 1 = 5$, so **both** its incoming branches — even the longer one through $c_1$ — are complete before it is filled. That "join after all its branches" placement is exactly what a reconvergent structure needs.

**Why the guarantee needs the subdivision.** The sibling property ("no successor of a vertex precedes any of its siblings") holds **because every real edge carries at least one interior point**: the first vertex on each of a split's outgoing edges is then an interior point whose **sole** predecessor is the split, so a split's children are **pairwise incomparable** — none is a descendant of another — and they all land in layer $L(\text{split})+1$. On a raw graph the property can be impossible for *any* ordering: with `p→v`, `p→w`, **and** `v→w`, the vertex `w` is both a *sibling* of `v` and a *successor* of `v`, so "successors after all siblings" would demand `w` before `w`. Subdivision removes exactly this case.

A vertex with **many parents** (a merge) has **no siblings** under subdivision — each of its parents is an interior point whose only child is the merge — so the sibling group machinery of §4.1a only ever engages at splits. The layering holds identically on $A$ (point mode) and on the line graph $L(A)$ (segment mode); the same order serves both.

### 4.1 Forward Pass: The Upstream Cost Table `D`

We sweep the source points in **topological order** (sources first, sinks last). Let $\text{reach}(v) = B_{\text{pred}}(v) \cup \{v\}$ be the target points an upstream neighbour can hand off from (either staying on $v$, or moving from an immediate predecessor of $v$). Then:

$$D[a][v] = E(a, v) + \sum_{p \in A_{\text{pred}}(a)} \frac{1}{\text{outdeg}(p)} \min_{x \in \text{reach}(v)} D[p][x]$$

* **Source** ($A_{\text{pred}}(a)=\emptyset$): empty sum → $D[a][v] = E(a,v)$ (free entry at every source).
* **Chain** (one predecessor, $\text{outdeg}=1$): a single term with factor 1 — the ordinary DTW step.
* **Merge** (more than one predecessor): the sum couples the roads. $D[c][v]$ is finite **only if every road can reach $v$**, forcing a merge onto one common target point — this is (V2), folded into the cost. The sum is a true cost precisely because the roads are independent (§2).
* **The Split Factor $\frac{1}{\text{outdeg}(p)}$:** a point feeding several successors would otherwise have its cost counted once **per** branch; dividing by its out-degree splits that cost equally down the branches, so **each point's cost is counted exactly once** in the total. On a tree the branches go to separate sinks and never rejoin, so this conserves the cost exactly.

#### The back-pointer every cell stores

Like classic DTW, the cost is only half the table — each cell must also **record how it was reached**, so the extraction (§5) can *follow* the optimum instead of guessing it. Alongside `D[a][v]` store `bp_D[a][v]` = **the list of cells whose `D`-value this cell was computed from**:

```text
bp_D[a][v] = [ (a', x'), (a'', x''), … ]      # the source cells this value was built from
```

That single list is the whole back-pointer — it is the classic-DTW arrow generalised from *one* predecessor cell to a **list** (a merge is fed by several). You never need a separate "kind" tag: the move type is read straight off **whose** source point appears in the list:

- **`[]` (empty)** — `a` is a **source** (`Apred(a)=∅`): nothing fed it, a free entry at `v`.
- **`[(a, v')]` — one pair whose source point is `a` itself** — a **coverage** step (the 1:N horizontal move): the same source point `a`, one B-step on from `v'` (`v' → v` a B-arc, `v' ∈ Bpred(v)`).
- **`[(p₁, x₁), (p₂, x₂), …]` — pairs whose source points are the predecessors of `a`** — an **advance**: for each predecessor `pᵢ ∈ Apred(a)`, the cell `xᵢ ∈ reach(v) = {v} ∪ Bpred(v)` it sat on to feed `v`.

**Worked micro-example** — a 3-point chain `a₀→a₁→a₂` matched onto `b₀→b₁→b₂`:

| cell | `bp_D[a][v]` | reads as |
|---|---|---|
| `D[a₀][·]` | `[]` | `a₀` is a source — free start, nothing before it |
| `D[a₁][b₁]` | `[(a₀, b₀)]` | to put `a₁` on `b₁`, its predecessor `a₀` was on `b₀` |
| `D[a₂][b₂]` | `[(a₁, b₁)]` | to put `a₂` on `b₂`, `a₁` was on `b₁` |

If B is **finer** than A (one source segment `a₀→a₁` over `b₀→b₁→b₂`), `a₁` covers a run — note the coverage pair's source point is `a₁`, the *row itself*:

| `D[a₁][b₁]` | `[(a₀, b₀)]` | advance — `a₁`'s real match (its run starts here) |
| `D[a₁][b₂]` | `[(a₁, b₁)]` | coverage — **same** `a₁`, one B-step on from `b₁` |

At a **merge** `p, q → m`, the cell is `bp_D[m][v] = [(p, x_p), (q, x_q)]` with **both** `x_p, x_q ∈ {v} ∪ Bpred(v)` — that both branches sit on `v` or one step before it *is* the merge coupling (V2), recorded in the one list.

**How the list is filled — while the row `a` is filled** (mirrors the recurrence exactly):

1. Compute the **advance** value `adv = E(a,v) + Σ_p (1/outdeg(p))·min_{x∈reach(v)} D[p][x]`; while taking each predecessor's inner `min`, **collect the winning `x`** so the list becomes `[(p, x_p) for p ∈ Apred(a)]` (empty for a source). Set `D[a][v] ← adv`.
2. Run the horizontal coverage relaxation: for every B-arc `v' → v` (`v' ∈ Bpred(v)`), if `α·E(a,v) + D[a][v'] < D[a][v]`, **lower** `D[a][v]` and **replace** the list with `[(a, v')]`.

So `bp_D[a][v]` always lists the exact cells that produced `D[a][v]`. There is **no guessing later** — the extraction reads this list. (The α/β weights of §4.1 change nothing here: a β-stall is just an advance pair `(p, v)`, a 1:N step is the coverage pair `(a, v')`.)

#### Coverage weights: 1:N with `α`, N:1 with `β`

When the two graphs are sampled at different densities, one point on one side must span a **run** on the other. Each direction is a DTW "stay" move with its own weight that prices the *extra* pairs it creates; the 1:1 diagonal always stays at full cost.

* **1:N — one source point covers a run of target points** ($a$ **stays** while B advances; the **horizontal (H)** move), so one A-point rides $v_0 \to \dots \to v_k$ within its row. Unweighted it pays $E(a,\cdot)$ at *every* covered point, so the cost of one point matching a B-stretch grows **linearly with how finely B is sampled**. The **horizontal weight $\alpha \le 1$** discounts each *extra* covered point to $\alpha\cdot E$.

* **N:1 — many source points collapse onto one target point** (A advances while $v$ **stays**; the **vertical (V)** move), so consecutive source points $p_0 \to \dots \to a$ all land on the *same* $v$ — a source sampled finer than the target. Unweighted it pays $E(\cdot,v)$ at *every* stacked source point, so the cost grows with how finely **A** is sampled. The **vertical weight $\beta \le 1$** discounts each *extra* stacked point to $\beta\cdot E$.

The two are exact mirrors: $\alpha$ prices coverage **along B** (a within-row extension); $\beta$ prices coverage **along A** (an extension along the topological source sweep). For each predecessor $p$ write its two ways into $v$:

$$\text{step}_p = \min_{x\in B_{\text{pred}}(v)} D[p][x] \quad(\text{$p$ advances into } v),\qquad \text{stall}_p = D[p][v] \quad(\text{$p$ is already on } v).$$

The cell is then the cheapest of a full-cost advance, a $\beta$-discounted stall, and an $\alpha$-discounted coverage step:

$$D[a][v] = \min\begin{cases}
E(a,v) + \sum_{p\in A_{\text{pred}}(a)} \tfrac{1}{\text{outdeg}(p)}\,\text{step}_p & \text{(D) every branch advances into } v \quad(\text{full } E) \\[4pt]
\beta\cdot E(a,v) + \displaystyle\min_{q\in A_{\text{pred}}(a)}\!\Big[\tfrac{\text{stall}_q}{\text{outdeg}(q)} + \sum_{p\neq q}\tfrac{\min(\text{stall}_p,\ \text{step}_p)}{\text{outdeg}(p)}\Big] & \text{(V) at least one branch stalls on } v \quad(\beta E) \\[8pt]
\alpha\cdot E(a,v) + \min_{v'\in B_{\text{pred}}(v)} D[a][v'] & \text{(H) 1:N coverage along B} \quad(\alpha E)
\end{cases}$$

**Why $\beta$ triggers on *one* stall, not all.** Point $a$ landing on $v$ becomes an *extra* point on $v$ — an N:1 stack, so $\beta\cdot E$ — the moment **any one** incoming branch is already sitting on $v$. The other branches are free to advance or stall, each as it prefers ($\min(\text{stall}_p,\text{step}_p)$). The inner $\min_q$ **forces at least one branch to stall**, so the $\beta$ discount is never taken on a pure advance (line (D) already covers that at full $E$). And $a$'s emission is discounted **once**, however many branches sit on $v$: $a$ contributes a single new pair $(a,v)$; each upstream branch's own emissions are already paid inside its $D[p][\cdot]$.

* **Chain** (one predecessor $p$): (D) $= E + \text{step}_p$, (V) $= \beta E + \text{stall}_p$ — the ordinary discounted stay.
* **Merge**: (V) lets one branch stall on $v$ (unlocking $\beta$) while the rest independently advance or stall; the predecessor **sum** still forces every road onto the one $v$ — that is (V2), unchanged by $\beta$.

Because $\text{step}_p \ge \min(\text{stall}_p,\text{step}_p)$ and $\text{stall}_q \ge \min(\dots)$, **$\alpha = \beta = 1$ is bit-for-bit** the point-to-point cost of §4.1: the discounted lines never beat the full-cost advance when the optimum has no stall, and reproduce it exactly when it does. Both weights are $\ge 0$ per step, so $D$ never decreases along a run (no cost "laundering"). Unrolling a run of drift $\delta$ costs $\delta(1 + \alpha k)$ for a 1:N B-run of length $k$, and $\delta(1 + \beta m)$ for an N:1 A-stack of $m$ extra points:

| run length | weight = 1 | weight = 0.5 | weight = 0 |
|---|---|---|---|
| 1 (1:1) | δ | δ | δ |
| 6 | 7δ | 4δ | δ |
| 30 | 31δ | 16δ | δ |

(the same schedule for $\alpha$ over a B-run and $\beta$ over an A-stack). Lower $\alpha$ when a source point legitimately spans a finely-sampled B stretch; lower $\beta$ when many source points legitimately fall on one B point. Both live **inside the decision cost**: they steer *which* matching is chosen (toward more coverage in that direction), never the reported drift, which stays the raw $\sum E$ of the chosen matching. **Use $\alpha, \beta \in (0, 1]$** (defaults $1$); the $\to 0$ limit charges a whole run essentially once but *over-collapses* — a free stay is never dominated, so the matching can degenerate (and, in segment mode, cease to be a valid warping). Keep both comfortably above $0$.

> **$\beta$ vs. the merge coupling.** $\beta$ discounts only $a$'s *emission* when $a$ stacks on $v$; the requirement that **all** incoming branches reach the *same* $v$ (V2) is the predecessor sum itself and is untouched. Coverage prices *how much a stacked point costs*; the junction rule fixes *where the branches must meet* — independent concerns.

#### Why the total cost is *not* $\sum_{\text{sinks}} \min_v D$

The tempting quantity $\sum_{\text{sinks}\ t} \min_v D[t][v]$ is a **lower bound**, not the cost — it is **optimistic at a split**. $D[a][v]$ for a branch $a$ folds in $a$'s ancestors, including the **split point its two branches share**, and it takes a $\min$ over that point's position. So $\min_v D[a]$ silently places the split wherever is cheapest **for branch $a$ alone**, and $\min_v D[b]$ wherever is cheapest **for branch $b$ alone** — nothing makes them agree. Their sum can put one point in two places at once:

```
split s → {a, b}, with s free to sit at σ or σ':
    branch a : cheap from σ  (0),  costly from σ' (10)
    branch b : cheap from σ' (0),  costly from σ  (10)

Σ_sinks min D  =  min D[a] + min D[b]  =  0 (a takes s=σ)  +  0 (b takes s=σ')  =  0     ← phantom

but s is ONE point:   s=σ → 0+10 = 10     s=σ' → 10+0 = 10     true minimum = 10
```

The `0` is a phantom — $s$ at $\sigma$ for $a$ and $\sigma'$ for $b$ *simultaneously*, which is no matching at all. The real total is read off the **consistent** matching $M$ extracted in §5 (its resolution of this same example is worked there).

### 4.1a V3 Coupling in the Forward Pass — Forbid-and-Rebuild

The forward sum (§4.1) couples **merges** (V2) but is *optimistic at splits*: each child of a split is filled **independently**, so two children can link back to **different** cells of the split — precisely a (V3) break, invisible to the forward table itself (§4.1's phantom). Swept in layer order (§4.0) — a split's children complete and grouped before any of their successors — one extra step couples splits **inside the forward pass**.

**The `forbidden` flag.** Each cell carries a boolean **`forbidden`**: when set, **no back-pointer may link *to* this cell** — every place the recurrence reads a neighbour cell (a predecessor's advance/stall source, a same-row coverage source) skips it. The flag is on the **cell** $(a, v)$, never on the vertex $a$: only that one landing of the split is barred, its other cells stay available.

**The step, per split $a$** (children $a_1, a_2, \dots$, all in layer $L(a)+1$):

1. **Build** each child's row with the normal recurrence (§4.1), skipping forbidden cells.
2. **Forbid non-shared exits.** As each child $a_i$ completes — **including the first** — collect the split cells it links to, $\text{links}(a_i) = \{(a, v) : (a,v) \in \text{bp}_D[a_i][\cdot]\}$, and mark **forbidden** every cell of $a$ **not** in it. After $a_1$ the allowed exits are exactly $a_1$'s; each later child narrows them. Net effect:
   $$\text{allowed exits of } a \;=\; \bigcap_i \text{links}(a_i).$$
3. **Rebuild affected rows — whole rows.** A newly-forbidden cell is dead for **all** children, past and future. Every earlier child whose row linked to it is **rebuilt from scratch**: re-run its full recurrence, skipping all forbidden cells, so each of its cells re-links to the best surviving exit (or goes to $\infty$ if none remains in its reach). It must be the *whole row*, not the one cell, because of the within-row 1:N coverage dependency $D[a_i][v_1] \to D[a_i][v_2]$ (§4.3): re-deriving the cell that advanced from the forbidden exit shifts every coverage cell that fed off it.
4. **Iterate to the fixed point.** A rebuilt row may re-link to a *different* exit; if some sibling does not share that one, it too is forbidden and the affected rows rebuild again. Repeat until no new cell is forbidden.

**Termination and cost.** The forbidden set only **grows**, and is bounded by the split's candidate set — so there are at most $|\text{cand}(a)|$ rounds, each rebuilding at most all children once: $O(|\text{cand}(a)| \cdot \sum_i \text{row}(a_i))$ worst-case for the split, and every cell is forbidden at most once overall. In practice the intersection stabilises in one or two rounds.

**Worked trace** — split $a$ with children $a_1, a_2$, candidates $\text{cand}(a) = \{v_1, v_2, v_3\}$:

| round | event | forbidden | allowed exits |
|---|---|---|---|
| 1 | build $a_1$: links to $\{v_1, v_2\}$ → forbid $v_3$ | $\{v_3\}$ | $\{v_1, v_2\}$ |
| 1 | build $a_2$: links to $\{v_1\}$ only → forbid $v_2$ | $\{v_2, v_3\}$ | $\{v_1\}$ |
| 2 | $a_1$ had linked $v_2$ → **rebuild $a_1$'s whole row** skipping $\{v_2, v_3\}$; it re-links via $v_1$ | $\{v_2, v_3\}$ | $\{v_1\}$ |
| 2 | rebuild introduced no exit outside $\{v_1\}$ → **fixed point** | | **$\{v_1\}$** |

Both children now leave from $v_1$ — every surviving exit is shared, which **is** (V3) at this split.

**Feasibility.** If the intersection empties — no cell of the split is usable by every child within the candidate gate — there is **no** (V3)-valid exit at this radius: raise the same feasibility error as Part 1.3 (*increase `match_radius_m`*), never return a silently-broken table.

**Multiple exits are fine — not a bug.** A split may keep **several** surviving exit options; the step never forces one. Its invariant is only that **every survivor works for all children**. The single exit is chosen later, by the traceback (§5), which commits the split to one surviving cell and routes every child through it — always feasible, because every kept option already works for all of them. (The traceback never commits a vertex to a forbidden cell.)

**It forbids a *cell*, never a *parent*** — so the step is **identical in point and segment mode**, and indifferent to how many parents a child has. In point mode a split's children have the split as sole predecessor (§4.0); in segment mode an outgoing segment of a merge+split **bipartite cluster** in $L(A)$ has several parent segments — it simply skips all forbidden cells across all of them and re-links through whatever survives. Same step, no special case.

**Invariant to check:** for every split and every surviving (non-forbidden) exit cell, **every** child of the split links to it — and the survivor set is non-empty. (Not the independent per-sink decode of `check_forward_v3`: that would wrongly flag two *different but individually valid* surviving options as a violation.)

**Implementation:** `forward_v3(A, B, α, β)` — the §4.0 layer sweep with this coupling; plain `forward` stays the uncoupled §4.1 sweep (bit-identical to `forward_v3` on a split-free source). The invariant is `check_split_exits(A)` (empty ⇒ consistent); the extraction seed and coverage gap-fill skip forbidden cells.

**Cross-table validation under the coupling.** Canonical order: **`forward_v3` first, then `backward`** — the backward pass reads the flags, so its pointers never target a forbidden cell; run the other way round, the unconstrained backward table can point into cells `forward_v3` later forbids (observed: committed-forbidden cells / reciprocity breaks). In that canonical order, over a 170-case sweep (fixed scenarios + random out-trees over cyclic targets, α/β down to 0.2):

* **§6c reachability — unaffected**: 0 failures, both tables, every case.
* **§6b reciprocity — never regressed**: the coupled pipeline fails only where the plain pipeline already fails (10 cases *fixed*, 0 broken); the residual shared failures under harsh weighting are the documented §6d complementarity — the **backward table's V2 corner**, which this forward-side coupling does not touch.
* Per-cell `validate_tables` / final-`M` counts shift by ±1–3 cases on **cyclic-B coverage runs** (a back-arc re-entering a covered cell reads as a local V1 cross) — the same pre-existing checker-sensitivity family both pipelines show; the coupling only reroutes which runs occur.

### 4.2 Backward Pass: The Downstream Cost Table `B`

To force splits to agree on a single physical location, we build a mirror table $B$ by sweeping the source in **reverse topological order** (sinks first) and summing over **successors**:

$$B[a][v] = E(a, v) + \sum_{s \in A_{\text{succ}}(a)} \frac{1}{\text{indeg}(s)} \min_{w \in \{v\} \cup B_{\text{succ}}(v)} B[s][w]$$

$B$ stores the **mirror back-pointer** `bp_B[a][v]` — the same kind of list, but over **successors**: `[]` for a sink; `[(a, w')]` (source point `a` *itself*, `w' ∈ Bsucc(v)`) for a downstream coverage step; or `[(s₁, w₁), (s₂, w₂), …]` (each successor `sᵢ` at `wᵢ ∈ {v} ∪ Bsucc(v)`) for an advance. It is populated exactly as $D$'s list is (§4.1), with successors in place of predecessors. The extraction reads `bp_D` upstream and `bp_B` downstream.

**(V3) is enforced by the shared cell index — and computing its cost is then a sum, not a joint search.** (V3) requires every successor of a split to leave from the *same* point, the split point $v$. That shared point **is the cell's index**: every successor's minimum in $B[a][v]$ is taken relative to the *same* $v$ (over $\{v\} \cup B_{\text{succ}}(v)$ = "continue from $v$"), so a matching in which two successors leave from *different* points is **not even representable** in a single cell. The traceback (§5.2) then commits one $v$ for the split and sends **all** its successors out of it — that is (V3), by construction; nothing is relaxed. This is the exact **mirror** of how the predecessor coupling in $D$ enforces (V2) at a merge — *each coupling enforces its own rule by construction*. What a single directed sweep cannot do is carry **both** rules at once: the successor coupling alone is exact for an **out-tree** (splits, no merges), the predecessor coupling alone for an in-tree (merges, no splits). A source with **both** junction types keeps both tables and threads them together in the joint traceback (§5) — that combination is not a repair, it is how the two by-construction couplings are reconciled.

What the tree buys is *efficiency*. Given $a$ fixed at $v$, the successors' downstream subtrees are disjoint (§2), so the V3-constrained cost **factorizes**: each successor's best is computed **independently** ($\min_w B[s][w]$, over its own subtree) and the results are **summed** — because the subtrees don't touch, that sum is the exact joint cost of all branches leaving $v$. So the V3-constrained cost is a **scalar per cell**, re-summed for each candidate $v$, never a coupled multi-successor state — and no joint Dijkstra is needed (that would be exponential state for no gain; Dijkstra is only for the *coverage* move, §4.3). The forward $D$ does the exact mirror for a merge's predecessors. This factorization is precisely what a diamond breaks: if two branches later rejoin they stop being independent given $v$, the sum mis-counts the shared part, and only then would you need extra conditioning or a genuine joint search.

The two tables are the two halves of the same quantity:

* $D[a][v]$ = best cost of everything **upstream** of $a$, with $a$ at $v$ → it couples **merges**.
* $B[a][v]$ = best cost of everything **downstream** of $a$, with $a$ at $v$ → it couples **splits**.

Combining them gives the true global cost of pinning any point $a$ to target point $v$:

$$\text{GlobalCost}(a, v) = D[a][v] + B[a][v] - E(a, v)$$

*(We subtract $E(a,v)$ once because point $a$'s own emission is counted in both $D$ and $B$.)*

### 4.3 Implementation — How the Tables Are Filled

The recurrences are evaluated as a DP with **two nested levels**: an outer sweep over the source points (rows) and, for each row, an inner computation over the target points (columns).

**Outer level — over the source $G_A$ (the rows): a plain topological sweep.** Fill $D$ row by row in **topological order** of $A$ (sources → sinks); fill $B$ in **reverse** topological order (sinks → sources). Because $A$ is a tree it always has such an order, so this level is a straight sweep — **never a shortest-path search**. Each row reads only already-filled neighbour rows: $D[a][\cdot]$ from the predecessor rows $D[p][\cdot]$, $B[a][\cdot]$ from the successor rows $B[s][\cdot]$.

**Inner level — over the target $G_B$ (the columns of one row), for a fixed source point $a$.** Two cases:

* **Point-to-point (no coverage).** With $\text{reach}(v) = B_{\text{pred}}(v) \cup \{v\}$, the cell
  $$D[a][v] = E(a,v) + \sum_{p \in A_{\text{pred}}(a)} \tfrac{1}{\text{outdeg}(p)} \min_{x \in \text{reach}(v)} D[p][x]$$
  reads only the *predecessor* rows — never $D[a][\cdot]$ itself. So the cells of a row are **independent**; fill them in any order with a **plain scan** over the candidate B-points. Cost $O(\text{band})$ per row, $O(|A| \times \text{band})$ overall. **No Dijkstra.**

* **N:1 coverage (the $\beta$ vertical move).** Alongside the full-cost advance (line (D), each predecessor takes $\text{step}_p$), take the $\beta$-discounted stall (line (V)):
  $$D[a][v] \leftarrow \min\Big(\, D[a][v],\ \ \beta\,E(a,v) + \min_{q\in A_{\text{pred}}(a)}\big[\tfrac{D[q][v]}{\text{outdeg}(q)} + \sum_{p\neq q}\tfrac{\min(D[p][v],\ \text{step}_p)}{\text{outdeg}(p)}\big] \,\Big).$$
  It reads only **already-filled predecessor rows** (never $D[a][\cdot]$), so it is a **plain per-cell update** in the outer topological sweep — coverage along A needs **no search**, because $A$ is a tree. The $\min_q$ forces one branch to stall on $v$, so $\beta$ is never charged on a pure advance; for a **chain** it collapses to $\beta E + D[p][v]$. It costs $O(\text{in-degree})$ per cell (equivalently $\sum_p \tfrac{\min(\text{stall}_p,\text{step}_p)}{\text{outdeg}(p)} + \min_q \tfrac{\text{stall}_q-\min(\text{stall}_q,\text{step}_q)}{\text{outdeg}(q)}$, the second term $0$ once any branch already stalls). Chaining is automatic: $D[q][v]$ may itself have come from *its* stall, so a run of $m$ stacked A-points accrues $\beta E$ each.

* **1:N coverage (the $\alpha$ horizontal move).** The horizontal term $\min_{v' \in B_{\text{pred}}(v)} D[a][v']$ makes a cell depend on **other cells in the same row**, along the B-edges — a within-row shortest path. Fill the row in two steps: (1) **inject** each B-point $v$ with its A-advance / $\beta$-stall cost above; (2) **relax** the horizontal edges $v' \to v$ with weight $\alpha\,E(a,v)$:
  $$D[a][v] \leftarrow \min\big(\, D[a][v],\ \ \alpha\,E(a,v) + D[a][v'] \,\big).$$
  Because those weights are **non-negative** ($\alpha E \ge 0$):
  * if the local B-graph is **acyclic**, one pass in **B-topological order** suffices;
  * if it **cycles** (roundabouts, one-way loops — $G_B$ may cycle), use **Dijkstra** (multi-source, one entry per injected point) or bounded iterative relaxation until convergence.

  This within-row relaxation is the **only** place a shortest-path search appears; with $\alpha = 1$ it is the plain per-point coverage charge, and with $\alpha = \beta = 1$ neither extension ever lowers a cell, so the point-to-point recurrence above is recovered exactly.

$B$ is filled the same way, with both moves mirrored: the $\beta$ stall over successors ($\sum_{s} D[s][v]$) and the $\alpha$ horizontal over $B_{\text{succ}}(v)$ on the reversed local B-graph.

---

## 5. The Traceback Stage: Extracting the Match Relation `M`

> **The default extraction is now the forward-only anchored extraction** — `extract(A, B, α, β)`,
> run after `prepare` + `forward_v3` (§4.1a): one table, two pointer types (`bpD` up, its transpose
> down), anchor enumeration with reject-and-retry, direct-cost selection. Protocol and measurements:
> `docs/tree_dtw_minimal_matching.md`, "Fork B realized". The two-table traceback specified in this
> section remains available as **`extract_two_table`** (it is what the §6b cross-table diagnostics
> compare against, and requires the §4.2 backward table).

### 5.1 Intuitive Concept

1. **Why one table isn't enough.** Each coupling enforces its own rule **by construction**: the predecessor sum in $D$ makes every merge's approaches meet at one point (V2), and — its exact mirror — the successor sum in $B$ makes every split's exits leave one point (V3). But a single directed sweep carries only **one** of the two. If you take just $D$ and read an independent $\arg\min$ per sink, the splits are left uncoordinated: two branches of a fork place it at **different** points — the phantom of §4.1.
2. **The fix — seed once, then *follow the stored back-pointers*.** There is exactly **one** $\arg\min$ in the whole extraction: the seed, which pins one representative point per source-tree component to its joint optimum $\arg\min_v\, D[a][v] + B[a][v] - E(a,v)$. From there the walk **reads the back-pointers `bp_D`/`bp_B` recorded during the passes** — it never re-minimises per point and never guesses a connection. Upstream it follows `bp_D` (a merge's approaches already agree on the cell — V2); downstream it follows `bp_B` (a split's exits all leave the one committed cell — V3); a source point's 1:N coverage run is read straight off its **same-source pairs** (the `(c, v')` entries). Because every transition was constrained to $\text{reach}(v)$ / $\{v\}\cup B_{\text{succ}}(v)$ **when the cell was written**, each committed point sits within one B-step of its neighbour, so the extracted matching is monotone **by construction**. (Reading a stored pointer, not re-solving, is the whole point: it is why classic DTW keeps back-pointers, and it is what makes the result the *actual* optimum the tables found rather than a re-guess of it.)

### 5.2 Extraction Pseudocode

```text
Algorithm: Extract-Tree-DTW-Matching(G_A, G_B, D, B, bp_D, bp_B, E)
Input:  the two cost tables AND their back-pointers bp_D, bp_B (§4.1 / §4.2)
Output: M — set of matched pairs (a, v)

M ← ∅ ;  Committed ← empty map ;  Queue ← empty

commit(x, w):                                     # pin source point x to target cell w, once
    if x ∉ Committed:
        Committed[x] ← w ;  M.add((x, w)) ;  Queue.enqueue(x)

# --- Step 1 — the ONLY argmin: seed one representative per source-tree component ---
for each weakly-connected component of G_A, pick a representative r:
    commit( r, argmin_v ( D[r][v] + B[r][v] − E(r, v) ) )

# --- Step 2 — FOLLOW the stored back-pointers until every point is committed ---
while Queue not empty:
    c ← Queue.dequeue() ;  v ← Committed[c]

    # (a) c's 1:N coverage run: follow the coverage pairs (a pair whose source point is c itself)
    head ← v ;  while bp_D[c][head] = [(c, v')]:  head ← v'        # same-c pair -> step back
    tail ← v ;  while bp_B[c][tail] = [(c, w')]:  tail ← w'        # same-c pair -> step forward
    for w on the B-path head → … → v → … → tail:  M.add((c, w))    # c covers this whole run

    # (b) UPSTREAM — the pairs stored at the run's head ARE c's predecessors (V2, no re-min)
    for (p, x) in bp_D[c][head]:   commit(p, x)

    # (c) DOWNSTREAM — the pairs stored at the run's tail ARE c's successors (V3, no re-min)
    for (s, w) in bp_B[c][tail]:   commit(s, w)

return M
```

Everything after the seed is a **list read, never a search**. At the run's `head`, `bp_D[c][head]` is exactly the list of predecessor cells `D[c][head]` was summed from — they already agree on `head`, so committing them all **is** (V2); symmetrically the list at `tail` sends every successor out of the one cell, which **is** (V3). A source point's coverage run is walked off the same-`c` pairs — exactly the run the DP priced. Because $G_A$ is a tree, the walk reaches every point on exactly one path and `commit` fixes it once; a split and a merge can never fight over a cell. The result is the exact optimum the tables encode, valid V1–V4 by the construction of §4.1/§4.2 — the extraction only *reads* it.

#### Worked walk — the order points are committed

Source `a₀ → a₁ → {a₂, a₃}` (a chain that splits at `a₁`), with the back-pointer lists the passes stored (no coverage, so every run is a single cell):

```text
bp_D[a₀][w₀] = []                 bp_B[a₀][w₀] = [(a₁,w₁)]
bp_D[a₁][w₁] = [(a₀,w₀)]          bp_B[a₁][w₁] = [(a₂,w₂),(a₃,w₃)]   # split → both exits
bp_D[a₂][w₂] = [(a₁,w₁)]          bp_B[a₂][w₂] = []                  # sink
bp_D[a₃][w₃] = [(a₁,w₁)]          bp_B[a₃][w₃] = []                  # sink
```

Seed the first uncommitted point, `a₀` → `commit(a₀, w₀)`, then drain the queue — each pop commits the neighbours named in its two lists, skipping any already committed:

| pop | `bp_D` (predecessors, upstream) | `bp_B` (successors, downstream) | queue after | committed |
|---|---|---|---|---|
| `a₀` | `[]` — source | `(a₁,w₁)` → commit `a₁` | `[a₁]` | `a₀, a₁` |
| `a₁` | `(a₀,w₀)` — already committed | `(a₂,w₂), (a₃,w₃)` → commit `a₂, a₃` | `[a₂, a₃]` | `a₀…a₃` |
| `a₂` | `(a₁,w₁)` — already | `[]` — sink | `[a₃]` | all |
| `a₃` | `(a₁,w₁)` — already | `[]` — sink | `[]` | all |

The walk **floods outward from the seed in both directions** — down `a₀→a₁`, then from `a₁` back up to `a₀` (already done) *and* down to **both** `a₂` and `a₃`; it is **not** a topological sweep. **Every point is committed exactly once** — the "already committed" guard turns each examined edge into at most one commit (siblings are reached by going up to the shared parent, then back down). The seed's identity is irrelevant: seeding the leaf `a₂` instead gives the order `a₂ → a₁ → {a₀, a₃}` but the **same** points and pins. A forest seeds one representative per component.

### 5.3 The Split from §4.1, Resolved

$B$ sums **both** branches at each candidate for $s$:

```
B[s][σ]  =  (a-branch from σ : 0)   +  (b-branch from σ : 10)  =  10
B[s][σ'] =  (a-branch from σ' : 10) +  (b-branch from σ' : 0)   =  10
```

Both are `10`, so the split commits to **one** point (say $s=\sigma$); branch $a$ takes its downstream match $v_a$ (cost 0) and branch $b$ is forced from $\sigma$ to $v_b$ (cost 10). The matching is $\{(s,\sigma),(a,v_a),(b,v_b)\}$ with cost **10** — the true minimum. The phantom `0` is gone because $B$ never let the two branches pick different $s$.

### 5.4 Calculating the Final Total Cost

You cannot find the true cost by summing the minima of the tables alone; you must wait until the consistent relation $M$ is generated. The reported cost is the raw sum over its matched pairs:

$$C(M) = \sum_{(a,v) \in M} E(a, v)$$

(With $\alpha < 1$ the *decision* cost inside $D$/$B$ is discounted, but the reported drift stays this raw $\sum E$ — $\alpha$ only steered which $M$ was chosen, §4.1.)

---

## 6. Splits and Merges — the Duality

| Feature | **Merge** (in-degree > 1) | **Split** (out-degree > 1) |
|---|---|---|
| **Meaning** | two roads **join** into one | one road **forks** into two |
| **Enforced rule** | **(V2)** all roads arrive at one target point | **(V3)** all exits leave one target point |
| **Coupled by** | **forward** sum over predecessors (in $D$) | **backward** sum over successors (in $B$), *or* **forward** forbid-and-rebuild (§4.1a) |
| **Traceback logic** | read stored $D$ back-pointers (descend into all) | filter options via $B$ lookahead (inherit into all) |
| **If done independently** | roads enter at different points | exits leave at different points |

Neither junction is harder than the other on a tree — each is handled exactly by summing at it in the appropriate direction, and each is consistent because its arms are independent (§2).

---

## 7. Guarantees

On a genuine directed tree, Tree-DTW returns a matching $M$ that is:

* **Valid** — (V1)–(V4) hold by construction: the forward sum enforces (V2) at merges, the backward sum enforces (V3) at splits, the monotone traceback enforces (V1), the boundary handles (V4).
* **Optimal** — the cost $C(M) = \sum_{(a,v)\in M} E(a,v)$ of the extracted matching is the exact minimum over all valid matchings. It is **not** the forward pass's $\sum_{\text{sinks}} \min_v D$, which is optimistic at a split; the consistent traceback of §5 couples **both** junctions — forward at merges, backward at splits — and on a tree, with no loop, that is exactly the minimum.
* **Efficient** — each of the forward and backward passes is $O(|G_A| \times \text{search band})$ and the traceback is $O(|G_A|)$. Nothing is carried between a junction's arms, because a tree's arms are independent (§2): separate branches never rejoin to complicate constraints, so no exponential blow-up.
* **Point-to-point (or 1:N)** — in the common case each point matches exactly one target point, so $M$ is a one-arrow-per-point map; under coverage a point's run contributes several pairs to $M$ (priced by $\alpha$, §4.1). Per-source-edge routes are read off $M$ by grouping consecutive matches by target edge.

The one thing Tree-DTW requires is that the source really is a **tree**. A source in which a road forks and later rejoins itself is not a tree: its merge-roads share an ancestor, the forward sum stops being a true cost (§2), and the guarantees above no longer hold — that structure needs a different treatment and is out of scope here.

---

## 8. Segment-to-Segment Matching — the Segment-State DP (`emission="segment"`)

> Implemented in `network_matching/tree_dtw.py`: the six parts of §9 run **unchanged** on the directed line graphs $L(G_A)$, $L(G_B)$ (`line_digraph`). `emission="point"` (§2–§7) is unchanged and stays the default.

### 8.1 Why a segment *state*, not a segment *cost*

The point-state DP of §2–§7 pairs **points**: a state is $(a, v)$ and the warping path is a chain of point-to-point arrows. You can dress up its *cost* with a heading term — score $a$ against $v$ using the midpoints of the micro-segments each vertex happens to own — but the **state is still a point pair**, and that has a concrete failure. On a coverage/stall move several source points collapse onto **one** target point, where there is no target *segment* at all; a heading penalty is then undefined and silently charged as zero, so the DP can dodge it by stalling. This is the documented graph-DTW finding (`weighted_emission.md` §9–§11).

True segment-to-segment matching makes the **state itself a segment pair**. Every state is two real directed segments, so the local cost — heading included — is always defined and **no move can bypass it**.

### 8.2 The lift: run the point-state algorithm on arcs

Both local digraphs already carry the segments we need. A **source micro-segment** is an *arc* of $G_A$ — a directed edge $s = (t \to h)$ between consecutive pooled source points; a **target arc** is an edge $e = (u \to v)$ of $G_B$. The segment-state DP is the §4–§5 algorithm run on the **arc line-graph** $L(G_A)$ — its nodes are the source arcs, with $s \to s'$ whenever $\text{head}(s) = \text{tail}(s')$ — against the target **arcs**. Everything else is a term-by-term substitution:

| point-state (§2–§7) | segment-state (§8) |
|---|---|
| source point $a$ | source arc $s = (t \to h)$ |
| target point $v$ | target arc $e = (u \to v)$ |
| $E(a, v) = \lVert a - v\rVert$ | $E(s, e) = \lVert \operatorname{mid}(s) - \operatorname{mid}(e)\rVert + \lambda\cdot\operatorname{circ}\!\big(\operatorname{bear}(s), \operatorname{bear}(e)\big)$ |
| predecessors $A_{\text{pred}}(a)$ | source arcs ending at $t$ (predecessors in $L(G_A)$) |
| reach $\{v\} \cup B_{\text{pred}}(v)$ | $\{e\} \cup \{p : \text{head}(p) = \text{tail}(e)\}$ — the same target arc (stall) or one adjacent before |
| horizontal coverage over $B_{\text{succ}}(v)$ | coverage over target arcs adjacent to $e$ |

$\operatorname{mid}(\cdot)$ is the segment midpoint; $\operatorname{bear}(\cdot)$ the compass bearing $(\deg\cdot\operatorname{atan2}(\Delta x, \Delta y)+360)\bmod 360$ ($0°=$ north); $\operatorname{circ}(\theta,\phi)=\min(|\theta-\phi|, 360-|\theta-\phi|)\in[0,180]$; $\lambda=$ `bearing_weight`. Because the substitution is exact, the forward table $D$ (§4.1), the backward table $B$ (§4.2), and the joint $D+B-E$ traceback (§5) carry over unchanged **in form** — only the index sets (arcs, not vertices) and $E$ differ. $L(G_A)$ is a directed acyclic graph whenever $G_A$ is ($s \to s' \to \cdots \to s$ would trace a directed cycle in $G_A$), so the topological sweeps of §4.3 still apply.

### 8.3 Emission paid by every state

Every hosted state $(s, e)$ pays $E(s, e)$ — **including on the N:1 stall (§8.5)** — so the §8.1 bypass is closed: a stall costs $\beta\cdot E$ for each additional source arc, never zero. In the current representation a **junction is a single vertex** (Part 1), so the line graph connects real segments directly — there are no stitch connectors, hence no free pass-through a state could park on (the free-connector collapse of `weighted_emission.md` §10 cannot occur).

### 8.4 Junctions — how segment-states couple at a merge and a split

The couplings are the §4/§6 couplings, read at the shared **target vertex** where the junction pins:

* **Merge** (source vertex $m$: incoming arcs $s_1, \dots, s_d$, one outgoing arc $s'$ riding $e = (u \to v)$). All approaches meet at one target point — here $u = \text{tail}(e)$. The **forward** table sums the predecessors, each minimised over its reach that **ends at $u$**: $D[s'][e]$ is finite only if *every* $s_i$ can end on a target arc whose head is $u$. That is (V2), in arc form.
* **Split** (a source vertex with one incoming arc and outgoing arcs $s_1, \dots, s_d$). All exits leave one target point $\sigma$. The **backward** table sums the successors over the shared cell — each $s_j$ rides a target arc with $\text{tail} = \sigma$. That is (V3), in arc form.
* **Merge *and* split at the same vertex.** The incoming and outgoing arcs of $m$ all pin to the **one** target vertex $\sigma$: incoming end at $\sigma$ (forward), outgoing start at $\sigma$ (backward), the joint traceback commits the single $\sigma$. In $L(G_A)$ these arcs form a small bipartite cluster (for a 2-in/2-out vertex, an *undirected* 4-cycle), but this is **not a reconvergence** — it is one shared pin, exactly the §5.2 vertex coupling applied to a vertex that is both merge and split. Independence (§2) is untouched: the incoming cones are disjoint, the outgoing cones are disjoint, and given $\sigma$ the two sides do not interact. So $L(G_A)$'s undirected cycles are harmless; the reconvergence that §7 forbids (in $G_A$ itself) is what would break independence, and that is still rejected by `NotATree`.

### 8.5 Moves and the coverage weights $\alpha$, $\beta$

Three moves generate every alignment — the §4.1 moves with arcs in place of points:

| move | meaning | weight |
|---|---|---|
| **both advance** | the next source arc rides an *adjacent* target arc (one step in $G_B$) — 1:1 | full $E$ |
| **A-advance, same arc** | consecutive source arcs ride the *same* target arc — **N:1**, a source denser than the target | $\beta\,E$ |
| **coverage** | one source arc spans a *run* of consecutive target arcs — **1:N**, a target denser than the source (a within-row shortest path over adjacent arcs) | $\alpha\,E$ |

$\beta$ is the (V) stall of §4.1 read with arcs: it discounts an *extra* source arc landing on the **same** target arc, triggered by ≥1 predecessor arc already on it (the same ≥1-stall, charge-once, force-one-stall rule). $\alpha$ is the (H) coverage. The coverage relaxation is the only shortest-path search — Dijkstra when $G_B$ cycles, a topological pass when it does not. $\alpha, \beta, \lambda$ (bearing weight, §8.2) are **call-time hyperparameters**; $\alpha = \beta = 1$ (defaults) charge every arc in full — segment mode's point-to-point-equivalent pricing.

### 8.6 Output — the matching lives on the line-graph (no point conversion)

Segment mode's matching **is** a relation on segments: $M_{\mathrm{seg}} \subseteq \mathrm{arcs}(G_A)\times\mathrm{arcs}(G_B)$, each source arc paired with the target arc it was committed to (§5) plus any target arcs on its 1:N coverage run. It is emphatically **not** collapsed to a point matching. A segment matcher emits segment pairs; a per-point $\varphi$ would re-introduce exactly the collapse §8.1 exists to prevent — several source points landing on one target point, where the segment and its heading vanish — so the segment matching is neither *produced* nor *validated* through a point conversion.

The extraction stores the §4.1/§4.2 back-pointers on the **arc** tables and runs the §5 walk over $L(G_A)$: seed one arc per component, then follow `bp_D`/`bp_B`. Because it only reads the stored pointers (written constrained to adjacent arcs), a split's arms leave the **one** committed target arc (V3) and a merge's arms meet on one (V2) *by construction*.

**Coverage partitions the target arcs.** Each source arc's 1:N run is read from the **forward-table** COVER chain only — the target arcs strictly after its predecessor's committed arc, up to and including its own. So every target arc belongs to **at most one** source arc: no run reaches past the next source arc, no two runs overlap, and there is no gap-fill. (Combining the forward *and* backward COVER chains — as a naïve seed-in-the-middle walk would — double-counts the arcs between two neighbours' anchors and is the one thing that breaks the partition.)

Outputs are all read **directly from $M_{\mathrm{seg}}$**:

* `routes` — per source edge, the target edges its arcs match, by grouping $M_{\mathrm{seg}}$ on source/target edge id.
* `segment_pairs` — per source arc, the middle-to-middle link (and endpoints) to its committed target arc, for the correspondence view.
* `M_seg` — the arc relation itself.

A per-point `M`/$\varphi$ may still be offered as a clearly-labelled interop convenience, but it is a *derived* object and is **never** validated in place of $M_{\mathrm{seg}}$.

### 8.7 Validation — V1–V4 on the line-graph (independent of the point validator)

Segment mode is validated by the four rules of §3 read **on the arc line-graph**, against $M_{\mathrm{seg}}$ — never by converting to a point matching and validating that. That conversion is the §8.1 collapse; it would test a derived object rather than the matching, and its over-assigned coverage produces phantom V1 crosses even when the segment matching is monotone and correct.

`check_rules(M_seg, L(G_A), L(G_B))` (Part 6 — the **same function** as point mode, on different graphs) uses the $L(G_A)$ adjacency (predecessor/successor **arcs**) and the target-arc adjacency ($B_{\text{pred}}$/$B_{\text{succ}}$ on arcs), each rule restricted to matched neighbours:

* **(V1) no cross** — for $(s,e)\in M_{\mathrm{seg}}$, no predecessor arc of $s$ is matched to a successor arc of $e$.
* **(V2) merge** — $s$ either continues a run ($e$ has a matched target-arc-predecessor also on $s$) or **every** matched predecessor arc of $s$ lands on $e$ or a target-arc-predecessor of $e$.
* **(V3) split** — symmetrically, over successors and target-arc-successors.
* **(V4) coverage** — every source arc appears in $M_{\mathrm{seg}}$.

This is exactly the per-cell check the table validator already applies (Part 6, `validate_tables`), now applied to the final arc matching. Point and segment validation are the same function on different graphs; **neither substitutes for the other**.

### 8.8 What is unchanged

(V1)–(V4) (§3) — now read on $L(G_A)$ (§8.7) — the coverage weights $\alpha$ and $\beta$ (§4.1), the tree-only requirement and `NotATree` (§7), and the exactness/efficiency guarantees (§7) hold verbatim: the segment-state DP is the same algorithm on $L(G_A)$, whose junction cones stay independent (§2) exactly when $G_A$ is a tree. Complexity is the same order — $|L(G_A)| = |{\it arcs}(G_A)| \approx |G_A|$ states against $\approx |G_B|$ target arcs, one forward and one backward sweep plus the per-row coverage relaxation. `emission="point"` remains the default and byte-for-byte the §2–§7 algorithm, validated by `check_rules` on its point $M$.

### 8.9 Implementation notes

* **Segment mode is not separate code.** It is the six parts of §9 run on `line_digraph(A)`, `line_digraph(B)` — `nx.line_graph` gives exactly the directed arc adjacency ($(a,b) \to (c,d)$ iff $b = c$); midpoint and bearing are attached to each L-node from the original endpoint coordinates (it copies no attributes itself).
* **One emission serves both modes and both sweeps.** $E$ = position distance + $\lambda\cdot\operatorname{circ}(\text{bearing})$ whenever both nodes carry a `bearing` (Part 2); it is invariant under reversing both graphs (every bearing rotates $180°$, $\operatorname{circ}$ unchanged), so $D$ and $B$ share it.
* **A junction is a vertex — there are no stitches.** A merge/split becomes several predecessors/successors of one L-node directly; the §8.4 couplings fall out of the ordinary sums with no special case, and there is no free connector to park on (§8.3).
* **Output and validation are native to $L(G_A)$.** The matching is $M_{\mathrm{seg}}$ on arcs (§8.6); validation is `check_rules` on $L(G_A)$/$L(G_B)$ (§8.7). Any per-point $M$/$\varphi$ is a clearly-derived interop convenience and is never validated in place of $M_{\mathrm{seg}}$.
* **The extraction is shared with point mode.** Both modes store `bp_D`/`bp_B` (§4.1/§4.2) and run the one §5 back-pointer walk — point mode over **vertices**, segment mode over **arcs**. Following the stored pointers is what makes each junction consistent by construction; coverage is read from the forward COVER chain only, so runs partition (§8.6).

---

## 9. Implementation — networkx (`network_matching/tree_dtw.py`)

The matcher is implemented on plain **`networkx.DiGraph`** objects, in six independently-verifiable parts: representation + candidates (Part 1), emission (Part 2), forward `D` (Part 3), backward `B` (Part 4), extraction (Part 5), validation (Part 6). **Segment mode is not separate code** — the same six parts run on the directed line graphs `L(A) = line_digraph(A)`, `L(B) = line_digraph(B)` (§8). The matching is the relation of §3: `M ⊆ V(A) × V(B)` (point) or `M ⊆ E(A) × E(B) = V(L(A)) × V(L(B))` (segment); the DP minimises the **decision cost** `Σ w(a,v)·E(a,v)` with `w = 1` on a 1:1 advance, `α` on a 1:N coverage step, `β` on an N:1 stall (§4.1), while the **reported** drift stays the raw `Σ E` of the chosen matching (§5.4).

### Part 1 — Representation & candidate gating

#### 1.1 Input graphs

* **A** (source tree) and **B** (target network) are `networkx.DiGraph`; node ids are any hashable value.
* Every node carries float coordinates in attributes **`x`, `y`**. Edges are the directed **segments**; a node's coordinates are its geometry (segments are straight between endpoints).
* **A must be a tree**: its underlying *undirected* graph is acyclic (`nx.is_forest(A.to_undirected())`). A directed reconvergence (a diamond) has an undirected cycle and is rejected — `NotATree` — because it breaks the junction independence of §2. B has no such restriction (it may cycle).
* A **junction is a vertex**: `out_degree > 1` is a split, `in_degree > 1` is a merge, both at once is a merge+split. Nothing is "coincident"; there are no stitches.

#### 1.2 Candidates — radius-gated, stored on the node

For an A-vertex `a`, a **candidate** is a B-vertex `v` it may match to, gated by distance:

* Parameter **`r = match_radius_m`** (default **20 m**); `cand(a) = { v ∈ V(B) : ‖a − v‖ ≤ r }`, found with a KD-tree over B's vertex coordinates.
* **Non-empty guarantee.** If fewer than `k_min` (default 1) B-vertices lie within `r`, the `k_min` nearest are included anyway, so no row is ever empty for a purely-geometric reason.

Each A-vertex stores its **own candidate table** as a node attribute — this *is* that vertex's row of the DP tables, filled progressively by the later parts:

```python
A.nodes[a]["cand"] = {
    v: {"E": ‖a − v‖,            # emission (Part 2)
        "D": +inf, "bpD": [],    # forward cost + back-pointers (Part 3)
        "B": +inf, "bpB": [],    # backward cost + back-pointers (Part 4)
        "forbidden": False}      # §4.1a: when set, no back-pointer may target this cell
    for v in cand(a)
}
```

`prepare(A, B, r)` validates the inputs, builds the KD-tree, and fills `E`; `D`/`bpD`/`B`/`bpB` are placeholders until Parts 3–4.

#### 1.3 Feasibility rule

Radius gating can make the *coupled* DP infeasible even when every row is individually non-empty: the warping is a chain, and `D[a][v]` needs a predecessor candidate in its reach `{v} ∪ Bpred(v)`. If `r` is smaller than the true A↔B drift somewhere along a path, that chain breaks and every cell of some vertex becomes `∞`. So:

* `r` should be **≥ the largest expected A↔B drift** (synthetic tests: < 2 m; NVDB↔OSM: 10–20 m).
* After the DP, if any A-vertex has **no finite** `D+B` entry, **raise** `ValueError("vertex … unreachable within r=…; increase match_radius_m")` — never return a broken match.
* The per-vertex check is **not sufficient at a merge/split**, whose arms are only coupled during the traceback: a vertex can have a finite `D+B` at its arg-min yet the coupled optimum still runs through an infeasible cell, recorded as a **severed back-pointer** (a `None` cell reference). The extraction (Part 5) guards this — a `None` while following `bpD`/`bpB` raises the same feasibility `ValueError` rather than dereferencing the missing cell.

### Part 2 — Emission `E` and the node-attribute contract

Both modes run the **same** DP on a graph whose nodes carry a **position** and — in segment mode — a **bearing**:

| mode | graph | a node is… | node attributes |
|---|---|---|---|
| **point** | `A`, `B` | a vertex | `x, y` (position, meters) |
| **segment** | `L(A)`, `L(B)` | a segment `(u, v)` | `x, y` = **segment midpoint**, `bearing` = **compass bearing** |

One emission formula serves both, using whatever the nodes carry:

```
E(a, v) = ‖pos(a) − pos(v)‖  +  λ · circ(bearing(a), bearing(v))     ← 2nd term only if both carry `bearing`
```

`bearing` = `(deg·atan2(Δx, Δy) + 360) mod 360` (0° = north); `circ(θ, φ) = min(|θ − φ|, 360 − |θ − φ|)`; `λ = bearing_weight`. Gating (Part 1.2) uses `pos` — midpoint-to-midpoint in segment mode. `E` is symmetric under reversing both graphs, so the forward and backward passes share it.

### Part 3 — Forward table `D` (upstream cost) — on the node

`D[a][v]` = minimum cost of matching the **upstream cone** of `a` (a and its ancestors) with `a` pinned at candidate `v`; `bpD[a][v]` records the cells it was computed from. Both live in `A.nodes[a]["cand"][v]`. A is swept in **topological order** (sources first; the layered order of §4.0 when the V3 coupling of §4.1a runs). For each predecessor `p ∈ pred(a)`, its two ways into `v` (both read from `p`'s gated candidate table):

```
step_p  = min_{x ∈ Bpred(v) ∩ cand(p)} D[p][x]      # p advances one B-arc into v
stall_p = D[p][v]   (∞ if v ∉ cand(p))              # p already sits on v
```

`D[a][v]` is the cheapest of a full-cost advance (D), a **β**-discounted stall (V), and an **α**-discounted coverage step (H) — the recurrence of §4.1:

```
D[a][v] = min {
  (D)  E(a,v) + Σ_{p∈pred(a)} step_p / outdeg(p)                                                   full E
  (V)  β·E(a,v) + min_{q∈pred(a)} [ stall_q/outdeg(q) + Σ_{p≠q} min(stall_p, step_p)/outdeg(p) ]    β·E
  (H)  α·E(a,v) + min_{v'∈ Bpred(v) ∩ cand(a)} D[a][v']                                             α·E
}
```

* **Source** (`pred(a)=∅`): empty sum → `D[a][v] = E(a,v)` (free entry). **Merge**: the predecessor **sum** is finite only if every predecessor can reach `v` — (V2) folded into the cost; `1/outdeg(p)` is the split factor (§4.1).
* All predecessor / B-neighbour look-ups are **intersected with the gated candidate sets**: a candidate pruned by `r` has `D = ∞`, so it can't be a `step`, `stall`, or coverage source; a chain the gate has severed leaves every cell `∞` and trips the feasibility rule (Part 1.3).

**Back-pointer `bpD[a][v]`** records exactly the cells that produced the winning line (§4.1's list): `[]` for a source; `[(p, x_p), …]` — one chosen cell per predecessor — for an advance/stall; `[(a, v')]` — same source, one B-arc back — for a coverage step.

**Line (H) is a within-row fixed point, iterated to convergence.** Unlike (D)/(V), which read only *predecessor* rows (already final under the topological sweep of A), (H) reads **other cells of the same row** — `D[a][v]` depends on `D[a][v']` for `v' ∈ Bpred(v) ∩ cand(a)`. B carries no order of its own, so the row cannot be filled in one arbitrary pass; it is **relaxed until it stops changing**:

```
repeat over cand(a) until no cell changes:
    for each B-arc v' → v with v', v ∈ cand(a):
        if D[a][v'] + α·E(a,v) < D[a][v]:
            D[a][v]  = D[a][v'] + α·E(a,v)        # lower the cost …
            bpD[a][v] = [(a, v')]                  # … and repoint the back-pointer in the SAME step
```

The cost and its back-pointer are updated **together** — `bpD[a][v]` always names the cell that produced the current `D[a][v]`, so the chain never desyncs from the value. Because the coverage weight `α·E ≥ 0`, the relaxation is a monotone descent bounded below, converging to the **unique least fixed point** — the same result whether B is acyclic or **cyclic** (a naive single pass would leave cells un-relaxed on a cycle; convergence does not). No order on `V(B)` is assumed; only B's arcs and cell costs drive it, with `border` (Part 4b) settling exact-cost ties deterministically.

### Part 4 — Backward table `B` (downstream cost) — on the node

`B[a][v]` mirrors Part 3 over the **downstream cone** (a and its descendants): the identical three-way `min` — same **α**/**β**, same emission `E` — with A and B **reversed**: `pred → succ`, `Bpred → Bsucc`, `outdeg → indeg`, swept in reverse topological order. Concretely `step_p = min_{x ∈ Bsucc(v) ∩ cand(p)} B[p][x]`, `stall_p = B[p][v]`, over `p ∈ succ(a)`. Stored in `A.nodes[a]["cand"][v]` as `B`, `bpB`. This is the split coupling (V3) exactly as `D` is the merge coupling (V2) (§4.2, §6).

#### 4b Deterministic argmin — a fixed B-vertex order

Every `argmin` in Parts 3–4 (the advance step's choice of predecessor cell, and the (H) relaxation's choice of coverage predecessor) is broken among **equal-cost** options by iteration order — which is arbitrary and, worse, can be broken **differently** by the forward and backward passes (they scan `Bpred` vs `Bsucc`). That makes the tables non-reproducible and lets `bpD`/`bpB` diverge on a tie.

`_b_order(B)` fixes one total order on B's vertices, `border = {v: rank}` (sorted by id, independent of insertion order), and both passes use it to break ties: among cells of equal cost the **smallest-`border` one wins**, and the (H) relaxation likewise keeps, on an exact-cost tie, the coverage predecessor of smallest `border`. This changes only *which* of two **exactly-equal** cells is stored — never a strictly-cheaper choice — so `D`/`B` **costs and optimality are unchanged**; only the tie is resolved, **identically in both passes**. Result: the tables are **deterministic** (invariant to B's dict order). *(It does not, by itself, make reciprocity hold under weighting — those failures are strict cost preferences, not ties, Part 6b/§4.2. The extraction seed is a separate order-dependent site, Part 5.)*

### Part 5 — Extraction → `M`

> **Default:** `extract(A, B, α, β)` is the forward-only anchored extraction (see the §5 banner and
> `docs/tree_dtw_minimal_matching.md`). The two-table traceback below is **`extract_two_table`**.

The joint two-table traceback (§5), reading only the stored back-pointers.

1. **Seed.** Pick any still-uncommitted A-vertex `r` and its `v* = argmin_{v ∈ cand(r)} ( D[r][v] + B[r][v] − E(r,v) )` (feasibility rule Part 1.3 if none is finite). Commit `r → v*`. This is the **only** arg-min in the whole extraction.
2. **Flood via back-pointers.** From a committed `(c, v)` walk the coverage run, then commit each predecessor in `bpD[c][·]` and each successor in `bpB[c][·]`, repeating until the queue drains. Following predecessors (up), successors (down), and siblings (down from a shared parent) reaches the seed's **entire weakly-connected component** — a single connected tree needs exactly **one** seed. The flood never crosses into a disconnected part (no back-pointer spans the gap). **Coverage is read from the forward COVER chain only** (never both chains — that over-assigns) — but a 1:N run can be recorded on the *backward* chain instead, which the forward-only read then **drops**, leaving an uncovered target cell between two committed neighbours.
3. **Re-seed only if needed.** If any A-vertex is still uncommitted, go to step 1 — this happens **only** when A is a forest (≥ 2 disconnected trees).
4. **Coverage gap-fill.** Once every vertex is committed, close the dropped-run gaps **from the committed pivots, not the cover chains**: for each source edge `p → c`, walk the B-path between `committed[p]` and `committed[c]` and assign each still-uncovered cell to the downstream vertex it is a candidate of. This partitions correctly — only real gaps *between committed pivots* are filled, never the phantom coverage the cover chains also hold — and adds nothing to an already-covered matching. *Known residual:* complex **split/merge** coverage gaps (not simple linear ones) are not yet closed by this pass.

Output is the relation on the graph, **never converted to another index space**: point mode `M ⊆ V(A) × V(B)`; segment mode the same walk on `L(A)` gives `M_seg ⊆ E(A) × E(B)`. Optional convenience outputs (grouping by a `road_id` edge attribute for road-level routes) are caller-side, clearly derived, never validated in place of `M`.

### Part 6 — Validation — V1–V4 on the graph (never a point conversion)

One checker, `check_rules(M, src, tgt)`, run on the **same graph the match lives on**: `src, tgt = A, B` for point mode; `L(A), L(B)` for segment mode. It tests exactly the four rules of §3, each **restricted to neighbours present in `M`** — so an unmatched target branch (allowed by V4) does not false-fire. `validate_tables(A, which)` additionally replays every finite cell's back-pointer chain and checks it is a legal warping in isolation.

#### 6b Cross-table agreement — reciprocity on `M`

`validate_tables` checks each `D`/`B` cell **in isolation**; it never checks that the two independently-computed tables **agree**. `check_reciprocity` does — on the committed matching only: whenever the **forward** table threads a source edge `p → c`, the **backward** table must thread the same edge, at the same committed cells.

For a committed vertex `c` (pivot `committed[c]`), let `head(c)` / `tail(c)` be its forward / backward **advance anchors** — `committed[c]` walked along its own COVER chain (a `bpD` / `bpB` list that is a single same-source pair `[(c, ·)]`) to the run's start / end. The real advance pointers live there: a vertex connects to its **predecessors at its run-start** `head(c)` and to its **successors at its run-end** `tail(c)`. The invariant, over every source edge `p → c`:

```
(p, tail(p)) ∈ bpD[c][head(c)]   ⟺   (c, head(c)) ∈ bpB[p][tail(p)]
```

This is the structural twin of the numeric agreement `g(a) = min_v ( D[a][v] + B[a][v] − E(a,v) )` being **constant** across a component (Part 5 seeds one representative precisely because it is): both certify the two passes found the *one* optimum. `check_reciprocity` returns the offending edges — empty ⇒ agree.

**Only on `M`, never table-wide.** Off the optimum the reciprocity is *false*: `D[a][v]` and `B[a][v]` optimise **differently-pinned** subproblems (best way *into* `a@v` vs best way *out of* `a@v`), so a table-wide check would flag correct tables. **Coverage is excluded**: same-source COVER pairs are read from the forward chain only (Part 5) and have no backward mirror, so the `head`/`tail` walk consumes them rather than testing them. A failure is a genuine forward/backward disagreement on the optimum — a bug in one pass, **not** something to repair by forcing the tables to agree (§4.2).

#### 6c Table reachability — sources ↔ sinks by back-pointers

A third, **per-table structural** test: each table's back-pointers must reconstruct the tree's own **source ↔ sink reachability**. For **every finite cell of every sink**, walk `bpD` **branching at every predecessor entry**; the terminal (empty-`bp`) cells' vertices must equal exactly the sink's `ancestor_sources`. Mirror for `B` from every source cell via `bpB` (terminals = `descendant_sinks`). Branching is the point: a **merge** makes `bpD` fork so one sink cell must reach **all** its upstream sources; a **split** makes `bpB` fork likewise. A COVER step (same-source pair) stays on its vertex; an ADVANCE moves on. A cell is invalid if the walk hits a `None` reference (severed) or its reached set ≠ the required set; `∞` cells are skipped. `check_reachability(A, which)` returns the invalid cells — empirically **0** over a 22 500-case α,β × point/segment sweep, certifying that extraction-level failures live **downstream of the tables**.

#### 6d Per-table coupling — forward vs V3, backward vs V2

The two passes enforce **complementary** rules: the **forward** table couples merges (V2) but is *optimistic at splits* (§4.2) — read on its own it **can violate V3**; the **backward** table couples splits (V3) and can violate **V2**. This is the **non-vacuous** test of that fact: `check_forward_v3(A, B)` reconstructs the **forward-only** matching (seed each sink at its arg-min `D`, follow `bpD`, union over sinks) and returns its V3 violations; `check_backward_v2` mirrors it (sources, `bpB`, V2).

On clean inputs at **α=β=1** both are empty. Under weighting (α<1/β<1) they fire — and this is **not a bug**: it is *why the second pass exists* and why the Part 5 traceback must couple both. It also explains the Part 6b reciprocity failures under weighting: where the forward table cuts the V3 corner and the backward cuts the V2 corner, the two disagree. (Over a 16 425-case α,β sweep: forward violates V3 on 4 317, backward violates V2 on 3 038; ~2 220 coincide with a reciprocity failure.) These checks are diagnostics of the two-pass complementarity, **not** invariants to enforce on a single table. *(The forward V3 coupling of §4.1a exists precisely to close the forward table's half of this gap during the build.)*

### Segment mode = the same six parts on the line-graph

`line_digraph(G)` = `nx.line_graph(G)` with each L-node `(u, v)` given the segment **midpoint** as `x, y` and the segment **bearing** (`nx.line_graph` on a `DiGraph` yields exactly the directed arc adjacency but copies no attributes). A merge+split vertex becomes a bipartite cluster in `L` (an undirected cycle) — harmless: one shared pin, not a reconvergence (§8.4). Candidates are gated by **midpoint** distance ≤ `r`; `E` is the segment emission (Part 2). Everything else — `D`, `B`, extraction, V1–V4 — is byte-for-byte the point-mode code with `L(A)`, `L(B)` in place of `A`, `B`.
