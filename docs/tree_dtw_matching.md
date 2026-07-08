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
Every time we pair a source point $a$ to a target point $v$, we pay a penalty called the **emission cost**, denoted $E(a, v)$. It usually represents the geographic distance or heading difference between the two points. A single source point may also cover a **run** of target points — the ordinary DTW "stay" move — priced along the run (see the coverage weight $\alpha$ in §4.1).

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

### 4.1 Forward Pass: The Upstream Cost Table `D`

We sweep the source points in **topological order** (sources first, sinks last). Let $\text{reach}(v) = B_{\text{pred}}(v) \cup \{v\}$ be the target points an upstream neighbour can hand off from (either staying on $v$, or moving from an immediate predecessor of $v$). Then:

$$D[a][v] = E(a, v) + \sum_{p \in A_{\text{pred}}(a)} \frac{1}{\text{outdeg}(p)} \min_{x \in \text{reach}(v)} D[p][x]$$

* **Source** ($A_{\text{pred}}(a)=\emptyset$): empty sum → $D[a][v] = E(a,v)$ (free entry at every source).
* **Chain** (one predecessor, $\text{outdeg}=1$): a single term with factor 1 — the ordinary DTW step.
* **Merge** (more than one predecessor): the sum couples the roads. $D[c][v]$ is finite **only if every road can reach $v$**, forcing a merge onto one common target point — this is (V2), folded into the cost. The sum is a true cost precisely because the roads are independent (§2).
* **The Split Factor $\frac{1}{\text{outdeg}(p)}$:** a point feeding several successors would otherwise have its cost counted once **per** branch; dividing by its out-degree splits that cost equally down the branches, so **each point's cost is counted exactly once** in the total. On a tree the branches go to separate sinks and never rejoin, so this conserves the cost exactly.

At every cell **store a back-pointer**: for each predecessor $p$, the $x \in \text{reach}(v)$ that achieved the inner $\min$. The backward traceback follows these.

#### 1:N coverage and the horizontal weight `α`

**Coverage** is a single point $a$ **staying** while B advances — the DTW "stay" move — so one A-point covers a run $v_0 \to \dots \to v_k$. It is realized by a **horizontal (H) move** ($a$ holds while B steps one arc), chained to ride the whole run. But the plain recurrence pays $E(a,\cdot)$ at **every** covered point, so the cost of one point matching a B-stretch grows **linearly with how finely B is sampled** — an arbitrary quantity. The **horizontal weight $\alpha \le 1$** fixes this: discount the emission to $\alpha \cdot E$, but **only when the point is reached by extending coverage** (a horizontal step); a genuinely new match (reached by an A-advance) pays full $E$:

```
D[a][v] = α·E(a,v) + min(
    (H)  min over v'∈Bpred(v)  D[a][v'],                                    # B advances, A STAYS  (coverage)
    (A)  Σ over a'∈Apred(a) (1/outdeg(a'))·min over x∈reach(v) D[a'][x]     # A advances           (new match)
)
    with  α = horizontal_weight (≤ 1)   if the min is the (H) term   (extending coverage)
          α = 1                         if the min is the (A) term   (a new point's first match)
```

Unrolling a run of drift $\delta$: $v_0$ is entered by A-advance (full $\delta$), the rest horizontally ($\alpha\cdot\delta$ each), so the coverage cost is $\delta \cdot (1 + \alpha k)$:

| B-run length k | α = 1 | α = 0.5 | α = 0 |
|---|---|---|---|
| 1 (1:1) | δ | δ | δ |
| 6 | 7δ | 4δ | δ |
| 30 | 31δ | 16δ | δ |

So $\alpha = 1$ is the plain per-point charge; $\alpha < 1$ discounts each *extra* covered point; $\alpha \to 0$ charges the run essentially **once** (sampling-independent). Because $\alpha \cdot E \ge 0$ is a **non-negative** per-step emission, $D$ never decreases along a run — there is **no cost "laundering"**, and the routing decision $\min(H, A)$ on the carried cost is unchanged; only the emission *charged* differs. $\alpha$ lives inside the DP's decision cost, so it shifts **which** matching is chosen (always toward *more* coverage), not the reported drift, which stays the raw $\sum E$ of the chosen matching. **Default $\alpha = 1$** (point-to-point pricing, unchanged); reach for $\alpha < 1$ only when 1:N cost scaling with B's sampling density is the problem, and keep it comfortably above 0 to avoid over-covering.

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

### 4.2 Backward Pass: The Downstream Cost Table `B`

To force splits to agree on a single physical location, we build a mirror table $B$ by sweeping the source in **reverse topological order** (sinks first) and summing over **successors**:

$$B[a][v] = E(a, v) + \sum_{s \in A_{\text{succ}}(a)} \frac{1}{\text{indeg}(s)} \min_{w \in \{v\} \cup B_{\text{succ}}(v)} B[s][w]$$

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

* **1:N coverage (the $\alpha$ horizontal move).** Now the horizontal term $\min_{v' \in B_{\text{pred}}(v)} D[a][v']$ makes a cell depend on **other cells in the same row**, along the B-edges — a within-row shortest path. Fill the row in two steps: (1) **inject** each B-point $v$ with its A-advance cost (the $(A)$ term, full emission); (2) **relax** the horizontal edges $v' \to v$ with weight $\alpha\,E(a,v)$:
  $$D[a][v] \leftarrow \min\big(\, D[a][v],\ \ \alpha\,E(a,v) + D[a][v'] \,\big).$$
  Because those weights are **non-negative** ($\alpha E \ge 0$):
  * if the local B-graph is **acyclic**, one pass in **B-topological order** suffices;
  * if it **cycles** (roundabouts, one-way loops — $G_B$ may cycle), use **Dijkstra** (multi-source, one entry per injected point) or bounded iterative relaxation until convergence.

  This within-row relaxation is the **only** place a shortest-path search appears; with $\alpha = 1$ it is the plain per-point coverage charge, and it never runs at all in the point-to-point case above.

$B$ is filled the same way, with the mirror horizontal move over $B_{\text{succ}}(v)$ on the reversed local B-graph.

---

## 5. The Traceback Stage: Extracting the Match Relation `M`

### 5.1 Intuitive Concept

1. **Why one table isn't enough.** Each coupling enforces its own rule **by construction**: the predecessor sum in $D$ makes every merge's approaches meet at one point (V2), and — its exact mirror — the successor sum in $B$ makes every split's exits leave one point (V3). But a single directed sweep carries only **one** of the two. If you take just $D$ and read an independent $\arg\min$ per sink, the splits are left uncoordinated: two branches of a fork place it at **different** points — the phantom of §4.1.
2. **The fix — thread both couplings together.** Perform an ordered traceback that **commits one coherent optimum** — *not* an independent $\arg\min$ per point (at ties, independent choices can stitch together pieces of *different* optima that don't fit). Each junction is pinned once, jointly, by both tables ($\arg\min_v D[a][v] + B[a][v] - E(a,v)$, §4.2). A **merge** is then traversed upstream by reading the back-pointers stored in $D$ — its approaches already agree (V2); a **split** uses $B$ so **all** its branches leave the one committed point (V3).

### 5.2 Extraction Pseudocode

```text
Algorithm: Extract-Tree-DTW-Matching(G_A, G_B, D, B, E)
Input:  G_A, G_B ; D (forward table with back-pointers) ; B (backward table) ; E
Output: M — set of matched pairs (a, v)

Initialize:
    M         ← ∅
    Committed ← empty map   (source point → committed target point)
    Queue     ← empty

Step 1 — Seed at the global optimum.
    Pick any point r (seed one per weakly-connected component if the source is a forest).
    Commit it to its whole-tree optimum:
        v_opt = argmin_v ( D[r][v] + B[r][v] − E(r, v) )
    Committed[r] ← v_opt ;  M.add((r, v_opt)) ;  Queue.enqueue(r)

Step 2 — Propagate outward until every point is committed.
While Queue is not empty:
    c ← Queue.dequeue() ;  v ← Committed[c]

    # DOWNSTREAM — commit EVERY successor (chains and splits alike)
    For each s in Asucc(c) with s not committed:
        legal = {v} ∪ Bsucc(v)                    # s must leave legally from c's point v  → (V3)
        w*    = argmin over w in legal of B[s][w] # cheapest downstream, given c fixed at v
        Committed[s] ← w* ;  M.add((s, w*)) ;  Queue.enqueue(s)

    # UPSTREAM — commit EVERY predecessor via the stored forward back-pointer
    For each p in Apred(c) with p not committed:
        w* = D[c][v].back_pointer_for(p)          # the point p used to feed c at v  → (V2)
        Committed[p] ← w* ;  M.add((p, w*)) ;  Queue.enqueue(p)

Return M
```

Two things make this correct on a tree. **Downstream is unconditional** — every successor is committed, not only at splits (a chain point has one successor, a split has several; both must be walked). At a split, all children are constrained to $\{v\} \cup B_{\text{succ}}(v)$, so they all leave from $c$'s committed point $v$ — that is (V3) — and each then takes its own cheapest downstream because their subtrees are disjoint. **Upstream reads $D$'s back-pointers**, which already agree on $v$ — that is (V2). Because the source is a tree, this single walk commits every point **exactly once**, consistently; there is no loop to force a split and a merge to fight over the same point.

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
| **Coupled by** | **forward** sum over predecessors (in $D$) | **backward** sum over successors (in $B$) |
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
