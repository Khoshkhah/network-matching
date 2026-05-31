# Dynamic Time Warping (DTW) in Spatial Road Matching

This document provides a deep, conceptual dive into the **Dynamic Time Warping (DTW)** algorithm, its application to 2D geographical coordinate lists, its mathematical properties, and how we handle boundary constraints (endpoint stretches) in road network matching.

---

## 1. Aligned Terminology

To prevent any ambiguity during design and implementation, we define and adhere to the following precise terminology:

1. **Node**: A raw, discrete coordinate vertex of the road geometry (from the input coordinate lists `coords_a` or `coords_b`).
2. **Link**: Each individual straight line segment connecting two sequential **Nodes** in a road geometry.
3. **Segment**: The entire road geometry representing a sequence of Nodes and intermediate Links (e.g. Segment A or Segment B).
4. **Node Projection**: The perpendicular projection of a **Node** of one Segment onto a **Link** of the other Segment.
5. **Point**: Any general position/coordinate along a Segment (which can be a raw **Node** or an interpolated **Node Projection**).

---

## 2. Concept: Shape Alignment via Time Series

In geospatial analysis, a directed road segment is represented as a sequence of 2D coordinates:
* **Source Segment A**: $A = [a_1, a_2, \dots, a_N]$ (composed of sequential Links $[a_i, a_{i+1}]$)
* **Destination Segment B**: $B = [b_1, b_2, \dots, b_M]$ (composed of sequential Links $[b_j, b_{j+1}]$)

If we want to compare their shapes, a direct point-to-point comparison (matching $a_i$ directly to $b_i$) fails because:
1. **Vertex Mismatch**: Source Segment A might have $3$ nodes (e.g. simplified straight line), while Destination Segment B has $20$ nodes (representing a detailed curve along the exact same path).
2. **Phase Shifts**: The vertices might be offset even if the curves are geometrically identical.

**Dynamic Time Warping (DTW)** solves this by calculating an optimal **warping path** that maps points on Source Segment A to points on Destination Segment B in a non-linear, flexible way.

```
       Source Segment A (3 nodes)
       a1--------------a2--------------a3
       |  \          / | \            /|
       |    \      /   |   \        /  |
       b1----b2--b3----b4----b5--b6----b7
       Destination Segment B (7 nodes)
```

In this alignment:
* $a_1$ matches $\{b_1, b_2\}$
* $a_2$ matches $\{b_3, b_4, b_5\}$
* $a_3$ matches $\{b_6, b_7\}$

This is a **many-to-many point mapping** that captures shape similarity regardless of sampling density.

---

## 3. The Endpoint Stretch Constraint (Partial Overlap Problem)

The **Standard DTW** algorithm enforces strict **boundary conditions**: the warping path *must* start at the very first nodes $(0, 0)$ and end at the very last nodes $(N-1, M-1)$. 

While this is perfect for matching whole curves of similar lengths, it creates a major issue when road segments only partially overlap. For example, if Source Segment A extends far beyond Destination Segment B, or vice versa, the standard boundary conditions force the unmatched ends of the segments to map to each other.

```
  A1 (Start)                         Segment A (Source)                                      A_N (End)
  =======================================================================================================>
                      B1 (Start)                  B_M (End)
                      ---------------------------->
                      Segment B (Destination)
```

In Standard DTW:
1. The warping path is forced to start at $(a_1, b_1)$ and end at $(a_N, b_M)$.
2. If the segments only partially overlap, the standard boundary conditions force the unmatched ends to map to each other (e.g. mapping $a_1$ to $b_1$ and $a_N$ to $b_M$, even if they are hundreds of meters apart).
3. This creates massive endpoint stretches, inflating the cumulative cost and causing correct matches to be rejected.

---

## 4. Progressive Source-to-Destination DTW DP

To handle partially overlapping segments while preserving DTW's shape-alignment benefits, we implement the **Progressive Node-and-Projection DTW** algorithm. The algorithm runs natively on the raw **Nodes** and **Node Projections** of the **Source A** and **Destination B** segments, with **strictly no length comparisons**.

### The Alignment Constraints & Monotonicity Rules

1. **Source-to-Destination Directional Matching**:
   - The first parameter to `dtw_align` is strictly the **Source (`coords_a`)** segment, and the second parameter is strictly the **Destination (`coords_b`)** segment.
   - The match ALWAYS starts at the beginning node of Source A ($a_0$, i.e. `coords_a[0]`) projected onto Destination B at distance $s_{b\_start}$.
   - We find the closest index $j_{start}$ of Destination's points corresponding to this projection.
   - The match starts at distance $0.0$ on Source A and $s_{b\_start}$ on Destination B.

2. **High-Resolution Node Pool Construction**:
   - **Source Points ($pts_a$)**: The sorted union of Source A's raw **Nodes** and the **Node Projections** of Destination B's nodes onto Source A.
   - **Destination Points ($pts_b$)**: The sorted union of Destination B's raw **Nodes** and the **Node Projections** of Source A's nodes onto Destination B (starting from $s_{b\_start}$).

3. **Dynamic Programming Step Transitions & Bounded Projection Areas**:
   At each step $(i, j)$ in the dynamic programming table, where the last matched node of Source A is $a_{i-1}$ and Destination B is $b_{j-1}$, we have three transition options for the next step:

   - **Vertical Step (Option A - Matching the next Node of Source A)**:
     - *Action*: We move to the next Node $a_i$ of Source A and project it onto Destination Segment B to get the matched distance $s_b$.
     - *Projection Area*: Restricted strictly to the active interval $[proj(a_{i-1}), b_j]$ along Destination Segment B. The projected position $s_b$ is mathematically bounded by:
       $$s_b = \max\left(proj(a_{i-1}), \min\left(s_{b,cand}, \text{dist}(b_j)\right)\right)$$
       where $s_{b,cand}$ is the full projection distance of $a_i$ on Segment B, and $\text{dist}(b_j)$ is the distance of Node $b_j$ along B.
     
   - **Horizontal Step (Option B - Matching the next Node of Destination B)**:
     - *Action*: We move to the next Node $b_j$ of Destination B and project it onto Source Segment A to get the matched distance $s_a$.
     - *Projection Area*: Restricted strictly to the interval $[proj(b_{j-1}), a_i]$ along Source Segment A. The projected position $s_a$ is mathematically bounded by:
       $$s_a = \max\left(proj(b_{j-1}), \min\left(s_{a,cand}, \text{dist}(a_i)\right)\right)$$
       where $s_{a,cand}$ is the full projection distance of $b_j$ on Segment A, and $\text{dist}(a_i)$ is the distance of Node $a_i$ along A.
     
   - **Diagonal Step (Option C - Direct Node-to-Node Match)**:
     - *Action*: We move to both the next Node $a_i$ of Source A and Node $b_j$ of Destination B, matching them directly to each other ($a_i \longleftrightarrow b_j$).
     - *Projection Area*: No continuous projection is performed; the transition cost is simply the Euclidean physical distance between raw Node $a_i$ and raw Node $b_j$.

    $$D(i, j) = d(pts_a[i], pts_b[j]) + \min \begin{cases} 
          D(i-1, j) & \text{(Vertical step - Node of Source A matches to Destination B)} \\
          D(i, j-1) & \text{(Horizontal step - Node of Destination B matches to Source A)} \\
          D(i-1, j-1) & \text{(Diagonal step - Node matches to Node)}
       \end{cases}$$

5. **Boundary-Based Stopping & Backtracking**:
   - The algorithm stops when either segment is completed. This corresponds to the boundaries of the dynamic table: the last row $i = N'-1$ or the last column $j = M'-1$.
   - We search these boundary cells to find the cell $(i_{end}, j_{end})$ that minimizes the cumulative cost $D(i, j)$, preferring more progress along the segments if costs are equal.
   - We backtrack from $(i_{end}, j_{end})$ to the initial start cell $(0, j_{start})$ to extract the optimal warping path.

6. **Alignment-Based Overlap Percentage**:
   - We measure what percentage of **Source Segment A** is matched, from the start ($0.0$) to the furthest matched point $pts_a[i_{end}]$:
     $$\text{matched\_len} = pts_a[i_{end}]$$
     $$\text{overlap\_pct} = \min\left(100.0, \max\left(0.0, \text{round}\left(\frac{\text{matched\_len}}{\text{len\_source\_a}} \times 100.0, 2\right)\right)\right)$$
   - This coverage-based metric completely replaces the old corridor box-matching calculations.

---

## 5. Map-Matching Strategies: Directed vs. Bidirectional

The matching framework supports two top-level matching strategies via the `DuckDBMapMatcher.match` API:

### 5.1 Directed Map-Matching (Default)
When running `match(bidirectional=False)`:
- Computes directed mapping from **Source Network A** to **Destination Network B**.
- For each segment in A, it evaluates candidate segments in B, computes the progressive DTW alignment from A to B, and ranks candidate segments by proximity and directional alignment.
- **Result Columns**: `[source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance, bearing_diff, overlap_pct, rank, is_best, match_type]`.
- *Use Case*: Ideal when you have a defined trajectory/GPS track (Source A) and want to find its corresponding segments on a base map road network (Destination B).

### 5.2 Bidirectional Map-Matching
When running `match(bidirectional=True)`:
- Executes directional matching in **both directions** simultaneously:
  1. **Direction 1 ($A \to B$)**: Evaluates segment coverage and alignment with A acting as the source and B acting as the destination.
  2. **Direction 2 ($B \to A$)**: Swaps the roles, evaluating segment coverage and alignment with B acting as the source and A acting as the destination.
- Unions both tables into a single output, adding a `direction` metadata column.
- **Result Columns**: `[source_id, dest_id, dtw_distance, max_dtw_distance, min_dtw_distance, bearing_diff, overlap_pct, rank, is_best, match_type, direction]` where `direction` is either `'A_to_B'` or `'B_to_A'`.
- *Use Case*: Essential for reciprocal shape matches, map conflation (merging two road networks A and B), and reconciling nested road splits (e.g. mapping a long road in A to split parallel roads in B, and mapping those split roads back to the main road in A).

