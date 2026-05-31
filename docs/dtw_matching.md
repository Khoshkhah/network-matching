# Dynamic Time Warping (DTW) in Spatial Road Matching

This document provides a deep, conceptual dive into the **Dynamic Time Warping (DTW)** algorithm, its application to 2D geographical coordinate lists, its mathematical properties, and how we handle boundary constraints (endpoint stretches) in road network matching.

---

## 1. Concept: Shape Alignment via Time Series

In geospatial analysis, a directed road segment is represented as a sequence of 2D coordinates:
* **Curve A**: $A = [a_1, a_2, \dots, a_N]$
* **Curve B**: $B = [b_1, b_2, \dots, b_M]$

If we want to compare their shapes, a direct point-to-point comparison (matching $a_i$ directly to $b_i$) fails because:
1. **Vertex Mismatch**: Curve A might have $3$ vertices (e.g. simplified straight line), while Curve B has $20$ vertices (representing a detailed curve along the exact same path).
2. **Phase Shifts**: The vertices might be offset even if the curves are geometrically identical.

**Dynamic Time Warping (DTW)** solves this by calculating an optimal **warping path** that maps points on Curve A to points on Curve B in a non-linear, flexible way.

```
       Curve A (3 vertices)
       a1--------------a2--------------a3
       |  \          / | \            /|
       |    \      /   |   \        /  |
       b1----b2--b3----b4----b5--b6----b7
       Curve B (7 vertices)
```

In this alignment:
* $a_1$ matches $\{b_1, b_2\}$
* $a_2$ matches $\{b_3, b_4, b_5\}$
* $a_3$ matches $\{b_6, b_7\}$

This is a **many-to-many vertex mapping** that captures shape similarity regardless of sampling density.

---

## 2. The Mathematical Steps

### Step 1: The Local Cost Matrix
First, we build a local distance matrix $M$ of size $N \times M$, where each cell $(i, j)$ represents the flat projected Euclidean distance (in meters) between point $a_i$ and point $b_j$:

$$d(a_i, b_j) = \sqrt{(x_{a_i} - x_{b_j})^2 + (y_{a_i} - y_{b_j})^2}$$

### Step 2: The Accumulated Cost Matrix (Dynamic Programming)
We calculate the accumulated cost matrix $D(i, j)$ using the following recurrence relation:

$$D(i, j) = d(a_i, b_j) + \min \begin{cases} 
      D(i-1, j) & \text{(Deletion / Skip A)} \\
      D(i, j-1) & \text{(Insertion / Skip B)} \\
      D(i-1, j-1) & \text{(Match)}
   \end{cases}$$

#### Boundary Conditions of Standard DTW:
* Start cell: $D(0, 0) = d(a_1, b_1)$
* First row: $D(0, j) = D(0, j-1) + d(a_1, b_j)$
* First column: $D(i, 0) = D(i-1, 0) + d(a_i, b_1)$

### Step 3: Backtracking the Warping Path
Starting from the top-right corner $(N-1, M-1)$, we backtrack through the accumulated cost matrix by choosing the adjacent cell with the lowest cost until we reach $(0, 0)$. This gives us the **Optimal Warping Path $P$**:

$$P = [p_1, p_2, \dots, p_K] \quad \text{where } p_k = (i_k, j_k)$$

---

## 3. The Endpoint Stretch Constraint (Split Road Problem)

The **Standard DTW** algorithm enforces strict **boundary conditions**: the warping path *must* start at $(0, 0)$ and end at $(N-1, M-1)$. 

While this is perfect for matching whole curves of similar lengths, it creates a major issue for **divided roads** (where a short road B matches a sub-section of a long road A).

```
  A1 (Start)                                 Segment A (Map A: Coarse)                                A_N (End)
  =======================================================================================================>
  B1 (Start)                  B_M (End)
  ---------------------------->
  Segment B (Map B: Fine Split)
```

In Standard DTW:
1. The start points are close, so $d(a_1, b_1)$ is small.
2. But the end of B ($b_M$) **is forced** to match the end of A ($a_N$).
3. Because $a_N$ is hundreds of meters away, the distance $d(a_N, b_M)$ is extremely large.
4. This endpoint stretch inflates the cumulative cost, causing the match to be **rejected**.

---

## 4. Reconciling the Page: Two Ways to Handle Splits in DTW

To resolve the split-road issue while preserving DTW's benefits, we have two distinct choices:

## 4. Continuous Piecewise Order-Preserving Projection Alignment

To combine the precision of continuous spatial projection with the strict directed-flow ordering of DTW, we define the **Continuous Piecewise Order-Preserving Alignment** framework.

### The Alignment Constraints

Let Segment $A$ have vertices $a_i \to a_{i+1}$ (in Map A), and Segment $B$ have vertices $b_j \to b_{j+1}$ (in Map B). 
If the point $a_i$ projects continuously onto Segment $B$ at the location $proj(a_i)$, the alignment path must evolve according to **Three Fundamental Rules**:

```
  Road A:       a_i -----------------------------------------> a_{i+1}
                 |                                             /
                 | (projected)                               / (next projection)
                 v                                         v
  Road B:  b_j --[ proj(a_i) ]------------------------------> b_{j+1}
```

#### Rule 1: A-Projections Advance (Monotonicity along B)
If the next vertex $a_{i+1}$ projects onto Segment $B$, the new projection $proj(a_{i+1})$ **must lie at or ahead** of $proj(a_i)$ (closer to $b_{j+1}$). It is mathematically forbidden from sliding backward toward $b_j$:
$$\text{Distance}(b_j, proj(a_{i+1})) \ge \text{Distance}(b_j, proj(a_i))$$

#### Rule 2: B-Vertices Interleave (Monotonicity along A)
If Road B is more detailed and has an intermediate vertex $b_{j+1}$ before $a_{i+1}$ is reached, then $b_{j+1}$ must project backward onto Segment $A$ at a location $proj(b_{j+1})$ that lies **strictly between** $a_i$ and $a_{i+1}$:
$$proj(b_{j+1}) \in [a_i, a_{i+1}]$$

#### Rule 3: Exact Vertex Match
The vertices of both roads align perfectly at their next junctions:
$$a_{i+1} \longleftrightarrow b_{j+1}$$

---

## 5. Implementation: The Continuous Subsequence DTW Engine

To implement this continuous piecewise alignment in Python efficiently without complex geometric solvers, we use the **Continuous Subsequence DTW** algorithm:

1. **High-Density Spatial Sampling**:
   We project both coordinate lists into flat UTM meters, and sample points along both lines at a **high-density, fixed spatial interval** (e.g. every $5\text{m}$).
   * This transforms a discrete vertex list into a dense, continuous sequence: $A_{\text{dense}} = [p_1, \dots, p_{N'}]$ and $B_{\text{dense}} = [q_1, \dots, q_{M'}]$.
2. **Determine Query vs. Target (Asymmetry)**:
   * Let the shorter road be the **Query** ($Q$) and the longer road be the **Target** ($T$).
3. **Open-Ended Boundary Conditions**:
   * We run Subsequence DTW on the dense point lists, allowing $Q$ to match *anywhere* along $T$ without start or end penalties on $T$.
4. **Calculated Match Metrics**:
   * The resulting warping path $P$ represents your **Continuous Piecewise Order-Preserving Alignment**, satisfying all three rules.
   * The final distance is the normalized sum of the projection distances along this optimal path.

