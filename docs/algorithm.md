# Directed Road Map Matching & Conflation Algorithm

This document defines the conceptual framework, mathematics, and execution stages of the progressive shape-alignment and directional road map matching algorithm.

---

## 1. The Core Problem

When matching and aligning two independent directed road networks (e.g. GPS trajectories or a secondary network to a master road network), we face three major constraints:
1. **Asymmetry in Segmentation**: A single road in Network A may be represented as a single long segment ($500\text{m}$), whereas in Network B it is divided into multiple short segments ($150\text{m}$ each).
2. **Missing Roads**: Some road segments exist in Network A but not in Network B (and vice-versa). The algorithm must identify these "unmatched" segments rather than forcing incorrect matches.
3. **Directed Lanes**: Roads run in specific directions (one-way, dual carriageway). Matches must align strictly with driving direction and avoid mapping opposing lanes together.
4. **Partial Overlap and Endpoint Stretch**: Standard matching methods fail when two segments overlap only in the middle or have long unmatched sections at either end, causing massive shape distortions and stretching.

---

## 2. The 3-Tier Architecture

To process large spatial datasets efficiently, we divide the matching pipeline into three modular tiers:

```
+-----------------------------------------------------------------+
| TIER 1: Spatial Candidate Selection (DuckDB Spatial)            |
| - Fast, approximate filtering using ST_DWithin on UTM coords   |
| - Discards 99% of non-matching road pairs in < 1ms              |
+-----------------------------------------------------------------+
                                |
                                v (Qualifying candidate pairs)
+-----------------------------------------------------------------+
| TIER 2: Progressive DP DTW Shape Alignment (Python / NumPy)     |
| - Constructs high-resolution node/projection pools              |
| - Selects natural symmetric start projected point               |
| - Populates pruned cost matrix and backtracks path              |
| - Computes exact distance & aligned coverage percentage         |
+-----------------------------------------------------------------+
                                |
                                v (Evaluated pair metrics)
+-----------------------------------------------------------------+
| TIER 3: Directional & Bidirectional Reconciliation             |
| - Directional: Filters candidates and ranks best destinations    |
| - Bidirectional: Runs independent runs both ways and unions     |
+-----------------------------------------------------------------+
```

---

## 3. Tier 1: Spatial Candidate Selection

Instead of comparing every segment in Network A to every segment in Network B, we use DuckDB's spatial index and parallel engine to instantly filter candidates.
- **Projected Coordinates**: Geometries are transformed from EPSG:4326 to a local metric UTM projection (`utm_srid`) to perform all distance calculations in meters.
- **Spatial Join**: We retrieve pairs that are within a baseline search window:
  $$\text{ST\_DWithin}(\text{Geom\_A}, \text{Geom\_B}, \text{max\_distance})$$

---

## 4. Tier 2: Progressive DP DTW Shape Alignment

To align shapes under partial overlaps without length-based classification or swapping, we implement the **Progressive Node-and-Projection DTW** algorithm. It is evaluated from **Source A** (matching input) to **Destination B** (target network).

### 4.1 Symmetric Overlap Start Selection
The starting boundary is determined dynamically by projecting the start nodes of each segment onto the other:
1. Project the start of Source A ($a_0$) onto Destination B to find $s_{b\_start}$.
2. Project the start of Destination B ($b_0$) onto Source A to find $s_{a\_start}$.
3. We select the start coordinate index $j_{start}$ along Destination B matching the projection of $a_0$. The DP matrix starts at distance $0.0$ on Source A and $s_{b\_start}$ on Destination B. This clips off unmatched pre-overlap sections naturally without using segment lengths.

### 4.2 High-Resolution Node Pool Construction
To avoid distortion and resolution mismatches, we build dense node pools:
* **Source Points ($pts_a$)**: The sorted union of Source A's raw Nodes and the perpendicular projections of Destination B's nodes onto Source A.
* **Destination Points ($pts_b$)**: The sorted union of Destination B's raw Nodes and the perpendicular projections of Source A's nodes onto Destination B (starting from $s_{b\_start}$).

### 4.3 Pruned Dynamic Programming Step Transitions
At each step $(i, j)$ in the cost matrix, where the last matched node of Source A is $a_{i-1}$ and Destination B is $b_{j-1}$, the cumulative cost $D(i, j)$ is computed by:

$$D(i, j) = d(pts_a[i], pts_b[j]) + \min \begin{cases} 
      D(i-1, j) & \text{(Vertical step - Node of Source A matches to Destination B)} \\
      D(i, j-1) & \text{(Horizontal step - Node of Destination B matches to Source A)} \\
      D(i-1, j-1) & \text{(Diagonal step - Node matches to Node)}
   \end{cases}$$

To prevent shape distortion:
- **Vertical Step Projections** are mathematically bounded to the active interval $[proj(a_{i-1}), b_j]$:
  $$s_b = \max\left(proj(a_{i-1}), \min\left(s_{b,cand}, \text{dist}(b_j)\right)\right)$$
- **Horizontal Step Projections** are mathematically bounded to the active interval $[proj(b_{j-1}), a_i]$:
  $$s_a = \max\left(proj(b_{j-1}), \min\left(s_{a,cand}, \text{dist}(a_i)\right)\right)$$

### 4.4 Boundary stopping and Backtracking
- The DP table construction stops as soon as either segment is fully traversed (the last row $i = N'-1$ or column $j = M'-1$).
- We identify the boundary cell $(i_{end}, j_{end})$ that minimizes the cumulative warping cost.
- Backtracking from $(i_{end}, j_{end})$ back to $(0, j_{start})$ yields the optimal warping path.

### 4.5 Aligned Overlap Percentage
The coverage percentage represents the proportion of Source A's total length ($len\_a$) that is aligned:

$$\text{matched\_len} = pts_a[i_{end}]$$
$$\text{overlap\_pct} = \min\left(100.0, \max\left(0.0, \text{round}\left(\frac{\text{matched\_len}}{\text{len\_a}} \times 100.0, 2\right)\right)\right)$$

---

## 5. Tier 3: Reconciliation and Strategies

Once the Tiers 1 and 2 metrics (average DTW distance, bearing difference, and aligned overlap percentage) are calculated, we resolve the relationships.

### 5.1 Directed Strategy (Default)
When executing directional matching:
- **No Cutoff Filtering**: The algorithm does not filter or discard matches based on arbitrary bearing, overlap, or distance thresholds. All spatial candidates generated in Tier 1 are kept and evaluated.
- **One Parameter Candidate Search**: The single parameter `max_distance` serves as the search radius to query the closest road segments as candidates.
- **Topological Ranking**: For each source segment A, all qualifying candidate destination segments B are ranked by alignment quality (`dtw_distance` ascending).
- **Match Flagging**: The closest segment is flagged with `rank = 1`, and other parallel qualifying segments are kept with a higher `rank` value. This handles split roads naturally while providing complete candidate alignment data.

### 5.2 Bidirectional Strategy
When executing bidirectional matching:
- The directional matching pipeline is run twice independently:
  1. **Direction 1 ($A \to B$)**: Evaluates alignment and coverage with Network A as the source and Network B as the destination.
  2. **Direction 2 ($B \to A$)**: Swaps roles, evaluating alignment and coverage with Network B as the source and Network A as the destination.
- The results of both runs are unified, and a `direction` column (valued `'A_to_B'` or `'B_to_A'`) is appended.
- This strategy provides a symmetrical, reciprocal conflation table, enabling complete network-to-network reconciliation.
