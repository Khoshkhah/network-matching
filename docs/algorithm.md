# Directed Road Map Matching & Conflation Algorithm

This document defines the conceptual framework, math, and steps for matching and conflating two directed road networks.

---

## 1. The Core Problem

When aligning two independent directed road maps (e.g. OpenStreetMap and a commercial provider), we face three major constraints:
1. **Asymmetry in Segmentation**: A single road in Map A may be represented as a single long edge ($500\text{m}$), whereas in Map B it is divided into multiple short edges ($150\text{m}$ each).
2. **Missing Roads**: Some roads exist in Map A but not in Map B (and vice-versa). The algorithm must identify these "unmatched" segments rather than forcing incorrect matches.
3. **Directed Lanes**: Roads run in specific directions (one-way, dual carriageway). We must guarantee that matches align in driving direction and do not map opposing lanes together.

---

## 2. The Two-Tier Architecture

To process large spatial datasets (e.g. hundreds of thousands of segments) efficiently, we divide the matching pipeline into two steps:

```
+-------------------------------------------------------------+
|                TIER 1: Spatial Index Filter                 |
| - Fast, approximate filtering using Bounding Boxes & R-Tree |
| - Discards 99% of non-matching road segments in < 1ms       |
+-------------------------------------------------------------+
                              |
                              v (2 to 5 candidates returned)
+-------------------------------------------------------------+
|                TIER 2: Precise Evaluation                   |
| - Calculates detailed metrics (Distance, Bearing, Overlap)  |
| - Computes a unified Similarity Score (0.0 to 1.0)          |
| - Selects the absolute best candidate or flags mismatch      |
+-------------------------------------------------------------+
```

---

## 3. Tier 1: Candidate Selection

Instead of comparing every segment in Map A to every segment in Map B, we use an **R-Tree (Spatial Index)** to instantly find nearby candidates.

### Bounding Box & Expanded Search Window
1. Every segment, regardless of shape, is represented by its **Bounding Box**—the tightest rectangle that encloses its coordinates:
   $$\text{Min}_x = \min(x), \quad \text{Max}_x = \max(x)$$
   $$\text{Min}_y = \min(y), \quad \text{Max}_y = \max(y)$$
2. We expand this box in all directions by a search radius (e.g., $25\text{m}$) to create the **Search Window**.
3. We query Map B's R-Tree with this window. The index instantly returns the row IDs of the segments whose bounding boxes intersect our window.

---

## 4. Tier 2: The Multi-Criteria Scoring System

For each candidate segment, we calculate three independent spatial measurements. If any measurement fails its threshold, the match is rejected. If all pass, we calculate a unified **Similarity Score** to pick the best match.

```
       Segment a (Map A)
===============================================> (Bearing: 90°)
       .......................................  <- Corridor Buffer (25m)
       ----------------->  ------------------>  <- Candidate Segments (Map B)
          Segment b1           Segment b2
```

### The 3 Core Metrics

#### A. Proximity / Normalized 2D DTW Distance ($D_{\text{dtw}}$)
* **What it measures**: The average physical alignment drift (in meters) along the entire shape of the two directed curves, ignoring vertex density mismatches.
* **How to calculate**:
  1. Project both segments' coordinate lists into a flat coordinate system (meters, e.g., UTM). Let $A = [a_1, \dots, a_N]$ and $B = [b_1, \dots, b_M]$.
  2. Build an $N \times M$ cost matrix where each cell $(i, j)$ is the flat Euclidean distance:
     $$d(a_i, b_j) = \sqrt{(x_{a_i} - x_{b_j})^2 + (y_{a_i} - y_{b_j})^2}$$
  3. Find the optimal warping path $P = [p_1, \dots, p_K]$ that minimizes the cumulative cost from $(1, 1)$ to $(N, M)$. Here, each $p_k = (i_k, j_k)$ represents a matched coordinate pair between Road A and Road B.
  4. Define the **Length of the Match ($K$)**: This is the number of discrete steps (alignment pairs) in the warping path $P$. It represents the total number of point-to-point associations made during the alignment. The value of $K$ is mathematically bounded by:
     $$\max(N, M) \le K \le N + M - 1$$
  5. Normalize the cumulative cost by dividing it by the length of the match $K$:
     $$D_{\text{dtw}} = \frac{1}{K} \sum_{k=1}^K d(a_{i_k}, b_{j_k})$$
* **Pass Criteria**: $D_{\text{dtw}} \le 20\text{m}$.
* **If Failed**: `"Too far apart (Normalized DTW distance exceeds 20m)."`
* **Why it is superior**: It is naturally sensitive to direction (opposing travel lanes result in a massive DTW cost due to crossing alignments) and resolves vertex-density mismatches automatically.

#### B. Directional Alignment ($\Delta\theta$)
* **What it measures**: Whether the vehicles on both roads are driving in the same direction.
* **How to calculate**: Calculate the absolute bearing (heading angle relative to North, $0^\circ$ to $360^\circ$) from the first to the last coordinate of each segment. The absolute difference is wrapped to a maximum of $180^\circ$:
  $$\Delta\theta = \min(|\theta_a - \theta_b|, 360^\circ - |\theta_a - \theta_b|)$$
* **Pass Criteria**: $\Delta\theta \le 30^\circ$.
* **If Failed**: `"Direction mismatch (bearing difference exceeds 30°)."`

#### C. Parallel Overlap Percentage ($O_{\text{pct}}$)
* **What it measures**: Whether the two segments run parallel along each other, or simply cross at an intersection.
* **How to calculate**: Create a corridor buffer of radius $25\text{m}$ around segment $a$. Intersect the geometry of candidate segment $b_i$ with this buffer.
  $$O_{\text{pct}} = \frac{\text{Length}(b_i \cap \text{Buffer}(a))}{\text{Length}(b_i)} \times 100$$
* **Pass Criteria**: $O_{\text{pct}} \ge 50\%$.
* **If Failed**: `"Insufficient parallel overlap."`

---

## 5. Unified Similarity Score ($S$)

If a candidate segment passes all three thresholds, its normalized sub-scores are combined into a weighted similarity score between $0.0$ and $1.0$:

$$S = w_d \cdot S_{\text{dist}} + w_\theta \cdot S_{\text{angle}} + w_o \cdot S_{\text{overlap}}$$

Where:
* $S_{\text{dist}} = \max\left(0, 1 - \frac{D_{\text{dtw}}}{\text{Max\_Distance}}\right)$
* $S_{\text{angle}} = \max\left(0, 1 - \frac{\Delta\theta}{\text{Max\_Angle}}\right)$
* $S_{\text{overlap}} = \frac{O_{\text{pct}}}{100}$
* **Weights**: $w_d = 0.3, \quad w_\theta = 0.3, \quad w_o = 0.4$ (adjustable based on priority).

---

## 6. Bidirectional Reconciliation & Split Road Detection

Instead of executing two separate, isolated matching runs (which would duplicate the heavy DTW math), we consolidate all evaluations into a single **De-duplicated Candidate Pairs Table**. This cuts computational cost in half and unifies all post-processing.

```
       Table A (Coarse)                 Table B (Fine)
     +------------------+             +------------------+
     |  Road Segments   |             |  Road Segments   |
     +------------------+             +------------------+
              |                                |
              +--------------+-----------------+
                             | (Query Spatial Index)
                             v
           +-----------------------------------+
           |    Unique Candidate Pairs (Set)   |  <- Ensures we never compute DTW
           |         e.g., (a1, b1)            |     between the same pair twice
           +-----------------------------------+
                             |
                             v (Compute DTW Distance ONCE)
     +---------------------------------------------------+
     |        Unified Candidate Evaluation Table         |
     | Columns: [edge_a_id, edge_b_id, dtw_distance]     |
     +---------------------------------------------------+
                             |
                             v (Reconciliation Rules)
     +---------------------------------------------------+
     |     Classified Matches & Quality Metrics          |
     | (Symmetric 1:1, 1:N Splits, Conflicts, No Match)  |
     +---------------------------------------------------+
```

---

## 7. The Reconciliation Rules

Once the **Unified Candidate Evaluation Table** is populated, we reconcile and classify the relationships using standard SQL or DataFrame queries:

### Step 1: Apply Strict Distance Threshold
Remove any candidate pair where the spatial alignment distance is too high:
$$\text{Filter out: } dtw\_distance > 20\text{m}$$
*Any road segment in Table A or B that has zero remaining candidate pairs after this step is immediately classified as **`NO_MATCH`** (Missing Road).*

### Step 2: Extract Best Choices (Unidirectional Argmin)
* **Best A $\to$ B**: For each `edge_a_id`, select the row with the **minimum** `dtw_distance`.
* **Best B $\to$ A**: For each `edge_b_id`, select the row with the **minimum** `dtw_distance`.

### Step 3: Classify by Relationship

#### A. Symmetric 1:1 Match (Agreement)
* **Condition**: A candidate pair `(a, b)` is the **Best A $\to$ B** choice AND the **Best B $\to$ A** choice.
* **Decision**: Approved as high-confidence match.
* **Explanation**: *"Reciprocal agreement: both networks agree that segment $a$ and segment $b$ are the absolute closest counterparts."*

#### B. 1:N Split (Divided Road)
* **Condition**: Multiple entries in the **Best B $\to$ A** list (e.g. `(b_1, a)`, `(b_2, a)`) point to the same `edge_a_id`.
* **Decision**: Approved as divided road match. 
* **Explanation**: *"Segment $a$ is a long road in Map A that is split into multiple parts in Map B."*

#### C. Topological Conflict (Disagreement)
* **Condition**: The **Best A $\to$ B** choice for $a_1$ is $b_1$, but the **Best B $\to$ A** choice for $b_1$ is a different road $a_2$.
* **Decision**: 
  * If $\text{Distance}(a_1, b_1) \ll \text{Distance}(a_2, b_1)$ by a significant margin, keep the shorter match.
  * Otherwise, reject and flag as **`CONFLICT`** for manual review (e.g., parallel service roads or lane crossings).
* **Explanation**: *"Topological conflict: parallel lane matching overlap mismatch."*

