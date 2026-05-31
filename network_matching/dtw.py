import numpy as np
from shapely.geometry import LineString, Point
from typing import List, Tuple, Dict

def dtw_align(
    coords_a: List[Tuple[float, float]], 
    coords_b: List[Tuple[float, float]], 
    step_meters: float = 5.0
) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]], Dict[str, float]]:
    """
    Executes the progressive DP-based DTW alignment from Source Segment A to Destination Segment B.
    
    The match starts at the beginning of the overlapping region (determined dynamically by projecting
    the beginning nodes of each segment onto the other with no length comparisons).
    
    Returns:
    - average_distance: Average physical alignment drift in meters (float).
    - warping_path: List of coordinate pairs [((xa, ya), (xb, yb)), ...] representing the alignment links.
    - metrics: Dictionary containing {'average': float, 'max': float, 'min': float, 'overlap_pct': float}.
    """
    if len(coords_a) < 2 or len(coords_b) < 2:
        return float('inf'), [], {
            "average": float('inf'), 
            "max": float('inf'), 
            "min": float('inf'), 
            "overlap_pct": 0.0
        }
        
    line_a = LineString(coords_a)
    line_b = LineString(coords_b)
    
    len_a = line_a.length
    len_b = line_b.length
    
    # 1. Determine the overlapping start point on both segments
    s_b_start = line_b.project(Point(coords_a[0]))
    s_a_start = line_a.project(Point(coords_b[0]))
    
    if s_a_start > 1e-3 and s_b_start <= 1e-3:
        start_on_a = s_a_start
        start_on_b = 0.0
    elif s_b_start > 1e-3 and s_a_start <= 1e-3:
        start_on_a = 0.0
        start_on_b = s_b_start
    else:
        start_on_a = s_a_start if s_a_start > s_b_start else 0.0
        start_on_b = s_b_start if s_b_start > s_a_start else 0.0
        
    # Calculate cumulative distances along raw nodes
    dist_a = [0.0]
    for i in range(1, len(coords_a)):
        dist_a.append(dist_a[-1] + Point(coords_a[i-1]).distance(Point(coords_a[i])))
        
    dist_b = [0.0]
    for j in range(1, len(coords_b)):
        dist_b.append(dist_b[-1] + Point(coords_b[j-1]).distance(Point(coords_b[j])))
        
    # 2. Build sorted unified points list along Source A (pts_a) starting from start_on_a
    points_a_pool = []
    for i in range(len(coords_a)):
        if dist_a[i] >= start_on_a:
            points_a_pool.append({"pt": Point(coords_a[i]), "dist": dist_a[i]})
    for pt_b_coords in coords_b:
        s_a = line_a.project(Point(pt_b_coords))
        if s_a >= start_on_a:
            points_a_pool.append({"pt": line_a.interpolate(s_a), "dist": s_a})
            
    points_a_pool = sorted(points_a_pool, key=lambda x: x["dist"])
    pts_a = []
    for p in points_a_pool:
        if not pts_a or p["dist"] - pts_a[-1]["dist"] > 1e-3:
            pts_a.append(p)
            
    # 3. Build sorted unified points list along Destination B (pts_b) starting from start_on_b
    points_b_pool = []
    for j in range(len(coords_b)):
        if dist_b[j] >= start_on_b:
            points_b_pool.append({"pt": Point(coords_b[j]), "dist": dist_b[j]})
    for pt_a_coords in coords_a:
        s_b = line_b.project(Point(pt_a_coords))
        if s_b >= start_on_b:
            points_b_pool.append({"pt": line_b.interpolate(s_b), "dist": s_b})
            
    points_b_pool = sorted(points_b_pool, key=lambda x: x["dist"])
    pts_b = []
    for p in points_b_pool:
        if not pts_b or p["dist"] - pts_b[-1]["dist"] > 1e-3:
            pts_b.append(p)
            
    N = len(pts_a)
    M = len(pts_b)
    
    if N == 0 or M == 0:
        return float('inf'), [], {
            "average": float('inf'), 
            "max": float('inf'), 
            "min": float('inf'), 
            "overlap_pct": 0.0
        }
        
    # 4. DP Cost Matrix
    D = np.full((N, M), float('inf'))
    D[0][0] = pts_a[0]["pt"].distance(pts_b[0]["pt"])
    
    # Populate row 0
    for j in range(1, M):
        D[0][j] = D[0][j-1] + pts_b[j]["pt"].distance(pts_a[0]["pt"])
        
    # Populate column 0
    for i in range(1, N):
        D[i][0] = D[i-1][0] + pts_a[i]["pt"].distance(pts_b[0]["pt"])
        
    # Populate rest of the DP cost matrix
    for i in range(1, N):
        for j in range(1, M):
            dist = pts_a[i]["pt"].distance(pts_b[j]["pt"])
            options = []
            if i > 0:
                options.append(D[i-1][j])
            if j > 0:
                options.append(D[i][j-1])
            if i > 0 and j > 0:
                options.append(D[i-1][j-1])
            if options:
                D[i][j] = dist + min(options)
                
    # 5. Boundary-Based Stopping Criteria
    best_cost = float('inf')
    best_cell = (0, 0)
    
    for j in range(M):
        if D[N-1][j] <= best_cost:
            best_cost = D[N-1][j]
            best_cell = (N-1, j)
            
    for i in range(N):
        if D[i][M-1] <= best_cost:
            best_cost = D[i][M-1]
            best_cell = (i, M-1)
            
    i_end, j_end = best_cell
    
    if best_cost == float('inf'):
        return float('inf'), [], {
            "average": float('inf'), 
            "max": float('inf'), 
            "min": float('inf'), 
            "overlap_pct": 0.0
        }
        
    # Backtrack to (0, 0)
    path = []
    curr = best_cell
    while curr != (0, 0):
        path.append(curr)
        i, j = curr
        neighbors = []
        if i > 0:
            neighbors.append(((i-1, j), D[i-1][j]))
        if j > 0:
            neighbors.append(((i, j-1), D[i][j-1]))
        if i > 0 and j > 0:
            neighbors.append(((i-1, j-1), D[i-1][j-1]))
        neighbors = sorted(neighbors, key=lambda x: x[1])
        curr = neighbors[0][0]
    path.append((0, 0))
    path.reverse()
    
    # Build warping path links
    warping_path = []
    for (i, j) in path:
        pt_a_coords = (float(pts_a[i]["pt"].x), float(pts_a[i]["pt"].y))
        pt_b_coords = (float(pts_b[j]["pt"].x), float(pts_b[j]["pt"].y))
        warping_path.append((pt_a_coords, pt_b_coords))
        
    # Calculate physical distances along the warping path
    distances = [Point(pa).distance(Point(pb)) for (pa, pb) in warping_path]
    average_distance = float(np.mean(distances))
    
    # Overlap percentage of Source A covered by the matched section
    matched_len = pts_a[i_end]["dist"] - start_on_a
    overlap_pct = min(100.0, max(0.0, round((matched_len / len_a) * 100.0, 2)))
    
    metrics = {
        "average": average_distance,
        "max": float(np.max(distances)),
        "min": float(np.min(distances)),
        "overlap_pct": overlap_pct
    }
    
    return average_distance, warping_path, metrics
