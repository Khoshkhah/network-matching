import numpy as np
from shapely.geometry import LineString, Point
from typing import List, Tuple, Dict

def dtw_align(coords_a: List[Tuple[float, float]], coords_b: List[Tuple[float, float]], step_meters: float = 5.0) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]], Dict[str, float]]:
    """
    Executes the TRUE Continuous Piecewise Order-Preserving Projection Alignment.
    
    1. Identifies the shorter road as Query (Q) and longer road as Target (T).
    2. Projects the real vertices of Q onto the continuous segments of T.
    3. Projects the real vertices of T onto the query Q, filtering out unmatched ends.
    4. Combines all bidirectional projections and applies a strict Monotonicity Filter.
    5. Returns the length-normalized average distance, the warping path links, and 
       a dictionary containing the 'average', 'max', and 'min' projection distances.
    
    Returns:
    - average_distance: Average physical alignment drift in meters (float).
    - warping_path: List of coordinate pairs [((xa, ya), (xb, yb)), ...] representing the projection links.
    - metrics: Dictionary containing {'average': float, 'max': float, 'min': float} distances in meters.
    """
    if len(coords_a) < 2 or len(coords_b) < 2:
        return float('inf'), [], {"average": float('inf'), "max": float('inf'), "min": float('inf')}
        
    line_a = LineString(coords_a)
    line_b = LineString(coords_b)
    
    len_a = line_a.length
    len_b = line_b.length
    
    # 1. Determine Query (shorter) vs Target (longer)
    if len_a >= len_b:
        query_line, target_line = line_b, line_a
        query_coords, target_coords = coords_b, coords_a
        is_a_target = True
    else:
        query_line, target_line = line_a, line_b
        query_coords, target_coords = coords_a, coords_b
        is_a_target = False
        
    len_q = query_line.length
    len_t = target_line.length
    
    candidate_pairs = []
    
    # 2. Project all Real Vertices of Query (Q) onto the Target (T)
    for q_pt in query_coords:
        pt = Point(q_pt)
        s_t = target_line.project(pt)
        proj_t = target_line.interpolate(s_t)
        s_q = query_line.project(pt)
        
        dist = pt.distance(proj_t)
        
        pt_t_coords = (float(proj_t.x), float(proj_t.y))
        pt_q_coords = q_pt
        
        candidate_pairs.append({
            "pt_a": pt_t_coords if is_a_target else pt_q_coords,
            "pt_b": pt_q_coords if is_a_target else pt_t_coords,
            "s_a": s_t if is_a_target else s_q,
            "s_b": s_q if is_a_target else s_t,
            "distance": dist
        })
        
    # 3. Project all Real Vertices of Target (T) onto the Query (Q)
    # Filter out unmatched ends of T (0.0 < s_q < len_q)
    for t_pt in target_coords:
        pt = Point(t_pt)
        s_q = query_line.project(pt)
        s_t = target_line.project(pt)
        
        if 0.0 < s_q < len_q:
            proj_q = query_line.interpolate(s_q)
            dist = pt.distance(proj_q)
            
            pt_t_coords = t_pt
            pt_q_coords = (float(proj_q.x), float(proj_q.y))
            
            candidate_pairs.append({
                "pt_a": pt_t_coords if is_a_target else pt_q_coords,
                "pt_b": pt_q_coords if is_a_target else pt_t_coords,
                "s_a": s_t if is_a_target else s_q,
                "s_b": s_q if is_a_target else s_t,
                "distance": dist
            })
            
    # 4. Sort all candidate pairs along the Query road B (s_b)
    candidate_pairs = sorted(candidate_pairs, key=lambda x: x["s_b"])
    
    # 5. Apply the strict Monotonicity (Order-Preserving) Filter
    approved_pairs = []
    
    for pair in candidate_pairs:
        if not approved_pairs:
            approved_pairs.append(pair)
        else:
            last_approved = approved_pairs[-1]
            if pair["s_a"] >= last_approved["s_a"]:
                approved_pairs.append(pair)
                
    if not approved_pairs:
        return float('inf'), [], {"average": float('inf'), "max": float('inf'), "min": float('inf')}
        
    # 6. Calculate length-normalized metrics (average, max, min)
    distances = [p["distance"] for p in approved_pairs]
    average_distance = float(np.mean(distances))
    max_distance = float(np.max(distances))
    min_distance = float(np.min(distances))
    
    metrics = {
        "average": average_distance,
        "max": max_distance,
        "min": min_distance
    }
    
    # Construct the warping path coordinate pairs
    warping_path = [((p["pt_a"]), (p["pt_b"])) for p in approved_pairs]
    
    return average_distance, warping_path, metrics
