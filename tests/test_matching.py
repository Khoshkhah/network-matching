import sys
import os
import pandas as pd

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from network_matching import DuckDBMapMatcher

def run_automated_test_suite():
    print("==================================================")
    print("     STARTING NETWORK MATCHING TEST SUITE")
    print("==================================================")
    
    # 1. Create simulated datasets in Pandas (coordinates in Longitude, Latitude - EPSG:4326)
    # Target UTM SRID is 32639 (covers Iran/Tehran area, coords around 51.4E, 35.7N)
    
    # Map A (Coarse Network)
    # - Road A1: 1 long road going straight North (approx 550m)
    # - Road A2: 1 road located very far away (unmatched)
    df_a = pd.DataFrame([
        {
            "id": "A1", 
            "geom_wkt": "LINESTRING(51.4000 35.7000, 51.4000 35.7050)"
        },
        {
            "id": "A2", 
            "geom_wkt": "LINESTRING(51.5000 35.8000, 51.5000 35.8050)"
        }
    ])
    
    # Map B (Fine Network)
    # - Road B1: Short road (split part 1), parallel to A1, shifted 9 meters East
    # - Road B2: Short road (split part 2), parallel to A1, shifted 9 meters East
    # - Road B3: Road parallel to A1 but pointing in OPPOSITE direction (Southbound)
    # - Road B4: Road located very far away (unmatched)
    df_b = pd.DataFrame([
        {
            "id": "B1", 
            "geom_wkt": "LINESTRING(51.4001 35.7000, 51.4001 35.7025)"
        },
        {
            "id": "B2", 
            "geom_wkt": "LINESTRING(51.4001 35.7025, 51.4001 35.7050)"
        },
        {
            "id": "B3", 
            "geom_wkt": "LINESTRING(51.4001 35.7050, 51.4001 35.7000)" # Southbound
        },
        {
            "id": "B4", 
            "geom_wkt": "LINESTRING(51.6000 35.9000, 51.6000 35.9050)"
        }
    ])
    
    print("\n[Step 1] Loading synthetic data sources...")
    
    # 2. Initialize the matcher
    matcher = DuckDBMapMatcher()
    
    # Register the raw DataFrames into the in-memory DuckDB connection
    matcher.conn.register("raw_df_a", df_a)
    matcher.conn.register("raw_df_b", df_b)
    
    # Convert raw text WKT columns to DuckDB Spatial geometries in-memory
    matcher.conn.execute("CREATE TABLE network_a AS SELECT id, ST_GeomFromText(geom_wkt) AS geom FROM raw_df_a")
    matcher.conn.execute("CREATE TABLE network_b AS SELECT id, ST_GeomFromText(geom_wkt) AS geom FROM raw_df_b")
    
    # 3. Configure the matcher sources
    matcher.configure_sources(
        source_a="network_a", id_col_a="id", geom_col_a="geom",
        source_b="network_b", id_col_b="id", geom_col_b="geom",
        utm_srid=32639
    )
    
    # Use typical thresholds: 20 meters radius, 30 degrees heading limit
    matcher.set_parameters(max_distance=20.0, max_angle=30.0, min_overlap=0.50)
    
    print("\n[Step 2] Executing Tier 1 Candidate Generation...")
    candidates = matcher.generate_candidate_pairs()
    print(f"Generated {len(candidates)} spatial candidate pairs.")
    print(candidates[["id_a", "id_b"]])
    
    print("\n[Step 3] Executing Tier 2 DTW Performance calculations...")
    evaluated = matcher.compute_dtw_metrics(candidates)
    print("Evaluation Metrics:")
    print(evaluated.to_string(index=False))
    
    print("\n[Step 4] Executing Post-Processing & Bidirectional Reconciliation...")
    results = matcher.reconcile_matches(evaluated)
    print("\nReconciliation Matches Results:")
    print(results.to_string(index=False))
    
    # 4. Verify Correctness Assertion
    print("\n==================================================")
    print("          VERIFICATION & CORRECTNESS CHECKS")
    print("==================================================")
    
    # Check 1: Did B1 and B2 successfully trigger a 1:N Split match with A1?
    split_matches = results[results["match_type"] == "1:N_SPLIT"]
    split_ids = sorted(split_matches["id_b"].tolist())
    if split_ids == ["B1", "B2"]:
        print("✅ [PASS] 1:N Split detected perfectly! Coarse road 'A1' matched fine parts 'B1' and 'B2'.")
    else:
        print("❌ [FAIL] 1:N Split failed detection. Got:", split_ids)
        
    # Check 2: Was B3 (opposite direction) filtered out?
    # B3 should NOT have any approved match in results because its bearing difference is 180 degrees.
    b3_matches = results[results["id_b"] == "B3"]
    if b3_matches.empty:
        print("✅ [PASS] Opposite direction road 'B3' successfully filtered out (Direction mismatch).")
    else:
        print("❌ [FAIL] Opposite direction road 'B3' was incorrectly matched!")
        
    # Check 3: Were unmatched roads (A2, B4) successfully ignored?
    a2_matches = results[results["id_a"] == "A2"]
    b4_matches = results[results["id_b"] == "B4"]
    if a2_matches.empty and b4_matches.empty:
        print("✅ [PASS] Missing/Unmatched roads ('A2', 'B4') successfully ignored (No Match).")
    else:
        print("❌ [FAIL] Unmatched roads were incorrectly included in matches!")
        
    print("\n==================================================")
    print("     TEST RUN COMPLETED SUCCESSFULY")
    print("==================================================")
 
if __name__ == "__main__":
    run_automated_test_suite()
