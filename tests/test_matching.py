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
    
    # Map A (Source Network A)
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
    
    # Map B (Destination Network B)
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
    
    # Use only one parameter: 20 meters candidate search radius
    matcher.set_parameters(max_distance=20.0)
    
    print("\n[Step 2] Executing Tier 1 Candidate Generation...")
    candidates = matcher.generate_candidate_pairs()
    print(f"Generated {len(candidates)} spatial candidate pairs.")
    print(candidates[["id_a", "id_b"]])
    
    print("\n[Step 3] Executing Tier 2 DTW Performance calculations...")
    evaluated = matcher.compute_dtw_metrics(candidates)
    print("Evaluation Metrics:")
    print(evaluated.to_string(index=False))

    print("\n[Step 4] Executing full match() pipeline (includes NO_MATCH rows for unmatched sources)...")
    results = matcher.match()   # uses all 3 tiers + appends NO_MATCH rows
    print("\nDirectional Match Results (source A → destination B):")
    print(results.to_string(index=False))

    # 4. Verify Correctness Assertion
    print("\n==================================================")
    print("          VERIFICATION & CORRECTNESS CHECKS")
    print("==================================================")

    # Check 1: Does the long source 'A1' match ALL 3 candidate destinations 'B1', 'B2', 'B3'
    # (since no cutoff filters are applied), with exactly one rank 1 match (the closest)?
    a1 = results[results["source_id"] == "A1"]
    a1_dests = sorted(a1["dest_id"].tolist())
    if a1_dests == ["B1", "B2", "B3"] and int((a1["rank"] == 1).sum()) == 1:
        print("✅ [PASS] Source 'A1' → all candidate destinations 'B1','B2','B3' kept; exactly one rank 1 match.")
    else:
        print("❌ [FAIL] A1 destinations wrong. Got:", a1_dests, "| rank 1 sum:", int((a1["rank"] == 1).sum()))

    # Check 2: Was B3 (opposite direction) kept as a candidate but not flagged as best?
    b3_match = results[results["dest_id"] == "B3"]
    if not b3_match.empty and b3_match.iloc[0]["rank"] != 1:
        print("✅ [PASS] Opposite direction road 'B3' kept as candidate but correctly not marked as best.")
    else:
        print("❌ [FAIL] Opposite direction road 'B3' was incorrectly ranked or missing!")

    # Check 3: Unmatched source 'A2' (no nearby B segments) must appear as a NO_MATCH row.
    #           Unmatched destination 'B4' must NOT appear as a dest_id.
    a2_rows = results[results["source_id"] == "A2"]
    b4_matches = results[results["dest_id"] == "B4"]
    a2_is_no_match = (
        len(a2_rows) == 1
        and a2_rows.iloc[0]["match_type"] == "NO_MATCH"
        and pd.isna(a2_rows.iloc[0]["dest_id"])
        and pd.isna(a2_rows.iloc[0]["dtw_distance"])
    )
    if a2_is_no_match and b4_matches.empty:
        print("✅ [PASS] Unmatched source 'A2' returned as NO_MATCH row; 'B4' not present as a dest.")
    else:
        print("❌ [FAIL] Unmatched road handling incorrect.",
              f"a2_rows={a2_rows[['source_id','dest_id','match_type']].to_dict()}",
              f"b4_present={not b4_matches.empty}")
        
    print("\n[Step 5] Applying resolve() decision strategies...")

    # Check 4: best_per_source (MANY-TO-ONE) -> each source keeps exactly its closest dest,
    #          sources are never reused, and unmatched 'A2' stays a NO_MATCH row.
    bps = matcher.resolve(results, strategy="best_per_source")
    print("\nbest_per_source assignment:")
    print(bps.to_string(index=False))
    a1_bps = bps[(bps["source_id"] == "A1") & (bps["match_type"] != "NO_MATCH")]
    a2_bps = bps[bps["source_id"] == "A2"]
    matched_bps = bps[bps["match_type"] != "NO_MATCH"]
    if (len(a1_bps) == 1 and int(a1_bps.iloc[0]["rank"]) == 1
            and not matched_bps["source_id"].duplicated().any()
            and len(a2_bps) == 1 and a2_bps.iloc[0]["match_type"] == "NO_MATCH"):
        print("✅ [PASS] best_per_source: 'A1' assigned its single closest dest; sources unique; 'A2' NO_MATCH.")
    else:
        print("❌ [FAIL] best_per_source produced an unexpected assignment.")

    # Check 5: one_to_one (GLOBAL UNIQUE) -> no source AND no destination is reused.
    o2o = matcher.resolve(results, strategy="one_to_one")
    matched_o2o = o2o[o2o["match_type"] != "NO_MATCH"]
    if (not matched_o2o["source_id"].duplicated().any()
            and not matched_o2o["dest_id"].duplicated().any()):
        print("✅ [PASS] one_to_one: every source and destination used at most once.")
    else:
        print("❌ [FAIL] one_to_one reused a source or destination.")

    # Check 6: best_per_dest (ONE-TO-MANY) -> each of B1/B2/B3 (whose only nearby source is A1)
    #          keeps its single best source 'A1'; destinations are unique, source 'A1' repeats.
    bpd = matcher.resolve(results, strategy="best_per_dest")
    matched_bpd = bpd[bpd["match_type"] != "NO_MATCH"]
    if (not matched_bpd["dest_id"].duplicated().any()
            and set(matched_bpd["dest_id"]) == {"B1", "B2", "B3"}
            and (matched_bpd["source_id"] == "A1").all()):
        print("✅ [PASS] best_per_dest: B1/B2/B3 each kept their single best source 'A1'.")
    else:
        print("❌ [FAIL] best_per_dest produced an unexpected assignment.")

    print("\n==================================================")
    print("     TEST RUN COMPLETED SUCCESSFULY")
    print("==================================================")


def run_symmetric_test_suite():
    """Validate the symmetric (two-way) split-aware reconciliation (match_symmetric)."""
    print("\n==================================================")
    print("     SYMMETRIC (TWO-WAY) MATCHING TEST SUITE")
    print("==================================================")

    # Synthetic scenario (coords ~51.4E 35.7N; UTM 32639):
    # - A_full <-> B_full : full mutual coverage         -> 1:1
    # - A_long -> B_h1,B_h2: B pieces tile the long road -> 1:N_SPLIT
    # - A_par   .. B_par   : only ~10% shared stretch    -> dropped by containment rule
    df_a = pd.DataFrame([
        {"id": "A_full", "geom_wkt": "LINESTRING(51.4000 35.7000, 51.4000 35.7050)"},
        {"id": "A_long", "geom_wkt": "LINESTRING(51.4100 35.7000, 51.4100 35.7050)"},
        {"id": "A_par",  "geom_wkt": "LINESTRING(51.4300 35.7000, 51.4300 35.7050)"},
    ])
    df_b = pd.DataFrame([
        {"id": "B_full", "geom_wkt": "LINESTRING(51.4001 35.7000, 51.4001 35.7050)"},
        {"id": "B_h1",   "geom_wkt": "LINESTRING(51.4101 35.7000, 51.4101 35.7025)"},
        {"id": "B_h2",   "geom_wkt": "LINESTRING(51.4101 35.7025, 51.4101 35.7050)"},
        {"id": "B_par",  "geom_wkt": "LINESTRING(51.4301 35.7045, 51.4301 35.7090)"},
    ])

    matcher = DuckDBMapMatcher()
    matcher.conn.register("sym_raw_a", df_a)
    matcher.conn.register("sym_raw_b", df_b)
    matcher.conn.execute("CREATE TABLE sym_a AS SELECT id, ST_GeomFromText(geom_wkt) AS geom FROM sym_raw_a")
    matcher.conn.execute("CREATE TABLE sym_b AS SELECT id, ST_GeomFromText(geom_wkt) AS geom FROM sym_raw_b")
    matcher.configure_sources(
        source_a="sym_a", id_col_a="id", geom_col_a="geom",
        source_b="sym_b", id_col_b="id", geom_col_b="geom", utm_srid=32639,
    )
    matcher.set_parameters(max_distance=25.0)

    sym = matcher.match_symmetric(max_dtw=12.0, max_angle=45.0, keep_overlap=70, sym_overlap=70)
    print("\nSymmetric match table:")
    print(sym.to_string(index=False))

    # Check 1: A_full <-> B_full is a clean 1:1.
    r = sym[(sym["a_id"] == "A_full") & (sym["b_id"] == "B_full")]
    if len(r) == 1 and r.iloc[0]["relation"] == "1:1" and r.iloc[0]["cardinality"] == "1:1":
        print("✅ [PASS] A_full <-> B_full classified as 1:1.")
    else:
        print("❌ [FAIL] A_full <-> B_full not classified as 1:1.")

    # Check 2: A_long keeps BOTH pieces B_h1 and B_h2, labelled split / 1:N_SPLIT.
    s = sym[sym["a_id"] == "A_long"]
    if (set(s["b_id"]) == {"B_h1", "B_h2"}
            and (s["relation"] == "split").all()
            and (s["cardinality"] == "1:N_SPLIT").all()):
        print("✅ [PASS] A_long preserved as a 1:N split across B_h1 and B_h2.")
    else:
        print("❌ [FAIL] A_long split not preserved.", set(s["b_id"]))

    # Check 3: A_par / B_par share only a short stretch -> dropped by the containment rule.
    if sym[sym["a_id"] == "A_par"].empty:
        print("✅ [PASS] Incidental A_par/B_par pair dropped (containment < keep_overlap).")
    else:
        print("❌ [FAIL] Incidental A_par/B_par pair was not dropped.")

    # Check 4: the containment keep-rule is never violated in the output.
    if (sym["containment"] >= 70).all():
        print("✅ [PASS] Every kept edge satisfies containment >= keep_overlap.")
    else:
        print("❌ [FAIL] An edge with containment < keep_overlap survived.")

    print("\n==================================================")
    print("     SYMMETRIC TEST RUN COMPLETED SUCCESSFULY")
    print("==================================================")


if __name__ == "__main__":
    run_automated_test_suite()
    run_symmetric_test_suite()

