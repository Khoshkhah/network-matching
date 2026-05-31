import os
import duckdb

def export_data_to_csv():
    # 1. Create data directory if it doesn't exist
    data_dir = "/home/kaveh/projects/network-matching/data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created/verified data directory: {data_dir}")
    
    # 2. Initialize in-memory DuckDB to bypass any active read-write file locks
    print("Initializing in-memory DuckDB with spatial extension...")
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    # 3. Attach databases in READ_ONLY mode
    db_path_a = "/home/kaveh/projects/osm-traffic-enrichment/db/sundbyberg.duckdb"
    db_path_b = "/home/kaveh/projects/fetching-sweden-data/data/processed/sundbyberg.duckdb"
    
    print(f"Attaching DB A (OSM) as read-only: {db_path_a}...")
    conn.execute(f"ATTACH '{db_path_a}' AS db_a (READ_ONLY);")
    
    print(f"Attaching DB B (Sweden NVDB) as read-only: {db_path_b}...")
    conn.execute(f"ATTACH '{db_path_b}' AS db_b (READ_ONLY);")
    
    # 4. Export OSM edges table
    osm_csv_path = os.path.join(data_dir, "osm_edges.csv")
    print(f"Exporting OSM edges directly to CSV: {osm_csv_path}...")
    conn.execute(f"COPY (SELECT * FROM db_a.driving.edges) TO '{osm_csv_path}' (HEADER, DELIMITER ',');")
    
    # 5. Standardize Sweden road geometries in-memory and export
    sweden_csv_path = os.path.join(data_dir, "sweden_edges.csv")
    print("Standardizing Sweden road geometries in-memory (duplicating bidirectional, reversing coordinates)...")
    conn.execute("CREATE SEQUENCE directed_id_seq;")
    conn.execute("""
        CREATE TABLE vehicle_edges_directed AS
        WITH RawDirected AS (
            -- Forward direction for all edges that are not reverse-only
            SELECT 
                edge_id AS original_edge_id,
                nvdb_id,
                name,
                geometry,
                FALSE AS is_reverse
            FROM db_b.main.vehicle_edges
            WHERE oneway != '-1'

            UNION ALL

            -- Reverse direction for bidirectional and reverse-only edges
            SELECT 
                edge_id AS original_edge_id,
                nvdb_id,
                name,
                ST_Reverse(geometry) AS geometry,
                TRUE AS is_reverse
            FROM db_b.main.vehicle_edges
            WHERE oneway = 'no' OR oneway = '-1'
        )
        SELECT 
            nextval('directed_id_seq')::BIGINT AS directed_id,
            *
        FROM RawDirected;
    """)
    
    print(f"Exporting preprocessed directed Sweden edges to CSV: {sweden_csv_path}...")
    conn.execute(f"COPY vehicle_edges_directed TO '{sweden_csv_path}' (HEADER, DELIMITER ',');")
    
    # Verify the generated files
    osm_size = os.path.getsize(osm_csv_path)
    sweden_size = os.path.getsize(sweden_csv_path)
    
    print("\n✅ Data successfully exported to project data directory!")
    print(f"--> OSM edges CSV: {osm_csv_path} ({osm_size / 1024:.1f} KB)")
    print(f"--> Sweden edges CSV: {sweden_csv_path} ({sweden_size / 1024:.1f} KB)")
    
    conn.close()

if __name__ == "__main__":
    export_data_to_csv()
