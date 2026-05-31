import duckdb

def preprocess_sweden_data(db_path):
    print(f"Connecting to Sweden NVDB Database: {db_path}...")
    conn = duckdb.connect(db_path)
    
    # Ensure spatial extension is loaded
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    print("Executing directed edge duplication and auto-increment unique ID generation...")
    conn.execute("""
        -- 1. Create a sequence for the auto-incrementing directed_id index
        CREATE SEQUENCE IF NOT EXISTS directed_id_seq;

        -- 2. Create the new standardized table with duplicated reverse edges
        CREATE OR REPLACE TABLE main.vehicle_edges_directed AS
        WITH RawDirected AS (
            -- Forward direction for all edges that are not reverse-only
            SELECT 
                edge_id AS original_edge_id,
                nvdb_id,
                name,
                geometry,
                FALSE AS is_reverse
            FROM main.vehicle_edges
            WHERE oneway != '-1'

            UNION ALL

            -- Reverse direction for bidirectional and reverse-only edges
            SELECT 
                edge_id AS original_edge_id,
                nvdb_id,
                name,
                ST_Reverse(geometry) AS geometry,
                TRUE AS is_reverse
            FROM main.vehicle_edges
            WHERE oneway = 'no' OR oneway = '-1'
        )
        SELECT 
            nextval('directed_id_seq')::BIGINT AS directed_id,
            *
        FROM RawDirected;

        -- 3. Clean up the sequence
        DROP SEQUENCE directed_id_seq;
    """)
    
    # Verify table row count
    original_count = conn.execute("SELECT COUNT(*) FROM main.vehicle_edges").fetchone()[0]
    directed_count = conn.execute("SELECT COUNT(*) FROM main.vehicle_edges_directed").fetchone()[0]
    
    print(f"✅ Preprocessing successful!")
    print(f"--> Original edges: {original_count}")
    print(f"--> Directed edges (including duplicated reverse lanes): {directed_count}")
    
    conn.close()

if __name__ == "__main__":
    db_path = "/home/kaveh/projects/fetching-sweden-data/data/processed/sundbyberg.duckdb"
    preprocess_sweden_data(db_path)
