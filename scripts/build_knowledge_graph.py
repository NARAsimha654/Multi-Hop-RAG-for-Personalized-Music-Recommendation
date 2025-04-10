from neo4j import GraphDatabase
import pandas as pd
import os
import time # Optional: for timing operations

# --- Neo4j Connection Details ---
# Use the Bolt URI shown in Neo4j Desktop
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Narasimha123" # <<<--- IMPORTANT: PUT YOUR PASSWORD HERE

# --- File Paths ---
# Make sure these paths point to your processed data files
MSD_CLEANED_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_subset_cleaned.csv' # Or .csv
MPD_TRACKS_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_unique_tracks.parquet'
MPD_PLAYLISTS_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_playlists.parquet'
MPD_PLAYLIST_TRACKS_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_playlist_tracks.parquet'
MATCHES_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_mpd_direct_matches.parquet'

# Try to establish connection
print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity() # Check if connection is valid
    print("Neo4j connection successful!")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    print("Please check Neo4j Desktop, URI, username, and password. Exiting.")
    # Exit if connection fails
    exit()

# Helper function to run queries (from previous example)
def run_cypher_query(tx, query, parameters=None):
    result = tx.run(query, parameters)
    # If the query modifies data, you might not need to return records
    # If it reads data (like MATCH...RETURN), collect results
    return [record for record in result] # Adjust as needed

def create_constraints(driver):
    # Use the specific database name if not default. Usually not needed for driver sessions.
    # driver.session(database="db-1")
    print("Creating constraints for data integrity and performance...")
    queries = [
        "CREATE CONSTRAINT unique_artist_name IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE;",
        "CREATE CONSTRAINT unique_msd_song_id IF NOT EXISTS FOR (s:MsdSong) REQUIRE s.song_id IS UNIQUE;",
        "CREATE CONSTRAINT unique_mpd_track_uri IF NOT EXISTS FOR (t:MpdTrack) REQUIRE t.track_uri IS UNIQUE;",
        "CREATE CONSTRAINT unique_playlist_pid IF NOT EXISTS FOR (p:Playlist) REQUIRE p.pid IS UNIQUE;"
    ]
    # Use try-execute-commit pattern for writes
    with driver.session(database="db-1") as session: # Specify database if needed
        for query in queries:
            try:
                print(f"  Executing: {query}")
                # execute_write handles transaction retries automatically
                session.execute_write(run_cypher_query, query)
            except Exception as e:
                # This might happen if constraint already exists in a slightly different form
                # Or if there's a syntax error. Neo4j driver often raises specific exceptions.
                print(f"  Warning/Error creating constraint: {e}") # Log warning
    print("Constraints checked/created.")

# --- Call the function ---
create_constraints(driver)

# --- Load Data into Pandas DataFrames ---
print("\nLoading processed data files into pandas DataFrames...")

try:
    # Adjust read function if you saved as CSV (e.g., pd.read_csv)
    if MSD_CLEANED_FILE.endswith('.parquet'):
        msd_df = pd.read_parquet(MSD_CLEANED_FILE)
    else:
        msd_df = pd.read_csv(MSD_CLEANED_FILE)
    print(f"  Loaded MSD Cleaned Data: {MSD_CLEANED_FILE} ({msd_df.shape[0]} rows)")

    if MPD_TRACKS_FILE.endswith('.parquet'):
        mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else:
        mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    print(f"  Loaded MPD Unique Tracks: {MPD_TRACKS_FILE} ({mpd_tracks_df.shape[0]} rows)")

    if MPD_PLAYLISTS_FILE.endswith('.parquet'):
        mpd_playlists_df = pd.read_parquet(MPD_PLAYLISTS_FILE)
    else:
        mpd_playlists_df = pd.read_csv(MPD_PLAYLISTS_FILE)
    print(f"  Loaded MPD Playlists: {MPD_PLAYLISTS_FILE} ({mpd_playlists_df.shape[0]} rows)")

    if MPD_PLAYLIST_TRACKS_FILE.endswith('.parquet'):
        mpd_playlist_tracks_df = pd.read_parquet(MPD_PLAYLIST_TRACKS_FILE)
    else:
        mpd_playlist_tracks_df = pd.read_csv(MPD_PLAYLIST_TRACKS_FILE)
    print(f"  Loaded MPD Playlist-Track Map: {MPD_PLAYLIST_TRACKS_FILE} ({mpd_playlist_tracks_df.shape[0]} rows)")

    if MATCHES_FILE.endswith('.parquet'):
         direct_matches_df = pd.read_parquet(MATCHES_FILE)
    else:
         direct_matches_df = pd.read_csv(MATCHES_FILE)
    print(f"  Loaded Direct Matches: {MATCHES_FILE} ({direct_matches_df.shape[0]} rows)")

    print("All data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Please ensure all necessary processed files exist at the specified paths.")
    # Exit if essential data is missing
    exit()
except Exception as e:
    print(f"An unexpected error occurred during file loading: {e}")
    # Exit on other loading errors
    exit()

# You can optionally display head() or info() for loaded dataframes here
# print("\nMSD DataFrame Head:")
# print(msd_df.head())
# print("\nMPD Tracks DataFrame Head:")
# print(mpd_tracks_df.head())
# etc.

# --- Step: Insert Artist Nodes ---
print("\nStep: Inserting Artist nodes...")

def insert_artists(driver, artist_list):
    """
    Inserts artist nodes into Neo4j using MERGE on the name property.
    Uses UNWIND for batching.
    """
    # Cypher query using UNWIND to process a list of artists
    # MERGE ensures that artists are created only if they don't exist,
    # based on the unique constraint on 'name'
    query = """
    UNWIND $artists AS artist_props
    MERGE (a:Artist {name: artist_props.name})
    RETURN count(a) as created_count
    """
    # Prepare data as a list of dictionaries
    # Neo4j driver expects parameters in a dictionary format
    artists_data = [{'name': name} for name in artist_list if pd.notna(name) and name] # Ensure name is not null/empty

    if not artists_data:
        print("  No valid artist names found to insert.")
        return

    print(f"  Preparing to insert/merge {len(artists_data)} unique artist nodes...")

    # Using execute_write for transactional safety
    with driver.session(database="db-1") as session:
        try:
            # Pass the list of dictionaries as the 'artists' parameter
            summary = session.execute_write(run_cypher_query, query, parameters={'artists': artists_data})
            # Note: run_cypher_query currently returns records, might not be needed for MERGE summary
            # You could also just run: session.execute_write(lambda tx: tx.run(query, artists=artists_data))
            print(f"  Artist node merge/creation process completed.")
            # The summary object from tx.run() often contains counters, but accessing them depends on query details.
            # For simplicity, we confirm completion. Precise count requires specific RETURN clause.

        except Exception as e:
            print(f"  Error inserting artist nodes: {e}")


# 1. Get unique artist names from both dataframes
print("  Extracting unique artist names from MSD and MPD data...")
msd_artists = set(msd_df['artist_name'].dropna().unique())
mpd_artists = set(mpd_tracks_df['artist_name'].dropna().unique())

# Combine the sets to get overall unique artists
unique_artists = list(msd_artists.union(mpd_artists))
# Remove any potential empty strings if they slipped through normalization/dropna
unique_artists = [name for name in unique_artists if name]

print(f"  Found {len(unique_artists)} unique artist names.")

# 2. Call the insertion function
insert_artists(driver, unique_artists)

# --- Remember to close the driver at the very end of your script ---
# driver.close() # Keep this at the end after all operations