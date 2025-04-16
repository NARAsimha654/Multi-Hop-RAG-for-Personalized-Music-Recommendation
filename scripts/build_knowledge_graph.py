# Import necessary libraries
from neo4j import GraphDatabase
import pandas as pd
import os
import time # Optional: for timing operations
import sys # To exit if connection fails

# --- Neo4j Connection Details ---
# Use the Bolt URI shown in Neo4j Desktop
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
# Use the actual password you set
NEO4J_PASSWORD = "Narasimha123" # <<<--- VERIFY YOUR PASSWORD

# --- File Paths ---
# Using raw strings (r'...') for better path handling
MSD_CLEANED_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_subset_cleaned.csv'
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'
MPD_PLAYLISTS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_playlists.parquet'
MPD_PLAYLIST_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_playlist_tracks.parquet'
MATCHES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_mpd_direct_matches.parquet'

# --- Neo4j Driver Setup ---
driver = None # Initialize driver variable
print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
try:
    # Increase connection timeout if needed, e.g., max_connection_lifetime=3600
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity() # Check if connection is valid
    print("Neo4j connection successful!")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    print("Please check Neo4j Desktop, URI, username, and password. Exiting.")
    sys.exit(1) # Exit if connection fails

# --- Helper Function to Run Cypher Queries ---
def run_cypher_query(tx, query, parameters=None):
    """ Helper function to execute a Cypher query within a transaction """
    result = tx.run(query, parameters)
    # Consume the result to ensure the query executes fully on the server side
    return result.consume()

# --- Function to Create Constraints ---
def create_constraints(driver):
    """ Creates unique constraints on node properties """
    print("Creating constraints for data integrity and performance...")
    db_name = "db-1"
    queries = [
        "CREATE CONSTRAINT unique_artist_name IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE;",
        "CREATE CONSTRAINT unique_msd_song_id IF NOT EXISTS FOR (s:MsdSong) REQUIRE s.song_id IS UNIQUE;",
        "CREATE CONSTRAINT unique_mpd_track_uri IF NOT EXISTS FOR (t:MpdTrack) REQUIRE t.track_uri IS UNIQUE;",
        "CREATE CONSTRAINT unique_playlist_pid IF NOT EXISTS FOR (p:Playlist) REQUIRE p.pid IS UNIQUE;"
    ]
    with driver.session(database=db_name) as session:
        for query in queries:
            try:
                print(f"  Executing: {query}")
                session.execute_write(run_cypher_query, query)
            except Exception as e:
                print(f"  Warning/Error creating constraint: {e}")
    print("Constraints checked/created.")

# --- Functions to Insert Nodes ---

def insert_artists(driver, artist_list):
    """ Inserts artist nodes into Neo4j using MERGE """
    print("\nStep: Inserting Artist nodes...")
    db_name = "db-1"
    query = """
    UNWIND $artists AS artist_props
    MERGE (a:Artist {name: artist_props.name})
    """
    artists_data = [{'name': name} for name in artist_list if pd.notna(name) and name]
    if not artists_data:
        print("  No valid artist names found to insert.")
        return
    batch_size = 5000
    total_artists = len(artists_data)
    print(f"  Preparing to insert/merge {total_artists} unique artist nodes in batches of {batch_size}...")
    with driver.session(database=db_name) as session:
        for i in range(0, total_artists, batch_size):
            batch = artists_data[i:min(i + batch_size, total_artists)]
            print(f"    Inserting batch {i // batch_size + 1}/{(total_artists + batch_size - 1) // batch_size} ({len(batch)} artists)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'artists': batch})
            except Exception as e:
                print(f"  Error inserting artist nodes batch starting at index {i}: {e}")
    print(f"  Artist node merge/creation process completed.")

def insert_msd_songs(driver, songs_df):
    """ Inserts MsdSong nodes into Neo4j using MERGE """
    print("\nStep: Inserting MsdSong nodes...")
    db_name = "db-1"
    required_cols = ['song_id', 'title', 'year', 'duration', 'key', 'loudness', 'mode', 'tempo', 'time_signature']
    if not all(col in songs_df.columns for col in required_cols):
        print("  Error: Missing required columns in msd_df for MsdSong insertion.")
        return
    songs_data = []
    for index, row in songs_df[required_cols].iterrows():
        if pd.isna(row['song_id']): continue
        song_props = row.astype(object).where(pd.notna(row), None).to_dict()
        songs_data.append(song_props)
    if not songs_data:
        print("  No valid MSD song data found to insert.")
        return
    batch_size = 5000
    total_songs = len(songs_data)
    print(f"  Preparing to insert/merge {total_songs} MsdSong nodes in batches of {batch_size}...")
    query = """
    UNWIND $songs AS song_props
    MERGE (s:MsdSong {song_id: song_props.song_id})
    SET s += song_props
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_songs, batch_size):
            batch = songs_data[i:min(i + batch_size, total_songs)]
            print(f"    Inserting batch {i // batch_size + 1}/{(total_songs + batch_size - 1) // batch_size} ({len(batch)} songs)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'songs': batch})
            except Exception as e:
                print(f"  Error inserting MsdSong nodes batch starting at index {i}: {e}")
    print(f"  MsdSong node merge/creation process completed.")

def insert_mpd_tracks(driver, tracks_df):
    """ Inserts MpdTrack nodes into Neo4j using MERGE """
    print("\nStep: Inserting MpdTrack nodes...")
    db_name = "db-1"
    required_cols = ['track_uri', 'track_name', 'artist_name', 'album_name', 'duration_ms']
    if not all(col in tracks_df.columns for col in required_cols):
        print("  Error: Missing required columns in mpd_tracks_df for MpdTrack insertion.")
        return
    tracks_data = []
    for index, row in tracks_df[required_cols].iterrows():
        if pd.isna(row['track_uri']): continue
        track_props = row.astype(object).where(pd.notna(row), None).to_dict()
        if track_props['duration_ms'] is not None:
            try: track_props['duration_ms'] = int(track_props['duration_ms'])
            except (ValueError, TypeError): track_props['duration_ms'] = None
        tracks_data.append(track_props)
    if not tracks_data:
        print("  No valid MPD track data found to insert.")
        return
    batch_size = 5000
    total_tracks = len(tracks_data)
    print(f"  Preparing to insert/merge {total_tracks} MpdTrack nodes in batches of {batch_size}...")
    query = """
    UNWIND $tracks AS track_props
    MERGE (t:MpdTrack {track_uri: track_props.track_uri})
    SET t += track_props
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_tracks, batch_size):
            batch = tracks_data[i:min(i + batch_size, total_tracks)]
            print(f"    Inserting batch {i // batch_size + 1}/{(total_tracks + batch_size - 1) // batch_size} ({len(batch)} tracks)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'tracks': batch})
            except Exception as e:
                print(f"  Error inserting MpdTrack nodes batch starting at index {i}: {e}")
    print(f"  MpdTrack node merge/creation process completed.")

def insert_playlists(driver, playlists_df):
    """ Inserts Playlist nodes into Neo4j using MERGE """
    print("\nStep: Inserting Playlist nodes...")
    db_name = "db-1"
    required_cols = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_artists', 'num_edits', 'duration_ms', 'description']
    if not all(col in playlists_df.columns for col in required_cols):
        print("  Error: Missing required columns in mpd_playlists_df for Playlist insertion.")
        return
    playlists_data = []
    for index, row in playlists_df[required_cols].iterrows():
        if pd.isna(row['pid']): continue
        playlist_props = row.astype(object).where(pd.notna(row), None).to_dict()
        try:
            playlist_props['pid'] = int(playlist_props['pid'])
            if playlist_props['modified_at'] is not None: playlist_props['modified_at'] = int(playlist_props['modified_at'])
            if playlist_props['num_tracks'] is not None: playlist_props['num_tracks'] = int(playlist_props['num_tracks'])
            if playlist_props['num_albums'] is not None: playlist_props['num_albums'] = int(playlist_props['num_albums'])
            if playlist_props['num_artists'] is not None: playlist_props['num_artists'] = int(playlist_props['num_artists'])
            if playlist_props['num_edits'] is not None: playlist_props['num_edits'] = int(playlist_props['num_edits'])
            if playlist_props['duration_ms'] is not None: playlist_props['duration_ms'] = int(playlist_props['duration_ms'])
        except (ValueError, TypeError) as e:
             print(f"  Warning: Type conversion error for PID {playlist_props['pid']}: {e}. Skipping row.")
             continue
        if isinstance(playlist_props['collaborative'], str):
            playlist_props['collaborative'] = playlist_props['collaborative'].lower() == 'true'
        elif playlist_props['collaborative'] is None:
             playlist_props['collaborative'] = False
        playlists_data.append(playlist_props)
    if not playlists_data:
        print("  No valid Playlist data found to insert.")
        return
    batch_size = 5000
    total_playlists = len(playlists_data)
    print(f"  Preparing to insert/merge {total_playlists} Playlist nodes in batches of {batch_size}...")
    query = """
    UNWIND $playlists AS playlist_props
    MERGE (p:Playlist {pid: playlist_props.pid})
    SET p += playlist_props
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_playlists, batch_size):
            batch = playlists_data[i:min(i + batch_size, total_playlists)]
            print(f"    Inserting batch {i // batch_size + 1}/{(total_playlists + batch_size - 1) // batch_size} ({len(batch)} playlists)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'playlists': batch})
            except Exception as e:
                print(f"  Error inserting Playlist nodes batch starting at index {i}: {e}")
    print(f"  Playlist node merge/creation process completed.")

# --- Functions to Create Relationships ---

def create_msd_song_artist_rels(driver, songs_df):
    """ Creates :BY_ARTIST relationship between MsdSong and Artist nodes """
    print("\nStep: Creating (MsdSong)-[:BY_ARTIST]->(Artist) relationships...")
    db_name = "db-1"
    # Prepare data: list of dictionaries with song_id and artist_name
    rel_data = songs_df[['song_id', 'artist_name']].dropna().to_dict('records')
    if not rel_data:
        print("  No valid MsdSong-Artist data found to create relationships.")
        return
    batch_size = 5000
    total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :BY_ARTIST relationships for MsdSongs in batches of {batch_size}...")
    # Cypher query: MATCH nodes, MERGE relationship
    query = """
    UNWIND $rels AS rel_props
    MATCH (s:MsdSong {song_id: rel_props.song_id})
    MATCH (a:Artist {name: rel_props.artist_name})
    MERGE (s)-[:BY_ARTIST]->(a)
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e:
                print(f"  Error creating MsdSong-Artist relationships batch starting at index {i}: {e}")
    print(f"  :BY_ARTIST relationship creation for MsdSongs completed.")

def create_mpd_track_artist_rels(driver, tracks_df):
    """ Creates :BY_ARTIST relationship between MpdTrack and Artist nodes """
    print("\nStep: Creating (MpdTrack)-[:BY_ARTIST]->(Artist) relationships...")
    db_name = "db-1"
    # Prepare data: track_uri and artist_name
    rel_data = tracks_df[['track_uri', 'artist_name']].dropna().to_dict('records')
    if not rel_data:
        print("  No valid MpdTrack-Artist data found to create relationships.")
        return
    batch_size = 5000
    total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :BY_ARTIST relationships for MpdTracks in batches of {batch_size}...")
    # Cypher query: MATCH nodes, MERGE relationship
    query = """
    UNWIND $rels AS rel_props
    MATCH (t:MpdTrack {track_uri: rel_props.track_uri})
    MATCH (a:Artist {name: rel_props.artist_name})
    MERGE (t)-[:BY_ARTIST]->(a)
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e:
                print(f"  Error creating MpdTrack-Artist relationships batch starting at index {i}: {e}")
    print(f"  :BY_ARTIST relationship creation for MpdTracks completed.")

def create_track_playlist_rels(driver, playlist_tracks_df):
    """ Creates :APPEARS_IN relationship between MpdTrack and Playlist nodes """
    print("\nStep: Creating (MpdTrack)-[:APPEARS_IN]->(Playlist) relationships...")
    db_name = "db-1"
    # Prepare data: track_uri, pid, and pos (position in playlist)
    rel_data = playlist_tracks_df[['track_uri', 'pid', 'pos']].dropna().to_dict('records')
    if not rel_data:
        print("  No valid Track-Playlist map data found to create relationships.")
        return
    batch_size = 10000 # Can often use larger batches for relationships
    total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :APPEARS_IN relationships in batches of {batch_size}...")
    # Cypher query: MATCH nodes, MERGE relationship with properties
    query = """
    UNWIND $rels AS rel_props
    MATCH (t:MpdTrack {track_uri: rel_props.track_uri})
    MATCH (p:Playlist {pid: rel_props.pid})
    MERGE (t)-[r:APPEARS_IN]->(p)
    SET r.position = rel_props.pos // Add position as a relationship property
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e:
                print(f"  Error creating Track-Playlist relationships batch starting at index {i}: {e}")
    print(f"  :APPEARS_IN relationship creation completed.")

def create_same_as_rels(driver, matches_df):
    """ Creates :SAME_AS relationship between matched MsdSong and MpdTrack nodes """
    print("\nStep: Creating (MsdSong)-[:SAME_AS]->(MpdTrack) relationships...")
    db_name = "db-1"
    # Prepare data: song_id and track_uri from the matches file
    # Ensure column names match your direct_matches_df
    required_cols = ['song_id', 'track_uri']
    if not all(col in matches_df.columns for col in required_cols):
         print(f"  Error: Missing required columns in matches_df for SAME_AS relationship. Need: {required_cols}")
         return
    rel_data = matches_df[required_cols].dropna().to_dict('records')
    if not rel_data:
        print("  No valid match data found to create :SAME_AS relationships.")
        return
    batch_size = 5000 # Matches count is small, one batch likely fine
    total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :SAME_AS relationships in batches of {batch_size}...")
    # Cypher query: MATCH nodes, MERGE relationship
    query = """
    UNWIND $rels AS rel_props
    MATCH (s:MsdSong {song_id: rel_props.song_id})
    MATCH (t:MpdTrack {track_uri: rel_props.track_uri})
    MERGE (s)-[:SAME_AS]->(t)
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try:
                session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e:
                print(f"  Error creating SAME_AS relationships batch starting at index {i}: {e}")
    print(f"  :SAME_AS relationship creation completed.")


# --- Main Execution Logic ---
if __name__ == "__main__": # Ensure code runs only when script is executed directly
    script_start_time = time.time()

    # 1. Create Constraints (Run only if needed, idempotent)
    create_constraints(driver)

    # 2. Load Data
    print("\nLoading processed data files into pandas DataFrames...")
    try:
        # Using raw strings (r'...') for paths to handle backslashes correctly
        if MSD_CLEANED_FILE.endswith('.parquet'): msd_df = pd.read_parquet(MSD_CLEANED_FILE)
        else: msd_df = pd.read_csv(MSD_CLEANED_FILE)
        print(f"  Loaded MSD Cleaned Data ({msd_df.shape[0]} rows)")

        if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
        else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
        print(f"  Loaded MPD Unique Tracks ({mpd_tracks_df.shape[0]} rows)")

        if MPD_PLAYLISTS_FILE.endswith('.parquet'): mpd_playlists_df = pd.read_parquet(MPD_PLAYLISTS_FILE)
        else: mpd_playlists_df = pd.read_csv(MPD_PLAYLISTS_FILE)
        print(f"  Loaded MPD Playlists ({mpd_playlists_df.shape[0]} rows)")

        if MPD_PLAYLIST_TRACKS_FILE.endswith('.parquet'): mpd_playlist_tracks_df = pd.read_parquet(MPD_PLAYLIST_TRACKS_FILE)
        else: mpd_playlist_tracks_df = pd.read_csv(MPD_PLAYLIST_TRACKS_FILE)
        print(f"  Loaded MPD Playlist-Track Map ({mpd_playlist_tracks_df.shape[0]} rows)")

        if MATCHES_FILE.endswith('.parquet'): direct_matches_df = pd.read_parquet(MATCHES_FILE)
        else: direct_matches_df = pd.read_csv(MATCHES_FILE)
        print(f"  Loaded Direct Matches ({direct_matches_df.shape[0]} rows)")
        print("All data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        if driver: driver.close() # Close driver before exiting
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        if driver: driver.close()
        sys.exit(1)

    # 3. Insert Nodes (Idempotent due to MERGE)
    # Extract unique artists first
    print("  Extracting unique artist names...")
    msd_artists = set(msd_df['artist_name'].dropna().unique())
    mpd_artists = set(mpd_tracks_df['artist_name'].dropna().unique())
    unique_artists = list(msd_artists.union(mpd_artists))
    unique_artists = [name for name in unique_artists if name] # Clean empty strings
    print(f"  Found {len(unique_artists)} unique artist names.")
    # Insert Nodes
    insert_artists(driver, unique_artists)
    insert_msd_songs(driver, msd_df)
    insert_mpd_tracks(driver, mpd_tracks_df)
    insert_playlists(driver, mpd_playlists_df)

    # 4. Create Relationships (Idempotent due to MERGE)
    create_msd_song_artist_rels(driver, msd_df)
    create_mpd_track_artist_rels(driver, mpd_tracks_df)
    create_track_playlist_rels(driver, mpd_playlist_tracks_df)
    create_same_as_rels(driver, direct_matches_df)

    # --- Final Steps ---
    script_end_time = time.time()
    print(f"\n--- Graph Build Process Complete (Nodes & Relationships) ---")
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")

    # Close the driver connection
    if driver:
        driver.close()
        print("\nNeo4j driver closed.")

