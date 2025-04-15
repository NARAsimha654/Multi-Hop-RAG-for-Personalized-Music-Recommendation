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
# Using paths provided by the user
# Using raw strings (r'...') or forward slashes for better path handling
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
    # For write operations, summary() can provide stats but consume() is simpler
    return result.consume()

# --- Function to Create Constraints ---
def create_constraints(driver):
    """ Creates unique constraints on node properties """
    print("Creating constraints for data integrity and performance...")
    # Define database name explicitly if needed (usually default works)
    db_name = "db-1"
    queries = [
        "CREATE CONSTRAINT unique_artist_name IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE;",
        "CREATE CONSTRAINT unique_msd_song_id IF NOT EXISTS FOR (s:MsdSong) REQUIRE s.song_id IS UNIQUE;",
        "CREATE CONSTRAINT unique_mpd_track_uri IF NOT EXISTS FOR (t:MpdTrack) REQUIRE t.track_uri IS UNIQUE;",
        "CREATE CONSTRAINT unique_playlist_pid IF NOT EXISTS FOR (p:Playlist) REQUIRE p.pid IS UNIQUE;"
    ]
    # Use try-with-resources for session management
    with driver.session(database=db_name) as session:
        for query in queries:
            try:
                print(f"  Executing: {query}")
                # execute_write handles transaction logic (begin, commit/rollback)
                session.execute_write(run_cypher_query, query)
            except Exception as e:
                # Catch potential errors like constraint already exists
                print(f"  Warning/Error creating constraint: {e}")
    print("Constraints checked/created.")

# --- Function to Insert Artists ---
def insert_artists(driver, artist_list):
    """ Inserts artist nodes into Neo4j using MERGE """
    print("\nStep: Inserting Artist nodes...")
    db_name = "db-1"
    # Cypher query using UNWIND for batching and MERGE for idempotency
    query = """
    UNWIND $artists AS artist_props
    MERGE (a:Artist {name: artist_props.name})
    """
    # Prepare data: list of dictionaries, ensure name is not null/empty
    artists_data = [{'name': name} for name in artist_list if pd.notna(name) and name]
    if not artists_data:
        print("  No valid artist names found to insert.")
        return

    batch_size = 5000 # Process in batches
    total_artists = len(artists_data)
    print(f"  Preparing to insert/merge {total_artists} unique artist nodes in batches of {batch_size}...")

    with driver.session(database=db_name) as session:
        for i in range(0, total_artists, batch_size):
            batch = artists_data[i:min(i + batch_size, total_artists)]
            print(f"    Inserting batch {i // batch_size + 1}/{(total_artists + batch_size - 1) // batch_size} ({len(batch)} artists)...")
            try:
                # Pass batch data as parameter
                session.execute_write(run_cypher_query, query, parameters={'artists': batch})
            except Exception as e:
                print(f"  Error inserting artist nodes batch starting at index {i}: {e}")
                # Consider adding more robust error handling if needed (e.g., logging failed batches)
    print(f"  Artist node merge/creation process completed.")

# --- Function to Insert MsdSongs ---
def insert_msd_songs(driver, songs_df):
    """ Inserts MsdSong nodes into Neo4j using MERGE """
    print("\nStep: Inserting MsdSong nodes...")
    db_name = "db-1"
    # Define columns to include as node properties
    required_cols = ['song_id', 'title', 'year', 'duration', 'key', 'loudness', 'mode', 'tempo', 'time_signature']
    # Check if all required columns exist in the DataFrame
    if not all(col in songs_df.columns for col in required_cols):
        print("  Error: Missing required columns in msd_df for MsdSong insertion.")
        print(f"  Required: {required_cols}")
        print(f"  Available: {list(songs_df.columns)}")
        return

    # Prepare data: Convert DataFrame rows to list of dictionaries, handle NaNs
    songs_data = []
    for index, row in songs_df[required_cols].iterrows():
        # Ensure song_id exists, as it's the key for MERGE
        if pd.isna(row['song_id']): continue
        # Convert row to dictionary, replacing pandas NaN with Python None
        song_props = row.astype(object).where(pd.notna(row), None).to_dict()
        # Optional: Ensure specific types if needed (driver usually handles numpy types)
        # if song_props['year'] is not None: song_props['year'] = int(song_props['year'])
        songs_data.append(song_props)

    if not songs_data:
        print("  No valid MSD song data found to insert.")
        return

    batch_size = 5000 # Adjust batch size based on memory/performance
    total_songs = len(songs_data)
    print(f"  Preparing to insert/merge {total_songs} MsdSong nodes in batches of {batch_size}...")

    # Cypher query: MERGE on song_id, SET properties
    query = """
    UNWIND $songs AS song_props
    MERGE (s:MsdSong {song_id: song_props.song_id})
    SET s += song_props // Add/update other properties from the map
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

# --- Function to Insert MpdTracks ---
def insert_mpd_tracks(driver, tracks_df):
    """ Inserts MpdTrack nodes into Neo4j using MERGE """
    print("\nStep: Inserting MpdTrack nodes...")
    db_name = "db-1"
    # Define columns for MpdTrack properties
    required_cols = ['track_uri', 'track_name', 'artist_name', 'album_name', 'duration_ms']
    if not all(col in tracks_df.columns for col in required_cols):
        print("  Error: Missing required columns in mpd_tracks_df for MpdTrack insertion.")
        return

    # Prepare data, ensuring track_uri is present and handling types
    tracks_data = []
    for index, row in tracks_df[required_cols].iterrows():
        # Skip if track_uri is missing (key for MERGE)
        if pd.isna(row['track_uri']): continue
        track_props = row.astype(object).where(pd.notna(row), None).to_dict()
        # Ensure duration_ms is integer or None
        if track_props['duration_ms'] is not None:
            try: track_props['duration_ms'] = int(track_props['duration_ms'])
            except (ValueError, TypeError): track_props['duration_ms'] = None # Handle conversion errors
        tracks_data.append(track_props)

    if not tracks_data:
        print("  No valid MPD track data found to insert.")
        return

    batch_size = 5000
    total_tracks = len(tracks_data)
    print(f"  Preparing to insert/merge {total_tracks} MpdTrack nodes in batches of {batch_size}...")

    # Cypher query: MERGE on track_uri, SET properties
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

# --- Function to Insert Playlists ---
def insert_playlists(driver, playlists_df):
    """ Inserts Playlist nodes into Neo4j using MERGE """
    print("\nStep: Inserting Playlist nodes...")
    db_name = "db-1"
    # Define columns for Playlist properties
    required_cols = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_artists', 'num_edits', 'duration_ms', 'description']
    if not all(col in playlists_df.columns for col in required_cols):
        print("  Error: Missing required columns in mpd_playlists_df for Playlist insertion.")
        return

    # Prepare data, ensuring pid is present and handling types
    playlists_data = []
    for index, row in playlists_df[required_cols].iterrows():
        # Skip if pid is missing (key for MERGE)
        if pd.isna(row['pid']): continue
        playlist_props = row.astype(object).where(pd.notna(row), None).to_dict()
        # Ensure numeric types are standard Python integers
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
             continue # Skip row if essential types fail conversion

        # Convert collaborative flag (often string 'true'/'false') to boolean
        if isinstance(playlist_props['collaborative'], str):
            playlist_props['collaborative'] = playlist_props['collaborative'].lower() == 'true'
        elif playlist_props['collaborative'] is None:
             playlist_props['collaborative'] = False # Default collaborative to False if missing

        playlists_data.append(playlist_props)

    if not playlists_data:
        print("  No valid Playlist data found to insert.")
        return

    batch_size = 5000 # 5000 playlists is small, one batch is fine
    total_playlists = len(playlists_data)
    print(f"  Preparing to insert/merge {total_playlists} Playlist nodes in batches of {batch_size}...")

    # Cypher query: MERGE on pid, SET properties
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


# --- Main Execution Logic ---
if __name__ == "__main__": # Ensure code runs only when script is executed directly
    start_time = time.time()

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

    # 3. Insert Nodes
    # Insert Artists
    print("  Extracting unique artist names...")
    msd_artists = set(msd_df['artist_name'].dropna().unique())
    mpd_artists = set(mpd_tracks_df['artist_name'].dropna().unique())
    unique_artists = list(msd_artists.union(mpd_artists))
    unique_artists = [name for name in unique_artists if name] # Clean empty strings
    print(f"  Found {len(unique_artists)} unique artist names.")
    insert_artists(driver, unique_artists)

    # Insert MsdSongs
    insert_msd_songs(driver, msd_df)

    # Insert MpdTracks
    insert_mpd_tracks(driver, mpd_tracks_df)

    # Insert Playlists
    insert_playlists(driver, mpd_playlists_df)

    # --- Final Steps ---
    end_time = time.time()
    print(f"\n--- Node Insertion Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # Close the driver connection
    if driver:
        driver.close()
        print("\nNeo4j driver closed.")

