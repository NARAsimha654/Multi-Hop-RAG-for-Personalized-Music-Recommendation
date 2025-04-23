# Import necessary libraries
from neo4j import GraphDatabase
import pandas as pd
import os
import time # Optional: for timing operations
import sys # To exit if connection fails
import pickle # For loading fuzzy matches if needed
import random # For random delay
from tqdm.auto import tqdm # For progress bar

# --- Neo4j Connection Details ---
# Ensure these are correct for your Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Narasimha123" # <<<--- VERIFY YOUR PASSWORD

# --- File Paths ---
# Use raw strings (r'...') or forward slashes for paths
# Ensure these paths point to your actual processed data files
MSD_CLEANED_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_subset_cleaned.csv'
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'
MPD_PLAYLISTS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_playlists.parquet'
MPD_PLAYLIST_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_playlist_tracks.parquet'
DIRECT_MATCHES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_mpd_direct_matches.parquet'
FUZZY_MATCHES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_mpd_fuzzy_matches.parquet'
GTZAN_FEATURES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\gtzan_librosa_features.parquet'


# --- Neo4j Driver Setup ---
driver = None # Initialize driver variable
print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
try:
    # Establish connection with the Neo4j database
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Verify the connection is active
    driver.verify_connectivity()
    print("Neo4j connection successful!")
except Exception as e:
    # Print error and exit if connection fails
    print(f"Error connecting to Neo4j: {e}")
    sys.exit(1)

# --- Helper Function to Run Cypher Queries ---
def run_cypher_query(tx, query, parameters=None):
    """
    Helper function to execute a Cypher query within a Neo4j transaction.
    Consumes the result to ensure execution, suitable for write operations.
    """
    result = tx.run(query, parameters)
    # Consume results to ensure execution for write queries
    return result.consume()

# --- Function to Create Constraints ---
def create_constraints(driver):
    """
    Creates unique constraints on node properties for data integrity and performance.
    These constraints ensure that properties like names or IDs are unique for specific node labels.
    """
    print("Creating constraints for data integrity and performance...")
    db_name = "db-1" # Specify the target database name if not default
    # List of Cypher queries to create constraints if they don't already exist
    queries = [
        "CREATE CONSTRAINT unique_artist_name IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE;",
        "CREATE CONSTRAINT unique_msd_song_id IF NOT EXISTS FOR (s:MsdSong) REQUIRE s.song_id IS UNIQUE;",
        "CREATE CONSTRAINT unique_mpd_track_uri IF NOT EXISTS FOR (t:MpdTrack) REQUIRE t.track_uri IS UNIQUE;",
        "CREATE CONSTRAINT unique_playlist_pid IF NOT EXISTS FOR (p:Playlist) REQUIRE p.pid IS UNIQUE;",
        "CREATE CONSTRAINT unique_genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;"
    ]
    # Execute each constraint query within a session
    with driver.session(database=db_name) as session:
        for query in queries:
            try:
                print(f"  Executing: {query}")
                # Use execute_write for schema modification queries
                session.execute_write(run_cypher_query, query)
            except Exception as e:
                # Log warnings if constraint already exists or other errors occur
                print(f"  Warning/Error creating constraint: {e}")
    print("Constraints checked/created.")

# --- Functions to Insert Nodes ---
# (Keep node insertion functions as they were before)
def insert_artists(driver, artist_list):
    """ Inserts Artist nodes into Neo4j using MERGE to avoid duplicates. """
    print("\nStep: Inserting Artist nodes...")
    db_name = "db-1"
    query = """
    UNWIND $artists AS artist_props
    MERGE (a:Artist {name: artist_props.name})
    """
    artists_data = [{'name': name} for name in artist_list if pd.notna(name) and name]
    if not artists_data: print("  No valid artist names found to insert."); return
    batch_size = 5000; total_artists = len(artists_data)
    print(f"  Preparing to insert/merge {total_artists} unique artist nodes...")
    with driver.session(database=db_name) as session:
        for i in range(0, total_artists, batch_size):
            batch = artists_data[i:min(i + batch_size, total_artists)]
            # print(f"    Inserting batch {i // batch_size + 1}/{(total_artists + batch_size - 1) // batch_size} ({len(batch)} artists)...") # Reduced verbosity
            try: session.execute_write(run_cypher_query, query, parameters={'artists': batch})
            except Exception as e: print(f"  Error inserting artist nodes batch {i}: {e}")
    print(f"  Artist node merge/creation process completed.")

def insert_msd_songs(driver, songs_df):
    """ Inserts MsdSong nodes into Neo4j using MERGE based on song_id. """
    print("\nStep: Inserting MsdSong nodes...")
    db_name = "db-1"
    required_cols = ['song_id', 'title', 'year', 'duration', 'key', 'loudness', 'mode', 'tempo', 'time_signature']
    if not all(col in songs_df.columns for col in required_cols): print("  Error: Missing required columns in msd_df."); return
    songs_data = []
    for index, row in songs_df[required_cols].iterrows():
        if pd.isna(row['song_id']): continue
        songs_data.append(row.astype(object).where(pd.notna(row), None).to_dict())
    if not songs_data: print("  No valid MSD song data found."); return
    batch_size = 5000; total_songs = len(songs_data)
    print(f"  Preparing to insert/merge {total_songs} MsdSong nodes...")
    query = "UNWIND $songs AS sp MERGE (s:MsdSong {song_id: sp.song_id}) SET s += sp"
    with driver.session(database=db_name) as session:
        for i in range(0, total_songs, batch_size):
            batch = songs_data[i:min(i + batch_size, total_songs)]
            # print(f"    Inserting batch {i // batch_size + 1}/{(total_songs + batch_size - 1) // batch_size} ({len(batch)} songs)...")
            try: session.execute_write(run_cypher_query, query, parameters={'songs': batch})
            except Exception as e: print(f"  Error inserting MsdSong nodes batch {i}: {e}")
    print(f"  MsdSong node merge/creation process completed.")

def insert_mpd_tracks(driver, tracks_df):
    """ Inserts MpdTrack nodes into Neo4j using MERGE based on track_uri. """
    print("\nStep: Inserting MpdTrack nodes...")
    db_name = "db-1"
    required_cols = ['track_uri', 'track_name', 'artist_name', 'album_name', 'duration_ms']
    if not all(col in tracks_df.columns for col in required_cols): print("  Error: Missing required columns in mpd_tracks_df."); return
    tracks_data = []
    for index, row in tracks_df[required_cols].iterrows():
        if pd.isna(row['track_uri']): continue
        track_props = row.astype(object).where(pd.notna(row), None).to_dict()
        if track_props['duration_ms'] is not None:
            try: track_props['duration_ms'] = int(track_props['duration_ms'])
            except: track_props['duration_ms'] = None
        tracks_data.append(track_props)
    if not tracks_data: print("  No valid MPD track data found."); return
    batch_size = 5000; total_tracks = len(tracks_data)
    print(f"  Preparing to insert/merge {total_tracks} MpdTrack nodes...")
    query = "UNWIND $tracks AS tp MERGE (t:MpdTrack {track_uri: tp.track_uri}) SET t += tp"
    with driver.session(database=db_name) as session:
        for i in range(0, total_tracks, batch_size):
            batch = tracks_data[i:min(i + batch_size, total_tracks)]
            # print(f"    Inserting batch {i // batch_size + 1}/{(total_tracks + batch_size - 1) // batch_size} ({len(batch)} tracks)...")
            try: session.execute_write(run_cypher_query, query, parameters={'tracks': batch})
            except Exception as e: print(f"  Error inserting MpdTrack nodes batch {i}: {e}")
    print(f"  MpdTrack node merge/creation process completed.")

def insert_playlists(driver, playlists_df):
    """ Inserts Playlist nodes into Neo4j using MERGE based on pid. """
    print("\nStep: Inserting Playlist nodes...")
    db_name = "db-1"
    required_cols = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_artists', 'num_edits', 'duration_ms', 'description']
    if not all(col in playlists_df.columns for col in required_cols): print("  Error: Missing required columns in mpd_playlists_df."); return
    playlists_data = []
    for index, row in playlists_df[required_cols].iterrows():
        if pd.isna(row['pid']): continue
        playlist_props = row.astype(object).where(pd.notna(row), None).to_dict()
        try:
            for col in ['pid', 'modified_at', 'num_tracks', 'num_albums', 'num_artists', 'num_edits', 'duration_ms']:
                 if playlist_props[col] is not None: playlist_props[col] = int(playlist_props[col])
        except: continue
        if isinstance(playlist_props['collaborative'], str): playlist_props['collaborative'] = playlist_props['collaborative'].lower() == 'true'
        elif playlist_props['collaborative'] is None: playlist_props['collaborative'] = False
        playlists_data.append(playlist_props)
    if not playlists_data: print("  No valid Playlist data found."); return
    batch_size = 5000; total_playlists = len(playlists_data)
    print(f"  Preparing to insert/merge {total_playlists} Playlist nodes...")
    query = "UNWIND $playlists AS pp MERGE (p:Playlist {pid: pp.pid}) SET p += pp"
    with driver.session(database=db_name) as session:
        for i in range(0, total_playlists, batch_size):
            batch = playlists_data[i:min(i + batch_size, total_playlists)]
            # print(f"    Inserting batch {i // batch_size + 1}/{(total_playlists + batch_size - 1) // batch_size} ({len(batch)} playlists)...")
            try: session.execute_write(run_cypher_query, query, parameters={'playlists': batch})
            except Exception as e: print(f"  Error inserting Playlist nodes batch {i}: {e}")
    print(f"  Playlist node merge/creation process completed.")

def insert_genres(driver, genre_list):
    """ Inserts Genre nodes into Neo4j using MERGE based on name. """
    print("\nStep: Inserting Genre nodes...")
    db_name = "db-1"
    query = "UNWIND $genres AS gp MERGE (g:Genre {name: gp.name})" # Corrected query
    genres_data = [{'name': name} for name in genre_list if pd.notna(name) and name]
    if not genres_data: print("  No valid genre names found."); return
    print(f"  Preparing to insert/merge {len(genres_data)} unique Genre nodes...")
    with driver.session(database=db_name) as session:
        try: session.execute_write(run_cypher_query, query, parameters={'genres': genres_data})
        except Exception as e: print(f"  Error inserting Genre nodes: {e}")
    print(f"  Genre node merge/creation process completed.")

# --- Functions to Create Relationships ---
# (Keep create_msd_song_artist_rels, create_mpd_track_artist_rels, create_track_playlist_rels, create_same_as_rels functions as they were)
def create_msd_song_artist_rels(driver, songs_df):
    """ Creates :BY_ARTIST relationship between MsdSong and Artist nodes. """
    print("\nStep: Creating (MsdSong)-[:BY_ARTIST]->(Artist) relationships...")
    db_name = "db-1"
    rel_data = songs_df[['song_id', 'artist_name']].dropna().to_dict('records')
    if not rel_data: print("  No valid MsdSong-Artist data found."); return
    batch_size = 5000; total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :BY_ARTIST relationships...")
    query = "UNWIND $rels AS rp MATCH (s:MsdSong {song_id: rp.song_id}) MATCH (a:Artist {name: rp.artist_name}) MERGE (s)-[:BY_ARTIST]->(a)"
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            # print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try: session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e: print(f"  Error creating MsdSong-Artist rels batch {i}: {e}")
    print(f"  :BY_ARTIST relationship creation for MsdSongs completed.")

def create_mpd_track_artist_rels(driver, tracks_df):
    """ Creates :BY_ARTIST relationship between MpdTrack and Artist nodes. """
    print("\nStep: Creating (MpdTrack)-[:BY_ARTIST]->(Artist) relationships...")
    db_name = "db-1"
    rel_data = tracks_df[['track_uri', 'artist_name']].dropna().to_dict('records')
    if not rel_data: print("  No valid MpdTrack-Artist data found."); return
    batch_size = 5000; total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :BY_ARTIST relationships...")
    query = "UNWIND $rels AS rp MATCH (t:MpdTrack {track_uri: rp.track_uri}) MATCH (a:Artist {name: rp.artist_name}) MERGE (t)-[:BY_ARTIST]->(a)"
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            # print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try: session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e: print(f"  Error creating MpdTrack-Artist rels batch {i}: {e}")
    print(f"  :BY_ARTIST relationship creation for MpdTracks completed.")

def create_track_playlist_rels(driver, playlist_tracks_df):
    """ Creates :APPEARS_IN relationship between MpdTrack and Playlist nodes, adding position property. """
    print("\nStep: Creating (MpdTrack)-[:APPEARS_IN]->(Playlist) relationships...")
    db_name = "db-1"
    rel_data = playlist_tracks_df[['track_uri', 'pid', 'pos']].dropna().to_dict('records')
    if not rel_data: print("  No valid Track-Playlist map data found."); return
    batch_size = 10000; total_rels = len(rel_data)
    print(f"  Preparing to create {total_rels} :APPEARS_IN relationships...")
    query = "UNWIND $rels AS rp MATCH (t:MpdTrack {track_uri: rp.track_uri}) MATCH (p:Playlist {pid: rp.pid}) MERGE (t)-[r:APPEARS_IN]->(p) SET r.position = rp.pos"
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = rel_data[i:min(i + batch_size, total_rels)]
            # print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try: session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e: print(f"  Error creating Track-Playlist rels batch {i}: {e}")
    print(f"  :APPEARS_IN relationship creation completed.")

def create_same_as_rels(driver, direct_matches_df, fuzzy_matches_df):
    """ Creates :SAME_AS relationship using BOTH direct and fuzzy matches, adding fuzzy_score property. """
    print("\nStep: Creating (MsdSong)-[:SAME_AS]->(MpdTrack) relationships...")
    db_name = "db-1"
    required_cols = ['song_id', 'track_uri']
    combined_matches = []
    direct_data = []
    fuzzy_data = []
    if direct_matches_df is not None and not direct_matches_df.empty:
        if all(col in direct_matches_df.columns for col in required_cols):
            direct_data = direct_matches_df[required_cols].dropna().to_dict('records')
            combined_matches.extend(direct_data)
            print(f"  Added {len(direct_data)} direct matches.")
        else: print("  Warning: Direct matches DataFrame missing required columns.")
    if fuzzy_matches_df is not None and not fuzzy_matches_df.empty:
        if all(col in fuzzy_matches_df.columns for col in required_cols + ['fuzzy_score']):
            fuzzy_data = fuzzy_matches_df[required_cols + ['fuzzy_score']].dropna(subset=required_cols).to_dict('records')
            combined_matches.extend(fuzzy_data)
            print(f"  Added {len(fuzzy_data)} fuzzy matches.")
        else: print("  Warning: Fuzzy matches DataFrame missing required columns (need 'fuzzy_score').")
    unique_match_tuples = set((d['song_id'], d['track_uri']) for d in combined_matches)
    final_match_data = []
    for match_tuple in unique_match_tuples:
        song_id, track_uri = match_tuple
        fuzzy_match_dict = next((f for f in fuzzy_data if f['song_id']==song_id and f['track_uri']==track_uri), None)
        if fuzzy_match_dict: final_match_data.append(fuzzy_match_dict)
        else:
            direct_match_dict = next((d for d in direct_data if d['song_id']==song_id and d['track_uri']==track_uri), None)
            if direct_match_dict: final_match_data.append(direct_match_dict) # Add fuzzy_score: None implicitly
    if not final_match_data: print("  No valid match data found."); return
    batch_size = 5000; total_rels = len(final_match_data)
    print(f"  Preparing to create {total_rels} unique :SAME_AS relationships...")
    query = """
    UNWIND $rels AS rp MATCH (s:MsdSong {song_id: rp.song_id}) MATCH (t:MpdTrack {track_uri: rp.track_uri})
    MERGE (s)-[r:SAME_AS]->(t)
    FOREACH (score IN CASE WHEN rp.fuzzy_score IS NOT NULL THEN [rp.fuzzy_score] ELSE [] END | SET r.fuzzy_score = score)
    """
    with driver.session(database=db_name) as session:
        for i in range(0, total_rels, batch_size):
            batch = final_match_data[i:min(i + batch_size, total_rels)]
            # print(f"    Creating relationship batch {i // batch_size + 1}/{(total_rels + batch_size - 1) // batch_size} ({len(batch)} relationships)...")
            try: session.execute_write(run_cypher_query, query, parameters={'rels': batch})
            except Exception as e: print(f"  Error creating SAME_AS rels batch {i}: {e}")
    print(f"  :SAME_AS relationship creation completed.")

# --- UPDATED Function to Create Co-occurrence Relationships (Iterative) ---
def create_cooccurrence_rels_iterative(driver, playlists_df):
    """
    Creates :CO_OCCURS_WITH relationship between MpdTracks iteratively,
    processing one playlist at a time to avoid memory errors.
    Increments a 'count' property on the relationship.
    """
    print(f"\nStep: Creating/Updating (MpdTrack)-[:CO_OCCURS_WITH]->(MpdTrack) relationships iteratively...")
    db_name = "db-1"

    # Get the list of playlist PIDs to iterate over
    if playlists_df is None or playlists_df.empty or 'pid' not in playlists_df.columns:
        print("  Error: Playlists DataFrame is missing or invalid. Cannot create co-occurrence relationships.")
        return
    # Ensure PIDs are integers for matching in Cypher
    try:
        playlist_pids = playlists_df['pid'].dropna().astype(int).unique().tolist()
    except ValueError:
        print("Error: Could not convert playlist PIDs to integers.")
        return

    if not playlist_pids:
        print("  No playlist PIDs found to process.")
        return

    print(f"  Processing {len(playlist_pids)} playlists one by one (this might take a while)...")
    start_tm = time.time()
    processed_count = 0
    error_count = 0

    # Cypher query to process pairs within a single playlist
    query = """
    MATCH (p:Playlist {pid: $pid})<-[:APPEARS_IN]-(t1:MpdTrack)
    WITH p, collect(t1) as tracks_in_playlist // Collect all tracks first
    UNWIND tracks_in_playlist as t1          // Unwind to process pairs
    UNWIND tracks_in_playlist as t2
    WITH t1, t2 WHERE id(t1) < id(t2)         // Ensure pairs are unique and no self-loops
    // Merge the relationship (creates if not exists)
    MERGE (t1)-[r:CO_OCCURS_WITH]-(t2)
    // Set count on creation, increment on match (using coalesce for safety)
    ON CREATE SET r.count = 1
    ON MATCH SET r.count = coalesce(r.count, 0) + 1
    """

    # Iterate through each playlist PID with a progress bar
    with driver.session(database=db_name) as session:
        for pid in tqdm(playlist_pids, desc="Processing Playlists for Co-occurrence", unit="playlist"):
            try:
                # Execute the query for the current playlist PID
                session.execute_write(run_cypher_query, query, parameters={'pid': pid})
                processed_count += 1
            except Exception as e:
                error_count += 1
                # Use tqdm.write to avoid messing up the progress bar
                tqdm.write(f"\n  Error processing playlist PID {pid} for co-occurrence: {e}")
                # Decide whether to continue or stop on error
                # continue

    end_tm = time.time()
    print(f"\n  Finished processing {processed_count} playlists for co-occurrence in {end_tm - start_tm:.2f} seconds.")
    if error_count > 0:
        print(f"  Encountered errors processing {error_count} playlists.")
    print(f"  :CO_OCCURS_WITH relationship creation/update process completed.")


# --- Main Execution Logic ---
if __name__ == "__main__":
    script_start_time = time.time()

    # 1. Create Constraints
    create_constraints(driver)

    # 2. Load Data
    print("\nLoading processed data files into pandas DataFrames...")
    # Initialize DataFrames
    msd_df = pd.DataFrame(); mpd_tracks_df = pd.DataFrame(); mpd_playlists_df = pd.DataFrame()
    mpd_playlist_tracks_df = pd.DataFrame(); direct_matches_df = pd.DataFrame(columns=['song_id', 'track_uri'])
    fuzzy_matches_df = pd.DataFrame(columns=['song_id', 'track_uri', 'fuzzy_score']); gtzan_features_df = None
    all_loaded = True
    try:
        # Load main dataframes
        if os.path.exists(MSD_CLEANED_FILE):
            if MSD_CLEANED_FILE.endswith('.parquet'): msd_df = pd.read_parquet(MSD_CLEANED_FILE)
            else: msd_df = pd.read_csv(MSD_CLEANED_FILE)
            print(f"  Loaded MSD Cleaned Data ({msd_df.shape[0]} rows)")
        else: print(f"  Warning: File not found {MSD_CLEANED_FILE}"); all_loaded=False

        if os.path.exists(MPD_TRACKS_FILE):
            if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
            else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
            print(f"  Loaded MPD Unique Tracks ({mpd_tracks_df.shape[0]} rows)")
        else: print(f"  Warning: File not found {MPD_TRACKS_FILE}"); all_loaded=False

        if os.path.exists(MPD_PLAYLISTS_FILE):
            if MPD_PLAYLISTS_FILE.endswith('.parquet'): mpd_playlists_df = pd.read_parquet(MPD_PLAYLISTS_FILE)
            else: mpd_playlists_df = pd.read_csv(MPD_PLAYLISTS_FILE)
            print(f"  Loaded MPD Playlists ({mpd_playlists_df.shape[0]} rows)")
        else: print(f"  Warning: File not found {MPD_PLAYLISTS_FILE}"); all_loaded=False

        if os.path.exists(MPD_PLAYLIST_TRACKS_FILE):
            if MPD_PLAYLIST_TRACKS_FILE.endswith('.parquet'): mpd_playlist_tracks_df = pd.read_parquet(MPD_PLAYLIST_TRACKS_FILE)
            else: mpd_playlist_tracks_df = pd.read_csv(MPD_PLAYLIST_TRACKS_FILE)
            print(f"  Loaded MPD Playlist-Track Map ({mpd_playlist_tracks_df.shape[0]} rows)")
        else: print(f"  Warning: File not found {MPD_PLAYLIST_TRACKS_FILE}"); all_loaded=False

        # Load matches files
        if os.path.exists(DIRECT_MATCHES_FILE):
            if DIRECT_MATCHES_FILE.endswith('.parquet'): direct_matches_df = pd.read_parquet(DIRECT_MATCHES_FILE)
            else: direct_matches_df = pd.read_csv(DIRECT_MATCHES_FILE)
            print(f"  Loaded Direct Matches ({direct_matches_df.shape[0]} rows)")
        else: print(f"  Warning: Direct matches file not found.")

        if os.path.exists(FUZZY_MATCHES_FILE):
            if FUZZY_MATCHES_FILE.endswith('.parquet'): fuzzy_matches_df = pd.read_parquet(FUZZY_MATCHES_FILE)
            else: fuzzy_matches_df = pd.read_csv(FUZZY_MATCHES_FILE)
            print(f"  Loaded Fuzzy Matches ({fuzzy_matches_df.shape[0]} rows)")
        else: print(f"  Warning: Fuzzy matches file not found.")

        # Load GTZAN Features
        if os.path.exists(GTZAN_FEATURES_FILE):
            if GTZAN_FEATURES_FILE.endswith('.parquet'): gtzan_features_df = pd.read_parquet(GTZAN_FEATURES_FILE)
            else: gtzan_features_df = pd.read_csv(GTZAN_FEATURES_FILE)
            print(f"  Loaded GTZAN Features ({gtzan_features_df.shape[0]} rows)")
        else: print(f"  Warning: GTZAN features file not found.")

        if all_loaded: print("All essential data files loaded successfully.")
        else: print("Warning: Some data files were missing.")

    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        if driver: driver.close()
        sys.exit(1)

    # 3. Insert Nodes (Idempotent due to MERGE)
    print("\n--- Inserting/Merging Nodes ---")
    # (Node insertion logic remains the same)
    if not msd_df.empty and not mpd_tracks_df.empty:
        print("  Extracting unique artist names...")
        msd_artists = set(msd_df['artist_name'].dropna().unique()); mpd_artists = set(mpd_tracks_df['artist_name'].dropna().unique())
        unique_artists = list(msd_artists.union(mpd_artists)); unique_artists = [name for name in unique_artists if name]
        print(f"  Found {len(unique_artists)} unique artist names.")
        insert_artists(driver, unique_artists)
    else: print("Skipping Artist insertion.")
    if not msd_df.empty: insert_msd_songs(driver, msd_df)
    else: print("Skipping MsdSong insertion.")
    if not mpd_tracks_df.empty: insert_mpd_tracks(driver, mpd_tracks_df)
    else: print("Skipping MpdTrack insertion.")
    if not mpd_playlists_df.empty: insert_playlists(driver, mpd_playlists_df)
    else: print("Skipping Playlist insertion.")
    if gtzan_features_df is not None and 'genre' in gtzan_features_df.columns:
        unique_genres = list(gtzan_features_df['genre'].dropna().unique())
        if unique_genres: insert_genres(driver, unique_genres)
        else: print("\nNo unique genres found in GTZAN data.")
    else: print("\nSkipping Genre node insertion.")


    # 4. Create Relationships (Idempotent due to MERGE)
    print("\n--- Creating/Merging Relationships ---")
    # (Calls to create_msd_song_artist_rels, etc. remain the same)
    if not msd_df.empty: create_msd_song_artist_rels(driver, msd_df)
    else: print("Skipping MsdSong-Artist relationships.")
    if not mpd_tracks_df.empty: create_mpd_track_artist_rels(driver, mpd_tracks_df)
    else: print("Skipping MpdTrack-Artist relationships.")
    if not mpd_playlist_tracks_df.empty: create_track_playlist_rels(driver, mpd_playlist_tracks_df)
    else: print("Skipping Track-Playlist relationships.")
    if not direct_matches_df.empty or not fuzzy_matches_df.empty:
         create_same_as_rels(driver, direct_matches_df, fuzzy_matches_df)
    else: print("Skipping SAME_AS relationships.")

    # --- Call the NEW iterative co-occurrence function ---
    if not mpd_playlists_df.empty and not mpd_playlist_tracks_df.empty:
        create_cooccurrence_rels_iterative(driver, mpd_playlists_df) # Pass playlists_df
    else:
        print("Skipping CO_OCCURS_WITH relationships as Playlist or Track-Playlist map data is missing.")

    # --- Final Steps ---
    script_end_time = time.time()
    print(f"\n--- Graph Build Process Complete ---")
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")

    # Close the driver connection
    if driver:
        driver.close()
        print("\nNeo4j driver closed.")

