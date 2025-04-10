import os
import glob
import json
import pandas as pd
import time

# --- Configuration ---
# IMPORTANT: Update this path to the directory containing your MPD JSON files
MPD_DATA_PATH = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\raw\\Spotify Million Playlist' # <<<--- UPDATE THIS PATH

# Output file paths
PLAYLISTS_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_playlists.parquet'
TRACKS_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_unique_tracks.parquet'
PLAYLIST_TRACKS_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_playlist_tracks.parquet'

# --- Functions ---

def find_json_files(data_dir):
    """Finds all JSON files (.json) in a directory."""
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    return json_files

def process_mpd_slice(json_file_path):
    """Processes a single MPD JSON slice file."""
    playlists_data = []
    playlist_track_map_data = []
    tracks_data_set = set() # Use a set to store unique track tuples initially

    print(f"  Processing file: {os.path.basename(json_file_path)}...")
    try:
        with open(json_file_path, 'r') as f:
            slice_data = json.load(f)

        for playlist in slice_data['playlists']:
            # 1. Extract Playlist Info
            playlist_info = {
                'pid': playlist.get('pid'),
                'name': playlist.get('name', ''), # Handle potentially missing names
                'collaborative': playlist.get('collaborative'),
                'modified_at': playlist.get('modified_at'),
                'num_tracks': playlist.get('num_tracks'),
                'num_albums': playlist.get('num_albums'),
                'num_artists': playlist.get('num_artists'),
                'num_edits': playlist.get('num_edits'),
                'duration_ms': playlist.get('duration_ms'),
                'description': playlist.get('description', ''), # Handle potentially missing descriptions
            }
            playlists_data.append(playlist_info)

            # 2. Extract Track Info and Playlist-Track Mapping
            for track in playlist['tracks']:
                # Add to playlist-track mapping
                playlist_track_map_data.append({
                    'pid': playlist.get('pid'),
                    'track_uri': track.get('track_uri'),
                    'pos': track.get('pos') # Position of track in playlist
                })

                # Add unique track info to set (using a tuple as set elements must be hashable)
                track_tuple = (
                    track.get('track_uri'),
                    track.get('track_name'),
                    track.get('artist_uri'),
                    track.get('artist_name'),
                    track.get('album_uri'),
                    track.get('album_name'),
                    track.get('duration_ms')
                )
                tracks_data_set.add(track_tuple)

    except Exception as e:
        print(f"Error processing file {json_file_path}: {e}")
        return None, None, None # Return None on error

    return playlists_data, playlist_track_map_data, tracks_data_set

# --- Main Processing Logic ---

start_time = time.time()

# 1. Find all JSON files
all_json_files = find_json_files(MPD_DATA_PATH)

if not all_json_files:
    print("No JSON files found. Please check the MPD_DATA_PATH.")
else:
    # 2. Process all files
    all_playlists = []
    all_playlist_track_maps = []
    all_unique_tracks_set = set() # Use a set to collect unique tracks across all files

    for json_path in all_json_files:
        playlists, playlist_tracks, tracks_set = process_mpd_slice(json_path)
        if playlists is not None: # Check if processing was successful
            all_playlists.extend(playlists)
            all_playlist_track_maps.extend(playlist_tracks)
            all_unique_tracks_set.update(tracks_set) # Union sets

    print(f"\nFinished processing all files.")
    print(f"Total playlists found: {len(all_playlists)}")
    print(f"Total track entries in playlists: {len(all_playlist_track_maps)}")
    print(f"Total unique tracks found: {len(all_unique_tracks_set)}")

    # 3. Create DataFrames
    if all_playlists and all_playlist_track_maps and all_unique_tracks_set:
        print("\nCreating DataFrames...")
        # Playlists DataFrame
        playlists_df = pd.DataFrame(all_playlists)
        print("\nPlaylists DataFrame Info:")
        playlists_df.info()
        print(playlists_df.head())

        # Playlist-Track Map DataFrame
        playlist_tracks_df = pd.DataFrame(all_playlist_track_maps)
        print("\nPlaylist-Track Map DataFrame Info:")
        playlist_tracks_df.info()
        print(playlist_tracks_df.head())

        # Unique Tracks DataFrame (convert set of tuples to list of dicts first)
        track_columns = ['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms']
        unique_tracks_list = [dict(zip(track_columns, t)) for t in all_unique_tracks_set]
        tracks_df = pd.DataFrame(unique_tracks_list)
        print("\nUnique Tracks DataFrame Info:")
        tracks_df.info()
        print(tracks_df.head())

        # 4. Save DataFrames
        print(f"\nSaving DataFrames to Parquet files...")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(PLAYLISTS_OUTPUT_FILE), exist_ok=True)
        try:
            playlists_df.to_parquet(PLAYLISTS_OUTPUT_FILE, index=False, engine='pyarrow')
            tracks_df.to_parquet(TRACKS_OUTPUT_FILE, index=False, engine='pyarrow')
            playlist_tracks_df.to_parquet(PLAYLIST_TRACKS_OUTPUT_FILE, index=False, engine='pyarrow')
            print(f"Saved playlists data to {PLAYLISTS_OUTPUT_FILE}")
            print(f"Saved unique tracks data to {TRACKS_OUTPUT_FILE}")
            print(f"Saved playlist-track mapping to {PLAYLIST_TRACKS_OUTPUT_FILE}")
        except ImportError:
            print("Note: 'pyarrow' not found. Consider installing it (`pip install pyarrow`) for efficient Parquet saving.")
            # Optionally add fallback to CSV here if needed
            # playlists_df.to_csv(PLAYLISTS_OUTPUT_FILE.replace('.parquet','.csv'), index=False)
            # ... etc.

    else:
        print("No data processed, cannot create or save DataFrames.")

end_time = time.time()
print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")