import pandas as pd
import lyricsgenius
import time
import os
import sys
from tqdm.auto import tqdm # Import tqdm for progress bar

# --- Configuration ---
# IMPORTANT: Put your Genius API Client Access Token here
GENIUS_API_TOKEN = "TuoRTZ6t8sxtQi5yt-MvWUOHlZe6DGbK35bdT_CP8ui78UQDeKB1mGN3m0E9k6nA" # <<<--- VERIFY YOUR TOKEN

# Input file
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# Output file
LYRICS_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_lyrics_full.parquet' # Changed name slightly for clarity

# File to save progress (optional, helps resume if script stops)
PROGRESS_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\lyrics_fetch_progress.pkl'

# --- Initialize API ---
if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN" or not GENIUS_API_TOKEN:
    print("Error: GENIUS_API_TOKEN not set. Please get a token from genius.com/api-clients.")
    sys.exit(1)

print("Initializing LyricsGenius API...")
# Adjust timeout, retries, and consider verbose=True for debugging API calls
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, timeout=20, retries=3, verbose=False, remove_section_headers=True, skip_non_songs=True)
print("API Initialized.")

# --- Load Track Data ---
print(f"Loading track data from {MPD_TRACKS_FILE}...")
try:
    if MPD_TRACKS_FILE.endswith('.parquet'):
        mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else:
        mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    # Select necessary columns and drop rows where essential info is missing
    # Keep track_uri even if name/artist is missing, handle in loop
    tracks_to_process_all = mpd_tracks_df[['track_uri', 'track_name', 'artist_name']].drop_duplicates(subset=['track_uri']) # Ensure unique URIs
    print(f"Loaded {len(tracks_to_process_all)} unique tracks to potentially process.")
except Exception as e:
    print(f"Error loading track data: {e}")
    sys.exit(1)

# --- Resume Logic (Optional but Recommended) ---
processed_uris = set()
lyrics_results = []
start_index = 0

if os.path.exists(PROGRESS_FILE):
    print(f"Found existing progress file: {PROGRESS_FILE}. Resuming...")
    try:
        with open(PROGRESS_FILE, 'rb') as f:
            progress_data = pickle.load(f)
            processed_uris = progress_data.get('processed_uris', set())
            lyrics_results = progress_data.get('lyrics_results', [])
            start_index = len(processed_uris) # Estimate where to start roughly
            print(f"  Resuming. Already processed approximately {start_index} tracks.")
            # Filter tracks_to_process to only include unprocessed ones
            tracks_to_process = tracks_to_process_all[~tracks_to_process_all['track_uri'].isin(processed_uris)].reset_index(drop=True)
            print(f"  {len(tracks_to_process)} tracks remaining.")
    except Exception as e:
        print(f"  Error reading progress file: {e}. Starting from scratch.")
        tracks_to_process = tracks_to_process_all.copy() # Use all tracks if resume fails
else:
    print("No progress file found. Starting from scratch.")
    tracks_to_process = tracks_to_process_all.copy()


# --- Fetch Lyrics with Progress Bar ---
# Determine the total number of items to process
total_to_process = len(tracks_to_process)
print(f"Fetching lyrics for {total_to_process} tracks...")

# Counters for summary
found_count = sum(1 for r in lyrics_results if r['status'] == 'FOUND') # Count previously found
not_found_count = sum(1 for r in lyrics_results if r['status'] == 'NOT_FOUND')
error_count = sum(1 for r in lyrics_results if r['status'] == 'ERROR')
missing_info_count = sum(1 for r in lyrics_results if r['status'] == 'MISSING_INFO')

start_time = time.time()

# Use tqdm to wrap the iteration
# initial=start_index helps if resuming, total=total_to_process shows correct endpoint
# desc sets a description for the progress bar
# unit='track' sets the unit label
with tqdm(total=total_to_process, desc="Fetching Lyrics", unit='track', initial=0) as pbar:
    for index, row in tracks_to_process.iterrows():
        track_uri = row['track_uri']
        title = row['track_name']
        artist = row['artist_name']
        lyrics = None
        status = "NOT_FOUND" # Default status

        if not title or not artist or pd.isna(title) or pd.isna(artist):
            status = "MISSING_INFO"
            missing_info_count += 1
        else:
            try:
                # Search for the song
                song = genius.search_song(title, artist)
                if song:
                    # Basic cleaning
                    cleaned_lyrics = '\n'.join(song.lyrics.split('\n')[1:])
                    cleaned_lyrics = cleaned_lyrics.replace('EmbedShare URLCopyEmbedCopy', '').strip()
                    # Check if lyrics are substantial (e.g., more than a few words)
                    if len(cleaned_lyrics.split()) > 5:
                         lyrics = cleaned_lyrics
                         status = "FOUND"
                         found_count += 1
                         # Use tqdm.write for occasional status updates if needed, avoids breaking bar
                         # pbar.write(f"  Found lyrics for: {artist} - {title}")
                    else:
                         # print(f"  Found lyrics too short for: {artist} - {title}")
                         status = "NOT_FOUND_SHORT"
                         not_found_count += 1

                else:
                    # pbar.write(f"  Lyrics NOT FOUND for: {artist} - {title}")
                    status = "NOT_FOUND"
                    not_found_count += 1

                # --- IMPORTANT: Respect API limits ---
                time.sleep(1.1) # Slightly increased delay

            except Exception as e:
                pbar.write(f"  Error fetching lyrics for {artist} - {title}: {e}")
                status = "ERROR"
                error_count += 1
                time.sleep(5) # Longer delay after an error

        # Append result and mark URI as processed
        lyrics_results.append({
            'track_uri': track_uri,
            'lyrics': lyrics,
            'status': status
        })
        processed_uris.add(track_uri)

        # Update progress bar
        pbar.update(1)

        # --- Optional: Save progress periodically ---
        if index % 500 == 0 and index > 0: # Save every 500 tracks
             pbar.write(f"Saving intermediate progress ({len(processed_uris)} processed)...")
             progress_data_to_save = {
                 'processed_uris': processed_uris,
                 'lyrics_results': lyrics_results
             }
             try:
                 with open(PROGRESS_FILE, 'wb') as f_prog:
                     pickle.dump(progress_data_to_save, f_prog)
                 pbar.write("Intermediate progress saved.")
             except Exception as e:
                 pbar.write(f"Error saving intermediate progress: {e}")


# --- Final Summary and Save ---
end_time = time.time()
total_processed_in_run = len(tracks_to_process) # How many were attempted in this run
print(f"\nFinished fetching lyrics run.")
print(f"Attempted {total_processed_in_run} tracks in this run in {end_time - start_time:.2f} seconds.")
print("\n--- Overall Stats ---")
print(f"Total Processed URIs: {len(processed_uris)}")
print(f"Lyrics Found: {found_count}")
print(f"Lyrics Not Found (or too short): {not_found_count}")
print(f"Missing Track/Artist Info: {missing_info_count}")
print(f"Errors Encountered: {error_count}")


if lyrics_results:
    # Convert final results list to DataFrame
    lyrics_df = pd.DataFrame(lyrics_results)
    # Optional: Drop duplicates just in case resume logic had issues
    lyrics_df = lyrics_df.drop_duplicates(subset=['track_uri'], keep='last')

    print(f"\nSaving final {len(lyrics_df)} lyric results...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(LYRICS_OUTPUT_FILE), exist_ok=True)
        # Save final results
        lyrics_df.to_parquet(LYRICS_OUTPUT_FILE, index=False)
        print(f"Final lyrics data saved to {LYRICS_OUTPUT_FILE}")

        # Clean up progress file after successful final save
        if os.path.exists(PROGRESS_FILE):
            print(f"Cleaning up progress file: {PROGRESS_FILE}")
            os.remove(PROGRESS_FILE)

    except Exception as e:
        print(f"Error saving final lyrics data: {e}")
        print(f"Progress file {PROGRESS_FILE} may still contain intermediate results.")

