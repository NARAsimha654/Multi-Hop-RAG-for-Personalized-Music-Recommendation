import pandas as pd
import lyricsgenius
import time
import os
import sys
from tqdm.auto import tqdm # Import tqdm for progress bar
import pickle
import requests # For potential future use or debugging
import random # For slight random delay variation

# --- Configuration ---
# IMPORTANT: Put your Genius API Client Access Token here
GENIUS_API_TOKEN = "TuoRTZ6t8sxtQi5yt-MvWUOHlZe6DGbK35bdT_CP8ui78UQDeKB1mGN3m0E9k6nA" # <<<--- VERIFY YOUR TOKEN

# --- Define Sample Size ---
SAMPLE_SIZE = 50000 # Process the first 50,000 unique tracks

# Input file (Use the path appropriate for your local machine)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# Output file (Consider naming it to reflect the sample size)
LYRICS_OUTPUT_FILE = f'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_lyrics_{SAMPLE_SIZE}.parquet'

# File to save progress (Specific to this sample size run)
PROGRESS_FILE = f'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\lyrics_fetch_progress_{SAMPLE_SIZE}.pkl'

# --- Initialize API ---
if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN" or not GENIUS_API_TOKEN:
    print("Error: GENIUS_API_TOKEN not set. Please get a token from genius.com/api-clients.")
    sys.exit(1)

print("Initializing LyricsGenius API...")
# Adjust timeout, retries. verbose=False is fine for long runs unless debugging.
genius = lyricsgenius.Genius(
    GENIUS_API_TOKEN,
    timeout=30, # Increased timeout slightly
    retries=3,
    verbose=False,
    remove_section_headers=True,
    skip_non_songs=True
    # We are not adding User-Agent here assuming local run is less likely to be blocked
)
print("API Initialized.")

# --- Load Track Data ---
print(f"Loading track data from {MPD_TRACKS_FILE}...")
try:
    if MPD_TRACKS_FILE.endswith('.parquet'):
        mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else:
        print(f"Error: Expecting Parquet file, found {MPD_TRACKS_FILE}")
        sys.exit(1)

    required_cols = ['track_uri', 'track_name', 'artist_name']
    if not all(col in mpd_tracks_df.columns for col in required_cols):
        print(f"Error: Input file {MPD_TRACKS_FILE} is missing required columns.")
        sys.exit(1)

    # Select necessary columns and drop duplicates by URI
    tracks_to_process_all = mpd_tracks_df[required_cols].drop_duplicates(subset=['track_uri'])
    print(f"Loaded {len(tracks_to_process_all)} total unique tracks.")

    # --- Apply Sample Size Limit ---
    if len(tracks_to_process_all) > SAMPLE_SIZE:
        print(f"Processing the first {SAMPLE_SIZE} unique tracks.")
        tracks_to_process_all = tracks_to_process_all.head(SAMPLE_SIZE).reset_index(drop=True)
    else:
        print(f"Processing all {len(tracks_to_process_all)} unique tracks (less than sample size).")
        SAMPLE_SIZE = len(tracks_to_process_all) # Adjust sample size if dataset is smaller

except FileNotFoundError:
    print(f"Error: Input file not found at {MPD_TRACKS_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading track data: {e}")
    sys.exit(1)

# --- Resume Logic ---
processed_uris = set()
lyrics_results = []
start_index = 0 # Keep track of how many were already processed

if os.path.exists(PROGRESS_FILE):
    print(f"Found existing progress file: {PROGRESS_FILE}. Resuming...")
    try:
        with open(PROGRESS_FILE, 'rb') as f:
            progress_data = pickle.load(f)
            processed_uris = progress_data.get('processed_uris', set())
            lyrics_results = progress_data.get('lyrics_results', [])
            start_index = len(processed_uris) # Count how many are done
            print(f"  Resuming. Already processed {start_index} tracks.")
            # Filter the *sampled* track list to only include unprocessed ones
            tracks_to_process = tracks_to_process_all[~tracks_to_process_all['track_uri'].isin(processed_uris)].reset_index(drop=True)
            print(f"  {len(tracks_to_process)} tracks remaining in this run (target total: {SAMPLE_SIZE}).")
    except Exception as e:
        print(f"  Error reading progress file: {e}. Starting from scratch for {SAMPLE_SIZE} tracks.")
        processed_uris = set() # Reset progress tracking
        lyrics_results = []
        start_index = 0
        tracks_to_process = tracks_to_process_all.copy() # Use the sampled tracks
else:
    print(f"No progress file found. Starting from scratch for {SAMPLE_SIZE} tracks.")
    tracks_to_process = tracks_to_process_all.copy()


# --- Fetch Lyrics with Progress Bar ---
# Determine the total number of items to process in *this run*
total_to_process_this_run = len(tracks_to_process)
print(f"Fetching lyrics for {total_to_process_this_run} tracks...")

# Initialize counters based on loaded results if resuming
found_count = sum(1 for r in lyrics_results if r['status'] == 'FOUND')
not_found_count = sum(1 for r in lyrics_results if r['status'] in ['NOT_FOUND', 'NOT_FOUND_SHORT'])
error_count = sum(1 for r in lyrics_results if r['status'] == 'ERROR' or r['status'] == 'ERROR_TIMEOUT')
missing_info_count = sum(1 for r in lyrics_results if r['status'] == 'MISSING_INFO')

run_start_time = time.time()
last_status = "Starting..." # Variable to hold status for postfix

# Use tqdm to wrap the iteration
with tqdm(total=total_to_process_this_run, desc="Fetching Lyrics", unit='track', initial=0) as pbar:
    # Use enumerate for periodic saving index
    for i, (index, row) in enumerate(tracks_to_process.iterrows()):
        track_uri = row['track_uri']
        title = row['track_name']
        artist = row['artist_name']
        lyrics = None
        status = "NOT_FOUND" # Default status for this iteration

        # Update description for current track
        pbar.set_description(f"Fetching: {str(artist)[:15]} - {str(title)[:20]}")
        # Update postfix with status from *previous* iteration
        pbar.set_postfix_str(f"Last: {last_status}", refresh=True)


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
                    # Check if lyrics are substantial
                    if len(cleaned_lyrics.split()) > 5:
                         lyrics = cleaned_lyrics
                         status = "FOUND"
                         found_count += 1
                    else:
                         status = "NOT_FOUND_SHORT"
                         not_found_count += 1
                else:
                    status = "NOT_FOUND"
                    not_found_count += 1

                # --- IMPORTANT: Respect API limits ---
                # Add a small random jitter to the delay
                time.sleep(1.1 + (random.random() * 0.4)) # e.g., 1.1 - 1.5 seconds

            except requests.exceptions.Timeout as timeout_err:
                # Print error using tqdm.write to avoid messing up the progress bar
                pbar.write(f"\n  TIMEOUT Error fetching lyrics for {artist} - {title}: {timeout_err}")
                status = "ERROR_TIMEOUT"
                error_count += 1
                time.sleep(10) # Longer delay after timeout
            except Exception as e:
                pbar.write(f"\n  GENERAL Error fetching lyrics for {artist} - {title}: {type(e).__name__} - {e}")
                status = "ERROR"
                error_count += 1
                time.sleep(5) # Longer delay after general error

        # Store status for the next iteration's postfix display
        last_status = status

        # Append result and mark URI as processed
        lyrics_results.append({
            'track_uri': track_uri,
            'lyrics': lyrics,
            'status': status
        })
        processed_uris.add(track_uri) # Add to set of processed URIs

        # Update progress bar for this run
        pbar.update(1)

        # --- Optional: Save progress periodically ---
        # Use 'i' from enumerate for periodic saving based on this run's progress
        # Save more frequently for long runs, e.g., every 200 tracks
        save_interval = 200
        if i > 0 and i % save_interval == 0:
             # Display overall progress percentage
             overall_processed = len(processed_uris)
             overall_progress_percent = (overall_processed / SAMPLE_SIZE) * 100
             pbar.write(f"\nSaving intermediate progress ({overall_processed}/{SAMPLE_SIZE} total processed [{overall_progress_percent:.1f}%])...")
             progress_data_to_save = {
                 'processed_uris': processed_uris,
                 'lyrics_results': lyrics_results
             }
             try:
                 # Ensure directory exists before saving
                 os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
                 with open(PROGRESS_FILE, 'wb') as f_prog:
                     pickle.dump(progress_data_to_save, f_prog)
                 pbar.write("Intermediate progress saved.\n")
             except Exception as e:
                 pbar.write(f"Error saving intermediate progress: {e}\n")


# --- Final Summary and Save ---
run_end_time = time.time()
print(f"\nFinished fetching lyrics run.")
print(f"Attempted {total_to_process_this_run} tracks in this run in {run_end_time - run_start_time:.2f} seconds.")
print("\n--- Overall Stats ---")
print(f"Target Sample Size: {SAMPLE_SIZE}")
print(f"Total Processed URIs: {len(processed_uris)}")
print(f"Lyrics Found: {found_count}")
print(f"Lyrics Not Found (or too short): {not_found_count}")
print(f"Missing Track/Artist Info: {missing_info_count}")
print(f"Errors Encountered (Timeout): {sum(1 for r in lyrics_results if r['status'] == 'ERROR_TIMEOUT')}")
print(f"Errors Encountered (Other): {sum(1 for r in lyrics_results if r['status'] == 'ERROR')}")


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

