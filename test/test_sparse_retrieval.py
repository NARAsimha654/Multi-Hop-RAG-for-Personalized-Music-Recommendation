import pandas as pd
import lyricsgenius
import time
import os
import sys

# --- Configuration ---
# IMPORTANT: Put your Genius API Client Access Token here
GENIUS_API_TOKEN = "TuoRTZ6t8sxtQi5yt-MvWUOHlZe6DGbK35bdT_CP8ui78UQDeKB1mGN3m0E9k6nA" # <<<--- PUT YOUR TOKEN HERE

# Input file
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# Output file
LYRICS_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_lyrics.parquet'

# --- Initialize API ---
if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN" or not GENIUS_API_TOKEN:
    print("Error: GENIUS_API_TOKEN not set. Please get a token from genius.com/api-clients.")
    sys.exit(1)

print("Initializing LyricsGenius API...")
# You might want to adjust timeout and retries
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, timeout=15, retries=2, verbose=False, remove_section_headers=True)
print("API Initialized.")

# --- Load Track Data ---
print(f"Loading track data from {MPD_TRACKS_FILE}...")
try:
    if MPD_TRACKS_FILE.endswith('.parquet'):
        mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else:
        mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    # Select necessary columns and drop rows where essential info is missing
    tracks_to_process = mpd_tracks_df[['track_uri', 'track_name', 'artist_name']].dropna().drop_duplicates(subset=['track_name', 'artist_name'])
    print(f"Loaded {len(tracks_to_process)} unique track/artist pairs to process.")
except Exception as e:
    print(f"Error loading track data: {e}")
    sys.exit(1)

# --- Fetch Lyrics ---
lyrics_results = []
# !!! START WITH A SMALL SAMPLE !!!
sample_size = 20 # Increase gradually after testing
print(f"Fetching lyrics for a sample of {sample_size} tracks...")

processed_count = 0
start_time = time.time()

for index, row in tracks_to_process.head(sample_size).iterrows():
    track_uri = row['track_uri']
    title = row['track_name']
    artist = row['artist_name']
    lyrics = None
    status = "NOT_FOUND" # Default status

    if not title or not artist:
        status = "MISSING_INFO"
    else:
        try:
            # Search for the song
            song = genius.search_song(title, artist)
            if song:
                # Basic cleaning (remove first line often like 'TrackName Lyrics')
                cleaned_lyrics = '\n'.join(song.lyrics.split('\n')[1:])
                # Remove embedding text like 'EmbedShare URLCopyEmbedCopy' if present
                cleaned_lyrics = cleaned_lyrics.replace('EmbedShare URLCopyEmbedCopy', '').strip()
                lyrics = cleaned_lyrics
                status = "FOUND"
                print(f"  Found lyrics for: {artist} - {title}")
            else:
                print(f"  Lyrics NOT FOUND for: {artist} - {title}")
                status = "NOT_FOUND"

            # --- IMPORTANT: Respect API limits ---
            time.sleep(1) # Add a delay between requests (e.g., 1 second)

        except Exception as e:
            print(f"  Error fetching lyrics for {artist} - {title}: {e}")
            status = "ERROR"
            time.sleep(5) # Longer delay after an error

    lyrics_results.append({
        'track_uri': track_uri,
        'lyrics': lyrics,
        'status': status
    })
    processed_count += 1

end_time = time.time()
print(f"\nFinished fetching sample lyrics.")
print(f"Processed {processed_count} tracks in {end_time - start_time:.2f} seconds.")

# --- Save Results ---
if lyrics_results:
    lyrics_df = pd.DataFrame(lyrics_results)
    print(f"\nSaving {len(lyrics_df)} lyric results...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(LYRICS_OUTPUT_FILE), exist_ok=True)
        # Save (consider appending if running multiple times on samples)
        lyrics_df.to_parquet(LYRICS_OUTPUT_FILE, index=False)
        # Or use CSV: lyrics_df.to_csv(LYRICS_OUTPUT_FILE.replace('.parquet','.csv'), index=False)
        print(f"Lyrics data saved to {LYRICS_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving lyrics data: {e}")