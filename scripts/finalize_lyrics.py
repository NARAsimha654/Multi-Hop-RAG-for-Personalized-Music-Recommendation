import pandas as pd
import pickle
import os
import sys

# --- Configuration ---
# Path to the progress file saved by the fetching script (the one with ~27k results)
# Assuming it's still named based on the 50k target run
PROGRESS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\lyrics_fetch_progress_50000.pkl'

# --- IMPORTANT: Set the desired final output file path ---
# Reflecting the actual data size (~27k)
FINAL_LYRICS_OUTPUT_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_lyrics_27k.parquet'

print(f"Loading progress data from: {PROGRESS_FILE}")

# --- Load Progress Data ---
processed_uris = set()
lyrics_results = []

if os.path.exists(PROGRESS_FILE):
    try:
        with open(PROGRESS_FILE, 'rb') as f:
            progress_data = pickle.load(f)
            # Load the list of dictionaries containing results
            lyrics_results = progress_data.get('lyrics_results', [])
            # Load the set of processed URIs (optional, mainly for count verification)
            processed_uris = progress_data.get('processed_uris', set())
            print(f"Loaded {len(lyrics_results)} results covering {len(processed_uris)} unique URIs.")
            # Store the actual number found for filename consistency if needed
            actual_lyrics_count = len(lyrics_results)
    except Exception as e:
        print(f"Error reading progress file {PROGRESS_FILE}: {e}")
        sys.exit(1)
else:
    print(f"Error: Progress file not found at {PROGRESS_FILE}. Cannot finalize.")
    sys.exit(1)

# --- Save Final Parquet File ---
if lyrics_results:
    # Convert the list of dictionaries to a DataFrame
    lyrics_df = pd.DataFrame(lyrics_results)

    # Optional: Drop duplicates based on track_uri just in case
    lyrics_df = lyrics_df.drop_duplicates(subset=['track_uri'], keep='last')
    print(f"DataFrame created with {len(lyrics_df)} unique track entries.")

    # --- Use the FINAL_LYRICS_OUTPUT_FILE variable ---
    print(f"\nSaving final {len(lyrics_df)} lyric results to {FINAL_LYRICS_OUTPUT_FILE}...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(FINAL_LYRICS_OUTPUT_FILE), exist_ok=True)
        # Save final results as Parquet
        lyrics_df.to_parquet(FINAL_LYRICS_OUTPUT_FILE, index=False)
        print(f"Final lyrics data saved successfully.")

        # Optional: Keep the .pkl file as a backup until you confirm everything works
        print(f"Progress file still available at: {PROGRESS_FILE}")

    except Exception as e:
        print(f"Error saving final lyrics data: {e}")
else:
    print("No lyrics results found in the progress file.")

print("\n--- Lyrics Finalization Complete ---")
