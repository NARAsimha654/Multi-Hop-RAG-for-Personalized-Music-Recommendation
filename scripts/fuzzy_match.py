import pandas as pd
import numpy as np
import os
import re
import time
from rapidfuzz import fuzz, process # For fuzzy matching
from tqdm.auto import tqdm # Progress bar
import sys

# --- Configuration ---
# Input Files
MSD_CLEANED_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_subset_cleaned.csv' # Or .csv
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'
DIRECT_MATCHES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_mpd_direct_matches.parquet'

# Output File
FUZZY_MATCHES_OUTPUT_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_mpd_fuzzy_matches.parquet'

# Fuzzy Matching Parameters
SIMILARITY_THRESHOLD = 88 # Score out of 100 (adjust based on results)
# Use fuzz.token_sort_ratio which handles word order differences
FUZZY_METHOD = fuzz.token_sort_ratio

# --- Helper Function: Text Normalization (same as before) ---
def normalize_text(text):
    """Normalizes text for matching: lowercase, remove bracketed content, punctuation, extra whitespace."""
    if text is None or pd.isna(text):
        return "" # Return empty string for None or NaN
    text = str(text).lower()
    # Remove content within common brackets (feat., remix, live, remastered, version, edit etc.)
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    # Remove punctuation except apostrophes (adjust regex as needed)
    text = re.sub(r'[^\w\s\']', '', text)
    # Replace multiple whitespace characters with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load Data ---
print("Loading data...")
try:
    # Load MSD Data
    if MSD_CLEANED_FILE.endswith('.parquet'): msd_df = pd.read_parquet(MSD_CLEANED_FILE)
    else: msd_df = pd.read_csv(MSD_CLEANED_FILE)
    print(f"Loaded MSD data ({msd_df.shape[0]} rows)")

    # Load MPD Tracks Data
    if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    print(f"Loaded MPD tracks data ({mpd_tracks_df.shape[0]} rows)")

    # Load Direct Matches
    if os.path.exists(DIRECT_MATCHES_FILE):
        if DIRECT_MATCHES_FILE.endswith('.parquet'): direct_matches_df = pd.read_parquet(DIRECT_MATCHES_FILE)
        else: direct_matches_df = pd.read_csv(DIRECT_MATCHES_FILE)
        print(f"Loaded Direct Matches ({direct_matches_df.shape[0]} rows)")
        matched_msd_ids = set(direct_matches_df['song_id'].unique())
        matched_mpd_uris = set(direct_matches_df['track_uri'].unique())
    else:
        print(f"Warning: Direct matches file not found at {DIRECT_MATCHES_FILE}. Assuming no direct matches.")
        matched_msd_ids = set()
        matched_mpd_uris = set()

except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during file loading: {e}")
    sys.exit(1)


# --- Prepare Data for Fuzzy Matching ---
print("\nPreparing data for fuzzy matching...")

# 1. Normalize text in both dataframes
msd_df['norm_artist_name'] = msd_df['artist_name'].apply(normalize_text)
msd_df['norm_title'] = msd_df['title'].apply(normalize_text)
msd_df['norm_match_key'] = msd_df['norm_artist_name'] + '||' + msd_df['norm_title']

mpd_tracks_df['norm_artist_name'] = mpd_tracks_df['artist_name'].apply(normalize_text)
mpd_tracks_df['norm_track_name'] = mpd_tracks_df['track_name'].apply(normalize_text)
mpd_tracks_df['norm_match_key'] = mpd_tracks_df['norm_artist_name'] + '||' + mpd_tracks_df['norm_track_name']

# 2. Filter out already matched items
msd_unmatched = msd_df[~msd_df['song_id'].isin(matched_msd_ids)].copy()
mpd_unmatched = mpd_tracks_df[~mpd_tracks_df['track_uri'].isin(matched_mpd_uris)].copy()

# 3. Filter out entries with empty normalized keys (cannot be matched)
msd_unmatched = msd_unmatched[msd_unmatched['norm_match_key'] != '||']
mpd_unmatched = mpd_unmatched[mpd_unmatched['norm_match_key'] != '||']

print(f"MSD songs remaining for fuzzy matching: {len(msd_unmatched)}")
print(f"MPD tracks remaining for fuzzy matching: {len(mpd_unmatched)}")

# --- Perform Fuzzy Matching ---
# This can be computationally intensive. We'll iterate through unmatched MSD songs
# and find the best match in the unmatched MPD tracks.

fuzzy_matches_list = []
# Create a list of choices for rapidfuzz process.extractOne
# Format: (normalized_key, mpd_track_uri)
mpd_choices = list(zip(mpd_unmatched['norm_match_key'], mpd_unmatched['track_uri']))

# Check if there are choices to match against
if not mpd_choices:
    print("No unmatched MPD tracks available to perform fuzzy matching against.")
else:
    print(f"\nStarting fuzzy matching (Threshold: {SIMILARITY_THRESHOLD})...")
    start_time = time.time()

    # Use tqdm for progress bar
    for index, row in tqdm(msd_unmatched.iterrows(), total=len(msd_unmatched), desc="Fuzzy Matching MSD"):
        msd_song_id = row['song_id']
        msd_norm_key = row['norm_match_key']
        msd_artist = row['artist_name'] # Original names for context
        msd_title = row['title']

        if not msd_norm_key: continue # Skip if key is empty

        # Use process.extractOne to find the best match above the threshold
        # It's generally faster than iterating and calling fuzz.ratio individually
        # We provide the query (msd_norm_key) and the choices (list of mpd keys)
        # scorer specifies the similarity function to use
        # score_cutoff sets the minimum similarity score
        best_match = process.extractOne(
            msd_norm_key,
            [choice[0] for choice in mpd_choices], # Extract just the keys for matching
            scorer=FUZZY_METHOD,
            score_cutoff=SIMILARITY_THRESHOLD
        )

        # best_match format: (matched_mpd_key, score, index_in_choices_list)
        if best_match:
            matched_mpd_key, score, match_index = best_match
            # Get the corresponding track_uri using the index
            matched_mpd_uri = mpd_choices[match_index][1]

            # Retrieve original MPD names for verification/context
            mpd_match_details = mpd_unmatched[mpd_unmatched['track_uri'] == matched_mpd_uri].iloc[0]
            mpd_artist = mpd_match_details['artist_name']
            mpd_track_name = mpd_match_details['track_name']

            # Append the match details
            fuzzy_matches_list.append({
                'song_id': msd_song_id,
                'track_uri': matched_mpd_uri,
                'fuzzy_score': score,
                'msd_artist': msd_artist,
                'msd_title': msd_title,
                'mpd_artist': mpd_artist,
                'mpd_track_name': mpd_track_name,
                'msd_norm_key': msd_norm_key,
                'mpd_norm_key': matched_mpd_key
            })

    end_time = time.time()
    print(f"Fuzzy matching completed in {end_time - start_time:.2f} seconds.")


# --- Save Fuzzy Matches ---
if fuzzy_matches_list:
    fuzzy_matches_df = pd.DataFrame(fuzzy_matches_list)
    print(f"\nFound {len(fuzzy_matches_df)} potential fuzzy matches.")
    print("Sample fuzzy matches:")
    print(fuzzy_matches_df[['msd_artist', 'msd_title', 'mpd_artist', 'mpd_track_name', 'fuzzy_score']].head())

    print(f"\nSaving fuzzy matches to {FUZZY_MATCHES_OUTPUT_FILE}...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(FUZZY_MATCHES_OUTPUT_FILE), exist_ok=True)
        # Save as parquet (or CSV)
        if FUZZY_MATCHES_OUTPUT_FILE.endswith('.parquet'):
             fuzzy_matches_df.to_parquet(FUZZY_MATCHES_OUTPUT_FILE, index=False)
        else:
             fuzzy_matches_df.to_csv(FUZZY_MATCHES_OUTPUT_FILE, index=False)
        print("Fuzzy matches saved successfully.")
    except Exception as e:
        print(f"Error saving fuzzy matches file: {e}")
else:
    print("\nNo fuzzy matches found above the threshold.")

print("\n--- Fuzzy Matching Script Finished ---")