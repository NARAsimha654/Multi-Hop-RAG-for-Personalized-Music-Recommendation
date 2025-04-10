import pandas as pd
import re # For regular expression-based normalization
import os # To ensure output directory exists
import time # To time the process

# --- Configuration ---
# IMPORTANT: Verify these paths point to your processed files
MSD_CLEANED_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_subset_cleaned.csv' # Or .csv
MPD_TRACKS_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_unique_tracks.parquet'
MATCHES_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_mpd_direct_matches.parquet' # Where to save matches

# --- Functions ---

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

# --- Main Execution ---

print("--- Starting Data Integration: MSD <-> MPD Direct Matching ---")
start_time = time.time()

# 1. Load Data
print("\nStep 1: Loading processed data...")
try:
    # Adjust read function if you saved as CSV
    if MSD_CLEANED_FILE.endswith('.parquet'):
        msd_df = pd.read_parquet(MSD_CLEANED_FILE)
    else:
        msd_df = pd.read_csv(MSD_CLEANED_FILE)

    if MPD_TRACKS_FILE.endswith('.parquet'):
        mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else:
         mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)

    print(f"  Loaded MSD data ({msd_df.shape[0]} rows)")
    print(f"  Loaded MPD tracks data ({mpd_tracks_df.shape[0]} rows)")
except Exception as e:
    print(f"Error loading data files: {e}")
    print("Please check file paths and format. Exiting.")
    exit()

print("\nMSD Data Sample (Relevant Columns):")
print(msd_df[['song_id', 'artist_name', 'title']].head())
print("\nMPD Tracks Data Sample (Relevant Columns):")
print(mpd_tracks_df[['track_uri', 'artist_name', 'track_name']].head())


# 2. Normalize Text Fields
print("\nStep 2: Normalizing artist and track names...")
msd_df['norm_artist_name'] = msd_df['artist_name'].apply(normalize_text)
msd_df['norm_title'] = msd_df['title'].apply(normalize_text)

mpd_tracks_df['norm_artist_name'] = mpd_tracks_df['artist_name'].apply(normalize_text)
mpd_tracks_df['norm_track_name'] = mpd_tracks_df['track_name'].apply(normalize_text)

print("\nNormalized MSD Sample:")
print(msd_df[['norm_artist_name', 'norm_title']].head())
print("\nNormalized MPD Sample:")
print(mpd_tracks_df[['norm_artist_name', 'norm_track_name']].head())


# 3. Create Match Keys
print("\nStep 3: Creating combined match keys...")
# Using '||' as a separator, ensure it doesn't naturally occur in your normalized names
msd_df['match_key'] = msd_df['norm_artist_name'] + '||' + msd_df['norm_title']
mpd_tracks_df['match_key'] = mpd_tracks_df['norm_artist_name'] + '||' + mpd_tracks_df['norm_track_name']

# Check for potential issues like empty keys
print(f"  MSD empty match keys: {(msd_df['match_key'] == '||').sum()}")
print(f"  MPD empty match keys: {(mpd_tracks_df['match_key'] == '||').sum()}")


# 4. Perform Direct Matching via Merge
print("\nStep 4: Performing direct merge on normalized keys...")
# Select relevant columns for the merge result, avoiding duplicate normalized columns
direct_matches = pd.merge(
    msd_df[['song_id', 'match_key', 'artist_name', 'title']], # Original MSD names
    mpd_tracks_df[['track_uri', 'match_key', 'artist_name', 'track_name']], # Original MPD names
    on='match_key',
    how='inner',
    suffixes=('_msd', '_mpd') # Add suffixes to original name columns if they might be identical
)

print(f"\nFound {len(direct_matches)} direct matches.")
if not direct_matches.empty:
    print("Direct Matches Sample:\n", direct_matches.head())
    # Check how many unique MSD songs and MPD tracks were matched
    print(f"  Unique MSD songs matched: {direct_matches['song_id'].nunique()}")
    print(f"  Unique MPD tracks matched: {direct_matches['track_uri'].nunique()}")


# 5. Save the Matches (Optional but recommended)
print(f"\nStep 5: Saving direct matches to {MATCHES_OUTPUT_FILE}...")
if not direct_matches.empty:
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(MATCHES_OUTPUT_FILE), exist_ok=True)
        # Save as parquet (or CSV)
        if MATCHES_OUTPUT_FILE.endswith('.parquet'):
             direct_matches.to_parquet(MATCHES_OUTPUT_FILE, index=False, engine='pyarrow')
        else:
             direct_matches.to_csv(MATCHES_OUTPUT_FILE, index=False)
        print("  Matches saved successfully.")
    except ImportError:
        print("  Note: 'pyarrow' not found. Consider saving as CSV or installing pyarrow.")
        # Fallback to CSV if needed
        # MATCHES_OUTPUT_FILE = MATCHES_OUTPUT_FILE.replace('.parquet','.csv')
        # direct_matches.to_csv(MATCHES_OUTPUT_FILE, index=False)
    except Exception as e:
        print(f"  Error saving matches file: {e}")
else:
    print("  No direct matches found to save.")


end_time = time.time()
print(f"\n--- Data Integration (Direct Match) Complete ---")
print(f"Total time: {end_time - start_time:.2f} seconds")