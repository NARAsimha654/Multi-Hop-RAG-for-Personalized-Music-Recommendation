import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os # To create output directory

# --- Configuration ---
# Paths to your data
MSD_CLEANED_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\msd_subset_cleaned.csv'
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# Output directory for embeddings
EMBEDDING_DIR = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\embeddings\\'
MSD_EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, 'msd_text_embeddings.npz')
MPD_EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, 'mpd_text_embeddings.npz')

# Ensure output directory exists
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# --- Load Data ---
print("Loading data...")
if MSD_CLEANED_FILE.endswith('.parquet'): msd_df = pd.read_parquet(MSD_CLEANED_FILE)
else: msd_df = pd.read_csv(MSD_CLEANED_FILE)

if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
print("Data loaded.")

# --- Load Sentence Transformer Model ---
print("Loading Sentence Transformer model (e.g., all-MiniLM-L6-v2)...")
# This will download the model the first time you run it
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# --- Generate Embeddings for MSD Songs ---
print("\nGenerating embeddings for MSD songs...")
# Prepare text: "Artist - Title"
# Handle potential missing values gracefully
msd_df['text_for_embedding'] = msd_df['artist_name'].fillna('') + ' - ' + msd_df['title'].fillna('')
msd_texts = msd_df['text_for_embedding'].tolist()
msd_ids = msd_df['song_id'].tolist()

if msd_texts:
    # Generate embeddings (this can take time depending on data size and hardware)
    msd_embeddings = model.encode(msd_texts, show_progress_bar=True)
    print(f"Generated {msd_embeddings.shape[0]} embeddings with dimension {msd_embeddings.shape[1]}")

    # Save embeddings and corresponding IDs
    # Using np.savez for simplicity, stores multiple arrays in one file
    np.savez(MSD_EMBEDDING_FILE, ids=np.array(msd_ids), embeddings=msd_embeddings)
    print(f"MSD embeddings saved to {MSD_EMBEDDING_FILE}")
else:
    print("No text data found for MSD embeddings.")


# --- Generate Embeddings for MPD Tracks ---
print("\nGenerating embeddings for MPD tracks...")
# Prepare text: "Artist - Track Name"
mpd_tracks_df['text_for_embedding'] = mpd_tracks_df['artist_name'].fillna('') + ' - ' + mpd_tracks_df['track_name'].fillna('')
# Filter out tracks with missing URIs if any (shouldn't happen based on previous processing)
mpd_tracks_df_valid = mpd_tracks_df.dropna(subset=['track_uri'])
mpd_texts = mpd_tracks_df_valid['text_for_embedding'].tolist()
mpd_uris = mpd_tracks_df_valid['track_uri'].tolist()

if mpd_texts:
    # Generate embeddings (this will take longer due to >100k tracks)
    mpd_embeddings = model.encode(mpd_texts, show_progress_bar=True)
    print(f"Generated {mpd_embeddings.shape[0]} embeddings with dimension {mpd_embeddings.shape[1]}")

    # Save embeddings and corresponding URIs
    np.savez(MPD_EMBEDDING_FILE, ids=np.array(mpd_uris), embeddings=mpd_embeddings)
    print(f"MPD embeddings saved to {MPD_EMBEDDING_FILE}")
else:
    print("No text data found for MPD embeddings.")

print("\n--- Text Embedding Generation Complete ---")