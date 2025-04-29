import pandas as pd
import numpy as np
import pickle
import os
import re
from rank_bm25 import BM25Okapi # Or BM25Plus, BM25L

# Optional: for more advanced tokenization
# import nltk
# nltk.download('punkt') # Download once if using nltk tokenizer
# from nltk.tokenize import word_tokenize

# --- Configuration ---
# Input lyrics file
LYRICS_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\mpd_lyrics_27k.parquet'

# Output directory for BM25 index and ID map
INDEX_DIR = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\bm25_indices\\'
BM25_INDEX_FILE = os.path.join(INDEX_DIR, 'mpd_lyrics_bm25_index_27k.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR, 'mpd_lyrics_bm25_id_map_27k.pkl')

# Ensure output directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Load Lyrics Data ---
print(f"Loading lyrics data from {LYRICS_FILE}...")
try:
    if LYRICS_FILE.endswith('.parquet'):
        lyrics_df = pd.read_parquet(LYRICS_FILE)
    else:
        lyrics_df = pd.read_csv(LYRICS_FILE)
    print(f"Loaded {lyrics_df.shape[0]} lyric entries.")
except FileNotFoundError:
    print(f"Error: Lyrics file not found at {LYRICS_FILE}")
    exit()
except Exception as e:
    print(f"Error loading lyrics file: {e}")
    exit()

# --- Prepare Corpus for BM25 ---
print("Preparing corpus for BM25...")

# Filter for rows where lyrics were successfully found
valid_lyrics_df = lyrics_df.dropna(subset=['lyrics'])
valid_lyrics_df = valid_lyrics_df[valid_lyrics_df['lyrics'].str.strip() != '']
# Optionally filter based on status column if you have one
# valid_lyrics_df = valid_lyrics_df[valid_lyrics_df['status'] == 'FOUND']

if valid_lyrics_df.empty:
    print("No valid lyrics found in the input file. Cannot build BM25 index.")
    exit()

print(f"Found {len(valid_lyrics_df)} tracks with valid lyrics.")

# Store the track URIs in the order they will be indexed
bm25_track_uris = valid_lyrics_df['track_uri'].tolist()

# Preprocess and tokenize lyrics
tokenized_corpus = []
print("Tokenizing lyrics (lowercase, basic punctuation removal, split by space)...")
for lyrics in valid_lyrics_df['lyrics']:
    # Lowercase
    processed = lyrics.lower()
    # Basic punctuation removal (keep spaces)
    processed = re.sub(r'[^\w\s]', '', processed)
    # Tokenize by splitting on whitespace
    tokens = processed.split()
    # Optional: Use nltk tokenizer for better results
    # tokens = word_tokenize(processed)
    tokenized_corpus.append(tokens)

print(f"Tokenization complete for {len(tokenized_corpus)} documents.")

# --- Build BM25 Index ---
print("\nBuilding BM25 index (BM25Okapi)...")
# Initialize BM25 with the tokenized corpus
bm25 = BM25Okapi(tokenized_corpus)
# Other options: BM25L(tokenized_corpus), BM25Plus(tokenized_corpus)
print("BM25 index built successfully.")
# You can inspect properties like bm25.corpus_size, bm25.avgdl etc. if needed

# --- Save Index Object and ID Map ---
print(f"\nSaving BM25 index object to {BM25_INDEX_FILE}...")
try:
    with open(BM25_INDEX_FILE, 'wb') as f_index:
        pickle.dump(bm25, f_index)
    print("BM25 index saved.")
except Exception as e:
    print(f"Error saving BM25 index: {e}")

print(f"Saving BM25 ID map to {BM25_ID_MAP_FILE}...")
try:
    with open(BM25_ID_MAP_FILE, 'wb') as f_map:
        pickle.dump(bm25_track_uris, f_map) # Save the list of track URIs
    print("BM25 ID map saved.")
except Exception as e:
    print(f"Error saving BM25 ID map: {e}")


print("\n--- BM25 Index Building Complete ---")