import pickle
import numpy as np
import os
import re
from rank_bm25 import BM25Okapi # Make sure this matches the class you saved

# Optional: for tokenization consistency
# import nltk
# from nltk.tokenize import word_tokenize

# --- Configuration ---
# Input index and ID map files
INDEX_DIR = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\bm25_indices\\'
BM25_INDEX_FILE = os.path.join(INDEX_DIR, 'mpd_lyrics_bm25_index.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR, 'mpd_lyrics_bm25_id_map.pkl')

# Path to MPD tracks data (optional, to display track names for context)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# --- Load Index and ID Map ---
print(f"Loading BM25 index object from {BM25_INDEX_FILE}...")
try:
    with open(BM25_INDEX_FILE, 'rb') as f_index:
        bm25 = pickle.load(f_index)
    print("BM25 index loaded.")
except FileNotFoundError:
    print(f"Error: BM25 index file not found at {BM25_INDEX_FILE}")
    exit()
except Exception as e:
    print(f"Error loading BM25 index: {e}")
    exit()

print(f"Loading BM25 ID map from {BM25_ID_MAP_FILE}...")
try:
    with open(BM25_ID_MAP_FILE, 'rb') as f_map:
        bm25_track_uris = pickle.load(f_map) # Load the list of track URIs
    print(f"BM25 ID map loaded. Contains {len(bm25_track_uris)} IDs.")
    # Validation
    if len(bm25_track_uris) != bm25.corpus_size:
         print("Warning: Number of IDs in map doesn't match corpus size in BM25 index!")

except FileNotFoundError:
    print(f"Error: BM25 ID map file not found at {BM25_ID_MAP_FILE}")
    exit()
except Exception as e:
    print(f"Error loading BM25 ID map: {e}")
    exit()

# Optional: Load track details for better output display
try:
    import pandas as pd
    if MPD_TRACKS_FILE.endswith('.parquet'):
        mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else:
        mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    # Create a quick lookup dictionary: track_uri -> (track_name, artist_name)
    track_details_lookup = mpd_tracks_df.set_index('track_uri')[['track_name', 'artist_name']].apply(tuple, axis=1).to_dict()
    print("Track details loaded for display.")
except Exception as e:
    print(f"Warning: Could not load track details for display: {e}")
    track_details_lookup = {} # Use empty dict if loading fails


# --- Tokenization Function (Consistent with Indexing) ---
def tokenize_query(text):
    """Applies the same tokenization used during indexing to the query."""
    processed = text.lower()
    processed = re.sub(r'[^\w\s]', '', processed)
    tokens = processed.split()
    # Optional: Use nltk tokenizer if used for indexing
    # tokens = word_tokenize(processed)
    return tokens

# --- Search Function ---
def search_lyrics_bm25(query_text, top_n=5):
    """
    Searches the BM25 index for tracks with lyrics matching the query_text.

    Args:
        query_text (str): The text query (keywords).
        top_n (int): The number of relevant tracks to return.

    Returns:
        list: A list of tuples, where each tuple is (track_uri, score).
              Returns None if search fails.
    """
    print(f"\nSearching for top {top_n} tracks with lyrics matching: '{query_text}'")
    try:
        # 1. Tokenize the query
        tokenized_query = tokenize_query(query_text)
        print(f"  Tokenized query: {tokenized_query}")

        # 2. Get scores for all documents in the corpus
        # doc_scores = bm25.get_scores(tokenized_query)

        # OR: Get top N documents directly (more efficient)
        # This returns the actual documents (token lists), not useful here
        # top_docs = bm25.get_top_n(tokenized_query, tokenized_corpus, n=top_n)

        # We need the scores and the original indices to map back to URIs
        # Let's calculate all scores and find top N indices manually
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the indices of the top N scores
        top_indices = np.argsort(doc_scores)[::-1][:top_n] # Get indices sorted descending by score

        # 3. Map indices back to track URIs and get scores
        results = []
        for i in top_indices:
            # Ensure index is valid and score is positive (BM25 scores can be 0 or negative)
            if i < len(bm25_track_uris) and doc_scores[i] > 0:
                track_uri = bm25_track_uris[i]
                score = doc_scores[i]
                results.append((track_uri, score))
            # else: # Optional: break if scores become non-positive
            #    break

        return results

    except Exception as e:
        print(f"An error occurred during BM25 search: {e}")
        return None

# --- Example Usage ---

# Example queries (keywords likely to be in lyrics)
query1 = "love heart soul"
query2 = "tell me important" # From the empire! empire! title/lyrics
query3 = "bus wheels" # From the kids song

results1 = search_lyrics_bm25(query1, top_n=3)
if results1:
    print("Results:")
    for uri, score in results1:
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        print(f"  URI: {uri}, Score: {score:.4f}, Name: {details[0]}, Artist: {details[1]}")

results2 = search_lyrics_bm25(query2, top_n=3)
if results2:
    print("Results:")
    for uri, score in results2:
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        print(f"  URI: {uri}, Score: {score:.4f}, Name: {details[0]}, Artist: {details[1]}")

results3 = search_lyrics_bm25(query3, top_n=3)
if results3:
    print("Results:")
    for uri, score in results3:
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        print(f"  URI: {uri}, Score: {score:.4f}, Name: {details[0]}, Artist: {details[1]}")