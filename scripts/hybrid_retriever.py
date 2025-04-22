# Import necessary libraries
import numpy as np
import faiss
import os
import pickle
import re
import pandas as pd
import time
import sys
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi # Or the specific BM25 class you used

# --- Configuration ---
# Paths to indices and maps
# Using raw strings for Windows paths
INDEX_DIR_FAISS = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\faiss_indices'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index.faiss')
MPD_FAISS_ID_MAP_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index_id_map.pkl')

INDEX_DIR_BM25 = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\bm25_indices'
BM25_INDEX_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_index.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_id_map.pkl')

# Path to MPD tracks data (needed for track details)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# --- Load Components ---
print("--- Loading Retrieval Components ---")
faiss_index = None
faiss_ids = []
bm25_index = None
bm25_ids = []
sbert_model = None
track_details_lookup = {}

try:
    # FAISS
    print("Loading FAISS components...")
    faiss_index = faiss.read_index(MPD_FAISS_INDEX_FILE)
    with open(MPD_FAISS_ID_MAP_FILE, 'rb') as f: faiss_ids = pickle.load(f)
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors).")

    # BM25
    print("Loading BM25 components...")
    # Handle case where BM25 files might not exist (e.g., if lyrics fetching failed)
    if os.path.exists(BM25_INDEX_FILE) and os.path.exists(BM25_ID_MAP_FILE):
        with open(BM25_INDEX_FILE, 'rb') as f: bm25_index = pickle.load(f)
        with open(BM25_ID_MAP_FILE, 'rb') as f: bm25_ids = pickle.load(f)
        print(f"BM25 index loaded ({getattr(bm25_index, 'corpus_size', 0)} docs).")
    else:
        print("Warning: BM25 index or ID map not found. Sparse retrieval will be skipped.")
        bm25_index = None
        bm25_ids = []

    # SBERT Model
    print("Loading Sentence Transformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT model loaded.")

    # Track Details Lookup
    print("Loading track details...")
    if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    track_details_lookup = mpd_tracks_df.set_index('track_uri')[['track_name', 'artist_name']].apply(tuple, axis=1).to_dict()
    print("Track details loaded.")

    print("\n--- All components loaded successfully ---")

except FileNotFoundError as fnf_error:
    print(f"Error loading file: {fnf_error}. Please check file paths.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading components: {e}")
    sys.exit(1)

# --- Tokenizer for BM25 Query ---
def tokenize_query(text):
    """ Basic tokenizer for BM25 query """
    processed = text.lower()
    processed = re.sub(r'[^\w\s]', '', processed)
    tokens = processed.split()
    return tokens

# --- Hybrid Search Function ---
def hybrid_search(query_text, top_k_dense=10, top_k_sparse=10, rerank_k=5, w_dense=0.5, w_sparse=0.5):
    """
    Performs hybrid search using FAISS and BM25, then reranks.
    Returns top K results as [(track_uri, final_score)].
    """
    print(f"\nPerforming hybrid search for: '{query_text}'")
    dense_results = {} # {uri: dense_similarity_score}
    try:
        query_embedding = sbert_model.encode([query_text])
        if query_embedding.dtype != np.float32: query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1: query_embedding = np.expand_dims(query_embedding, axis=0)
        distances, indices = faiss_index.search(query_embedding, top_k_dense)
        if len(indices) > 0:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx >= 0 and idx < len(faiss_ids):
                    uri = faiss_ids[idx]
                    dist = distances[0][i]
                    score = 1.0 / (1.0 + dist) # Convert distance to similarity
                    dense_results[uri] = score
    except Exception as e:
        print(f"  Error during FAISS search: {e}")

    sparse_results = {} # {uri: bm25_score}
    # Only run BM25 if the index was loaded successfully
    if bm25_index and bm25_ids:
        try:
            tokenized_query = tokenize_query(query_text)
            doc_scores = bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[::-1][:top_k_sparse]
            for i in top_indices:
                if i < len(bm25_ids) and doc_scores[i] > 0:
                    uri = bm25_ids[i]
                    score = doc_scores[i]
                    sparse_results[uri] = score
        except Exception as e:
            print(f"  Error during BM25 search: {e}")
    else:
        # Skip BM25 if index not loaded
        pass

    # Combine and Rerank
    combined_scores = {}
    all_retrieved_uris = set(dense_results.keys()) | set(sparse_results.keys())
    print(f"  FAISS returned {len(dense_results)} results.")
    print(f"  BM25 returned {len(sparse_results)} results (with score > 0).")
    print(f"  Total unique results from both retrievers: {len(all_retrieved_uris)}")

    # Combine scores using weights
    for uri in all_retrieved_uris:
        dense_score = dense_results.get(uri, 0)
        bm25_score = sparse_results.get(uri, 0) # Use raw BM25 score
        # Simple weighted sum (consider normalizing later if needed)
        final_score = (w_dense * dense_score) + (w_sparse * bm25_score)
        combined_scores[uri] = final_score

    # Sort by final score (descending)
    reranked_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

    return reranked_results[:rerank_k] # Return top K results after reranking


# --- Interactive Query Loop ---
if __name__ == "__main__":
    print("\n--- Interactive Hybrid Search ---")
    print("Enter your music query (e.g., 'sad acoustic songs', 'artist - title')")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            # Get user input
            user_query = input("\nEnter query: ")

            # Check for exit command
            if user_query.lower() in ['quit', 'exit']:
                print("Exiting...")
                break

            # Perform the hybrid search
            # You can adjust parameters here if needed (e.g., weights, k values)
            final_results = hybrid_search(
                user_query,
                top_k_dense=15,  # Fetch slightly more initially
                top_k_sparse=15,
                rerank_k=10,     # Return top 10 final results
                w_dense=0.6,     # Example weights (adjust based on testing)
                w_sparse=0.4
            )

            # Display the results
            if final_results:
                print("\n--- Top Hybrid Results ---")
                for i, (uri, score) in enumerate(final_results):
                    details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
                    print(f"  {i+1}. Score: {score:.4f} | URI: {uri} | Name: {details[0]}, Artist: {details[1]}")
            else:
                print("No relevant results found.")

        except EOFError: # Handle Ctrl+D
             print("\nExiting...")
             break
        except KeyboardInterrupt: # Handle Ctrl+C
             print("\nExiting...")
             break
        except Exception as e:
             print(f"An error occurred: {e}")
             # Optionally continue the loop or break
             # break

    print("\n--- Search Session Ended ---")

