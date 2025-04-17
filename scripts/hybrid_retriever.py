import numpy as np
import faiss
import os
import pickle
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi # Or the specific BM25 class you used

# --- Configuration ---
# Paths to indices and maps
INDEX_DIR_FAISS = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\faiss_indices\\'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index.faiss')
MPD_FAISS_ID_MAP_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index_id_map.pkl')

INDEX_DIR_BM25 = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\bm25_indices\\'
BM25_INDEX_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_index.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_id_map.pkl')

# Optional: Track details for display
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'


# --- Load Components ---
print("Loading all retrieval components...")
try:
    # FAISS
    faiss_index = faiss.read_index(MPD_FAISS_INDEX_FILE)
    with open(MPD_FAISS_ID_MAP_FILE, 'rb') as f: faiss_ids = pickle.load(f)
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors).")

    # BM25
    with open(BM25_INDEX_FILE, 'rb') as f: bm25_index = pickle.load(f)
    with open(BM25_ID_MAP_FILE, 'rb') as f: bm25_ids = pickle.load(f)
    print(f"BM25 index loaded ({bm25_index.corpus_size} docs).")

    # SBERT Model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT model loaded.")

    # Optional Track Details
    import pandas as pd
    if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    track_details_lookup = mpd_tracks_df.set_index('track_uri')[['track_name', 'artist_name']].apply(tuple, axis=1).to_dict()
    print("Track details loaded.")

except Exception as e:
    print(f"Error loading components: {e}")
    exit()

# --- Tokenizer for BM25 Query ---
def tokenize_query(text):
    processed = text.lower()
    processed = re.sub(r'[^\w\s]', '', processed)
    tokens = processed.split()
    return tokens

# --- Hybrid Search Function ---
def hybrid_search(query_text, top_k_dense=10, top_k_sparse=10, rerank_k=5, w_dense=0.5, w_sparse=0.5):
    """
    Performs hybrid search using FAISS and BM25, then reranks.

    Args:
        query_text (str): User query.
        top_k_dense (int): How many results to fetch from FAISS.
        top_k_sparse (int): How many results to fetch from BM25.
        rerank_k (int): How many final results to return after reranking.
        w_dense (float): Weight for dense score.
        w_sparse (float): Weight for sparse score.

    Returns:
        list: Top rerank_k results as [(track_uri, final_score)].
    """
    print(f"\nPerforming hybrid search for: '{query_text}'")

    # 1. Dense Search (FAISS)
    dense_results = {} # Store as {uri: score}
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
                    # Convert distance to similarity score (higher is better)
                    # Simple inverse: 1 / (1 + distance) - adjust as needed
                    score = 1.0 / (1.0 + dist)
                    dense_results[uri] = score
        print(f"  FAISS returned {len(dense_results)} results.")
    except Exception as e:
        print(f"  Error during FAISS search: {e}")

    # 2. Sparse Search (BM25)
    sparse_results = {} # Store as {uri: score}
    try:
        tokenized_query = tokenize_query(query_text)
        # Get scores for *all* docs in the BM25 index
        doc_scores = bm25_index.get_scores(tokenized_query)

        # Get top N indices based on scores
        top_indices = np.argsort(doc_scores)[::-1][:top_k_sparse]

        for i in top_indices:
            if i < len(bm25_ids) and doc_scores[i] > 0: # Only consider positive scores
                uri = bm25_ids[i]
                score = doc_scores[i]
                sparse_results[uri] = score
        print(f"  BM25 returned {len(sparse_results)} results (with score > 0).")
    except Exception as e:
        print(f"  Error during BM25 search: {e}")

    # 3. Combine and Rerank (Simple Weighted Sum Example)
    combined_scores = {}
    all_retrieved_uris = set(dense_results.keys()) | set(sparse_results.keys())
    print(f"  Total unique results from both retrievers: {len(all_retrieved_uris)}")

    # Normalize BM25 scores (optional, but helpful if scales differ wildly)
    # Simple approach: divide by max score if max > 0
    max_bm25_score = max(sparse_results.values()) if sparse_results else 0
    norm_sparse_results = {uri: (score / max_bm25_score if max_bm25_score > 0 else 0)
                           for uri, score in sparse_results.items()}

    for uri in all_retrieved_uris:
        dense_score = dense_results.get(uri, 0) # Default to 0 if not found
        sparse_score_norm = norm_sparse_results.get(uri, 0) # Use normalized score

        # Weighted sum
        final_score = (w_dense * dense_score) + (w_sparse * sparse_score_norm)
        combined_scores[uri] = final_score

    # Sort by final score (descending)
    reranked_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

    # Return top K
    return reranked_results[:rerank_k]


# --- Example Usage ---
query = "love song ballad"
final_results = hybrid_search(query, top_k_dense=10, top_k_sparse=10, rerank_k=5, w_dense=0.6, w_sparse=0.4)

if final_results:
    print("\n--- Top Hybrid Results ---")
    for uri, score in final_results:
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        print(f"  Score: {score:.4f} | URI: {uri} | Name: {details[0]}, Artist: {details[1]}")