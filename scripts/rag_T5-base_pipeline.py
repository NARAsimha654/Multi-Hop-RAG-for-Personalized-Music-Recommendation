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
from transformers import T5Tokenizer, T5ForConditionalGeneration # Using T5 as example

# --- Configuration ---
# Paths to indices and maps
INDEX_DIR_FAISS = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\faiss_indices\\'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index.faiss')
MPD_FAISS_ID_MAP_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index_id_map.pkl')

INDEX_DIR_BM25 = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\bm25_indices\\'
# --- IMPORTANT: Update these paths when full lyrics/BM25 index is ready ---
# Using SAMPLE index for now
BM25_INDEX_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_index_27k.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_id_map_27k.pkl')

# Path to MPD tracks data (needed for track details)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# --- Path to Emotion Features ---
# Use the sample file for now, update when full emotion features are generated
EMOTION_FEATURES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_emotion_features_sample_vad.parquet'

# RAG Model Configuration
GENERATOR_MODEL_NAME = 't5-base' # Keep t5-base

# RRF Configuration
RRF_K = 60 # Constant for RRF calculation

# --- Load All Components ---
print("--- Loading RAG Pipeline Components ---")
faiss_index = None
faiss_ids = []
bm25_index = None
bm25_ids = []
sbert_model = None
tokenizer = None
generator_model = None
track_details_lookup = {}
emotion_lookup = {} # Dictionary to store emotion features

try:
    # FAISS
    print("Loading FAISS components...")
    faiss_index = faiss.read_index(MPD_FAISS_INDEX_FILE)
    with open(MPD_FAISS_ID_MAP_FILE, 'rb') as f: faiss_ids = pickle.load(f)
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors).")

    # BM25
    print("Loading BM25 components...")
    if os.path.exists(BM25_INDEX_FILE) and os.path.exists(BM25_ID_MAP_FILE):
        with open(BM25_INDEX_FILE, 'rb') as f: bm25_index = pickle.load(f)
        with open(BM25_ID_MAP_FILE, 'rb') as f: bm25_ids = pickle.load(f)
        print(f"BM25 index loaded ({getattr(bm25_index, 'corpus_size', 0)} docs).")
    else:
        print(f"Warning: BM25 index or ID map file not found in {INDEX_DIR_BM25}. Sparse retrieval will be skipped.")
        bm25_index = None
        bm25_ids = []

    # SBERT Model
    print(f"Loading Sentence Transformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT model loaded.")

    # Generator Model (T5)
    print(f"Loading Generator model ({GENERATOR_MODEL_NAME})...")
    tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    generator_model = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL_NAME)
    print("Generator model loaded.")

    # Track Details Lookup
    print("Loading track details...")
    if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    track_details_lookup = mpd_tracks_df.set_index('track_uri')[['track_name', 'artist_name']].apply(tuple, axis=1).to_dict()
    print("Track details loaded.")

    # --- Load Emotion Features ---
    print("Loading emotion features...")
    if os.path.exists(EMOTION_FEATURES_FILE):
        if EMOTION_FEATURES_FILE.endswith('.parquet'):
            emotion_df = pd.read_parquet(EMOTION_FEATURES_FILE)
        else:
            emotion_df = pd.read_csv(EMOTION_FEATURES_FILE)
        # Create a lookup dictionary: track_uri -> {feature_name: value}
        # Ensure track_uri is the index for faster lookup
        emotion_df = emotion_df.set_index('track_uri')
        emotion_lookup = emotion_df.to_dict('index') # Converts df to dict like {index_val: {col1: val1, col2: val2}}
        print(f"Emotion features loaded for {len(emotion_lookup)} tracks.")
    else:
        print(f"Warning: Emotion features file not found at {EMOTION_FEATURES_FILE}. Emotion context will be unavailable.")
        emotion_lookup = {} # Keep it empty if file not found

    print("\n--- All components loaded successfully ---")

except FileNotFoundError as fnf_error:
    print(f"Error loading file: {fnf_error}. Please check file paths.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading components: {e}")
    sys.exit(1)

# --- Tokenizer for BM25 Query ---
def tokenize_query(text):
    processed = text.lower(); processed = re.sub(r'[^\w\s]', '', processed); return processed.split()

# --- Hybrid Search Function with RRF (Remains the same) ---
def hybrid_search_rrf(query_text, top_k_dense=50, top_k_sparse=50, rerank_k=10, rrf_k_const=60):
    # (Implementation is the same as the previous version)
    print(f"\nPerforming hybrid search (RRF) for: '{query_text}'")
    dense_ranked_list = []
    try:
        query_embedding = sbert_model.encode([query_text])
        if query_embedding.dtype != np.float32: query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1: query_embedding = np.expand_dims(query_embedding, axis=0)
        distances, indices = faiss_index.search(query_embedding, top_k_dense)
        if len(indices) > 0:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx >= 0 and idx < len(faiss_ids): dense_ranked_list.append(faiss_ids[idx])
        # print(f"  FAISS returned {len(dense_ranked_list)} results.") # Reduced verbosity
    except Exception as e: print(f"  Error during FAISS search: {e}")
    sparse_ranked_list = []
    if bm25_index and bm25_ids:
        try:
            tokenized_query = tokenize_query(query_text)
            doc_scores = bm25_index.get_scores(tokenized_query)
            sorted_indices = np.argsort(doc_scores)[::-1]
            for i in sorted_indices:
                if i < len(bm25_ids) and doc_scores[i] > 0: sparse_ranked_list.append(bm25_ids[i])
                if len(sparse_ranked_list) >= top_k_sparse: break
            # print(f"  BM25 returned {len(sparse_ranked_list)} results (with score > 0).") # Reduced verbosity
        except Exception as e: print(f"  Error during BM25 search: {e}")
    else: print("  BM25 index not available, skipping sparse search.")
    rrf_scores = {}
    all_retrieved_uris = set(dense_ranked_list) | set(sparse_ranked_list)
    # print(f"  Total unique results from both retrievers: {len(all_retrieved_uris)}") # Reduced verbosity
    for uri in all_retrieved_uris:
        score = 0.0
        try: rank = dense_ranked_list.index(uri) + 1; score += 1.0 / (rrf_k_const + rank)
        except ValueError: pass
        try: rank = sparse_ranked_list.index(uri) + 1; score += 1.0 / (rrf_k_const + rank)
        except ValueError: pass
        rrf_scores[uri] = score
    reranked_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return reranked_results[:rerank_k]


# --- UPDATED RAG Prompt Formatting with Emotion ---
def format_rag_prompt(query, retrieved_items):
    """ Formats the prompt including basic emotion info. """
    context = ""
    if not retrieved_items: context = "No relevant tracks found."
    else:
        context_parts = []
        max_context_items = 5 # Limit context length
        for i, (uri, score) in enumerate(retrieved_items[:max_context_items]):
            details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
            # --- Add emotion info ---
            emo_info = emotion_lookup.get(uri) # Get emotion dict for this URI
            emotion_str = ""
            if emo_info:
                # Select key emotion scores to include
                vader_score = emo_info.get('vader_compound')
                valence = emo_info.get('vad_valence')
                arousal = emo_info.get('vad_arousal')
                # Format the emotion string (only include if score exists)
                parts = []
                if vader_score is not None: parts.append(f"Sentiment:{vader_score:.2f}")
                if valence is not None: parts.append(f"V:{valence:.2f}")
                if arousal is not None: parts.append(f"A:{arousal:.2f}")
                if parts: emotion_str = f" (Emotions: {', '.join(parts)})" # Add brackets for clarity
            # --- End of emotion addition ---
            context_parts.append(f"Track {i+1}: {details[1]} - {details[0]}{emotion_str}") # Append emotion string
        context = ". ".join(context_parts)

    # Using Option 2 prompt
    prompt = f"Recommend one music track similar to the query, considering the context.\n\nQuery: {query}\n\nContext: {context}\n\nRecommendation:"
    return prompt

# --- RAG Generation Function (Remains the same, uses updated prompt) ---
def generate_recommendation(query_text, max_length=100):
    """ Performs retrieval (RRF), formats prompt, and generates response using T5. """
    retrieved_items = hybrid_search_rrf(query_text, top_k_dense=50, top_k_sparse=50, rerank_k=5, rrf_k_const=RRF_K)
    prompt = format_rag_prompt(query_text, retrieved_items) # Calls updated prompt function
    print(f"\n--- Generated Prompt ---\n{prompt}\n------------------------")
    try:
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
        output_ids = generator_model.generate(
            input_ids, max_length=max_length, min_length=5, num_beams=5,
            early_stopping=True, no_repeat_ngram_size=2
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Sorry, I encountered an error trying to generate a response."


# --- Example Usage ---
if __name__ == "__main__":
    if not all([faiss_index, faiss_ids, sbert_model, tokenizer, generator_model]):
         print("Exiting because not all components were loaded successfully.")
         sys.exit(1)

    query1 = "Suggest a song similar to Queen - Bohemian Rhapsody"
    recommendation1 = generate_recommendation(query1)
    print(f"\n--- Generated Recommendation (Query 1) ---")
    print(recommendation1)

    print("\n" + "="*50 + "\n") # Separator

    query2 = "What's a good chill R&B song?"
    recommendation2 = generate_recommendation(query2)
    print(f"\n--- Generated Recommendation (Query 2) ---")
    print(recommendation2)

    print("\n--- RAG Script Finished ---")
