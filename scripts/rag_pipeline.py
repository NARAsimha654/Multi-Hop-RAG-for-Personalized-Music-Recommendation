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
BM25_INDEX_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_index.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_id_map.pkl')

# Path to MPD tracks data (needed for track details)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# RAG Model Configuration
# --- Changed to t5-base for potentially better results ---
GENERATOR_MODEL_NAME = 't5-base'
# GENERATOR_MODEL_NAME = 't5-small'

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

try:
    # FAISS
    print("Loading FAISS components...")
    faiss_index = faiss.read_index(MPD_FAISS_INDEX_FILE)
    with open(MPD_FAISS_ID_MAP_FILE, 'rb') as f: faiss_ids = pickle.load(f)
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors).")

    # BM25
    print("Loading BM25 components...")
    # Ensure BM25 files exist, handle FileNotFoundError gracefully
    if not os.path.exists(BM25_INDEX_FILE) or not os.path.exists(BM25_ID_MAP_FILE):
        print(f"Warning: BM25 index or ID map file not found in {INDEX_DIR_BM25}. Proceeding without BM25.")
        bm25_index = None # Set to None if files are missing
        bm25_ids = []
    else:
        with open(BM25_INDEX_FILE, 'rb') as f: bm25_index = pickle.load(f)
        with open(BM25_ID_MAP_FILE, 'rb') as f: bm25_ids = pickle.load(f)
        print(f"BM25 index loaded ({getattr(bm25_index, 'corpus_size', 0)} docs).")

    # SBERT Model (for query embedding)
    print(f"Loading Sentence Transformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT model loaded.")

    # Generator Model (T5)
    print(f"Loading Generator model ({GENERATOR_MODEL_NAME})...")
    # Consider adding cache_dir='./model_cache' to from_pretrained if needed
    tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    generator_model = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL_NAME)
    print("Generator model loaded.")

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

# --- Hybrid Search Function (Updated Scoring) ---
def hybrid_search(query_text, top_k_dense=10, top_k_sparse=10, rerank_k=5, w_dense=0.5, w_sparse=0.5):
    """ Performs hybrid search and returns top K results with scores. """
    print(f"Performing hybrid search for: '{query_text}'")
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
                    score = 1.0 / (1.0 + dist) # Convert distance to similarity (higher is better)
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
                # Check index bounds and if score is positive
                if i < len(bm25_ids) and doc_scores[i] > 0:
                    uri = bm25_ids[i]
                    score = doc_scores[i] # Use raw BM25 score (higher is better)
                    sparse_results[uri] = score
        except Exception as e:
            print(f"  Error during BM25 search: {e}")
    else:
        print("  BM25 index not available, skipping sparse search.")


    combined_scores = {}
    all_retrieved_uris = set(dense_results.keys()) | set(sparse_results.keys())
    print(f"  FAISS returned {len(dense_results)} results.")
    print(f"  BM25 returned {len(sparse_results)} results (with score > 0).")
    print(f"  Total unique results from both retrievers: {len(all_retrieved_uris)}")


    # Combine scores using weights (without complex normalization for now)
    for uri in all_retrieved_uris:
        dense_score = dense_results.get(uri, 0)
        # Use raw BM25 score, default to 0 if not found
        bm25_score = sparse_results.get(uri, 0)

        # Simple weighted sum (adjust weights as needed)
        # Consider normalizing if scales are vastly different later
        final_score = (w_dense * dense_score) + (w_sparse * bm25_score)
        combined_scores[uri] = final_score

    # Sort by final score (descending)
    reranked_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

    return reranked_results[:rerank_k] # Return top K results after reranking


# --- RAG Prompt Formatting ---
def format_rag_prompt(query, retrieved_items):
    """ Formats the prompt for the generator model. """
    context = ""
    if not retrieved_items:
        context = "No relevant tracks found."
    else:
        context_parts = []
        # Limit number of context items included in prompt
        max_context_items = 5
        for i, (uri, score) in enumerate(retrieved_items[:max_context_items]):
            # Fetch details using the pre-loaded lookup
            details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
            # Format context string clearly: "Artist - Title"
            context_parts.append(f"{details[1]} - {details[0]}") # Simplified context
        context = ". ".join(context_parts)

    # --- Experiment with different prompt structures here ---
    # Option 1: Original simple instruction (often poor results)
    # prompt = f"Answer the following query based on the provided context.\n\nQuery: {query}\n\nContext: {context}\n\nAnswer:"

    # Option 2: More direct recommendation task (Selected as default)
    prompt = f"Recommend one music track similar to the query, considering the context.\n\nQuery: {query}\n\nContext: {context}\n\nRecommendation:"

    # Option 3: Question-answering style
    # prompt = f"Based on these tracks: {context}. What is a good song recommendation for someone asking for '{query}'?"

    return prompt

# --- RAG Generation Function ---
def generate_recommendation(query_text, max_length=100):
    """ Performs retrieval, formats prompt, and generates response using T5. """
    # 1. Retrieve relevant tracks using hybrid search
    # Fetch top 5 results after reranking
    retrieved_items = hybrid_search(query_text, top_k_dense=10, top_k_sparse=10, rerank_k=5)

    # 2. Format the prompt for the generator model
    prompt = format_rag_prompt(query_text, retrieved_items)
    print(f"\n--- Generated Prompt ---\n{prompt}\n------------------------")

    # 3. Generate output using the T5 model
    try:
        # Encode the prompt, ensuring max length suitable for the model
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids

        # Generate output IDs using beam search (example)
        # Ensure the model and input_ids are on the same device (e.g., CPU or GPU) if using GPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # generator_model.to(device)
        # input_ids = input_ids.to(device)

        output_ids = generator_model.generate(
            input_ids,
            max_length=max_length, # Max length of the generated output text
            min_length=5,          # Minimum length
            num_beams=5,           # Number of beams for beam search
            early_stopping=True,   # Stop early if end token is generated
            no_repeat_ngram_size=2 # Prevent repeating phrases
            # Other parameters to try: temperature=0.7, top_k=50, top_p=0.9
        )

        # Decode the output IDs to text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Sorry, I encountered an error trying to generate a response."


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure components loaded before running examples
    if not all([faiss_index, faiss_ids, sbert_model, tokenizer, generator_model]):
         print("Exiting because not all components were loaded successfully.")
         if driver: driver.close() # Close Neo4j driver if it was opened
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

    # --- Clean up ---
    # Close Neo4j driver if it was opened (though it's not used in this script)
    # if driver: driver.close() # No 'driver' variable defined here
    print("\n--- RAG Script Finished ---")


