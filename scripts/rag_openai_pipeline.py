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
from tqdm.auto import tqdm # Progress bar (optional for loading)

# LangChain specific imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables from .env file ---
# Needs: pip install python-dotenv
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file into environment
print("Attempted to load variables from .env file.")
# --- End Load Environment Variables ---


# --- Debug: Check Environment Variable Immediately ---
# Get the key from the environment RIGHT NOW (after loading .env)
api_key_from_env = os.getenv('OPENAI_API_KEY')
print(f"--- Debug Start ---")
# Mask key for printing safety
print(f"Value of OPENAI_API_KEY as seen by Python: {'*' * (len(api_key_from_env) - 4) + api_key_from_env[-4:] if api_key_from_env else None}")
print(f"Is the key found? {'Yes' if api_key_from_env else 'No'}")
print(f"--- End Debug ---")
# --- End Debug ---


# --- Configuration ---
# Paths to indices and maps
INDEX_DIR_FAISS = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\faiss_indices\\'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index.faiss')
MPD_FAISS_ID_MAP_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index_id_map.pkl')

INDEX_DIR_BM25 = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\bm25_indices\\'
# --- IMPORTANT: Use the paths for your ~27k index files ---
BM25_INDEX_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_index_27k.pkl') # UPDATE FILENAME
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_id_map_27k.pkl') # UPDATE FILENAME

# Path to MPD tracks data (needed for track details)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# Path to Emotion Features (Use the one generated from ~27k lyrics)
EMOTION_FEATURES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_emotion_features_sample_vad.parquet' # UPDATE FILENAME

# LangChain / LLM Configuration
LLM_MODEL_NAME = "gpt-3.5-turbo" # Or "gpt-4" etc. Requires OPENAI_API_KEY env var

# RRF Configuration
RRF_K_CONST = 60


# --- Load All Retrieval Components ---
# (Loading logic remains the same as before)
print("--- Loading Retrieval Components ---")
faiss_index = None; faiss_ids = []; bm25_index = None; bm25_ids = []
sbert_model = None; track_details_lookup = {}; emotion_lookup = {}
try:
    print("Loading FAISS components...")
    faiss_index = faiss.read_index(MPD_FAISS_INDEX_FILE)
    with open(MPD_FAISS_ID_MAP_FILE, 'rb') as f: faiss_ids = pickle.load(f)
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors).")

    print("Loading BM25 components...")
    if os.path.exists(BM25_INDEX_FILE) and os.path.exists(BM25_ID_MAP_FILE):
        with open(BM25_INDEX_FILE, 'rb') as f: bm25_index = pickle.load(f)
        with open(BM25_ID_MAP_FILE, 'rb') as f: bm25_ids = pickle.load(f)
        print(f"BM25 index loaded ({getattr(bm25_index, 'corpus_size', 0)} docs).")
    else: print(f"Warning: BM25 files not found. Sparse retrieval skipped.")

    print(f"Loading Sentence Transformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT model loaded.")

    # NOTE: LLM Initialization moved after key check below

    print("Loading track details...")
    if MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(MPD_TRACKS_FILE)
    else: mpd_tracks_df = pd.read_csv(MPD_TRACKS_FILE)
    track_details_lookup = mpd_tracks_df.set_index('track_uri')[['track_name', 'artist_name']].apply(tuple, axis=1).to_dict()
    print("Track details loaded.")
    print("Loading Emotion Features...");
    if os.path.exists(EMOTION_FEATURES_FILE):
        if EMOTION_FEATURES_FILE.endswith('.parquet'): emotion_df = pd.read_parquet(EMOTION_FEATURES_FILE)
        else: emotion_df = pd.read_csv(EMOTION_FEATURES_FILE)
        emotion_lookup = emotion_df.set_index('track_uri').to_dict('index')
        print(f"Emotion features loaded for {len(emotion_lookup)} tracks.")
    else: print(f"Warning: Emotion features file not found.")

    print("\n--- All retrieval components loaded successfully ---")

except FileNotFoundError as fnf_error: print(f"Error loading file: {fnf_error}"); sys.exit(1)
except Exception as e: print(f"Error loading components: {e}"); sys.exit(1)


# --- Helper Functions (Define BEFORE use in Chain) ---

def tokenize_query(text):
    """ Basic tokenizer for BM25 query """
    processed = text.lower(); processed = re.sub(r'[^\w\s]', '', processed); return processed.split()

def hybrid_search_rrf(query_text, top_k_dense=50, top_k_sparse=50, rerank_k=10, rrf_k_const=60):
    """ Performs hybrid search and returns top K URIs using RRF. """
    # print(f"\nPerforming hybrid search (RRF) for: '{query_text}'") # Optional print
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
        except Exception as e: print(f"  Error during BM25 search: {e}")
    # else: print("  BM25 index not available, skipping sparse search.") # Reduced verbosity
    rrf_scores = {}
    all_retrieved_uris = set(dense_ranked_list) | set(sparse_ranked_list)
    for uri in all_retrieved_uris:
        score = 0.0
        try: rank = dense_ranked_list.index(uri) + 1; score += 1.0 / (rrf_k_const + rank)
        except ValueError: pass
        try: rank = sparse_ranked_list.index(uri) + 1; score += 1.0 / (rrf_k_const + rank)
        except ValueError: pass
        rrf_scores[uri] = score
    reranked_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    # Return only the URIs for the retriever function
    return [uri for uri, score in reranked_results[:rerank_k]]

def format_context_for_llm(retrieved_uris):
    """ Formats retrieved URIs into a string with details and emotions. """
    if not retrieved_uris: return "No relevant tracks found."
    context_parts = []
    max_context_items = 5 # Limit context length for the prompt
    for i, uri in enumerate(retrieved_uris[:max_context_items]):
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        emo_info = emotion_lookup.get(uri)
        emotion_str = ""
        if emo_info:
            parts = []
            vader_score = emo_info.get('vader_compound')
            valence = emo_info.get('vad_valence')
            arousal = emo_info.get('vad_arousal')
            if vader_score is not None: parts.append(f"Sentiment:{vader_score:.2f}")
            if valence is not None: parts.append(f"V:{valence:.2f}")
            if arousal is not None: parts.append(f"A:{arousal:.2f}")
            if parts: emotion_str = f" (Emotions: {', '.join(parts)})"
        context_parts.append(f"Track {i+1}: {details[1]} - {details[0]}{emotion_str}")
    return ". ".join(context_parts)


# --- Setup LangChain ---
# Check if the key was loaded successfully from the .env file
if not api_key_from_env:
    print("Error: OPENAI_API_KEY was NOT found by Python after attempting to load .env file.")
    print("Please ensure the .env file exists in the project root and contains the key.")
    sys.exit(1) # Exit if key is definitely missing

# Initialize the LLM - Now relying on LangChain to find the key from the environment
try:
    print(f"Initializing ChatOpenAI with model '{LLM_MODEL_NAME}' (using environment variable)...")
    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.7)
    print("ChatOpenAI initialized successfully.")
except Exception as e:
     print(f"Error initializing ChatOpenAI using environment variable: {e}")
     print("Ensure your API key in the .env file is correct and you have internet access.")
     sys.exit(1)


# Define the prompt template
template = """
You are a helpful music recommender assistant.
Answer the user's question based only on the provided context.
If the context doesn't contain the answer, say you don't have enough information but try to suggest something plausible based on the query type if possible.
Do not make up information not present in the context.

Context:
{context}

Question:
{question}

Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Define the RAG chain using LangChain Expression Language (LCEL)
# --- Ensure helper functions are defined before this point ---
retriever_runnable = RunnableLambda(lambda x: hybrid_search_rrf(x['question']))

chain = (
    RunnablePassthrough.assign(
        # Pass the output of the retriever_runnable to format_context_for_llm
        context=(retriever_runnable | RunnableLambda(format_context_for_llm))
    )
    | prompt_template # Apply the prompt template
    | llm             # Call the language model
    | StrOutputParser() # Parse the output as a string
)


# --- Interactive Query Loop ---
if __name__ == "__main__":
    print("\n--- Interactive RAG Music Recommender (LangChain + OpenAI) ---")
    print("Enter your music query (e.g., 'sad acoustic songs', 'artist - title')")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_query = input("\nEnter query: ")
            if user_query.lower() in ['quit', 'exit']: print("Exiting..."); break
            if not user_query: continue

            print("Generating recommendation...")
            start_invoke_time = time.time()
            # Invoke the chain with the user query
            response = chain.invoke({"question": user_query})
            end_invoke_time = time.time()

            print(f"\n--- Recommendation ({end_invoke_time - start_invoke_time:.2f}s) ---")
            print(response) # Print the final response from the LLM

        except EOFError: print("\nExiting..."); break
        except KeyboardInterrupt: print("\nExiting..."); break
        except Exception as e: print(f"An error occurred: {e}")

    print("\n--- Session Ended ---")

