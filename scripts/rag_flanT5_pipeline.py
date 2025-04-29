# Import necessary libraries
import numpy as np
import faiss
import os
import pickle
import re
import pandas as pd
import time
from operator import itemgetter # <-- ADD THIS IMPORT
import sys
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi # Or the specific BM25 class you used
from tqdm.auto import tqdm # Progress bar (optional for loading)

# --- LangChain specific imports ---
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory # Import memory

# --- Transformers imports for local pipeline ---
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# --- Configuration ---
# Paths to indices and maps
INDEX_DIR_FAISS = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\faiss_indices\\'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index.faiss')
MPD_FAISS_ID_MAP_FILE = os.path.join(INDEX_DIR_FAISS, 'mpd_text_index_id_map.pkl')

INDEX_DIR_BM25 = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\bm25_indices\\'
# --- Use the paths for your ~27k index files ---
BM25_INDEX_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_index_27k.pkl')
BM25_ID_MAP_FILE = os.path.join(INDEX_DIR_BM25, 'mpd_lyrics_bm25_id_map_27k.pkl')

# Path to MPD tracks data (needed for track details)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'

# Path to Emotion Features
EMOTION_FEATURES_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_emotion_features_sample_vad.parquet'

# --- Local LLM Configuration ---
LOCAL_LLM_MODEL_NAME = "google/flan-t5-base"

# RRF Configuration
RRF_K_CONST = 60

# --- Load All Retrieval Components ---
print("--- Loading Retrieval Components ---")
# (Loading logic remains the same)
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
    # (Implementation is the same as the previous version)
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
    # else: print("  BM25 index not available, skipping sparse search.")
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
    return [uri for uri, score in reranked_results[:rerank_k]] # Return only URIs

def format_context_for_llm(retrieved_uris):
    """ Formats retrieved URIs into a string with details and emotions. """
    if not retrieved_uris: return "No relevant tracks found."
    context_parts = []
    max_context_items = 5
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


# --- Setup LangChain with Local LLM and Memory ---
print(f"\n--- Initializing Local LLM Pipeline ({LOCAL_LLM_MODEL_NAME}) ---")
try:
    hf_pipeline = pipeline(
        "text2text-generation", model=LOCAL_LLM_MODEL_NAME, tokenizer=LOCAL_LLM_MODEL_NAME,
        max_length=100, device=-1 # Use CPU
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("Local LLM pipeline initialized successfully.")
except Exception as e:
     print(f"Error initializing local LLM pipeline: {e}"); sys.exit(1)

# --- Initialize Conversation Memory ---
# return_messages=True is important for chat models/prompts
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template with placeholders for history and context
# Using MessagesPlaceholder for chat history
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful music recommender assistant. Answer the user's question based on the conversation history and the provided context tracks. If the context isn't sufficient, say so."),
    MessagesPlaceholder(variable_name="chat_history"), # Where history messages go
    ("human", "Context Tracks:\n{context}\n\nQuestion:\n{question}"), # Combine context and question for human turn
])

# Define the RAG chain components
# Retriever runnable remains the same
retriever_runnable = RunnableLambda(lambda x: hybrid_search_rrf(x['question']))

# Runnable to load history (needed because memory isn't directly part of the input dict)
load_memory = RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")

# Define the core RAG steps using RunnableParallel for clarity
rag_steps = RunnableParallel(
    # Pass question through, retrieve context, format context
    {"context": retriever_runnable | RunnableLambda(format_context_for_llm),
     "question": RunnablePassthrough(), # Pass original question dict through
     "chat_history": load_memory # Load history here
     }
)

# Define the full chain
chain = (
    rag_steps # Prepare context, question, and history
    | prompt_template # Apply the prompt template
    | llm             # Call the local LLM pipeline
    | StrOutputParser() # Parse the output as a string
)


# --- Interactive Query Loop with Memory ---
if __name__ == "__main__":
    print("\n--- Interactive RAG Music Recommender (Local LLM + Memory) ---")
    print(f"Using model: {LOCAL_LLM_MODEL_NAME}")
    print("Enter your music query (e.g., 'sad acoustic songs', 'artist - title')")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_query = input("\nEnter query: ")
            if user_query.lower() in ['quit', 'exit']: print("Exiting..."); break
            if not user_query: continue

            print("Generating recommendation...")
            start_invoke_time = time.time()

            # Prepare inputs for the chain (now includes the question)
            inputs = {"question": user_query}

            # Invoke the chain - it will internally use the memory
            response = chain.invoke(inputs)
            end_invoke_time = time.time()

            # --- Save conversation context ---
            # Manually save the human query and AI response to memory
            memory.save_context(inputs, {"output": response})
            # print("\nMemory:", memory.load_memory_variables({})) # Optional: print memory state

            print(f"\n--- Recommendation ({end_invoke_time - start_invoke_time:.2f}s) ---")
            print(response) # Print the final response from the LLM

        except EOFError: print("\nExiting..."); break
        except KeyboardInterrupt: print("\nExiting..."); break
        except Exception as e: print(f"An error occurred during generation: {e}")

    print("\n--- Session Ended ---")
