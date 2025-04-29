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
from operator import itemgetter

# --- LangChain specific imports ---
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

# --- Transformers imports for local pipeline ---
from transformers import pipeline # Removed T5Tokenizer, T5ForConditionalGeneration as pipeline handles it

# --- Neo4j Import ---
from neo4j import GraphDatabase

# --- Configuration ---
# Neo4j Connection Details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Narasimha123" # <<<--- VERIFY YOUR PASSWORD
DB_NAME = "db-1"

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

# Local LLM Configuration
LOCAL_LLM_MODEL_NAME = "google/flan-t5-base"

# RRF Configuration
RRF_K_CONST = 60

# --- Neo4j Driver Setup ---
neo4j_driver = None
print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("Neo4j connection successful!")
except Exception as e:
    print(f"Warning: Error connecting to Neo4j: {e}. Graph context retrieval will be skipped.")
    # Don't exit, allow script to run without graph context if connection fails

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


# --- Helper Functions ---

def tokenize_query(text):
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
    # Return only the URIs for the retriever function
    return [uri for uri, score in reranked_results[:rerank_k]]

# --- NEW: Function to get context from Neo4j ---
def get_graph_context(tx, track_uris):
    """ Queries Neo4j for related info about a list of track URIs """
    if not track_uris:
        return {} # Return empty dict if no URIs provided

    # Query to find artists and co-occurring tracks for the input URIs
    # OPTIONAL: Add genre, SAME_AS links etc. if needed
    cypher_query = """
    UNWIND $uris AS trackUri
    MATCH (t:MpdTrack {track_uri: trackUri})
    // Get Artist info (including MusicBrainz data if available)
    OPTIONAL MATCH (t)-[:BY_ARTIST]->(a:Artist)
    // Get Co-occurring tracks (limit to top N by count for context)
    OPTIONAL MATCH (t)-[co:CO_OCCURS_WITH]-(co_t:MpdTrack)
    WITH t, a, co_t, co.count AS co_count
    ORDER BY co_count DESC
    WITH t, a, collect({name: co_t.track_name, artist: co_t.artist_name, count: co_count})[..3] AS top_cooccurring // Limit to top 3 co-occurring
    // Collect results per input track URI
    RETURN t.track_uri AS input_uri,
           a.name AS artist_name,
           a.artistType AS artist_type,
           a.country AS artist_country,
           top_cooccurring
    """
    results = tx.run(cypher_query, parameters={'uris': track_uris})
    # Structure the results into a dictionary: {input_uri: {details}}
    graph_context_map = {}
    for record in results:
        graph_context_map[record["input_uri"]] = {
            "artist_name": record["artist_name"],
            "artist_type": record["artist_type"],
            "artist_country": record["artist_country"],
            "cooccurring": record["top_cooccurring"] # List of dicts
        }
    return graph_context_map

# --- UPDATED: Function to Format Context for LLM ---
def format_context_for_llm(retrieved_uris, graph_context_map):
    """ Formats retrieved URIs into a string including graph context. """
    if not retrieved_uris: return "No relevant tracks found."

    context_parts = []
    max_context_items = 5 # Limit context length for the prompt
    for i, uri in enumerate(retrieved_uris[:max_context_items]):
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        base_info = f"Track {i+1}: {details[1]} - {details[0]}" # Artist - Title

        # Add Emotion Info
        emo_info = emotion_lookup.get(uri)
        if emo_info:
            parts = []
            vader = emo_info.get('vader_compound'); val = emo_info.get('vad_valence'); aro = emo_info.get('vad_arousal')
            if vader is not None: parts.append(f"Sent:{vader:.2f}")
            if val is not None: parts.append(f"V:{val:.2f}")
            if aro is not None: parts.append(f"A:{aro:.2f}")
            if parts: base_info += f" (Emotions: {', '.join(parts)})"

        # Add Graph Context Info
        graph_info = graph_context_map.get(uri)
        if graph_info:
            graph_parts = []
            if graph_info.get('artist_type'): graph_parts.append(f"Type:{graph_info['artist_type']}")
            if graph_info.get('artist_country'): graph_parts.append(f"Country:{graph_info['artist_country']}")
            # Add top co-occurring track simplified
            if graph_info.get('cooccurring') and len(graph_info['cooccurring']) > 0:
                 co_track = graph_info['cooccurring'][0] # Get the top one
                 graph_parts.append(f"Co-occurs with:{co_track.get('artist', '?')} - {co_track.get('name', '?')}")
            if graph_parts: base_info += f" (Graph: {'; '.join(graph_parts)})"

        context_parts.append(base_info)
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

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful music recommender assistant. Answer the user's question based on the conversation history and the provided context tracks (including their details like emotions or co-occurring songs). If the context isn't sufficient, say so."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context Tracks:\n{context}\n\nQuestion:\n{question}"),
])

# --- Define the RAG chain with Graph Context Retrieval ---

# 1. Retriever gets initial candidate URIs
retriever_runnable = RunnableLambda(lambda x: hybrid_search_rrf(x['question']))

# 2. Function to fetch graph context using Neo4j driver
def fetch_graph_context_for_uris(uris):
    if not neo4j_driver or not uris:
        return {} # Return empty if no driver or no URIs
    with neo4j_driver.session(database=DB_NAME) as session:
        return session.execute_read(get_graph_context, uris)

# 3. Combine retrieved URIs and graph context before formatting
def combine_context(inputs):
    retrieved_uris = inputs['uris']
    graph_context_map = inputs['graph_context']
    return format_context_for_llm(retrieved_uris, graph_context_map)

# 4. Define the chain structure
chain = (
    # Start with the input dictionary {"question": user_query}
    RunnablePassthrough.assign(
        uris=retriever_runnable # Run retriever first to get candidate URIs
    )
    | RunnablePassthrough.assign(
        graph_context=(RunnableLambda(lambda x: x['uris']) | RunnableLambda(fetch_graph_context_for_uris)) # Fetch graph context based on URIs
    )
    | RunnablePassthrough.assign(
        context=RunnableLambda(combine_context) # Format final context string using URIs and graph_context
    )
    | RunnableParallel( # Prepare final inputs for the prompt template
        {"context": itemgetter("context"),
         "question": itemgetter("question"),
         "chat_history": RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
        }
      )
    | prompt_template # Apply the prompt template
    | llm             # Call the local LLM pipeline
    | StrOutputParser() # Parse the output as a string
)


# --- Interactive Query Loop with Memory ---
if __name__ == "__main__":
    print("\n--- Interactive RAG Music Recommender (Local LLM + Memory + KG Context) ---")
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
            inputs = {"question": user_query} # Input for the chain
            response = chain.invoke(inputs)   # Invoke the chain
            end_invoke_time = time.time()

            # Save conversation context
            memory.save_context(inputs, {"output": response})

            print(f"\n--- Recommendation ({end_invoke_time - start_invoke_time:.2f}s) ---")
            print(response)

        except EOFError: print("\nExiting..."); break
        except KeyboardInterrupt: print("\nExiting..."); break
        except Exception as e: print(f"An error occurred during generation: {e}")

    print("\n--- Session Ended ---")
    # Close Neo4j driver connection
    if neo4j_driver:
        neo4j_driver.close()
        print("\nNeo4j driver closed.")

