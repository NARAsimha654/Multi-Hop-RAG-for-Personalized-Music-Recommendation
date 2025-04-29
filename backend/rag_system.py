# rag_system.py

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
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from operator import itemgetter

# LangChain specific imports
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage # For manual history formatting

# Transformers imports for local pipeline
from transformers import pipeline

# Neo4j Import
from neo4j import GraphDatabase

# --- Configuration Class (Optional but good practice) ---
class RAGConfig:
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

    # Neo4j Connection Details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Narasimha123" # <<<--- VERIFY YOUR PASSWORD
    DB_NAME = "db-1"


# --- RAG System Class ---
class MusicRAGSystem:
    def __init__(self, config: RAGConfig):
        """Initializes the RAG system by loading all components."""
        self.config = config
        print("--- Initializing Music RAG System ---")
        self._load_components()
        self._setup_langchain()
        print("--- RAG System Initialized Successfully ---")

    def _load_components(self):
        """Loads all necessary data, models, and indices."""
        self.faiss_index = None; self.faiss_ids = []; self.bm25_index = None; self.bm25_ids = []
        self.sbert_model = None; self.track_details_lookup = {}; self.emotion_lookup = {}
        self.neo4j_driver = None

        try:
            # Neo4j Driver
            print(f"Connecting to Neo4j at {self.config.NEO4J_URI}...")
            self.neo4j_driver = GraphDatabase.driver(self.config.NEO4J_URI, auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD))
            self.neo4j_driver.verify_connectivity()
            print("Neo4j connection successful!")

            # FAISS
            print("Loading FAISS components...")
            self.faiss_index = faiss.read_index(self.config.MPD_FAISS_INDEX_FILE)
            with open(self.config.MPD_FAISS_ID_MAP_FILE, 'rb') as f: self.faiss_ids = pickle.load(f)
            print(f"FAISS index loaded ({self.faiss_index.ntotal} vectors).")

            # BM25
            print("Loading BM25 components...")
            if os.path.exists(self.config.BM25_INDEX_FILE) and os.path.exists(self.config.BM25_ID_MAP_FILE):
                with open(self.config.BM25_INDEX_FILE, 'rb') as f: self.bm25_index = pickle.load(f)
                with open(self.config.BM25_ID_MAP_FILE, 'rb') as f: self.bm25_ids = pickle.load(f)
                print(f"BM25 index loaded ({getattr(self.bm25_index, 'corpus_size', 0)} docs).")
            else: print(f"Warning: BM25 files not found. Sparse retrieval skipped.")

            # SBERT Model
            print(f"Loading Sentence Transformer model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SBERT model loaded.")

            # Track Details Lookup
            print("Loading track details...")
            if self.config.MPD_TRACKS_FILE.endswith('.parquet'): mpd_tracks_df = pd.read_parquet(self.config.MPD_TRACKS_FILE)
            else: mpd_tracks_df = pd.read_csv(self.config.MPD_TRACKS_FILE)
            self.track_details_lookup = mpd_tracks_df.set_index('track_uri')[['track_name', 'artist_name']].apply(tuple, axis=1).to_dict()
            print("Track details loaded.")

            # Emotion Features Lookup
            print("Loading Emotion Features...");
            if os.path.exists(self.config.EMOTION_FEATURES_FILE):
                if self.config.EMOTION_FEATURES_FILE.endswith('.parquet'): emotion_df = pd.read_parquet(self.config.EMOTION_FEATURES_FILE)
                else: emotion_df = pd.read_csv(self.config.EMOTION_FEATURES_FILE)
                self.emotion_lookup = emotion_df.set_index('track_uri').to_dict('index')
                print(f"Emotion features loaded for {len(self.emotion_lookup)} tracks.")
            else: print(f"Warning: Emotion features file not found.")

            # Local LLM Pipeline
            print(f"Initializing Local LLM Pipeline ({self.config.LOCAL_LLM_MODEL_NAME})...")
            # Note: Consider loading model to GPU if available (device=0)
            hf_pipeline = pipeline(
                "text2text-generation", model=self.config.LOCAL_LLM_MODEL_NAME, tokenizer=self.config.LOCAL_LLM_MODEL_NAME,
                max_length=100, device=-1 # Use CPU
            )
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
            print("Local LLM pipeline initialized successfully.")

        except FileNotFoundError as fnf_error: print(f"Error loading file: {fnf_error}"); raise
        except Exception as e: print(f"Error loading components: {e}"); raise

    def _run_neo4j_query(self, tx, query, parameters=None):
        """ Helper to run read queries against Neo4j """
        result = tx.run(query, parameters)
        return [record for record in result]

    def _get_graph_context(self, tx, track_uris):
        """ Queries Neo4j for related info about a list of track URIs """
        if not track_uris: return {}
        cypher_query = """
        UNWIND $uris AS trackUri
        MATCH (t:MpdTrack {track_uri: trackUri})
        OPTIONAL MATCH (t)-[:BY_ARTIST]->(a:Artist)
        OPTIONAL MATCH (t)-[co:CO_OCCURS_WITH]-(co_t:MpdTrack)
        WITH t, a, co_t, co.count AS co_count ORDER BY co_count DESC
        WITH t, a, collect({name: co_t.track_name, artist: co_t.artist_name, count: co_count})[..3] AS top_cooccurring
        RETURN t.track_uri AS input_uri, a.name AS artist_name,
               a.artistType AS artist_type, a.country AS artist_country, top_cooccurring
        """
        results = self._run_neo4j_query(tx, cypher_query, parameters={'uris': track_uris})
        graph_context_map = {
            record["input_uri"]: {
                "artist_name": record["artist_name"], "artist_type": record["artist_type"],
                "artist_country": record["artist_country"], "cooccurring": record["top_cooccurring"]
            } for record in results
        }
        return graph_context_map

    def _fetch_graph_context_for_uris(self, uris):
        """ Wrapper to execute graph context query in a session """
        if not self.neo4j_driver or not uris: return {}
        with self.neo4j_driver.session(database=self.config.DB_NAME) as session:
            return session.execute_read(self._get_graph_context, uris)

    def _tokenize_query(self, text):
        processed = text.lower(); processed = re.sub(r'[^\w\s]', '', processed); return processed.split()

    def _hybrid_search_rrf(self, query_text, top_k_dense=50, top_k_sparse=50, rerank_k=10):
        """ Performs hybrid search and returns top K URIs using RRF. """
        dense_ranked_list = []
        try:
            query_embedding = self.sbert_model.encode([query_text])
            if query_embedding.dtype != np.float32: query_embedding = query_embedding.astype(np.float32)
            if query_embedding.ndim == 1: query_embedding = np.expand_dims(query_embedding, axis=0)
            distances, indices = self.faiss_index.search(query_embedding, top_k_dense)
            if len(indices) > 0:
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    if idx >= 0 and idx < len(self.faiss_ids): dense_ranked_list.append(self.faiss_ids[idx])
        except Exception as e: print(f"  Error during FAISS search: {e}")

        sparse_ranked_list = []
        if self.bm25_index and self.bm25_ids:
            try:
                tokenized_query = self._tokenize_query(query_text)
                doc_scores = self.bm25_index.get_scores(tokenized_query)
                sorted_indices = np.argsort(doc_scores)[::-1]
                for i in sorted_indices:
                    if i < len(self.bm25_ids) and doc_scores[i] > 0: sparse_ranked_list.append(self.bm25_ids[i])
                    if len(sparse_ranked_list) >= top_k_sparse: break
            except Exception as e: print(f"  Error during BM25 search: {e}")

        rrf_scores = {}
        all_retrieved_uris = set(dense_ranked_list) | set(sparse_ranked_list)
        for uri in all_retrieved_uris:
            score = 0.0
            try: rank = dense_ranked_list.index(uri) + 1; score += 1.0 / (self.config.RRF_K_CONST + rank)
            except ValueError: pass
            try: rank = sparse_ranked_list.index(uri) + 1; score += 1.0 / (self.config.RRF_K_CONST + rank)
            except ValueError: pass
            rrf_scores[uri] = score
        reranked_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [uri for uri, score in reranked_results[:rerank_k]]

    def _format_context_for_llm(self, retrieved_uris, graph_context_map):
        """ Formats retrieved URIs into a string with details, emotions and graph context. """
        if not retrieved_uris: return "No relevant tracks found."
        context_parts = []
        max_context_items = 5
        for i, uri in enumerate(retrieved_uris[:max_context_items]):
            details = self.track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
            base_info = f"Track {i+1}: {details[1]} - {details[0]}"
            emo_info = self.emotion_lookup.get(uri)
            if emo_info:
                parts = []
                vader = emo_info.get('vader_compound'); val = emo_info.get('vad_valence'); aro = emo_info.get('vad_arousal')
                if vader is not None: parts.append(f"Sentiment:{vader:.2f}")
                if val is not None: parts.append(f"V:{val:.2f}")
                if aro is not None: parts.append(f"A:{aro:.2f}")
                if parts: base_info += f" (Emotions: {', '.join(parts)})"
            graph_info = graph_context_map.get(uri)
            if graph_info:
                graph_parts = []
                if graph_info.get('artist_type'): graph_parts.append(f"Type:{graph_info['artist_type']}")
                if graph_info.get('artist_country'): graph_parts.append(f"Country:{graph_info['artist_country']}")
                if graph_info.get('cooccurring') and len(graph_info['cooccurring']) > 0:
                     co_track = graph_info['cooccurring'][0]
                     graph_parts.append(f"Co-occurs with:{co_track.get('artist', '?')} - {co_track.get('name', '?')}")
                if graph_parts: base_info += f" (Graph: {'; '.join(graph_parts)})"
            context_parts.append(base_info)
        return ". ".join(context_parts)

    def _setup_langchain(self):
        """Sets up the LangChain RAG chain."""
        # Initialize Conversation Memory (will be managed per request in API)
        # For direct use, create one here
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful music recommender assistant. Answer the user's question based on the conversation history and the provided context tracks (including their details like emotions or co-occurring songs). If the context isn't sufficient, say so."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context Tracks:\n{context}\n\nQuestion:\n{question}"),
        ])

        # Define the RAG chain components
        # Note: Using methods bound to 'self' within lambdas
        retriever_runnable = RunnableLambda(lambda x: self._hybrid_search_rrf(x['question']))
        fetch_graph_context_runnable = RunnableLambda(lambda uris: self._fetch_graph_context_for_uris(uris))
        format_context_runnable = RunnableLambda(lambda inputs: self._format_context_for_llm(inputs['uris'], inputs['graph_context']))
        load_memory_runnable = RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history")

        # Define the full chain
        self.chain = (
            RunnablePassthrough.assign(uris=retriever_runnable)
            | RunnablePassthrough.assign(graph_context=(RunnableLambda(lambda x: x['uris']) | fetch_graph_context_runnable))
            | RunnablePassthrough.assign(context=format_context_runnable)
            | RunnableParallel(
                {"context": itemgetter("context"),
                 "question": itemgetter("question"),
                 "chat_history": load_memory_runnable}
              )
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        print("LangChain RAG chain setup complete.")

    def generate_response(self, user_query: str, chat_history: list = None):
        """
        Generates a response for a given query, managing chat history.

        Args:
            user_query (str): The user's input query.
            chat_history (list): List of previous BaseMessage objects (optional).

        Returns:
            str: The generated response from the LLM.
        """
        # --- Manage Memory ---
        # Clear memory for each new request if not passing history,
        # OR load provided history. For API, history should be passed.
        self.memory.clear() # Clear memory for stateless operation in API context
        if chat_history:
            # Manually load history if provided (e.g., from API request)
            # Assumes chat_history is list of {'role': 'human'/'ai', 'content': '...'}
            loaded_messages = []
            for msg in chat_history:
                 if msg.get('role') == 'human':
                     loaded_messages.append(HumanMessage(content=msg['content']))
                 elif msg.get('role') == 'ai':
                      loaded_messages.append(AIMessage(content=msg['content']))
            self.memory.chat_memory.messages = loaded_messages # Directly set messages

        print(f"Generating response for query: '{user_query}'")
        start_invoke_time = time.time()
        inputs = {"question": user_query} # Input dictionary for the chain
        try:
            response = self.chain.invoke(inputs)
            end_invoke_time = time.time()
            print(f"LLM generation took {end_invoke_time - start_invoke_time:.2f}s")

            # --- IMPORTANT for API ---
            # In a real API, you wouldn't save context here.
            # The calling function would handle history persistence.
            # self.memory.save_context(inputs, {"output": response})

            return response
        except Exception as e:
            print(f"An error occurred during chain invocation: {e}")
            return "Sorry, I encountered an error generating a response."

    def close_neo4j(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            print("Closing Neo4j driver connection...")
            self.neo4j_driver.close()
            self.neo4j_driver = None


# --- Example Usage (if run directly) ---
if __name__ == "__main__":
    config = RAGConfig()
    try:
        rag_system = MusicRAGSystem(config)

        print("\n--- Interactive Test ---")
        print("Type 'quit' or 'exit' to stop.")
        # Simple history management for direct testing
        history = []
        while True:
            query = input("\nEnter query: ")
            if query.lower() in ['quit', 'exit']: break
            if not query: continue

            # Pass history (as list of dicts) - not implemented here yet
            # For direct testing, memory state persists within the object
            response = rag_system.generate_response(query)
            print(f"\n--- Response ---")
            print(response)
            # Manually update history for next turn in this test loop
            # rag_system.memory.save_context({"question": query}, {"output": response})

    except Exception as e:
        print(f"Failed to initialize or run RAG system: {e}")
    finally:
        # Ensure Neo4j connection is closed if system was initialized
        if 'rag_system' in locals() and rag_system.neo4j_driver:
            rag_system.close_neo4j()

