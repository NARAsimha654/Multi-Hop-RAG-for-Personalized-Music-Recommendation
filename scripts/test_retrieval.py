import numpy as np
import faiss
import os
import pickle # To load the ID map
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Paths to the index and ID map
INDEX_DIR = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\faiss_indices\\'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR, 'mpd_text_index.faiss')
MPD_ID_MAP_FILE = os.path.join(INDEX_DIR, 'mpd_text_index_id_map.pkl')

# Path to MPD tracks data (optional, to display track names for context)
MPD_TRACKS_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_unique_tracks.parquet'


# --- Load FAISS Index, ID Map, and Model ---

print("Loading FAISS index...")
try:
    index = faiss.read_index(MPD_FAISS_INDEX_FILE)
    print(f"Index loaded successfully. Contains {index.ntotal} vectors.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit()

print(f"Loading ID map from {MPD_ID_MAP_FILE}...")
try:
    with open(MPD_ID_MAP_FILE, 'rb') as f:
        mpd_ids_in_order = pickle.load(f) # Load the list of track_uris
    print(f"ID map loaded successfully. Contains {len(mpd_ids_in_order)} IDs.")
    # Basic validation
    if len(mpd_ids_in_order) != index.ntotal:
        print("Warning: Number of IDs in map doesn't match number of vectors in index!")

except FileNotFoundError:
    print(f"Error: ID map file not found at {MPD_ID_MAP_FILE}")
    exit()
except Exception as e:
    print(f"Error loading ID map file: {e}")
    exit()

print("Loading Sentence Transformer model (e.g., all-MiniLM-L6-v2)...")
try:
    # Ensure you load the *exact same* model used for indexing
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
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


# --- Search Function ---

def search_similar_tracks(query_text, top_k=5):
    """
    Searches the FAISS index for tracks similar to the query_text.

    Args:
        query_text (str): The text query (e.g., "artist - title").
        top_k (int): The number of similar tracks to return.

    Returns:
        list: A list of tuples, where each tuple is (track_uri, distance).
              Returns None if search fails.
    """
    print(f"\nSearching for top {top_k} tracks similar to: '{query_text}'")
    try:
        # 1. Get the embedding for the query text
        query_embedding = model.encode([query_text]) # Pass as a list

        # Ensure it's float32 and 2D array for FAISS search
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
             query_embedding = np.expand_dims(query_embedding, axis=0) # Make it (1, dimension)

        # 2. Search the index
        # index.search returns distances (D) and indices (I) of neighbours
        distances, indices = index.search(query_embedding, top_k)

        # 3. Map indices back to track URIs
        results = []
        if len(indices) > 0:
            for i in range(top_k):
                result_index = indices[0][i]
                # Check if index is valid
                if result_index >= 0 and result_index < len(mpd_ids_in_order):
                    track_uri = mpd_ids_in_order[result_index]
                    distance = distances[0][i]
                    results.append((track_uri, distance))
                else:
                    print(f"  Warning: Invalid index {result_index} found at position {i}.")
        return results

    except Exception as e:
        print(f"An error occurred during search: {e}")
        return None

# --- Example Usage ---

# Example queries:
query1 = "Queen - Bohemian Rhapsody"
query2 = "relaxing acoustic guitar music"
query3 = "upbeat electronic dance track"

results1 = search_similar_tracks(query1, top_k=5)
if results1:
    print("Results:")
    for uri, dist in results1:
        details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
        print(f"  URI: {uri}, Distance: {dist:.4f}, Name: {details[0]}, Artist: {details[1]}")

results2 = search_similar_tracks(query2, top_k=5)
if results2:
    print("Results:")
    for uri, dist in results2:
         details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
         print(f"  URI: {uri}, Distance: {dist:.4f}, Name: {details[0]}, Artist: {details[1]}")

results3 = search_similar_tracks(query3, top_k=5)
if results3:
    print("Results:")
    for uri, dist in results3:
         details = track_details_lookup.get(uri, ('Unknown Track', 'Unknown Artist'))
         print(f"  URI: {uri}, Distance: {dist:.4f}, Name: {details[0]}, Artist: {details[1]}")