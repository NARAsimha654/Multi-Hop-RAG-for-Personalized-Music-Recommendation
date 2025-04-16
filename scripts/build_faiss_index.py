import numpy as np
import faiss
import os
import pickle # Or use json for saving the ID list

# --- Configuration ---
# Input embedding file
MPD_EMBEDDING_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\embeddings\mpd_text_embeddings.npz'

# Output directory for FAISS index and ID map
INDEX_DIR = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\faiss_indices\\'
MPD_FAISS_INDEX_FILE = os.path.join(INDEX_DIR, 'mpd_text_index.faiss')
MPD_ID_MAP_FILE = os.path.join(INDEX_DIR, 'mpd_text_index_id_map.pkl') # Using pickle to save list

# Ensure output directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Load Embeddings and IDs ---
print(f"Loading embeddings and IDs from {MPD_EMBEDDING_FILE}...")
try:
    data = np.load(MPD_EMBEDDING_FILE)
    mpd_ids = data['ids']       # Array of track_uris
    mpd_embeddings = data['embeddings'] # Embeddings as numpy array
    print(f"Loaded {len(mpd_ids)} IDs and {mpd_embeddings.shape[0]} embeddings.")
    # Ensure embeddings are float32
    if mpd_embeddings.dtype != np.float32:
        print("Converting embeddings to float32...")
        mpd_embeddings = mpd_embeddings.astype(np.float32)

except FileNotFoundError:
    print(f"Error: Embedding file not found at {MPD_EMBEDDING_FILE}")
    exit()
except Exception as e:
    print(f"Error loading .npz file: {e}")
    exit()

# --- Build FAISS Index ---
if mpd_embeddings.shape[0] > 0:
    dimension = mpd_embeddings.shape[1] # Should be 384
    print(f"\nBuilding FAISS index (IndexFlatL2) with dimension {dimension}...")

    # Create the index - IndexFlatL2 uses Euclidean distance
    index = faiss.IndexFlatL2(dimension)

    # Check if the index is trained (IndexFlatL2 doesn't require training)
    print(f"Index is trained: {index.is_trained}")

    # Add the vectors to the index
    print(f"Adding {mpd_embeddings.shape[0]} vectors to the index...")
    index.add(mpd_embeddings)

    # Print total vectors in the index
    print(f"Total vectors in index: {index.ntotal}")

    # --- Save Index and ID Map ---
    print(f"\nSaving FAISS index to {MPD_FAISS_INDEX_FILE}...")
    faiss.write_index(index, MPD_FAISS_INDEX_FILE)
    print("Index saved.")

    print(f"Saving ID map to {MPD_ID_MAP_FILE}...")
    # Save the list of IDs (in the same order embeddings were added)
    with open(MPD_ID_MAP_FILE, 'wb') as f:
        pickle.dump(list(mpd_ids), f) # Save as a list
    print("ID map saved.")

else:
    print("No embeddings found to build the index.")

print("\n--- FAISS Index Building Complete ---")