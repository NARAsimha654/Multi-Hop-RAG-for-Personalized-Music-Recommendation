import os
import h5py
import pandas as pd
from tqdm import tqdm

# ---------- HELPER TO EXTRACT INFO FROM A SINGLE FILE ----------

def extract_metadata_from_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            title = f['metadata']['songs'][0][f['metadata']['songs'].dtype.names.index('title')].decode('utf-8')
            artist = f['metadata']['songs'][0][f['metadata']['songs'].dtype.names.index('artist_name')].decode('utf-8')
            song_id = f['metadata']['songs'][0][f['metadata']['songs'].dtype.names.index('song_id')].decode('utf-8')
            artist_id = f['metadata']['songs'][0][f['metadata']['songs'].dtype.names.index('artist_id')].decode('utf-8')
            release = f['metadata']['songs'][0][f['metadata']['songs'].dtype.names.index('release')].decode('utf-8')
            year = int(f['musicbrainz']['songs'][0][f['musicbrainz']['songs'].dtype.names.index('year')])
            return {
                'song_id': song_id,
                'title': title,
                'artist': artist,
                'artist_id': artist_id,
                'release': release,
                'year': year
            }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# ---------- MAIN FUNCTION TO RECURSIVELY SCAN THE FOLDER ----------

def load_msd_metadata(root_dir):
    metadata = []
    for root, dirs, files in os.walk(root_dir):
        for file in tqdm(files):
            if file.endswith(".h5"):
                file_path = os.path.join(root, file)
                song_data = extract_metadata_from_h5(file_path)
                if song_data:
                    metadata.append(song_data)
    return pd.DataFrame(metadata)

# ---------- EXECUTE FROM NOTEBOOK OR SCRIPT ----------

if __name__ == "__main__":
    root_folder = "C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\Dataset\\MillionSongSubset"
    output_csv = "C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_metadata.csv"
    
    df = load_msd_metadata(root_folder)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} songs to {output_csv}")
