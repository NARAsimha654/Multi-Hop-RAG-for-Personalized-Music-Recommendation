import pandas as pd
import numpy as np
import librosa # For audio feature extraction
import os
import glob # For finding files
import time
from tqdm.auto import tqdm # Progress bar
import sys

# --- Configuration ---
# Path to the GTZAN 'genres_original' directory
GTZAN_GENRES_PATH = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\raw\GTZAN\genres_original'

# Output file for extracted features
FEATURES_OUTPUT_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\gtzan_librosa_features.parquet'

# Audio parameters
SAMPLE_RATE = 22050 # Standard sample rate for GTZAN
N_MFCC = 20         # Number of MFCC coefficients to extract
HOP_LENGTH = 512    # Samples between successive frames
N_FFT = 2048        # Length of the FFT window

# --- Feature Extraction Function ---
def extract_features(file_path):
    """
    Extracts various audio features from a given audio file using Librosa.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict or None: A dictionary containing features (mean and std for time-varying ones),
                      or None if the file cannot be processed.
    """
    features = {}
    try:
        # Load audio file - load full duration (GTZAN tracks are ~30s)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True) # Ensure mono

        # --- Extract Features ---
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo[0] if hasattr(tempo, '__len__') else tempo # Handle scalar tempo

        # Chroma Features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_std'] = np.std(chroma_stft)

        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_std'] = np.std(spec_cent)

        # Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        features['spectral_bandwidth_std'] = np.std(spec_bw)

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)

        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        # Calculate mean and std for each MFCC coefficient
        for i in range(N_MFCC):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i,:])
            features[f'mfcc{i+1}_std'] = np.std(mfccs[i,:])

        return features

    except Exception as e:
        # Print error for the specific file but continue processing others
        print(f"\nError processing file {os.path.basename(file_path)}: {e}")
        return None # Indicate failure

# --- Main Processing Logic ---
print(f"Starting feature extraction from: {GTZAN_GENRES_PATH}")
start_time = time.time()

all_features_list = []
error_files = []

# Find all genre subdirectories
genre_folders = [f for f in os.listdir(GTZAN_GENRES_PATH) if os.path.isdir(os.path.join(GTZAN_GENRES_PATH, f))]

if not genre_folders:
    print(f"Error: No genre subdirectories found in {GTZAN_GENRES_PATH}")
    sys.exit(1)

print(f"Found genres: {', '.join(genre_folders)}")

# Iterate through genres and files
for genre in genre_folders:
    genre_path = os.path.join(GTZAN_GENRES_PATH, genre)
    # Find all .wav files (adjust extension if needed, e.g., '*.au')
    audio_files = glob.glob(os.path.join(genre_path, '*.wav'))

    if not audio_files:
        print(f"Warning: No .wav files found in genre folder: {genre}")
        continue

    print(f"\nProcessing genre: {genre} ({len(audio_files)} files)")
    # Use tqdm for progress within each genre
    for file_path in tqdm(audio_files, desc=f"Genre {genre}", unit="file"):
        # Extract features
        features = extract_features(file_path)

        if features:
            # Add filename and genre label
            features['filename'] = os.path.basename(file_path)
            features['genre'] = genre
            all_features_list.append(features)
        else:
            error_files.append(file_path)

# --- Create and Save DataFrame ---
if all_features_list:
    features_df = pd.DataFrame(all_features_list)
    print(f"\nSuccessfully extracted features for {len(features_df)} files.")
    if error_files:
        print(f"Encountered errors processing {len(error_files)} files.")

    print(f"\nSaving features to {FEATURES_OUTPUT_FILE}...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(FEATURES_OUTPUT_FILE), exist_ok=True)
        # Save as Parquet (recommended) or CSV
        if FEATURES_OUTPUT_FILE.endswith('.parquet'):
             features_df.to_parquet(FEATURES_OUTPUT_FILE, index=False)
        else:
             features_df.to_csv(FEATURES_OUTPUT_FILE, index=False)
        print("Features saved successfully.")
    except Exception as e:
        print(f"Error saving features file: {e}")

else:
    print("\nNo features were extracted. Check input path and audio files.")
    if error_files:
        print(f"Encountered errors processing {len(error_files)} files.")

end_time = time.time()
print(f"\n--- Feature Extraction Finished ---")
print(f"Total time: {end_time - start_time:.2f} seconds")

