import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_gtzan(dataset_path, save_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"🔍 Processing GTZAN dataset at {dataset_path}...")

    genres, audio_paths, sample_rates, durations, audio_data = [], [], [], [], []
    genre_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    print(f"Found {len(genre_folders)} genre folders")
    for genre in tqdm(genre_folders):
        genre_dir = os.path.join(dataset_path, genre)
        audio_files = [f for f in os.listdir(genre_dir) if f.endswith(('.wav', '.au'))]
        print(f"  Genre '{genre}': {len(audio_files)} audio files")
        for audio_file in audio_files:
            file_path = os.path.join(genre_dir, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                genres.append(genre)
                audio_paths.append(file_path)
                sample_rates.append(sr)
                durations.append(duration)
                audio_data.append(y)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

    df = pd.DataFrame({
        'genre': genres,
        'audio_path': audio_paths,
        'sample_rate': sample_rates,
        'duration': durations
    })

    ds = {
        'audio': audio_data,
        'genre': genres,
        'sample_rate': sample_rates,
        'path': audio_paths
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n✅ Saved metadata to: {save_path}")

    print("\n--- GTZAN Dataset Summary ---")
    print(f"Total samples: {len(df)}")
    print(f"Available genres: {sorted(df['genre'].unique())}")
    genre_counts = df['genre'].value_counts().sort_index()
    print("\nSamples per genre:")
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count}")
    if durations:
        print(f"\nDuration stats: avg={np.mean(durations):.2f}s, min={np.min(durations):.2f}s, max={np.max(durations):.2f}s")

    return df, ds

def visualize_genre_distribution(df, save_path):
    plt.figure(figsize=(12, 6))
    genre_counts = df['genre'].value_counts().sort_index()
    genre_counts.plot(kind='bar', color='skyblue')
    plt.title('🎵 GTZAN Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Saved plot to: {save_path}")
    plt.show()

def extract_audio_sample(ds, index=0):
    print(f"\n🎧 Extracting sample {index}...")
    if index < len(ds['audio']):
        audio = ds['audio'][index]
        genre = ds['genre'][index]
        print(f"Genre: {genre}")
        print(f"Audio shape: {audio.shape}")
        print(f"Audio dtype: {audio.dtype}")
        print(f"First 10 values: {audio[:10]}")
        return audio
    else:
        print(f"Index {index} out of range!")
        return None

def extract_audio_features(df, n_samples=None, save_path=None):
    if n_samples is None or n_samples > len(df):
        sample_df = df
        print(f"\n🎛️ Extracting features for all {len(df)} samples...")
    else:
        sample_df = df.sample(n=n_samples)
        print(f"\n🎛️ Extracting features for {n_samples} samples...")

    features_list = []
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        try:
            y, sr = librosa.load(row['audio_path'], sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_vars = np.var(mfccs, axis=1)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])

            feature_dict = {
                'genre': row['genre'],
                'file_path': row['audio_path'],
                'tempo': tempo,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zcr
            }
            for i, mfcc_val in enumerate(mfcc_means):
                feature_dict[f'mfcc_mean_{i}'] = mfcc_val
            for i, mfcc_val in enumerate(mfcc_vars):
                feature_dict[f'mfcc_var_{i}'] = mfcc_val

            features_list.append(feature_dict)
        except Exception as e:
            print(f"⚠️ Error extracting features from {row['audio_path']}: {e}")

    features_df = pd.DataFrame(features_list)
    if save_path and not features_df.empty:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        features_df.to_csv(save_path, index=False)
        print(f"✅ Saved features to: {save_path}")

    return features_df

if __name__ == "__main__":
    DATASET_PATH = r"C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\Dataset\GTZAN\genres_original"
    METADATA_PATH = r"C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\gtzan_metadata.csv"
    FEATURES_PATH = r"C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\gtzan_features.csv"
    PLOT_PATH = r"C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\results\gtzan_genre_distribution.png"

    df, ds = load_and_explore_gtzan(dataset_path=DATASET_PATH, save_path=METADATA_PATH)

    print("\n📋 Metadata Preview:")
    print(df.head())

    visualize_genre_distribution(df, save_path=PLOT_PATH)

    sample_audio = extract_audio_sample(ds, 0)

    features_df = extract_audio_features(
        df,
        n_samples=None,  # ✅ Extract from all 100 songs
        save_path=FEATURES_PATH
    )

    print("\n🚀 GTZAN metadata and features ready for use!")
