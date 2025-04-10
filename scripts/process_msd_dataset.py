import os
import glob
import h5py
import pandas as pd
import time # To time the processing
import numpy as np # For NaN

# --- Configuration ---
# IMPORTANT: Update this path to where your MSD Subset data is located
MSD_SUBSET_PATH = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\raw\\MillionSongSubset\\' # <<<--- UPDATE THIS PATH

# Output file paths
RAW_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_subset_full_raw.parquet' # Using Parquet for efficiency
CLEANED_OUTPUT_FILE = 'C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\msd_subset_cleaned.parquet'

# --- Functions (reuse from previous step) ---

def find_h5_files(root_dir):
    """Recursively finds all HDF5 files (.h5) in a directory."""
    print(f"Searching for .h5 files in: {root_dir}")
    h5_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    print(f"Found {len(h5_files)} HDF5 files.")
    return h5_files

def extract_msd_features(h5_file_path):
    """Extracts selected features from a single MSD HDF5 file."""
    features = {}
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Metadata group
            features['artist_name'] = h5_file['metadata']['songs']['artist_name'][0].decode('utf-8')
            features['title'] = h5_file['metadata']['songs']['title'][0].decode('utf-8')
            features['release'] = h5_file['metadata']['songs']['release'][0].decode('utf-8')
            features['song_id'] = h5_file['metadata']['songs']['song_id'][0].decode('utf-8')

            # Analysis group
            features['duration'] = h5_file['analysis']['songs']['duration'][0]
            features['key'] = h5_file['analysis']['songs']['key'][0]
            features['key_confidence'] = h5_file['analysis']['songs']['key_confidence'][0]
            features['loudness'] = h5_file['analysis']['songs']['loudness'][0]
            features['mode'] = h5_file['analysis']['songs']['mode'][0]
            features['mode_confidence'] = h5_file['analysis']['songs']['mode_confidence'][0]
            features['tempo'] = h5_file['analysis']['songs']['tempo'][0]
            features['time_signature'] = h5_file['analysis']['songs']['time_signature'][0]
            features['time_signature_confidence'] = h5_file['analysis']['songs']['time_signature_confidence'][0]

            # MusicBrainz group
            features['year'] = h5_file['musicbrainz']['songs']['year'][0]

    except Exception as e:
        # Log error for the specific file but continue processing others
        # print(f"Error processing file {os.path.basename(h5_file_path)}: {e}")
        return None # Indicate failure
    return features

# --- Main Processing Logic ---

start_time = time.time()

# 1. Find all HDF5 files
all_h5_files = find_h5_files(MSD_SUBSET_PATH)

if not all_h5_files:
    print("No HDF5 files found. Please check the MSD_SUBSET_PATH.")
else:
    # 2. Process all files
    print(f"Processing {len(all_h5_files)} files...")
    all_features_list = []
    processed_count = 0
    error_count = 0
    for i, h5_path in enumerate(all_h5_files):
        song_features = extract_msd_features(h5_path)
        if song_features:
            all_features_list.append(song_features)
            processed_count += 1
        else:
            error_count += 1
        # Optional: Print progress
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_h5_files)} files...")

    print(f"\nSuccessfully extracted features for {processed_count} songs.")
    if error_count > 0:
        print(f"Encountered errors processing {error_count} files.")

    # 3. Create DataFrame
    if all_features_list:
        full_msd_df = pd.DataFrame(all_features_list)
        print("\nFull Raw DataFrame Info:")
        full_msd_df.info()

        # 4. Save Raw DataFrame
        print(f"\nSaving raw data to {RAW_OUTPUT_FILE}...")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(RAW_OUTPUT_FILE), exist_ok=True)
        try:
            # Use pyarrow engine for parquet if available
             full_msd_df.to_parquet(RAW_OUTPUT_FILE, index=False, engine='pyarrow')
             # If pyarrow is not installed or causes issues, try 'fastparquet'
             # Or save as CSV: full_msd_df.to_csv(RAW_OUTPUT_FILE.replace('.parquet', '.csv'), index=False)
        except ImportError:
             print("Note: 'pyarrow' not found. Saving as CSV instead.")
             RAW_OUTPUT_FILE = RAW_OUTPUT_FILE.replace('.parquet', '.csv')
             full_msd_df.to_csv(RAW_OUTPUT_FILE, index=False)

        print("Raw data saved.")


        # 5. Clean Data (Specifically 'year')
        print("\nCleaning data...")
        cleaned_msd_df = full_msd_df.copy()

        # Replace 0 values in 'year' with NaN (Not a Number)
        # This explicitly marks them as missing/invalid
        original_year_zeros = (cleaned_msd_df['year'] == 0).sum()
        if original_year_zeros > 0:
             print(f"Replacing {original_year_zeros} zero values in 'year' column with NaN.")
             cleaned_msd_df['year'] = cleaned_msd_df['year'].replace(0, np.nan)
        else:
            print("No zero values found in 'year' column to replace.")

        # Optional: Add other cleaning steps here if needed
        # e.g., cleaned_msd_df['artist_name'] = cleaned_msd_df['artist_name'].str.strip()

        print("Data cleaning complete.")
        print("\nCleaned DataFrame Info:")
        cleaned_msd_df.info()
        print("\nMissing values in cleaned data:")
        print(cleaned_msd_df.isnull().sum())


        # 6. Save Cleaned DataFrame
        print(f"\nSaving cleaned data to {CLEANED_OUTPUT_FILE}...")
         # Ensure the output directory exists
        os.makedirs(os.path.dirname(CLEANED_OUTPUT_FILE), exist_ok=True)
        try:
            # Use pyarrow engine for parquet if available
            cleaned_msd_df.to_parquet(CLEANED_OUTPUT_FILE, index=False, engine='pyarrow')
        except ImportError:
            print("Note: 'pyarrow' not found. Saving as CSV instead.")
            CLEANED_OUTPUT_FILE = CLEANED_OUTPUT_FILE.replace('.parquet', '.csv')
            cleaned_msd_df.to_csv(CLEANED_OUTPUT_FILE, index=False)
        print("Cleaned data saved.")

    else:
        print("No features were extracted, cannot create DataFrame.")

end_time = time.time()
print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")