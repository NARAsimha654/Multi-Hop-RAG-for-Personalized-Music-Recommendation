import pandas as pd
import os
import sys

# Import your custom modules
from fetch_gtzan_lyrics import setup_genius_api, process_gtzan_dataset
from build_gtzan_rag_dataset import load_data, create_rag_dataset

def main():
    """Main function to process the GTZAN dataset with hardcoded paths"""
    
    # === MODIFY THESE VALUES ===
    metadata_path = "C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\gtzan_metadata.csv"
    features_path = "C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\gtzan_features.csv"
    lyrics_output_path = "C:\\Narasimha\\KLETU Related\\6th Semester Related\\GenAI and NLP\\GenAI\\Course Project\\GitHub Repo\\Multi-Hop-RAG-for-Personalized-Music-Recommendation\\data\\processed\\gtzan_lyrics.csv"
    api_key = "TuoRTZ6t8sxtQi5yt-MvWUOHlZe6DGbK35bdT_CP8ui78UQDeKB1mGN3m0E9k6nA"  # Replace with your actual Genius API key

    sample_size = None           # Example: 20 for testing
    skip_lyrics_extraction = False  # Set to True if you’ve already extracted lyrics
    # ============================

    # File existence checks
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file '{metadata_path}' not found.")
        sys.exit(1)

    if not os.path.exists(features_path):
        print(f"Error: Features file '{features_path}' not found.")
        sys.exit(1)

    # Step 1: Lyrics extraction
    if not skip_lyrics_extraction:
        print(f"Extracting lyrics for songs in {metadata_path}...")
        lyrics_df = process_gtzan_dataset(metadata_path, api_key, lyrics_output_path, sample_size)
        print(f"Lyrics extraction complete. Results saved to {lyrics_output_path}")
    else:
        if not os.path.exists(lyrics_output_path):
            print(f"Error: Lyrics file '{lyrics_output_path}' not found and skip_lyrics_extraction=True")
            sys.exit(1)
        print(f"Skipping lyrics extraction. Using existing file: {lyrics_output_path}")

    # Step 2: Merge data into final RAG dataset
    print("\nMerging data into final RAG dataset...")
    metadata_df, features_df, lyrics_df = load_data(metadata_path, features_path, lyrics_output_path)
    rag_dataset = create_rag_dataset(metadata_df, features_df, lyrics_df)

    # Final output summary
    print("\n✅ Process completed successfully!")
    print("The following files were created:")
    print("1. gtzan_lyrics.csv         - Song lyrics")
    print("2. gtzan_rag_dataset.csv    - Full dataset (CSV)")
    print("3. gtzan_rag_dataset.jsonl  - Full dataset (JSONL)")
    print("4. gtzan_mood_sentiment.csv - Lyrics sentiment/emotion analysis")

if __name__ == "__main__":
    main()
