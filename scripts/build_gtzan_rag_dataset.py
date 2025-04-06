import pandas as pd
import json
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def load_data(metadata_file, features_file, lyrics_file):
    """Load all datasets"""
    metadata_df = pd.read_csv(metadata_file)
    features_df = pd.read_csv(features_file)
    lyrics_df = pd.read_csv(lyrics_file)
    
    return metadata_df, features_df, lyrics_df

def analyze_sentiment(lyrics):
    """Analyze sentiment of lyrics using VADER"""
    try:
        # Download VADER lexicon if not already present
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        
        sid = SentimentIntensityAnalyzer()
        
        # If no lyrics found, return neutral sentiment
        if lyrics in ["LYRICS_NOT_FOUND", "N/A"]:
            return 0.5  # Neutral sentiment
        
        sentiment_scores = sid.polarity_scores(lyrics)
        # Compound score ranges from -1 (very negative) to 1 (very positive)
        # Normalize to 0-1 range for easier integration
        normalized_score = (sentiment_scores['compound'] + 1) / 2
        return round(normalized_score, 2)
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return 0.5  # Return neutral on error

def determine_mood(tempo, sentiment_score):
    """Determine mood based on tempo and sentiment"""
    # Define tempo ranges
    slow_tempo = 90
    fast_tempo = 140
    
    # Simple rule-based mood determination
    if tempo < slow_tempo:
        if sentiment_score < 0.4:
            return "Melancholic"
        elif sentiment_score > 0.6:
            return "Peaceful"
        else:
            return "Relaxed"
    elif tempo > fast_tempo:
        if sentiment_score < 0.4:
            return "Angry"
        elif sentiment_score > 0.6:
            return "Exuberant"
        else:
            return "Energetic"
    else:  # Medium tempo
        if sentiment_score < 0.4:
            return "Somber"
        elif sentiment_score > 0.6:
            return "Cheerful"
        else:
            return "Balanced"

def generate_summary(genre, tempo, sentiment_score, mood):
    """Generate a simple summary based on song characteristics"""
    tempo_description = "slow" if tempo < 90 else "moderate" if tempo < 140 else "fast"
    
    sentiment_description = "melancholic" if sentiment_score < 0.4 else \
                           "emotionally balanced" if sentiment_score < 0.6 else \
                           "uplifting"
    
    summary = f"A {tempo_description} {genre} track with a {mood.lower()} mood. "
    summary += f"The song has a {sentiment_description} quality with a tempo of {tempo:.1f} BPM."
    
    return summary

def create_rag_dataset(metadata_df, features_df, lyrics_df, output_csv="gtzan_rag_dataset.csv", output_jsonl="gtzan_rag_dataset.jsonl"):
    """Create the final RAG dataset by merging all data"""
    # Merge dataframes on file_path
    merged_df = features_df.merge(lyrics_df, on=['file_path', 'genre'])
    
    # Add song ID
    merged_df['id'] = [f"song_{i:04d}" for i in range(1, len(merged_df) + 1)]
    
    # Calculate sentiment score
    merged_df['sentiment_score'] = merged_df['lyrics'].apply(analyze_sentiment)
    
    # Determine mood
    merged_df['mood'] = merged_df.apply(lambda x: determine_mood(x['tempo'], x['sentiment_score']), axis=1)
    
    # Generate summary
    merged_df['summary'] = merged_df.apply(
        lambda x: generate_summary(x['genre'], x['tempo'], x['sentiment_score'], x['mood']), 
        axis=1
    )
    
    # Reorder columns for better readability
    column_order = ['id', 'file_path', 'genre', 'title', 'artist']
    
    # Add all feature columns in order
    feature_cols = [col for col in features_df.columns if col not in ['file_path', 'genre']]
    column_order.extend(feature_cols)
    
    # Add remaining columns
    column_order.extend(['lyrics', 'sentiment_score', 'mood', 'summary'])
    
    # Reorder and select only columns we have
    final_columns = [col for col in column_order if col in merged_df.columns]
    final_df = merged_df[final_columns]
    
    # Save as CSV
    final_df.to_csv(output_csv, index=False)
    print(f"Dataset saved as CSV: {output_csv}")
    
    # Save as JSONL
    with open(output_jsonl, 'w') as f:
        for _, row in final_df.iterrows():
            # Convert row to dictionary and handle NaN values
            row_dict = row.to_dict()
            for k, v in row_dict.items():
                if isinstance(v, (np.float64, np.float32)):
                    row_dict[k] = float(v)
                elif pd.isna(v):
                    row_dict[k] = None
            
            f.write(json.dumps(row_dict) + '\n')
    
    print(f"Dataset saved as JSONL: {output_jsonl}")
    
    # Also create a separate mood analysis file if desired
    mood_df = final_df[['id', 'file_path', 'genre', 'sentiment_score', 'mood']]
    mood_df.to_csv("gtzan_mood_sentiment.csv", index=False)
    print(f"Mood analysis saved as: gtzan_mood_sentiment.csv")
    
    return final_df

# Example usage
# metadata_df, features_df, lyrics_df = load_data("gtzan_metadata.csv", "gtzan_features.csv", "gtzan_lyrics.csv")
# rag_dataset = create_rag_dataset(metadata_df, features_df, lyrics_df)