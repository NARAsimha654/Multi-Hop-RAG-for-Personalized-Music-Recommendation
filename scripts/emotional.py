import pandas as pd
import numpy as np
import os
import sys
import re # For tokenization
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # VADER
from nrclex import NRCLex # NRCLex for basic emotions
from tqdm.auto import tqdm # Progress bar
import pickle # To potentially save lexicon dictionary

# Configure tqdm to work well with pandas apply
tqdm.pandas()

# --- Configuration ---
# Input file (the SMALL sample lyrics file you created earlier)
LYRICS_SAMPLE_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_lyrics_27k.parquet' # Adjust if you named it differently

# --- IMPORTANT: Update this path to your NRC-VAD lexicon file ---
NRC_VAD_LEXICON_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\raw\NRC-VAD-Lexicon-v2.1\NRC-VAD-Lexicon-v2.1.txt' # <<<--- UPDATE THIS PATH

# Output file for emotion features from the sample
EMOTION_FEATURES_OUTPUT_FILE = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\mpd_emotion_features_sample_vad.parquet' # Added _vad

# Optional: Pre-processed lexicon file path
PREPROCESSED_LEXICON_PKL = r'C:\Narasimha\KLETU Related\6th Semester Related\GenAI and NLP\GenAI\Course Project\GitHub Repo\Multi-Hop-RAG-for-Personalized-Music-Recommendation\data\processed\nrc_vad_lexicon.pkl'


# --- Function to Load VAD Lexicon ---
def load_vad_lexicon(filepath, pkl_path=None):
    """ Loads the NRC-VAD lexicon from the text file or a pre-processed pickle file. """
    vad_lexicon = {}
    # Try loading pre-processed pickle first
    if pkl_path and os.path.exists(pkl_path):
        print(f"Loading pre-processed VAD lexicon from {pkl_path}...")
        try:
            with open(pkl_path, 'rb') as f:
                vad_lexicon = pickle.load(f)
            if vad_lexicon:
                 print(f"Loaded {len(vad_lexicon)} terms from pickle.")
                 return vad_lexicon
            else:
                 print("Pickle file was empty. Loading from text file.")
        except Exception as e:
            print(f"Error loading pickle {pkl_path}: {e}. Loading from text file.")

    # Load from original text file
    print(f"Loading VAD lexicon from {filepath}...")
    try:
        # Use encoding that handles potential special characters if any
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f) # Skip header if necessary (assuming no header based on description)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    term, v_str, a_str, d_str = parts
                    try:
                        # Store scores as floats
                        vad_lexicon[term] = {
                            'valence': float(v_str),
                            'arousal': float(a_str),
                            'dominance': float(d_str)
                        }
                    except ValueError:
                        # Handle potential conversion errors for specific lines
                        print(f"Warning: Could not parse scores for term '{term}' in VAD lexicon.")
                        continue
        print(f"Loaded {len(vad_lexicon)} terms from VAD lexicon text file.")

        # Save pre-processed lexicon to pickle for faster loading next time
        if pkl_path and vad_lexicon:
            print(f"Saving pre-processed lexicon to {pkl_path}...")
            try:
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(vad_lexicon, f)
            except Exception as e:
                print(f"Error saving lexicon pickle: {e}")

        return vad_lexicon

    except FileNotFoundError:
        print(f"Error: NRC-VAD lexicon file not found at {filepath}")
        return None # Return None if file not found
    except Exception as e:
        print(f"Error reading VAD lexicon file: {e}")
        return None


# --- Load Sample Lyrics Data ---
print(f"Loading sample lyrics data from {LYRICS_SAMPLE_FILE}...")
try:
    if LYRICS_SAMPLE_FILE.endswith('.parquet'):
        lyrics_df = pd.read_parquet(LYRICS_SAMPLE_FILE)
    else:
        lyrics_df = pd.read_csv(LYRICS_SAMPLE_FILE) # Add CSV handling if needed

    # Filter for rows where lyrics were actually found
    lyrics_df = lyrics_df.dropna(subset=['lyrics'])
    lyrics_df = lyrics_df[lyrics_df['lyrics'].str.strip() != '']
    if 'status' in lyrics_df.columns:
        lyrics_df = lyrics_df[lyrics_df['status'] == 'FOUND']

    print(f"Loaded {len(lyrics_df)} tracks with valid lyrics from sample.")

    if lyrics_df.empty:
        print("No valid lyrics found in the sample file to process.")
        sys.exit(0)

except FileNotFoundError:
    print(f"Error: Sample lyrics file not found at {LYRICS_SAMPLE_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading sample lyrics file: {e}")
    sys.exit(1)

# --- Initialize Analyzers ---
print("Initializing sentiment and emotion analyzers...")
vader_analyzer = SentimentIntensityAnalyzer()
print("VADER analyzer initialized.")

# --- Load NRC-VAD Lexicon ---
vad_lexicon = load_vad_lexicon(NRC_VAD_LEXICON_FILE, PREPROCESSED_LEXICON_PKL)
if vad_lexicon is None:
    print("Proceeding without VAD analysis.")
    # Optionally exit if VAD is critical: sys.exit(1)

# --- Define Analysis Functions ---

def get_vader_sentiment(text):
    """ Calculates VADER sentiment scores, returns only compound score. """
    if not isinstance(text, str) or not text.strip(): return None
    try:
        return vader_analyzer.polarity_scores(text).get('compound', None)
    except Exception as e: print(f"VADER Error: {e}"); return None

def get_nrc_emotions(text):
    """ Calculates NRCLex emotion frequencies. """
    if not isinstance(text, str) or not text.strip(): return None
    try: return NRCLex(text).affect_frequencies
    except Exception as e: print(f"NRCLex Error: {e}"); return None

# --- Define VAD Calculation Function ---
def calculate_vad_scores(text, lexicon):
    """ Calculates average VAD scores for text based on a lexicon. """
    if not isinstance(text, str) or not text.strip() or lexicon is None:
        # Return default dictionary if no text or lexicon
        return {'avg_valence': None, 'avg_arousal': None, 'avg_dominance': None}

    # Simple tokenization (lowercase, split by space after removing punctuation)
    processed_text = text.lower()
    processed_text = re.sub(r'[^\w\s]', '', processed_text)
    tokens = processed_text.split()

    v_scores, a_scores, d_scores = [], [], []
    found_words = 0

    for token in tokens:
        if token in lexicon:
            scores = lexicon[token]
            v_scores.append(scores['valence'])
            a_scores.append(scores['arousal'])
            d_scores.append(scores['dominance'])
            found_words += 1

    # Calculate average scores only if words were found in the lexicon
    if found_words > 0:
        avg_v = np.mean(v_scores) if v_scores else None
        avg_a = np.mean(a_scores) if a_scores else None
        avg_d = np.mean(d_scores) if d_scores else None
        return {'avg_valence': avg_v, 'avg_arousal': avg_a, 'avg_dominance': avg_d}
    else:
        # Return default if no words matched
        return {'avg_valence': None, 'avg_arousal': None, 'avg_dominance': None}


# --- Apply Analysis Functions ---
print("\nApplying VADER sentiment analysis...")
lyrics_df['vader_compound'] = lyrics_df['lyrics'].progress_apply(get_vader_sentiment)
print("VADER analysis complete.")

print("\nApplying NRCLex emotion analysis...")
lyrics_df['nrc_emotions_dict'] = lyrics_df['lyrics'].progress_apply(get_nrc_emotions) # Keep dict temporarily
print("NRCLex analysis complete.")

# Apply VAD analysis only if lexicon was loaded
if vad_lexicon:
    print("\nApplying VAD analysis...")
    # Apply function returns a dictionary, store it first
    lyrics_df['vad_scores_dict'] = lyrics_df['lyrics'].progress_apply(lambda x: calculate_vad_scores(x, vad_lexicon))
    # Extract scores into separate columns
    lyrics_df['vad_valence'] = lyrics_df['vad_scores_dict'].apply(lambda x: x.get('avg_valence'))
    lyrics_df['vad_arousal'] = lyrics_df['vad_scores_dict'].apply(lambda x: x.get('avg_arousal'))
    lyrics_df['vad_dominance'] = lyrics_df['vad_scores_dict'].apply(lambda x: x.get('avg_dominance'))
    lyrics_df = lyrics_df.drop(columns=['vad_scores_dict']) # Drop temporary dict column
    print("VAD analysis complete.")
else:
    # Create empty columns if VAD lexicon wasn't loaded
    lyrics_df['vad_valence'] = None
    lyrics_df['vad_arousal'] = None
    lyrics_df['vad_dominance'] = None


# --- Process Emotion Results (Flatten NRCLex dictionary) ---
print("\nProcessing NRCLex emotion results...")
all_emotion_keys = set()
try:
    for emo_dict in lyrics_df['nrc_emotions_dict'].dropna():
        all_emotion_keys.update(emo_dict.keys())

    # Create new columns for each emotion
    nrc_cols = []
    for emotion in sorted(list(all_emotion_keys)):
        col_name = f'nrc_{emotion}'
        lyrics_df[col_name] = lyrics_df['nrc_emotions_dict'].apply(lambda x: x.get(emotion, 0.0) if isinstance(x, dict) else 0.0)
        nrc_cols.append(col_name) # Keep track of created columns
    print(f"Created columns for NRCLex emotions: {nrc_cols}")
    # Drop the original dictionary column
    lyrics_df = lyrics_df.drop(columns=['nrc_emotions_dict'])

except Exception as e:
    print(f"Could not flatten NRCLex emotion dictionaries: {e}")
    nrc_cols = [] # Ensure list is empty if flattening fails

# --- Save Results ---
print("\nPreview of results:")
# Define output columns including VAD
output_cols = ['track_uri', 'vader_compound', 'vad_valence', 'vad_arousal', 'vad_dominance'] + nrc_cols
# Ensure columns exist before selecting
output_cols = [col for col in output_cols if col in lyrics_df.columns]
print(lyrics_df[output_cols].head())


print(f"\nSaving emotion features to {EMOTION_FEATURES_OUTPUT_FILE}...")
try:
    os.makedirs(os.path.dirname(EMOTION_FEATURES_OUTPUT_FILE), exist_ok=True)
    final_df_to_save = lyrics_df[output_cols]
    if EMOTION_FEATURES_OUTPUT_FILE.endswith('.parquet'):
         final_df_to_save.to_parquet(EMOTION_FEATURES_OUTPUT_FILE, index=False)
    else:
         final_df_to_save.to_csv(EMOTION_FEATURES_OUTPUT_FILE, index=False)
    print("Emotion features saved successfully.")
except Exception as e:
    print(f"Error saving emotion features file: {e}")

print("\n--- Emotion Feature Extraction (VADER, NRCLex, VAD - Sample) Finished ---")
