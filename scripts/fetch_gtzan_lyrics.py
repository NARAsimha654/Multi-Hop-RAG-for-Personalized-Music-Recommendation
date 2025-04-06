import pandas as pd
import lyricsgenius as lg
import time
import re
import os
from tqdm import tqdm

def setup_genius_api(api_key):
    """Setup Genius API client with rate limiting"""
    genius = lg.Genius(api_key, 
                       timeout=15,
                       sleep_time=0.5,  # Sleep between requests to avoid rate limits
                       verbose=False)
    return genius

def extract_info_from_filename(filename):
    """Extract genre and track number from GTZAN filename format"""
    # Example: genres_original/blues/blues.00001.wav
    pattern = r'genres_original/(\w+)/\w+\.(\d+)\.wav'
    match = re.search(pattern, filename)
    
    if match:
        genre = match.group(1)
        track_num = match.group(2)
        return genre, track_num
    else:
        # Alternative pattern if files are structured differently
        filename_only = os.path.basename(filename)
        parts = filename_only.split('.')
        if len(parts) >= 3:
            genre = parts[0]
            track_num = parts[1]
            return genre, track_num
    
    return None, None

def create_search_query(filename, genre):
    """Create search query for Genius based on filename and genre"""
    # For GTZAN, we might not have proper titles/artists
    # So we'll search by genre and hope for relevant results
    genre, track_num = extract_info_from_filename(filename)
    
    if genre:
        # Simple search query with genre
        return f"{genre} song"
    
    # Fallback to just the filename without extension
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext

def fetch_lyrics_for_song(genius, filename, genre, max_retries=3):
    """Attempt to fetch lyrics for a song using Genius API"""
    search_query = create_search_query(filename, genre)
    
    # Initialize with empty/default values
    title = "Unknown"
    artist = "Unknown"
    lyrics = "LYRICS_NOT_FOUND"
    
    for attempt in range(max_retries):
        try:
            # Search for song
            search_results = genius.search_songs(search_query)
            
            if search_results['hits'] and len(search_results['hits']) > 0:
                # Get the first hit
                song_info = search_results['hits'][0]['result']
                song_id = song_info['id']
                title = song_info['title']
                artist = song_info['primary_artist']['name']
                
                # Fetch the complete song with lyrics
                song = genius.song(song_id)
                
                if song and 'lyrics' in song:
                    lyrics = song['lyrics']
                    # Clean up lyrics by removing Genius annotations/headers
                    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Verse], [Chorus], etc.
                    lyrics = re.sub(r'\d+Embed$', '', lyrics)  # Remove Embed numbers
                    lyrics = lyrics.strip()
                    
                    # If we still have "Lyrics not found", revert to default value
                    if "Lyrics not found" in lyrics:
                        lyrics = "LYRICS_NOT_FOUND"
                    
                    return title, artist, lyrics
            
            # If we didn't return yet but there's no error, wait and retry
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching lyrics for {filename}: {str(e)}")
            time.sleep(2)  # Wait a bit longer on error
    
    # If all retries failed, return default values
    return title, artist, lyrics

def process_gtzan_dataset(metadata_file, api_key, output_file="gtzan_lyrics.csv", sample_size=None):
    """Process the entire GTZAN dataset and extract lyrics for each song"""
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Sample a subset if specified (for testing)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Setup Genius API
    genius = setup_genius_api(api_key)
    
    # Create output dataframe
    lyrics_df = pd.DataFrame(columns=['audio_path', 'genre', 'title', 'artist', 'lyrics'])
    
    # Process each song
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching lyrics"):
        file_path = row['audio_path']
        genre = row['genre']
        
        # Get lyrics
        title, artist, lyrics = fetch_lyrics_for_song(genius, file_path, genre)
        
        # Add to dataframe
        lyrics_df.loc[idx] = [file_path, genre, title, artist, lyrics]
        
        # Save periodically
        if idx % 10 == 0:
            lyrics_df.to_csv(output_file, index=False)
    
    # Final save
    lyrics_df.to_csv(output_file, index=False)
    print(f"Lyrics saved to {output_file}")
    
    return lyrics_df

# Usage:
# api_key = "YOUR_GENIUS_API_KEY"  # Replace with your actual API key
# process_gtzan_dataset("gtzan_metadata.csv", api_key)