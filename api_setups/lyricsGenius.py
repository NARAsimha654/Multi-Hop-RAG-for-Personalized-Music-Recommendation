import lyricsgenius

# Initialize Genius API
ACCESS_TOKEN = "TuoRTZ6t8sxtQi5yt-MvWUOHlZe6DGbK35bdT_CP8ui78UQDeKB1mGN3m0E9k6nA"
genius = lyricsgenius.Genius(ACCESS_TOKEN)

# Search for a song and get lyrics
song = genius.search_song("Die Tryin", "Drake")
print(song.lyrics)
