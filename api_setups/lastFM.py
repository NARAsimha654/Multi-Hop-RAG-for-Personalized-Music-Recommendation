import pylast

API_KEY = "8c9f3de5d0c55169230f419c266e3418"
API_SECRET = "d5f85d594dd64b6f1720d3998a054770"

network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)

# Example: Get top tracks of an artist
artist = network.get_artist("Drake")
top_tracks = artist.get_top_tracks()

for track in top_tracks:
    print(track.item.title)
