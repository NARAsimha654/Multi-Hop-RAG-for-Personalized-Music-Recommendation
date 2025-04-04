import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Your Spotify API credentials
CLIENT_ID = "a3e5c6fdfd184e45b5ed62e682868e2e"
CLIENT_SECRET = "0818f3d02e854b39b0feb88ad57889ee"

# Authenticate
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Fetch access token
print(auth_manager.get_access_token(as_dict=False))

