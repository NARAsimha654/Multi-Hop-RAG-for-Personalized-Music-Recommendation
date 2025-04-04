import musicbrainzngs

# Set up your app info (required by MusicBrainz API)
musicbrainzngs.set_useragent("MultiHopRAG", "1.0", "01fe22bci018@kletech.ac.in")

artist_id = "bb396e90-c9e3-42c5-b1d4-caf7965fe28f"  # Drake' ID
artist_info = musicbrainzngs.get_artist_by_id(artist_id, includes=["releases"])

print(f"Artist Name: {artist_info['artist']['name']}")
print("Releases:")
for release in artist_info["artist"]["release-list"]:
    print(f"- {release['title']} ({release['date']})")
