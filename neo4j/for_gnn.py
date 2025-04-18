from neo4j import GraphDatabase

# Connection info
uri = "bolt://localhost:7687"  # Change if using Aura/cloud
username = "neo4j"
password = "Narasimha123"

driver = GraphDatabase.driver(uri, auth=(username, password))

def create_music_graph(session):
    query = """
    CALL gds.graph.project(
        'musicGraph',
        {
            MsdSong: { label: 'MsdSong' },
            MpdTrack: { label: 'MpdTrack' },
            Artist: { label: 'Artist' },
            Playlist: { label: 'Playlist' }
        },
        {
            BY_ARTIST_MSD: {
                type: 'BY_ARTIST',
                orientation: 'UNDIRECTED',
                sourceNode: 'MsdSong',
                targetNode: 'Artist'
            },
            BY_ARTIST_MPD: {
                type: 'BY_ARTIST',
                orientation: 'UNDIRECTED',
                sourceNode: 'MpdTrack',
                targetNode: 'Artist'
            },
            APPEARS_IN: {
                type: 'APPEARS_IN',
                orientation: 'UNDIRECTED',
                sourceNode: 'MpdTrack',
                targetNode: 'Playlist'
            },
            SAME_AS: {
                type: 'SAME_AS',
                orientation: 'UNDIRECTED',
                sourceNode: 'MsdSong',
                targetNode: 'MpdTrack'
            }
        }
    )
    YIELD graphName, nodeCount, relationshipCount
    """
    result = session.run(query)
    return result.single()

# Run it
with driver.session() as session:
    try:
        graph_info = create_music_graph(session)
        print("Graph Created:")
        print("Graph Name:", graph_info["graphName"])
        print("Node Count:", graph_info["nodeCount"])
        print("Relationship Count:", graph_info["relationshipCount"])
    except Exception as e:
        print("Graph creation failed:", str(e))

driver.close()
