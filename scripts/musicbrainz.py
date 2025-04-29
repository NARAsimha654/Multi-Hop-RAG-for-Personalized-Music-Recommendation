# Import necessary libraries
from neo4j import GraphDatabase
import musicbrainzngs # MusicBrainz API client
import time
import sys
import os
from tqdm.auto import tqdm # Progress bar

# --- Neo4j Connection Details ---
# Ensure these are correct for your Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Narasimha123" # <<<--- VERIFY YOUR PASSWORD
DB_NAME = "db-1" # Specify database name if not default

# --- MusicBrainz Configuration ---
# Set a user agent string for MusicBrainz API access (Required by their policy)
# Replace 'MyAppName' and '0.1' with your application's name and version
# Replace 'YourContactInfo' with your email or website
try:
    musicbrainzngs.set_useragent(
        "MusicRAG_Project", "0.1", "https://github.com/NARAsimha654/Multi-Hop-RAG-for-Personalized-Music-Recommendation.git" # <<<--- UPDATE CONTACT INFO
    )
    print("MusicBrainz User-Agent set.")
except Exception as e:
    print(f"Error setting MusicBrainz User-Agent: {e}")
    # Decide if you want to exit or continue without it (API might reject requests)
    # sys.exit(1)

# --- Neo4j Driver Setup ---
driver = None
print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Neo4j connection successful!")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    sys.exit(1)

# --- Helper Function to Run Cypher Queries ---
def run_cypher_query(tx, query, parameters=None):
    """ Helper function to execute a Cypher query within a transaction """
    result = tx.run(query, parameters)
    # For read queries, collect results; for writes, consume
    # Adjust based on whether you expect data back
    try:
        # If query returns data, collect it
        return [record for record in result]
    except Exception:
        # If query is write-only or returns no data, consume summary
        return result.consume()

# --- Function to Fetch Artist Names from Neo4j ---
def get_artist_names_from_neo4j(driver):
    """ Fetches all distinct artist names from the Neo4j graph. """
    print("\nFetching artist names from Neo4j...")
    query = "MATCH (a:Artist) RETURN DISTINCT a.name AS name"
    artist_names = []
    with driver.session(database=DB_NAME) as session:
        try:
            results = session.execute_read(run_cypher_query, query)
            artist_names = [record["name"] for record in results if record and record["name"]]
            print(f"Found {len(artist_names)} distinct artist names.")
        except Exception as e:
            print(f"Error fetching artist names: {e}")
    return artist_names

# --- Function to Update Artist Node in Neo4j ---
def update_artist_in_neo4j(driver, artist_name, mb_data):
    """ Updates an Artist node with data fetched from MusicBrainz. """
    # Cypher query to match the artist by name and set new properties
    # Using SET ensures we add/overwrite properties without deleting existing ones
    query = """
    MATCH (a:Artist {name: $artist_name})
    SET a += $properties // Add properties from the map
    RETURN id(a) // Return something to confirm update
    """
    # Prepare properties map, ensuring values are primitive types
    properties_to_set = {}
    if mb_data.get('mbid'): properties_to_set['mbid'] = mb_data['mbid']
    if mb_data.get('type'): properties_to_set['artistType'] = mb_data['type'] # Use different property name if desired
    if mb_data.get('country'): properties_to_set['country'] = mb_data['country']
    if mb_data.get('begin_date'): properties_to_set['beginDate'] = mb_data['begin_date']
    if mb_data.get('end_date'): properties_to_set['endDate'] = mb_data['end_date']
    if mb_data.get('gender'): properties_to_set['gender'] = mb_data['gender']

    if not properties_to_set: # Don't run query if no properties to set
        return False

    with driver.session(database=DB_NAME) as session:
        try:
            summary = session.execute_write(
                run_cypher_query,
                query,
                parameters={'artist_name': artist_name, 'properties': properties_to_set}
            )
            # Check if any properties were actually set (depends on consume/return logic)
            # For simplicity, assume success if no exception
            return True
        except Exception as e:
            print(f"\n  Error updating artist '{artist_name}' in Neo4j: {e}")
            return False


# --- Main Enrichment Logic ---
if __name__ == "__main__":
    script_start_time = time.time()

    # 1. Get artist names from Neo4j
    artist_names = get_artist_names_from_neo4j(driver)

    if not artist_names:
        print("No artist names found in Neo4j to enrich.")
    else:
        print(f"\nStarting MusicBrainz enrichment for {len(artist_names)} artists...")
        update_count = 0
        error_count = 0
        not_found_count = 0

        # 2. Iterate and query MusicBrainz
        # Use tqdm for progress bar
        for artist_name in tqdm(artist_names, desc="Enriching Artists", unit="artist"):
            mb_data = {} # Dictionary to hold fetched data for this artist
            try:
                # Search for the artist on MusicBrainz
                # Limit=1 gets the most likely match
                # Includes 'aliases', 'tags', 'ratings' can add more info but increases response size
                result = musicbrainzngs.search_artists(artist=artist_name, limit=1)

                # Check if results were found and if score is high enough (e.g., 100)
                if result.get('artist-list') and result['artist-list'][0].get('ext:score') == '100':
                    mb_artist = result['artist-list'][0]
                    mb_data['mbid'] = mb_artist.get('id')
                    mb_data['type'] = mb_artist.get('type') # e.g., Person, Group
                    mb_data['country'] = mb_artist.get('country')
                    # Extract life span info if available
                    if 'life-span' in mb_artist:
                        mb_data['begin_date'] = mb_artist['life-span'].get('begin')
                        mb_data['end_date'] = mb_artist['life-span'].get('ended') # Note: 'ended' is boolean
                        if mb_artist['life-span'].get('ended') == 'true':
                             mb_data['end_date'] = mb_artist['life-span'].get('end') # Get actual end date if ended
                        else:
                             mb_data['end_date'] = None # Explicitly set to None if not ended

                    # Extract gender if available (usually for Person type)
                    mb_data['gender'] = mb_artist.get('gender')

                    # 3. Update Neo4j node if data was found
                    if mb_data.get('mbid'): # Check if we got at least the ID
                        if update_artist_in_neo4j(driver, artist_name, mb_data):
                            update_count += 1
                        else:
                            error_count += 1 # Count update errors separately
                    else:
                        not_found_count +=1 # Count cases where MB search didn't yield usable data

                else:
                    # No confident match found on MusicBrainz
                    not_found_count += 1

            except musicbrainzngs.WebServiceError as exc:
                # Handle specific MusicBrainz API errors (e.g., rate limiting, server errors)
                print(f"\n  MusicBrainz API Error for '{artist_name}': {exc}")
                error_count += 1
                time.sleep(5) # Longer sleep after API error
            except Exception as e:
                # Handle other potential errors (network, etc.)
                print(f"\n  General Error processing '{artist_name}': {e}")
                error_count += 1
                time.sleep(2) # Short sleep after general error

            finally:
                # --- IMPORTANT: MusicBrainz Rate Limiting ---
                # Adhere to the 1 request per second policy
                time.sleep(1.1) # Sleep for slightly over 1 second

        print("\n--- MusicBrainz Enrichment Summary ---")
        print(f"Artists processed: {len(artist_names)}")
        print(f"Artists updated in Neo4j: {update_count}")
        print(f"Artists not found/matched on MusicBrainz: {not_found_count}")
        print(f"Errors encountered (API or Update): {error_count}")


    # --- Final Steps ---
    script_end_time = time.time()
    print(f"\n--- Enrichment Script Finished ---")
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")

    # Close the driver connection
    if driver:
        driver.close()
        print("\nNeo4j driver closed.")

