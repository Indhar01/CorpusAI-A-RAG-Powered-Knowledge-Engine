from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import chromadb

collections_cache = {}

def get_or_create_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """Cache collections to avoid repeated get_collection calls"""
    if collection_name not in collections_cache:
        collections_cache[collection_name] = client.get_or_create_collection(
            name=collection_name
        )
    return collections_cache[collection_name]

def query_collection(client: chromadb.Client, collection_name: str, query_text: str, n_results: int = 5) -> Dict[str, Any]:
    """Query a single collection"""
    try:
        collection = get_or_create_collection(client, collection_name)
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'collection': collection_name,
            'results': results
        }
    except Exception as e:
        print(f"Error querying collection {collection_name}: {str(e)}")
        return None

def query_multiple_collections(
    client: chromadb.Client,
    collections: List[str],
    query_text: str,
    n_results: int = 5,
    distance_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Query multiple collections concurrently using ThreadPoolExecutor"""
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list to store futures
        futures = []
        
        # Submit tasks and store futures with collection names
        for collection_name in collections:
            future = executor.submit(
                query_collection, 
                client, 
                collection_name, 
                query_text, 
                n_results
            )
            futures.append((future, collection_name))
        
        # Process completed futures
        for future, collection_name in futures:
            try:
                result = future.result()
                if result and result['results']['distances'][0]:  # Check if there are any results
                    collection_results = {
                        'collection': result['collection'],
                        'documents': [],
                        'metadatas': [],
                        'distances': []
                    }
                    
                    # Filter based on distance threshold
                    for doc, meta, dist in zip(
                        result['results']['documents'][0],
                        result['results']['metadatas'][0],
                        result['results']['distances'][0]
                    ):
                        if dist <= distance_threshold:
                            collection_results['documents'].append(doc)
                            collection_results['metadatas'].append(meta)
                            collection_results['distances'].append(dist)
                    
                    if collection_results['documents']:  # Only add if there are valid results
                        results.append(collection_results)
            except Exception as e:
                print(f"Error processing results for collection {collection_name}: {str(e)}")
                
    return results

def process_query(
    client: chromadb.Client,
    collections: List[str],
    query_text: str,
    n_results: int = 5,
    distance_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Process queries across multiple collections"""
    # Convert collection objects to names if needed
    collection_names = [
        coll.name if hasattr(coll, 'name') else str(coll)
        for coll in collections
    ]
    
    return query_multiple_collections(
        client,
        collection_names,
        query_text,
        n_results,
        distance_threshold
    )