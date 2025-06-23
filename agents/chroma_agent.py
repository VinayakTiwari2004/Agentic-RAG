# chroma_agent.py

from chromadb import PersistentClient

chroma_client = PersistentClient(path="./chroma_storage")

def search_chroma(query, collection_name):
    collection = chroma_client.get_collection(collection_name)
    results = collection.query(query_texts=[query], n_results=5)

    return [
        {"chunk": doc, "score": score}
        for doc, score in zip(results["documents"][0], results["distances"][0])
    ]
