from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load embedding model
embedding_func = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en")

# Initialize ChromaDB client
chroma_client = PersistentClient(path="./chroma_storage")

def search_chroma(query: str, collection_name: str, top_k: int = 15):  
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    return [
        {
            "chunk": doc,
            "score": round((1 - dist) * 100, 2),  
            "metadata": metadata
        }
        for doc, dist, metadata in zip(results["documents"][0], results["distances"][0], results["metadatas"][0])
    ]
