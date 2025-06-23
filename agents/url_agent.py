# agents/url_agent.py

import requests
from bs4 import BeautifulSoup
from chromadb import PersistentClient

# Initialize ChromaDB persistent client
chroma_client = PersistentClient(path="./chroma_storage")

def crawl_and_store_url(url: str, collection_name: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator="\n")

    # Break into chunks (optional: can customize later)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Store in ChromaDB under the specified collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for idx, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"{collection_name}_chunk_{idx}"])

    return f" {len(chunks)} chunks stored in ChromaDB under '{collection_name}'"
