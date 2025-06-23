# agents/doc_agent.py

import fitz  # PyMuPDF
import uuid
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ✅ new

#  ChromaDB client & embedding model
chroma_client = PersistentClient(path="./chroma_storage")
embedding_func = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en")


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


#  Updated chunking using Langchain
def split_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def ingest_document(file_path):
    collection_name = str(uuid.uuid4())

    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)
    embeddings = embedding_func(chunks)

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{collection_name}_{i}" for i in range(len(chunks))]
    )

    return {
        "message": f" {len(chunks)} chunks stored in ChromaDB",
        "collection_id": collection_name
    }
