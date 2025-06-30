import fitz  # PyMuPDF
import uuid
import os  # for extracting filename
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.constants import UNIVERSAL_KB_UUID  # Import constant

# OCR and format detection
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import docx2txt
from pathlib import Path
import mimetypes

# Set Tesseract path (adjust as per your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ChromaDB client & embedding model
chroma_client = PersistentClient(path="./chroma_storage")
embedding_func = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en")


def extract_text_from_file(file_path):
    file_ext = Path(file_path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(file_path)

    if file_ext == ".pdf":
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
        if text.strip():
            return text

        # OCR fallback for scanned PDF
        images = convert_from_path(file_path)
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image)
        return ocr_text

    elif file_ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    elif file_ext in [".doc", ".docx"]:
        return docx2txt.process(file_path)

    elif file_ext == ".txt":
        return Path(file_path).read_text()

    else:
        return "[Unsupported file format]"


def split_text(text, chunk_size=1000, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def ingest_document(file_path):
    collection_name = str(uuid.uuid4())

    # Extract clean filename (e.g., warning1.pdf)
    filename = os.path.basename(file_path)

    text = extract_text_from_file(file_path)
    chunks = split_text(text)
    embeddings = embedding_func(chunks)

    metadatas = [{"filename": filename} for _ in chunks]  # same metadata for all chunks

    # Store in per-document collection
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{collection_name}_{i}" for i in range(len(chunks))],
        metadatas=metadatas  # added metadata
    )

    # Also store in master knowledge base
    master_collection = chroma_client.get_or_create_collection(
        name=UNIVERSAL_KB_UUID,
        embedding_function=embedding_func
    )
    master_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{UNIVERSAL_KB_UUID}_{uuid.uuid4()}_{i}" for i in range(len(chunks))],
        metadatas=metadatas  # added metadata
    )

    return {
        "message": f"{len(chunks)} chunks stored in ChromaDB",
        "collection_id": collection_name
    }