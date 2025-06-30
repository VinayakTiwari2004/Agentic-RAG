import asyncio
import sys
import uuid
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig, BrowserConfig

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from core.constants import UNIVERSAL_KB_UUID

from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added for better chunking

# Windows compatibility fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize ChromaDB client and embedding model
chroma_client = PersistentClient(path="./chroma_storage")
embedding_func = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en")


async def crawl_and_extract_text(url: str):
    browser_cfg = BrowserConfig()
    run_cfg = CrawlerRunConfig(
        exclude_all_images=True,
        remove_overlay_elements=True,
        process_iframes=True
    )
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        if not result.success:
            raise Exception(f"Crawl failed: {result.error_message}")
        text = result.markdown or result.raw_html or ""
        return text


def crawl_and_store_url(url: str, collection_name: str):
    # Uses fixed asyncio.run with Windows event loop
    text = asyncio.run(crawl_and_extract_text(url))

    # Use semantic-aware text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)

    # Embed chunks
    embeddings = embedding_func(chunks)

    # Add to individual URL collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{collection_name}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source_url": url} for _ in range(len(chunks))]  # Add metadata
    )

    # Also store in master universal collection
    master = chroma_client.get_or_create_collection(
        name=UNIVERSAL_KB_UUID,
        embedding_function=embedding_func
    )
    master.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{UNIVERSAL_KB_UUID}_{uuid.uuid4()}_{i}" for i in range(len(chunks))],
        metadatas=[{"source_url": url} for _ in range(len(chunks))]  # Add metadata
    )

    return f"{len(chunks)} chunks stored in ChromaDB under '{collection_name}'"