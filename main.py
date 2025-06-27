import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4

from core.orchestrator import get_response
from agents.url_agent import crawl_and_store_url
from agents.doc_agent import ingest_document
import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


app = FastAPI()

# =====================
#  Request & Response Schemas
# =====================

class QueryRequest(BaseModel):
    query: str
    agents: List[str]
    url_collections: Optional[List[str]] = None
    doc_collections: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True


class QueryResponse(BaseModel):
    llm_response: str
    data_sources: List[str]
    chunks: List[Dict[str, Any]]

    class Config:
        arbitrary_types_allowed = True


class UrlPushRequest(BaseModel):
    urls: List[str]  # Accepts multiple URLs

    class Config:
        arbitrary_types_allowed = True


class UrlPushResponse(BaseModel):
    message: str
    collection_id: str

    class Config:
        arbitrary_types_allowed = True


class DocPushRequest(BaseModel):
    file_paths: List[str]  # Accepts multiple file paths

    class Config:
        arbitrary_types_allowed = True


class DocPushResponse(BaseModel):
    message: str
    collection_id: str

    class Config:
        arbitrary_types_allowed = True


# =====================
#  Endpoints
# =====================

@app.post("/push/url", response_model=List[UrlPushResponse])
def push_urls(data: UrlPushRequest):
    responses = []
    for url in data.urls:
        collection_id = str(uuid4())
        message = crawl_and_store_url(url, collection_id)
        responses.append({
            "message": message,
            "collection_id": collection_id
        })
    return responses


@app.post("/push/document", response_model=List[DocPushResponse])
async def push_documents(data: DocPushRequest):
    responses = []
    for path in data.file_paths:
        result = ingest_document(path)
        responses.append(result)
    return responses


@app.post("/query", response_model=QueryResponse)
async def query_llm(payload: QueryRequest):
    result = get_response(
        user_query=payload.query,
        agents=payload.agents,
        url_collections=payload.url_collections,
        doc_collections=payload.doc_collections
    )
    return QueryResponse(**result)
