# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from core.orchestrator import get_response
from agents.url_agent import crawl_and_store_url
from agents.doc_agent import ingest_document

from uuid import uuid4

app = FastAPI()

# =====================
#  Request & Response Schemas
# =====================

class QueryRequest(BaseModel):
    query: str
    agents: List[str]  # ["chromadb", "document", "tavily"]
    url_collection: Optional[str] = None
    doc_collection: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class QueryResponse(BaseModel):
    llm_response: str
    data_sources: List[str]
    chunks: List[Dict[str, Any]]

    class Config:
        arbitrary_types_allowed = True


class UrlPushRequest(BaseModel):
    url: str

    class Config:
        arbitrary_types_allowed = True


class UrlPushResponse(BaseModel):
    message: str
    collection_id: str

    class Config:
        arbitrary_types_allowed = True


class DocPushRequest(BaseModel):
    file_path: str

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

@app.post("/push/url", response_model=UrlPushResponse)
async def push_url(data: UrlPushRequest):
    collection_id = str(uuid4())
    crawl_and_store_url(data.url, collection_id)
    return {
        "message": f" URL data from {data.url} stored in ChromaDB collection: {collection_id}",
        "collection_id": collection_id
    }

@app.post("/push/document", response_model=DocPushResponse)
async def push_document(data: DocPushRequest):
    result = ingest_document(data.file_path)
    return result  #  directly return the dict from doc_agent

@app.post("/query", response_model=QueryResponse)
async def query_llm(payload: QueryRequest):
    result = get_response(
        user_query=payload.query,
        agents=payload.agents,
        url_collection=payload.url_collection,
        doc_collection=payload.doc_collection
    )
    return QueryResponse(**result)
