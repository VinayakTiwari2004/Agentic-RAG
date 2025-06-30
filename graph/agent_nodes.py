from typing import Dict, Any
from agents.chroma_agent import search_chroma
from agents.tavily_agent import tavily_search
from core.constants import UNIVERSAL_KB_UUID
from utils.reranker import rerank_chunks  #  Import your reranking utility


def chroma_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    collection_ids = state.get("url_collections", [])

    all_results = []
    for collection in collection_ids:
        results = search_chroma(query, collection)
        all_results.extend(results)

    if all_results:
        reranked = rerank_chunks(query, all_results, top_n=10)

        state["chunks"] += [
            {
                "agent": "crawler",  
                "chunk": r["chunk"],
                "score": round(r["rerank_score"], 2),
                "source_url": r.get("metadata", {}).get("source_url")
            }
            for r in reranked
        ]
        state["context_parts"].append("\n".join(r["chunk"] for r in reranked))
        state["data_sources"].append("crawler")  

    return state



def document_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    collection_ids = state.get("doc_collections", [])

    all_results = []
    for collection in collection_ids:
        results = search_chroma(query, collection)
        all_results.extend(results)

    if all_results:
        #  Rerank and limit to top 10
        reranked = rerank_chunks(query, all_results, top_n=10)

        #  Add to state
        state["chunks"] += [
            {
                "agent": "chromadb_doc",
                "chunk": r["chunk"],
                "score": round(r["rerank_score"], 2),
                "filename": r.get("metadata", {}).get("filename")
            }
            for r in reranked
        ]
        state["context_parts"].append("\n".join(r["chunk"] for r in reranked))
        state["data_sources"].append("chromadb_doc")

    return state


def tavily_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    results = tavily_search(query)

    if results:
        state["chunks"] += [
            {
                "agent": "tavily",
                "url": r["url"],
                "content": r["content"]
            }
            for r in results
        ]
        state["context_parts"].append("\n".join(r["content"] for r in results))
        state["data_sources"].append("tavily")

    return state


def universal_kb_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    results = search_chroma(query, UNIVERSAL_KB_UUID)

    if results:
        #  Rerank and limit to top 10
        reranked = rerank_chunks(query, results, top_n=10)

        #  Add to state
        state["chunks"] += [
            {
                "agent": "universal_kb",
                "chunk": r["chunk"],
                "score": round(r["rerank_score"], 2),
                **r.get("metadata", {})
            }
            for r in reranked
        ]
        state["context_parts"].append("\n".join(r["chunk"] for r in reranked))
        state["data_sources"].append("universal_kb")

    return state
