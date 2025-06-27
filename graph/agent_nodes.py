from typing import Dict, Any
from agents.chroma_agent import search_chroma
from agents.tavily_agent import tavily_search
from core.constants import UNIVERSAL_KB_UUID


def chroma_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    collection_ids = state.get("url_collections", [])

    all_results = []
    for collection in collection_ids:
        results = search_chroma(query, collection)
        all_results.extend(results)
        state["chunks"] += [
            {
                "agent": "chromadb_url",
                "chunk": r["chunk"],
                "score": r["score"],
                "source_url": r.get("metadata", {}).get("source_url")  #  only source_url
            }
            for r in results
        ]
        state["context_parts"].append("\n".join(r["chunk"] for r in results))

    if all_results:
        state["data_sources"].append("chromadb_url")
    return state


def document_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    collection_ids = state.get("doc_collections", [])

    all_results = []
    for collection in collection_ids:
        results = search_chroma(query, collection)
        all_results.extend(results)
        state["chunks"] += [
            {
                "agent": "chromadb_doc",
                "chunk": r["chunk"],
                "score": r["score"],
                "filename": r.get("metadata", {}).get("filename")  #  only filename
            }
            for r in results
        ]
        state["context_parts"].append("\n".join(r["chunk"] for r in results))

    if all_results:
        state["data_sources"].append("chromadb_doc")
    return state


def tavily_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]
    results = tavily_search(query)

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

    # Dynamically include whatever metadata is present
    state["chunks"] += [
        {
            "agent": "universal_kb",
            "chunk": r["chunk"],
            "score": r["score"],
            **r.get("metadata", {})  #  Merge filename or source_url if present
        }
        for r in results
    ]
    state["context_parts"].append("\n".join(r["chunk"] for r in results))
    state["data_sources"].append("universal_kb")
    return state
