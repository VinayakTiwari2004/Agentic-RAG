# graph/agent_nodes.py

from typing import Dict, Any

from agents.chroma_agent import search_chroma
from agents.tavily_agent import tavily_search


# Wrapper node for ChromaDB (URL collection)
def chroma_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]  
    collection = state.get("url_collection")

    if not collection:
        return state

    results = search_chroma(query, collection)

    state["chunks"] += [{"agent": "chromadb_url", "chunk": r["chunk"], "score": r["score"]} for r in results]
    state["context_parts"].append("\n".join(r["chunk"] for r in results))
    state["data_sources"].append("chromadb_url")
    return state


# Wrapper node for ChromaDB (Document collection)
def document_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]  
    collection = state.get("doc_collection")

    if not collection:
        return state

    results = search_chroma(query, collection)

    state["chunks"] += [{"agent": "chromadb_doc", "chunk": r["chunk"], "score": r["score"]} for r in results]
    state["context_parts"].append("\n".join(r["chunk"] for r in results))
    state["data_sources"].append("chromadb_doc")
    return state


# Wrapper node for Tavily
def tavily_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["user_query"]  
    results = tavily_search(query)

    state["chunks"] += [{"agent": "tavily", "url": r["url"], "content": r["content"]} for r in results]
    state["context_parts"].append("\n".join(r["content"] for r in results))
    state["data_sources"].append("tavily")
    return state
