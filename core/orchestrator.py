# core/orchestrator.py

from graph.query_graph import build_query_graph
from llm.granite_llm import query_local_llm

def get_response(user_query: str, agents: list, url_collections=None, doc_collections=None) -> dict:
    print(" Incoming Query:", user_query)
    print(" Active Agents:", agents)

    # Initial state
    state = {
        "user_query": user_query,
        "url_collections": url_collections or [],
        "doc_collections": doc_collections or [],
        "chunks": [],
        "context_parts": [],
        "data_sources": []
    }

    # Dynamically build and run graph
    query_graph = build_query_graph(agents)
    final_state = query_graph.invoke(state)

    combined_context = "\n".join(final_state["context_parts"])
    llm_response = query_local_llm(combined_context, user_query)

    return {
        "llm_response": llm_response,
        "data_sources": final_state["data_sources"],
        "chunks": final_state["chunks"]
    }
