# graph/query_graph.py

from langgraph.graph import StateGraph, END
from graph.agent_nodes import chroma_node, document_node, tavily_node, universal_kb_node
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    user_query: str
    url_collections: List[str]
    doc_collections: List[str]
    chunks: List[Dict[str, Any]]
    context_parts: List[str]
    data_sources: List[str]

def build_query_graph(selected_agents: List[str]):
    builder = StateGraph(AgentState)

    agent_node_map = {
        "crawler": ("crawler", chroma_node),  #  updated key and node_name
        "document": ("document", document_node),
        "tavily": ("tavily", tavily_node),
        "universal_kb": ("universal_kb", universal_kb_node)
    }

    prev_node = None

    for agent in selected_agents:
        node_name, node_fn = agent_node_map.get(agent, (None, None))
        if node_name is None:
            continue

        builder.add_node(node_name, node_fn)

        if prev_node:
            builder.add_edge(prev_node, node_name)

        prev_node = node_name

    if prev_node:
        builder.add_edge(prev_node, END)
        builder.set_entry_point(selected_agents[0])

    return builder.compile()
