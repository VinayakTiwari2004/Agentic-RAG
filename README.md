
---

```markdown
Multi-Agent Coordination Protocol (MCP)

A modular FastAPI-based system that uses multiple intelligent agents to answer natural language queries using:
- Local document content  
- Crawled website data  
- Web search results via Tavily API  
- LLM (Granite) via Ollama  

---

How It Works

The system supports multi-source querying through 4 independent agents:
1. URL Agent: Crawls web content and stores in ChromaDB.
2. Document Agent: Extracts text from local PDFs and stores in ChromaDB.
3. Tavily Agent: Uses the Tavily search API to get real-time answers from the internet.
4. Universal KB Agent: Queries from a master collection that stores all pushed URLs and documents together.

These agents are orchestrated via [LangGraph](https://github.com/langchain-ai/langgraph), and their outputs are passed to a local Granite LLM (served via Ollama) for final answer generation.

---

Project Structure

```

MCP/
├── agents/ # Agent logic (URL, doc, Tavily, etc.)
├── core/ # Orchestration logic
├── graph/ # LangGraph-based dynamic flow builder
├── llm/ # Granite LLM and prompt templates
├── sample_doc/ # Sample PDF used for testing
├── utils/ # Reranker logic (e.g., reranker.py)
├── main.py # FastAPI entrypoint
├── requirements.txt # Python package dependencies
├── conda_env.yml # Conda environment (optional)
└── .gitignore

````

---

Setup Instructions

1. Create and activate conda environment

```bash
#  Create a new conda environment (will use your default Python version)
conda create -n mcp_env

#  Activate the environment
conda activate mcp_env
````


2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. Start the Ollama LLM server (ensure `granite3.1-moe` is pulled)

```bash
ollama run granite3.1-moe
```

4. Run the FastAPI server

```bash
uvicorn main:app --reload
```

Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to access Swagger UI.

---

Available API Endpoints

`POST /push/url`

Stores content from one or more URLs into ChromaDB.

Request:

```json
{
  "urls": ["https://timesofindia.indiatimes.com/"]
}
```

Response:

```json
[
  {
    "message": "X chunks stored in ChromaDB under 'uuid'",
    "collection_id": "uuid"
  }
]
```

---

`POST /push/document`

Extracts text from one or more local PDF files and stores in ChromaDB.

Request:

```json
{
  "file_paths": ["./sample_doc/latestnews.pdf"]
}
```

Response:

```json
[
  {
    "message": "X chunks stored in ChromaDB",
    "collection_id": "uuid"
  }
]
```

---

`POST /query`

Query one or more agents (URL, document, Tavily, universal KB) and get an LLM-generated answer.

Request:

```json
{
  "query": "What is the latest sports news?",
  "agents": ["crawler", "document", "tavily", "universal_kb"],
  "url_collections": ["uuid1"],
  "doc_collections": ["uuid2"]
}
```

Response:

```json
{
  "llm_response": "The latest sports news is...",
  "data_sources": ["crawler", "chromadb_doc", "tavily", "universal_kb"],
  "chunks": [
    {
      "agent": "crawler",
      "chunk": "...",
      "score": 0.91,
      "source_url": "https://example.com"
    },
    {
      "agent": "chromadb_doc",
      "chunk": "...",
      "score": 0.89,
      "filename": "latestnews.pdf"
    },
    {
      "agent": "universal_kb",
      "chunk": "...",
      "score": 0.92,
      "source_url": "https://example.com"
    },
    {
      "agent": "tavily",
      "url": "https://result.from.api",
      "content": "Some paragraph from the internet"
    }
  ]
}
```

Each chunk contains relevant metadata:

* `"source_url"` if it came from a URL
* `"filename"` if it came from a document
* `"url"` and `"content"` if it came from Tavily

---

Notes

* The system supports dynamic agent selection via LangGraph.
* ChromaDB is used for semantic chunk storage and retrieval.
* A universal knowledge base (UNIVERSAL\_KB\_UUID) is used for cross-source querying.
* Final response is generated using Granite LLM served via Ollama.
* Tavily allows real-time search results from the open web.

---

Sample Input Files
You can store any PDF file you want to ingest inside the `sample_doc/` directory. This directory is intended for storing sample documents used during testing.

---


