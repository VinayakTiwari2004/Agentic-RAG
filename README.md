A modular FastAPI-based system that uses multiple intelligent agents to answer natural language queries using:
- 📄 Local document content
- 🌐 Crawled website data
- 🔎 Web search results via Tavily API
- 🧠 LLM (Granite) via Ollama

---

  How It Works

The system supports multi-source querying through 3 independent agents:
1. URL Agent: Crawls web content and stores in ChromaDB.
2. Document Agent: Extracts text from local PDFs and stores in ChromaDB.
3. Tavily Agent: Uses the Tavily search API to get real-time answers from the internet.

These agents are orchestrated via [LangGraph](https://github.com/langchain-ai/langgraph) and their outputs are passed to a local Granite LLM (served via Ollama) for final answer generation.

---

 Project Structure

```

MCP/
├── agents/             # Agent logic (URL, doc, Tavily, etc.)
├── core/               # Orchestration logic
├── graph/              # LangGraph-based dynamic flow builder
├── llm/                # Granite LLM and prompt templates
├── sample\_doc/         # Sample PDF used for testing
├── main.py             # FastAPI entrypoint
├── requirements.txt    # Python package dependencies
├── conda\_env.yml       # Conda environment (optional)
└── .gitignore

````

---

## ⚙️ Setup Instructions

###  1. Create and activate conda environment

```bash
#  Create a new conda environment (will use your default Python version)
conda create -n mcp_env

#  Activate the environment
conda activate mcp_env
````

### 📦 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

 3. Start the Ollama LLM server (ensure granite3.1-moe is pulled)

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

Stores content from a URL to ChromaDB.

```json
{
  "url": "https://timesofindia.indiatimes.com/"
}
```

🡒 Response:

```json
{
  "message": "URL data from ... stored in ChromaDB collection: <uuid>",
  "collection_id": "<uuid>"
}
```

---

 `POST /push/document`

Extracts text from a local PDF file and stores in ChromaDB.

```json
{
  "file_path": "./sample_doc/latestnews.pdf"
}
```

🡒 Response:

```json
{
  "message": "X chunks stored in ChromaDB",
  "collection_id": "<uuid>"
}
```

---

 `POST /query`

Query one or more agents (URL, document, Tavily) and get an LLM-generated answer.

```json
{
  "query": "What is the latest sports news?",
  "agents": ["chromadb", "document", "tavily"],
  "url_collection": "<uuid>",
  "doc_collection": "<uuid>"
}
```

🡒 Response:

```json
{
  "llm_response": "The latest sports news is...",
  "data_sources": ["chromadb_url", "chromadb_doc", "tavily"],
  "chunks": [ ... ]
}
```

---

 Notes

* The project supports dynamic agent composition: you can pass any combination of agents.
* ChromaDB is used for semantic storage and retrieval.
* LangGraph ensures the agents are executed in sequence based on the user query.
* Granite LLM (via Ollama) generates the final user-friendly response from all gathered context.

---

 Sample Input Files

 `sample_doc/latestnews.pdf` – A sample document containing latest news articles for testing document agent.

---


