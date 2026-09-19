# Lexigraph-AI

### Subtitle
**Production-grade Hybrid RAG System for Enterprise Legal Document Intelligence**

---

## Overview
The Legal Contract Intelligence Platform is a state-of-the-art Retrieval-Augmented Generation (RAG) system built to parse, index, search, and analyze legal contracts. It leverages hybrid search (sparse BM25 + dense vector embeddings), cross-encoder reranking, and LangGraph-guided conversational agents to deliver highly accurate Q&A, summarizing, extraction, and comparison workflows with zero hallucinations and strict citation enforcement.

---

## Key Features
- **Conversational AI Chatbot**: Interactive Claude/ChatGPT style chat interface.
- **LLM-powered Intent Classification**: Classifies user queries (QUESTION, COMPARE, EXTRACT, SUMMARIZE, CLAUSE_SEARCH) dynamically on the server side using GPT-4o-mini.
- **Structure-aware Document Chunking**: Segments legal contracts at natural section and clause boundaries while preserving hierarchical metadata.
- **Hybrid Retrieval Strategy**: Fuses vector embeddings (`BAAI/bge-base-en-v1.5`) and lexical tokens using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-orders candidates with `cross-encoder/ms-marco-MiniLM-L6-v2` and maps raw logits into a scaled confidence metric.
- **Citation Enforcer**: Extracts, verifies, and formats inline citations (e.g. `[DocName, PageX, ClauseY]`), stripping ungrounded claims.
- **Developer & Debug Mode**: Toggle dashboard display of intent, confidence ratings, and latency breakdowns.
- **Dockerized Deployments**: Clean container builds for FastAPI and Streamlit frontend.

---

## Tech Stack
- **Languages**: Python 3.13
- **Web APIs**: FastAPI, Uvicorn
- **Frontend Dashboard**: Streamlit
- **Agent Orchestrator**: LangGraph, LangChain
- **Databases & Stores**: ChromaDB, Pickle (for BM25 sparse indices)
- **Deep Learning**: PyTorch, SentenceTransformers (BAAI/bge-base-en-v1.5, cross-encoder/ms-marco-MiniLM-L6-v2)

---

## Project Structure
```
.
├── app/
│   ├── api/            # API Endpoints (health, chat, documents)
│   ├── chunking/       # Structure-aware Chunker
│   ├── config/         # Environment Settings and Logger Config
│   ├── confidence/     # Rerank & Hybrid confidence logic
│   ├── evaluation/     # LLM-judge Quality Evaluator
│   ├── graph/          # State and Orchestrator Graph builder
│   ├── indexing/       # BM25 pickle and ChromaDB indexers
│   ├── ingestion/      # Ingest pipelines & PDF Loader
│   ├── models/         # Pydantic schemas for document properties
│   ├── nodes/          # Process nodes (classify, retrieve, generate, citations)
│   ├── prompts/        # Context prompts for QA, Summarization, and Clause Search
│   ├── schemas/        # API request & response models
│   └── main.py         # Main FastAPI Server
├── docs/               # System architecture documentation
├── scripts/            # CLI Command Line Tools (Evaluate runner)
├── streamlit_app/      # Streamlit Frontend UI
├── tests/              # Pytest integration/unit test suite
├── docker-compose.yml  # Docker compose config
└── Dockerfile          # Python application Dockerfile
```

---

## How Hybrid Retrieval Works
1. **Direct Clause Lookup**: Scans queries for patterns like `Clause 7.1` or `Section 3` and matches them directly in ChromaDB.
2. **Dense Vector Match**: Computes embeddings for queries and locates cosine similarity matches inside Chroma.
3. **Lexical Sparse Match**: Evaluates keyword coverage using BM25.
4. **RRF Rank Fusion**: Combines rankings from dense and sparse lookups.
5. **Reranker**: Reranks top merged candidates using a Cross-Encoder transformer model.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legal-contract-intelligence.git
   cd legal-contract-intelligence
   ```
2. Setup virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Write configuration values to `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

---

## Running Locally

### Start Backend API Server
```bash
PYTHONPATH=. .venv/bin/uvicorn app.main:app --port 8000 --reload
```

### Start Streamlit Frontend UI
```bash
.venv/bin/streamlit run streamlit_app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker Deployment
Run the entire platform inside containerized services using:
```bash
docker compose up --build
```

---

## GitHub Actions CI
The workflow in `.github/workflows/ci.yml` runs all unit tests on pull requests and pushes to `main`.

## 🎬 Demo

![Legal Contract Intelligence Demo](assets/demo.gif)