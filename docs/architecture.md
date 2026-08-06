# Architecture Documentation - Legal Contract Intelligence Platform

This document outlines the system architecture, ingestion pipelines, hybrid retrieval schemas, and orchestration workflows.

---

## 1. Overall System Architecture

The overall pipeline manages the flow of incoming queries, parses intents using LLMs, coordinates vector/lexical retrieval, reranks results, generates answers, and enforces formatting constraints.

```mermaid
graph TD
    User([User Client]) -->|Query / Chat| API[FastAPI Server]
    API -->|message| Graph[LangGraph Workflow]
    
    subgraph LangGraph Orchestrator
        Graph -->|START| Classify[Intent Classification Node]
        Classify -->|Conditional Route| Retrieve[Retrieve Node]
        Retrieve --> Validate[Retrieval Validation Node]
        Validate -->|PASSED| Generate[Generation Node]
        Validate -->|FAILED| Format[Response Formatter Node]
        Generate --> Citations[Citation Enforcer Node]
        Citations --> Format
        Format -->|END| Graph
    end
    
    Retrieve -->|Search request| Engine[Hybrid Retrieval Engine]
    Engine -->|Lexical Match| BM25[(BM25 Pickle Index)]
    Engine -->|Semantic Match| Chroma[(ChromaDB Persistent Store)]
    Engine --> Rerank[Cross-Encoder Reranker]
    
    Generate -->|Synthesize prompts| OpenAI[GPT-4o-mini LLM]
    
    Format -->|Unified JSON Response| API
    API -->|Response payload| User
```

---

## 2. Document Ingestion Pipeline

Ingestion reads PDF documents, extracts dynamic metadata, segments text using structure-aware rules, and updates indices.

```mermaid
graph LR
    PDF[PDF Contract] --> Parse[PyPDF Parser]
    Parse --> Metadata[Dynamic Metadata Extractor]
    Parse --> Segment[Structure-aware Chunker]
    
    Segment -->|Chunks + Metadata| Embed[Dense Embeddings BGE]
    Embed --> Chroma[(ChromaDB Persistent Store)]
    
    Segment -->|Tokenized Text| BM25Index[BM25 Indexer]
    BM25Index --> BM25[(BM25 Pickle Store)]
```

---

## 3. Hybrid Retrieval Pipeline

The retrieval strategy combines dense vector similarity matching and sparse BM25 lookup rank scores using Reciprocal Rank Fusion (RRF), followed by Cross-Encoder rerank scoring.

```mermaid
graph TD
    Query([Query]) --> Direct{Direct Clause / Section Lookup?}
    Direct -->|Yes| DirectMatch[Chroma metadata query]
    
    Query --> Lexical[BM25 Search]
    Query --> Semantic[Chroma Vector Search]
    
    Lexical --> RRF[Reciprocal Rank Fusion RRF]
    Semantic --> RRF
    
    DirectMatch --> Merge[Merge Candidates]
    RRF --> Merge
    
    Merge --> Rerank[Cross-Encoder ms-marco-MiniLM-L6-v2]
    Rerank --> Score[Sigmoid Logit Scaled Confidence]
    Score --> Output([Retrieved Chunks])
```

---

## 4. LangGraph QA Workflow

The orchestration graph uses conditional edges to route states dynamically based on validation passes and intent flags.

```mermaid
stateDiagram-v2
    [*] --> Classify_Node
    Classify_Node --> Retrieve_Node : Conditional Edge (Intent Route)
    Retrieve_Node --> Validate_Node
    
    state check_validation <<choice>>
    Validate_Node --> check_validation
    check_validation --> Generate_Node : PASSED
    check_validation --> Format_Node : FAILED
    
    Generate_Node --> Citations_Node
    Citations_Node --> Format_Node
    Format_Node --> [*]
```
