# Changelog

All notable changes to the Legal Contract Intelligence Platform project will be documented in this file.

---

## [v1.0.0] - 2026-08-06

### Features
- **ChatGPT-Style Frontend Redesign**: Conversational chat interface supporting multi-document upload, delete buttons, and system resets in the sidebar.
- **Server-Side Intent Classification**: Integrated a structured LLM classification node inside the LangGraph workflow using GPT-4o-mini to route queries into QUESTION, CLAUSE_SEARCH, COMPARE, EXTRACT, and SUMMARIZE paths.
- **Synthesized RAG Responses**: Enhanced system prompts to merge, align, and synthesize information across multiple supporting contract documents.
- **Developer Debug Mode**: Added a sidebar toggle to inspect classified intent, normalized confidence ratings, validation status, retrieved count metrics, and latency details.

### Architecture
- **Reciprocal Rank Fusion (RRF)**: Sparsely tokenized BM25 ranking fused with dense vector search similarity ranking.
- **Cross-Encoder Reranking**: Re-orders fused candidates to scale confidence ratings.
- **LangGraph Coordination**: Strict citation enforcement nodes stripping ungrounded claims.

### Evaluation & Testing
- **LLM-Judge Evaluation Suite**: Custom pipeline calculating Faithfulness, Answer Relevancy, Context Precision, and Context Recall metrics using GPT-4o-mini judges.
- **Pytest Suite**: Complete unit and integration test coverage ($32/32$ tests green).

### Known Limitations
- Reranker models are executed locally on CPU/MPS, which may cause higher latencies during local runs compared to GPU servers.
- OpenAI structured output completions require active internet connection and valid credentials.
