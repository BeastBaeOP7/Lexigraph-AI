import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.nodes.generate import get_openai_client
from app.retrieval.bm25.bm25_search import BM25Searcher
from app.config.logging import logger

class IntentClassification(BaseModel):
    intent: str = Field(..., description="One of: QUESTION, CLAUSE_SEARCH, COMPARE, EXTRACT, SUMMARIZE, UNKNOWN")
    documents: List[str] = Field(default_factory=list, description="List of matched document names mentioned in the query.")
    reason: str = Field(..., description="Brief explanation for classification.")

EXTRACTION_PROMPTS = {
    "obligations": "What are the primary obligations and responsibilities of the parties in this contract?",
    "payment_terms": "What are the payment terms, invoicing schedules, and pricing details?",
    "termination_conditions": "What are the termination conditions, notice periods, and breach consequences?",
    "renewal_clauses": "What are the renewal terms, automatic extension clauses, and options?"
}

def classify_intent_node(state: GraphState) -> Dict[str, Any]:
    """
    LLM-powered node that classifies user query intent and extracts target document restrictions.
    """
    query = state.get("user_query", "")
    logger.info(f"Classifying query intent: {query}")
    
    active_docs = []
    try:
        searcher = BM25Searcher()
        active_docs = list(set(chunk.document_name for chunk in searcher.chunks))
    except Exception:
        pass
        
    client = get_openai_client()
    system_prompt = (
        "You are an intelligent legal contract AI assistant classifier.\n"
        "Your task is to classify the user request into one of the following intents:\n"
        "- QUESTION: A general question or lookup request about contract terms. (Default choice if the query is document-related but does not clearly belong to another intent).\n"
        "- CLAUSE_SEARCH: A request to locate, explain, or search for specific section/clause numbers.\n"
        "- COMPARE: A comparative request between two or more documents.\n"
        "- EXTRACT: An information extraction request (obligations, payment terms, termination, renewal).\n"
        "- SUMMARIZE: A request to summarize a document.\n"
        "- UNKNOWN: Fallback ONLY for queries completely unrelated to the documents (e.g. general chit-chat, programming questions, weather).\n\n"
        f"Active Indexed Documents: {active_docs}\n\n"
        "Few-shot Examples:\n"
        "1. QUESTION:\n"
        "   - Input: 'What is my probation period?' -> Intent: QUESTION\n"
        "   - Input: 'What are my working hours?' -> Intent: QUESTION\n"
        "   - Input: 'Can I work remotely?' -> Intent: QUESTION\n"
        "   - Input: 'What happens if I resign?' -> Intent: QUESTION\n"
        "2. CLAUSE_SEARCH:\n"
        "   - Input: 'Explain Clause 4.2' -> Intent: CLAUSE_SEARCH\n"
        "   - Input: 'What does Section 5 say?' -> Intent: CLAUSE_SEARCH\n"
        "3. COMPARE:\n"
        "   - Input: 'Compare confidentiality clauses.' -> Intent: COMPARE\n"
        "   - Input: 'How does NDA differ from Vendor Agreement?' -> Intent: COMPARE\n"
        "4. EXTRACT:\n"
        "   - Input: 'Extract payment terms.' -> Intent: EXTRACT\n"
        "   - Input: 'Extract employee obligations.' -> Intent: EXTRACT\n"
        "5. SUMMARIZE:\n"
        "   - Input: 'Summarize the Employee Handbook.' -> Intent: SUMMARIZE\n"
        "6. UNKNOWN:\n"
        "   - Input: 'How do I post company code on GitHub?' -> Intent: UNKNOWN\n"
        "   - Input: 'What is the capital of France?' -> Intent: UNKNOWN\n\n"
        "Match any user mentions of documents to the exact filenames in the active list. "
        "Output structured JSON matching the schema."
    )
    
    intent = "UNKNOWN"
    docs = []
    
    try:
        res = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format=IntentClassification,
            temperature=0.0
        )
        parsed = res.choices[0].message.parsed
        if parsed:
            intent = parsed.intent.upper().strip()
            docs = parsed.documents
            logger.info(f"Classified Intent: {intent}, Matched Docs: {docs}")
    except Exception as e:
        logger.error(f"Intent classification failed: {e}. Defaulting to General QA.")
        intent = "QUESTION" # Default to QUESTION on classification failure
        
    metadata = state.get("metadata", {})
    if "filter" not in metadata:
        metadata["filter"] = {}
        
    if docs:
        metadata["filter"]["document_name"] = docs
        
    if intent == "SUMMARIZE":
        state["user_query"] = "Provide a comprehensive summary of this document highlighting the key terms, dates, parties, and core purpose."
        state["metadata"] = metadata
        state["intent"] = "SUMMARIZE"
        return state
        
    if intent == "EXTRACT":
        query_lower = query.lower()
        ext_type = "obligations"
        if any(kw in query_lower for kw in ["payment", "fee", "pricing", "invoice"]):
            ext_type = "payment_terms"
        elif any(kw in query_lower for kw in ["termination", "terminate", "breach"]):
            ext_type = "termination_conditions"
        elif any(kw in query_lower for kw in ["renewal", "renew", "extension"]):
            ext_type = "renewal_clauses"
            
        state["user_query"] = EXTRACTION_PROMPTS[ext_type]
        metadata["extraction_type"] = ext_type
        state["metadata"] = metadata
        state["intent"] = "EXTRACT"
        return state

    state["intent"] = intent
    state["metadata"] = metadata
    return state
