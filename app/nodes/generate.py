import os
import re
import traceback
from typing import Dict, Any
from openai import OpenAI

from app.config.settings import settings
from app.config.logging import logger
from app.graph.state import GraphState

QA_SYSTEM_PROMPT = """You are a professional legal contract assistant.
Your task is to answer the user's question based strictly on the provided contract context chunks.

Guidelines:
1. Base your answers ONLY on the provided context chunks.
2. If multiple retrieved documents contain relevant information, synthesize the information across all supporting documents. Do not answer from only the first document.
3. Merge duplicate information, highlight differences when appropriate, and produce one coherent answer.
4. Cite every document that materially contributed to the answer inline using [DocName, PageX, ClauseY] or [DocName, PageX, SectionZ].
5. Never hallucinate or invent clauses, facts, or citations.
6. If the context is insufficient or does not contain the answer, explicitly state: "I couldn't find enough information in the uploaded documents."

Context Chunks:
{context}
"""

SUMMARIZE_SYSTEM_PROMPT = """You are a professional legal assistant.
Your task is to summarize the provided contract context chunks into concise, clear bullet points.

Guidelines:
1. Base your summary ONLY on the provided context chunks.
2. Be concise, clear, and preserve factual accuracy. Do not use outside knowledge.
3. Format the summary as a list of bullet points.
4. Cite your sources inline using [DocName, PageX, ClauseY] or [DocName, PageX, SectionZ].
5. If the context is empty, state: "I couldn't find enough information in the uploaded documents."

Context Chunks:
{context}
"""

CLAUSE_SEARCH_SYSTEM_PROMPT = """You are a professional legal assistant.
Your task is to locate and explain the specific clause requested in plain English based strictly on the provided context chunks.

Guidelines:
1. Explain ONLY the requested clause. Do not summarize the entire document.
2. Base your explanation strictly on the provided context chunks.
3. Cite your sources inline using [DocName, PageX, ClauseY] or [DocName, PageX, SectionZ].
4. If the context does not contain the requested clause, state: "I couldn't find enough information in the uploaded documents."

Context Chunks:
{context}
"""

def get_openai_client() -> OpenAI:
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Generates the grounded response from validated context chunks using GPT."""
    chunks = state.get("validated_chunks", [])
    errors = list(state.get("errors", []))
    intent = state.get("intent", "UNKNOWN")
    query = state.get("user_query", "")

    if not chunks:
        return {
            "generated_answer": state.get("generated_answer", ""),
            "errors": errors
        }

    # Format the context chunks for the prompt
    context_blocks = []
    for chunk in chunks:
        doc_ref = chunk.document_name
        page_ref = f"Page {chunk.page_number}"
        clause_id = chunk.clause_number or chunk.section_number or "N/A"
        context_blocks.append(
            f"[{doc_ref}, {page_ref}, Clause {clause_id}]:\n{chunk.text}"
        )
    context_str = "\n\n".join(context_blocks)

    # Determine system prompt based on intent
    if intent == "SUMMARIZE":
        logger.info("Entering Summarize Node")
        logger.info(f"Chunks Retrieved: {len(chunks)}")
        system_prompt = SUMMARIZE_SYSTEM_PROMPT.format(context=context_str)
        logger.info(f"Prompt Length: {len(system_prompt)}")
    elif intent == "CLAUSE_SEARCH":
        logger.info("Entering Clause Search Node")
        req_match = re.search(r'\b(?:clause|section|cl\.?|sec\.?)\s*(\d+(?:\.\d+)*)\b', query, re.IGNORECASE)
        requested_clause = req_match.group(1) if req_match else "N/A"
        retrieved_clause = chunks[0].clause_number if chunks else "N/A"
        logger.info(f"Requested Clause: {requested_clause}")
        logger.info(f"Retrieved Clause: {retrieved_clause}")
        system_prompt = CLAUSE_SEARCH_SYSTEM_PROMPT.format(context=context_str)
        logger.info(f"Prompt Length: {len(system_prompt)}")
    else:
        system_prompt = QA_SYSTEM_PROMPT.format(context=context_str)

    try:
        logger.info("Calling GPT")
        client = get_openai_client()
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens
        )
        answer = response.choices[0].message.content or ""
        logger.info("GPT Response: " + answer[:100] + "...")
        
        if intent == "SUMMARIZE":
            logger.info("Leaving Summarize Node")
        elif intent == "CLAUSE_SEARCH":
            logger.info("Leaving Clause Search Node")

        return {
            "generated_answer": answer.strip(),
            "errors": errors
        }
    except Exception as e:
        # Print full traceback on LLM call exceptions
        tb = traceback.format_exc()
        print(f"Exception during LLM execution:\n{tb}")
        logger.error(f"OpenAI call failed: {e}")
        errors.append(f"OpenAI call failed: {str(e)}\nTraceback:\n{tb}")
        
        # Reraise/fail instead of silent fallback if requested by task details
        raise e
