import re
from typing import Dict, Any, List
from app.graph.state import GraphState

def enforce_citations_node(state: GraphState) -> Dict[str, Any]:
    """Parses and validates citations inside the generated text against retrieved chunks."""
    answer = state.get("generated_answer", "")
    chunks = state.get("validated_chunks", [])
    errors = list(state.get("errors", []))

    if not answer or not chunks:
        return {
            "citations": [],
            "generated_answer": answer,
            "errors": errors
        }

    # Matches [DocName, PageX, ClauseY] or [DocName, PageX, SectionY] or [DocName, PageX]
    pattern = r'\[([^,\]]+),\s*(?:Page\s*)?(\d+)(?:,\s*(?:Clause|Section|cl\.?|sec\.?)?\s*([^\]]+))?\]'
    
    verified_citations = []
    
    def clean_name(name: str) -> str:
        return name.lower().replace(".pdf", "").strip()

    def replacer(match: re.Match) -> str:
        doc_part = match.group(1).strip()
        page_part = match.group(2).strip()
        clause_part = match.group(3).strip().lower() if match.group(3) else ""
        
        doc_key = clean_name(doc_part)
        
        is_valid = False
        matched_chunk = None
        
        # Look for matching validated chunks
        for chunk in chunks:
            c_doc_key = clean_name(chunk.document_name)
            c_page = str(chunk.page_number)
            c_clause = str(chunk.clause_number or "").lower().strip()
            c_sec = str(chunk.section_number or "").lower().strip()
            
            if c_doc_key == doc_key and c_page == page_part:
                if not clause_part or clause_part == c_clause or clause_part == c_sec or clause_part in c_clause or clause_part in c_sec:
                    is_valid = True
                    matched_chunk = chunk
                    break
                    
        # Fallback to document + page match
        if not is_valid:
            for chunk in chunks:
                c_doc_key = clean_name(chunk.document_name)
                c_page = str(chunk.page_number)
                if c_doc_key == doc_key and c_page == page_part:
                    is_valid = True
                    matched_chunk = chunk
                    break

        if is_valid and matched_chunk:
            citation_info = {
                "document_name": matched_chunk.document_name,
                "page_number": matched_chunk.page_number,
                "section_number": matched_chunk.section_number,
                "section_title": matched_chunk.section_title,
                "clause_number": clause_part.upper().strip() if clause_part else matched_chunk.clause_number
            }
            if citation_info not in verified_citations:
                verified_citations.append(citation_info)
            return match.group(0)
        else:
            errors.append(f"Removed unverified citation tag: {match.group(0)}")
            return ""

    cleaned_answer = re.sub(pattern, replacer, answer)
    # Re-align spacing in case we removed items
    cleaned_answer = re.sub(r'\s{2,}', ' ', cleaned_answer).strip()

    return {
        "generated_answer": cleaned_answer,
        "citations": verified_citations,
        "errors": errors
    }
