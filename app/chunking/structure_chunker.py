import hashlib
import re
from typing import List, Tuple, Dict, Any, Optional

from app.config.settings import settings
from app.models.document import DocumentChunk

class StructureChunker:
    """Structure-aware chunker that splits legal documents while detecting headings/clauses."""
    
    # Allow leading decorations like dashes, stars, or bullets
    SECTION_WORD_PAT = re.compile(
        r'^\s*[-*•\s]*(?:ARTICLE|Article|SECTION|Section)\s+([IVXLCDM\d]+(?:\.\d+)*)(?:[\s.:-]+(.*))?$',
        re.IGNORECASE
    )
    SECTION_NUM_PAT = re.compile(
        r'^\s*[-*•\s]*(\d+)[\s.:]+([A-Z][A-Z\s,;]{3,})$'
    )
    CLAUSE_PAT = re.compile(
        r'^\s*[-*•\s]*(?:\bClause\b\s*)?(\d+\.\d+(?:\.\d+)*|\([a-z0-9]{1,2}\)|[a-z]\.)\s*(.*)$',
        re.IGNORECASE
    )

    @classmethod
    def chunk_document(
        cls,
        pages_text: List[Tuple[int, str]],
        document_name: str,
        document_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunks page-by-page text while identifying section/clause transitions.
        
        Returns:
            List[DocumentChunk]
        """
        chunks: List[DocumentChunk] = []
        
        current_section_num: Optional[str] = None
        current_section_title: Optional[str] = None
        current_clause_num: Optional[str] = None
        
        chunk_size = settings.chunk_size
        
        for page_num, text in pages_text:
            lines = text.split("\n")
            buffer: List[str] = []
            buffer_len = 0
            
            for line_idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                
                is_section = False
                is_clause = False
                sec_num, sec_title = None, None
                cl_num = None
                
                # Check for Article/Section heading
                sec_word_match = cls.SECTION_WORD_PAT.match(stripped)
                if sec_word_match:
                    is_section = True
                    sec_num = sec_word_match.group(1)
                    sec_title = sec_word_match.group(2).strip() if sec_word_match.group(2) else None
                    if not sec_title and line_idx + 1 < len(lines):
                        next_line = lines[line_idx + 1].strip()
                        if next_line and not cls.SECTION_WORD_PAT.match(next_line) and not cls.CLAUSE_PAT.match(next_line) and len(next_line) < 100:
                            sec_title = next_line
                else:
                    sec_num_match = cls.SECTION_NUM_PAT.match(stripped)
                    if sec_num_match:
                        is_section = True
                        sec_num = sec_num_match.group(1)
                        sec_title = sec_num_match.group(2).strip()
                
                # Check for Clause
                if not is_section:
                    clause_match = cls.CLAUSE_PAT.match(stripped)
                    if clause_match:
                        is_clause = True
                        cl_num = clause_match.group(1)
                
                # Boundary check: if we hit a new section/clause, or buffer exceeds chunk size
                if (is_section or is_clause or (buffer_len + len(stripped) > chunk_size)) and buffer:
                    chunk_text = "\n".join(buffer)
                    chunk_id = hashlib.sha256(f"{document_name}_{page_num}_{chunk_text}".encode()).hexdigest()
                    
                    # Align metadata with the dominant clause number in the chunk text
                    detected_clause = current_clause_num
                    clause_matches = re.findall(r'\b(\d+\.\d+(?:\.\d+)*)\b', chunk_text)
                    if clause_matches:
                        detected_clause = clause_matches[0]
                        
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document_name=document_name,
                        page_number=page_num,
                        section_number=current_section_num,
                        section_title=current_section_title,
                        clause_number=detected_clause,
                        text=chunk_text,
                        metadata=document_metadata
                    ))
                    buffer = []
                    buffer_len = 0
                
                # Update current active structure state
                if is_section:
                    current_section_num = sec_num
                    current_section_title = sec_title
                    current_clause_num = None
                elif is_clause:
                    current_clause_num = cl_num
                
                buffer.append(stripped)
                buffer_len += len(stripped)
            
            # Flush remaining buffer at page end
            if buffer:
                chunk_text = "\n".join(buffer)
                chunk_id = hashlib.sha256(f"{document_name}_{page_num}_{chunk_text}".encode()).hexdigest()
                
                detected_clause = current_clause_num
                clause_matches = re.findall(r'\b(\d+\.\d+(?:\.\d+)*)\b', chunk_text)
                if clause_matches:
                    detected_clause = clause_matches[0]
                    
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page_number=page_num,
                    section_number=current_section_num,
                    section_title=current_section_title,
                    clause_number=detected_clause,
                    text=chunk_text,
                    metadata=document_metadata
                ))
                
        return chunks
