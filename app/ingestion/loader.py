import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pypdf

from app.config.logging import logger

class PDFProcessingError(Exception):
    """Exception raised for errors during PDF ingestion/processing."""
    pass

class PDFLoader:
    """PDF loader that extracts page-by-page text and dynamic metadata from a legal PDF."""
    
    @staticmethod
    def load(file_path: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """
        Loads a PDF file, extracts text page-by-page, and parses dynamic metadata.
        
        Returns:
            Tuple[List[Tuple[page_number, text]], metadata_dict]
        """
        if not file_path.exists():
            raise PDFProcessingError(f"File not found: {file_path}")
            
        try:
            reader = pypdf.PdfReader(file_path)
            # Force checking the length of pages to trigger decryption/parsing error early if corrupted
            if len(reader.pages) == 0:
                raise PDFProcessingError(f"PDF is empty: {file_path}")
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise e
            raise PDFProcessingError(f"Corrupted or invalid PDF file {file_path}: {e}") from e

        pages_text: List[Tuple[int, str]] = []
        
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append((i + 1, text))
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1} of {file_path}: {e}")
                pages_text.append((i + 1, ""))

        # Check if the overall document has no text
        total_len = sum(len(txt) for _, txt in pages_text)
        if total_len == 0:
            raise PDFProcessingError(f"PDF has no readable text: {file_path}")

        first_page_text = pages_text[0][1] if pages_text else ""
        metadata = PDFLoader.extract_metadata(first_page_text, file_path.name)
        
        return pages_text, metadata

    @staticmethod
    def extract_metadata(first_page_text: str, filename: str) -> Dict[str, Any]:
        """Dynamically extracts document title, parties, and dates from the first page text."""
        metadata: Dict[str, Any] = {
            "title": None,
            "parties": [],
            "effective_date": None
        }
        
        if not first_page_text.strip():
            metadata["title"] = filename
            return metadata

        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        
        # 1. Title Extraction
        title_candidates = []
        for line in lines[:10]:
            # Clean and test
            if line.isupper() and 5 <= len(line) <= 100:
                title_candidates.append(line)
            elif any(kw in line.upper() for kw in ["AGREEMENT", "CONTRACT", "LEASE", "DEED", "UNDERTAKING", "POLICY"]):
                if len(line) <= 120:
                    title_candidates.append(line)
        
        if title_candidates:
            metadata["title"] = title_candidates[0]
        else:
            metadata["title"] = Path(filename).stem.replace("_", " ").replace("-", " ").title()

        # 2. Date Extraction
        date_patterns = [
            r'(?i)(?:effective\s+date|dated\s+as\s+of|date\s+of\s+this\s+agreement|this\s+agreement\s+is\s+entered\s+into\s+on)\s*[:\-\b]*\s*([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2} [A-Za-z]+ \d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})',
            r'(?i)dated\s+(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,\s+\d{4})',
            r'(?i)effective\s+as\s+of\s+([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2} [A-Za-z]+ \d{4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, first_page_text)
            if match:
                metadata["effective_date"] = match.group(1).strip()
                break

        # 3. Parties Extraction
        intro_text = first_page_text[:2000]
        party_regex = r'\b([A-Z][A-Za-z0-9&\'\s]+(?:\bLLC\b|\bInc\.\b|\bIncorporated\b|\bCorp\.\b|\bCorporation\b|\bCo\.\b|\bCompany\b|\bLtd\.\b|\bLimited\b|\bPartnership\b|\bLP\b))\b'
        parties_found = re.findall(party_regex, intro_text)
        
        seen = set()
        unique_parties = []
        for p in parties_found:
            p_clean = p.strip()
            p_lower = p_clean.lower()
            if p_lower not in seen and len(p_clean) > 2:
                seen.add(p_lower)
                unique_parties.append(p_clean)
        
        # Fallback party search if none found with designators
        if not unique_parties:
            between_match = re.search(r'(?i)(?:between|among)\s+([A-Z][a-zA-Z\s,]+?)\s+and\s+([A-Z][a-zA-Z\s,]+?)(?:,|\bis\b|\bcollectively\b|\b"parties"\b|\.|\n)', intro_text)
            if between_match:
                for grp in between_match.groups():
                    parts = re.split(r'(?i),|\ba\b|\ban\b|\bthe\b|\bcorporation\b|\bcompany\b', grp)
                    candidate = parts[0].strip()
                    if candidate and candidate[0].isupper() and len(candidate) > 2:
                        unique_parties.append(candidate)
                        
        metadata["parties"] = unique_parties[:5]
        
        return metadata
