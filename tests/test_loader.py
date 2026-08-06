from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.loader import PDFLoader, PDFProcessingError

def test_extract_metadata_nominal():
    sample_text = (
        "SERVICES AGREEMENT\n\n"
        "This Services Agreement (the \"Agreement\") is dated as of January 15, 2026,\n"
        "by and between Acme Corporation, a Delaware corporation (\"Client\"), and\n"
        "Beta Consultants LLC, a California limited liability company (\"Provider\").\n"
    )
    metadata = PDFLoader.extract_metadata(sample_text, "acme_beta_agreement.pdf")
    
    assert metadata["title"] == "SERVICES AGREEMENT"
    assert metadata["effective_date"] == "January 15, 2026"
    assert "Acme Corporation" in metadata["parties"]
    assert "Beta Consultants LLC" in metadata["parties"]

def test_extract_metadata_fallback():
    sample_text = ""
    metadata = PDFLoader.extract_metadata(sample_text, "simple-contract.pdf")
    assert metadata["title"] == "simple-contract.pdf"

@patch("app.ingestion.loader.pypdf.PdfReader")
def test_load_empty_pdf(mock_reader_cls, tmp_path):
    mock_reader = MagicMock()
    mock_reader.pages = []
    mock_reader_cls.return_value = mock_reader
    
    dummy_file = tmp_path / "empty.pdf"
    dummy_file.write_text("dummy pdf contents")
    
    with pytest.raises(PDFProcessingError, match="PDF is empty"):
        PDFLoader.load(dummy_file)

@patch("app.ingestion.loader.pypdf.PdfReader")
def test_load_corrupted_pdf(mock_reader_cls, tmp_path):
    mock_reader_cls.side_effect = Exception("Invalid file format")
    
    dummy_file = tmp_path / "corrupted.pdf"
    dummy_file.write_text("garbage")
    
    with pytest.raises(PDFProcessingError, match="Corrupted or invalid PDF file"):
        PDFLoader.load(dummy_file)
