from app.chunking.structure_chunker import StructureChunker

def test_structure_chunker_legal_headings():
    pages_text = [
        (1, (
            "ARTICLE I\n"
            "DEFINITIONS\n"
            "1.1. Affiliate. 'Affiliate' means any entity controlling, controlled by, or under common control.\n"
            "1.2. Agreement. 'Agreement' means this services contract.\n"
            "SECTION 2. SERVICES\n"
            "Clause 2.1. Provider shall perform services specified in Exhibit A.\n"
            "(a) First clause detail.\n"
            "(b) Second clause detail.\n"
        ))
    ]
    
    metadata = {"title": "Test Contract", "parties": [], "effective_date": None}
    chunks = StructureChunker.chunk_document(pages_text, "test.pdf", metadata)
    
    # Assert section number and section title detections
    # ARTICLE I / DEFINITIONS should set current_section_num="I" and current_section_title="DEFINITIONS"
    # 1.1 should set clause_number="1.1"
    # SECTION 2. SERVICES should set current_section_num="2" and current_section_title="SERVICES"
    # Clause 2.1 should set clause_number="2.1"
    # (a) should set clause_number="(a)" or similar depending on match
    
    assert len(chunks) > 0
    
    # Check that metadata matches
    for chunk in chunks:
        assert chunk.document_name == "test.pdf"
        assert chunk.metadata == metadata

    # Find the chunk containing Section 2
    sec_2_chunks = [c for c in chunks if c.section_number == "2"]
    assert len(sec_2_chunks) > 0
    assert sec_2_chunks[0].section_title == "SERVICES"
    
    # Check that clause numbers are detected
    clauses = [c.clause_number for c in chunks if c.clause_number is not None]
    assert len(clauses) > 0
    assert "1.1" in clauses or "1.2" in clauses or "2.1" in clauses
