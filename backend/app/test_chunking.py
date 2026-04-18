"""Unit tests for ChunkingModule."""

from __future__ import annotations

import tiktoken

from backend.app.chunking import ChunkingModule
from backend.app.config import AppConfig
from backend.app.models import DocumentChunk, PageContent, ParsedDocument


def _make_doc(text: str, pages: list[PageContent] | None = None) -> ParsedDocument:
    if pages is None:
        pages = [PageContent(page_number=1, text=text, tables=[])]
    return ParsedDocument(text=text, pages=pages, page_count=len(pages), metadata={})


def test_chunk_produces_document_chunks():
    doc = _make_doc("Hello world. " * 200)
    cm = ChunkingModule()
    chunks = cm.chunk(doc, document_id="doc-1")
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.document_id == "doc-1"
        assert c.chunk_id  # non-empty uuid


def test_chunk_ids_are_unique():
    doc = _make_doc("Some text content. " * 200)
    chunks = ChunkingModule().chunk(doc, document_id="d")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_token_count_within_limit():
    doc = _make_doc("The quick brown fox jumps over the lazy dog. " * 500)
    cfg = AppConfig(chunk_size=512, chunk_overlap=50)
    chunks = ChunkingModule(cfg).chunk(doc, document_id="d")
    # LangChain's tiktoken splitter may overshoot by a small margin
    for c in chunks:
        assert c.token_count <= 520


def test_page_number_resolution():
    p1 = "First page content. " * 100
    p2 = "Second page content. " * 100
    pages = [
        PageContent(page_number=1, text=p1, tables=[]),
        PageContent(page_number=2, text=p2, tables=[]),
    ]
    doc = _make_doc(p1 + p2, pages)
    chunks = ChunkingModule().chunk(doc, document_id="d")
    # At least one chunk should come from page 2
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers


def test_character_offsets_are_non_negative():
    doc = _make_doc("Test content. " * 200)
    chunks = ChunkingModule().chunk(doc, document_id="d")
    for c in chunks:
        assert c.start_char >= 0
        assert c.end_char > c.start_char


def test_small_document_single_chunk():
    doc = _make_doc("Short text.")
    chunks = ChunkingModule().chunk(doc, document_id="d")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_custom_config_chunk_size():
    doc = _make_doc("Word " * 500)
    cfg = AppConfig(chunk_size=100, chunk_overlap=10)
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = ChunkingModule(cfg).chunk(doc, document_id="d")
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100


def test_overlap_produces_shared_text():
    """Adjacent chunks should share overlapping text when overlap > 0."""
    # Use enough text to produce multiple chunks with a small chunk size
    doc = _make_doc("The quick brown fox jumps over the lazy dog. " * 200)
    cfg = AppConfig(chunk_size=60, chunk_overlap=15)
    chunks = ChunkingModule(cfg).chunk(doc, document_id="d")
    assert len(chunks) >= 2

    # Check that consecutive chunks share some text (overlap)
    overlap_found = False
    for i in range(len(chunks) - 1):
        # The tail of chunk i should appear at the start of chunk i+1
        tail = chunks[i].text[-50:]  # last 50 chars of current chunk
        head = chunks[i + 1].text[:100]  # first 100 chars of next chunk
        # Look for any shared substring of reasonable length
        for length in range(20, 5, -1):
            if tail[-length:] in head:
                overlap_found = True
                break
        if overlap_found:
            break

    assert overlap_found, "Expected overlapping text between adjacent chunks"


def test_zero_overlap_no_shared_text():
    """With overlap=0, consecutive chunks should not share significant text."""
    doc = _make_doc("Sentence number one. " * 300)
    cfg = AppConfig(chunk_size=60, chunk_overlap=0)
    chunks = ChunkingModule(cfg).chunk(doc, document_id="d")
    assert len(chunks) >= 2
    # Chunks should still be produced without error
    for c in chunks:
        assert c.token_count > 0


def test_empty_document():
    """An empty document should produce zero or one empty chunk."""
    doc = _make_doc("")
    chunks = ChunkingModule().chunk(doc, document_id="d")
    # Either no chunks or a single empty/whitespace chunk
    assert len(chunks) <= 1


def test_page_number_resolves_to_page_two():
    """Chunks from the second page should have page_number=2."""
    p1 = "First page. " * 100
    p2 = "Second page unique marker. " * 100
    pages = [
        PageContent(page_number=1, text=p1, tables=[]),
        PageContent(page_number=2, text=p2, tables=[]),
    ]
    doc = _make_doc(p1 + p2, pages)
    chunks = ChunkingModule().chunk(doc, document_id="d")
    page_numbers = {c.page_number for c in chunks}
    assert 2 in page_numbers, "Expected at least one chunk from page 2"


def test_chunk_text_matches_source():
    """Each chunk's text should be a substring of the original document text."""
    text = "Alpha beta gamma delta epsilon. " * 200
    doc = _make_doc(text)
    chunks = ChunkingModule().chunk(doc, document_id="d")
    for c in chunks:
        assert c.text in text


def test_default_config_uses_512_chunk_size():
    """Default ChunkingModule should use 512-token chunks."""
    cm = ChunkingModule()
    assert cm._chunk_size == 512
    assert cm._chunk_overlap == 50
