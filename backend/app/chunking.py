"""Document chunking using LangChain RecursiveCharacterTextSplitter with tiktoken."""

from __future__ import annotations

from uuid import uuid4

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import AppConfig
from backend.app.models import DocumentChunk, ParsedDocument


class ChunkingModule:
    """Split a ParsedDocument into overlapping token-based chunks."""

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        self._chunk_size = cfg.chunk_size
        self._chunk_overlap = cfg.chunk_overlap
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def chunk(self, document: ParsedDocument, document_id: str) -> list[DocumentChunk]:
        """Split *document* into ``DocumentChunk`` objects."""
        raw_chunks = self._splitter.split_text(document.text)
        result: list[DocumentChunk] = []
        search_start = 0

        for chunk_text in raw_chunks:
            start_char = document.text.find(chunk_text, search_start)
            if start_char == -1:
                start_char = document.text.find(chunk_text)
            end_char = start_char + len(chunk_text) if start_char != -1 else len(chunk_text)

            page_number = self._resolve_page(document, start_char)
            token_count = len(self._encoding.encode(chunk_text))

            result.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    text=chunk_text,
                    document_id=document_id,
                    page_number=page_number,
                    token_count=token_count,
                    start_char=max(start_char, 0),
                    end_char=end_char,
                )
            )
            if start_char != -1:
                search_start = start_char + 1

        return result

    @staticmethod
    def _resolve_page(document: ParsedDocument, char_offset: int) -> int:
        """Return the 1-based page number that contains *char_offset*."""
        if char_offset < 0:
            return 1
        running = 0
        for page in document.pages:
            page_len = len(page.text)
            if running + page_len > char_offset:
                return page.page_number
            running += page_len
        # Fallback: last page
        return document.pages[-1].page_number if document.pages else 1
