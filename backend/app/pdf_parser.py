"""PDF parsing module using PyMuPDF.

Extracts text and table structure from PDF documents, validates file
integrity, and enforces size limits.
"""

from __future__ import annotations

import os

import fitz  # PyMuPDF

from backend.app.config import AppConfig
from backend.app.logging_utils import get_logger
from backend.app.models import PageContent, ParsedDocument

logger = get_logger("pdf_parser")


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed."""


class PDFParser:
    """Extract text and structure from PDF files.

    Parameters
    ----------
    config:
        Application configuration providing ``max_pdf_size_mb``.
    """

    def __init__(self, config: AppConfig) -> None:
        self._max_bytes = config.max_pdf_size_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a PDF file and return structured content.

        Raises
        ------
        PDFParseError
            If the file exceeds the size limit, is corrupted, or is
            password-protected.
        """
        self._validate_file_size(file_path)
        doc = self._open_document(file_path)

        try:
            pages: list[PageContent] = []
            full_text_parts: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text") or ""
                tables = self._extract_tables(page)

                page_content = PageContent(
                    page_number=page_num + 1,
                    text=text,
                    tables=tables,
                )
                pages.append(page_content)

                # Build full document text including table markdown
                full_text_parts.append(text)
                if tables:
                    full_text_parts.extend(tables)

            metadata = doc.metadata or {}
        finally:
            doc.close()

        full_text = "\n".join(full_text_parts)

        logger.info(
            "Parsed PDF %s: %d pages extracted", file_path, len(pages)
        )

        return ParsedDocument(
            text=full_text,
            pages=pages,
            page_count=len(pages),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_file_size(self, file_path: str) -> None:
        """Raise ``PDFParseError`` if the file exceeds the size limit."""
        try:
            size = os.path.getsize(file_path)
        except OSError as exc:
            raise PDFParseError(
                f"Cannot read file '{file_path}': {exc}"
            ) from exc

        if size > self._max_bytes:
            limit_mb = self._max_bytes / (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            raise PDFParseError(
                f"PDF file exceeds the {limit_mb:.0f} MB size limit "
                f"(file is {actual_mb:.1f} MB)"
            )

    def _open_document(self, file_path: str) -> fitz.Document:
        """Open the PDF, checking for corruption and password protection."""
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise PDFParseError(
                f"Failed to open PDF '{file_path}': file may be corrupted"
            ) from exc

        if doc.is_encrypted:
            doc.close()
            raise PDFParseError(
                f"PDF '{file_path}' is password-protected and cannot be parsed"
            )

        return doc

    def _extract_tables(self, page: fitz.Page) -> list[str]:
        """Detect tables on a page and return them as markdown strings."""
        try:
            tables = page.find_tables()
        except Exception:
            return []

        md_tables: list[str] = []
        for table in tables:
            try:
                df = table.to_pandas()
                md_tables.append(df.to_markdown(index=False))
            except Exception:
                # Fall back to manual markdown if pandas conversion fails
                md = self._table_to_markdown(table)
                if md:
                    md_tables.append(md)

        return md_tables

    @staticmethod
    def _table_to_markdown(table: fitz.table.Table) -> str:
        """Convert a PyMuPDF Table to a markdown string without pandas."""
        try:
            rows = table.extract()
        except Exception:
            return ""

        if not rows:
            return ""

        def _cell(val: object) -> str:
            return str(val).strip() if val is not None else ""

        header = rows[0]
        lines = ["| " + " | ".join(_cell(c) for c in header) + " |"]
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(_cell(c) for c in row) + " |")

        return "\n".join(lines)
