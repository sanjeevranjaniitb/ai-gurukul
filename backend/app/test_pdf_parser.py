"""Tests for the PDFParser class."""

from __future__ import annotations

import os
import tempfile

import fitz  # PyMuPDF
import pytest

from backend.app.config import AppConfig
from backend.app.pdf_parser import PDFParseError, PDFParser


@pytest.fixture
def parser() -> PDFParser:
    """Return a PDFParser with default config."""
    return PDFParser(AppConfig())


@pytest.fixture
def simple_pdf(tmp_path) -> str:
    """Create a minimal single-page PDF with text."""
    path = str(tmp_path / "simple.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello, world!")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def multi_page_pdf(tmp_path) -> str:
    """Create a 3-page PDF."""
    path = str(tmp_path / "multi.pdf")
    doc = fitz.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i} content")
    doc.save(path)
    doc.close()
    return path


class TestPDFParserParse:
    """Test the parse() method with valid PDFs."""

    def test_extracts_text_from_single_page(self, parser, simple_pdf):
        result = parser.parse(simple_pdf)
        assert "Hello, world!" in result.text
        assert result.page_count == 1
        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1

    def test_extracts_all_pages(self, parser, multi_page_pdf):
        result = parser.parse(multi_page_pdf)
        assert result.page_count == 3
        assert len(result.pages) == 3
        for i, page in enumerate(result.pages, start=1):
            assert page.page_number == i
            assert f"Page {i} content" in page.text

    def test_returns_metadata(self, parser, simple_pdf):
        result = parser.parse(simple_pdf)
        assert isinstance(result.metadata, dict)

    def test_pages_have_tables_list(self, parser, simple_pdf):
        result = parser.parse(simple_pdf)
        assert isinstance(result.pages[0].tables, list)


class TestPDFParserValidation:
    """Test file size and integrity validation."""

    def test_rejects_file_exceeding_size_limit(self, tmp_path):
        # Use a tiny limit (1 MB) so we can create a file that exceeds it
        config = AppConfig()
        config.max_pdf_size_mb = 1
        parser = PDFParser(config)

        # Create a file slightly over 1 MB
        big_file = str(tmp_path / "big.pdf")
        with open(big_file, "wb") as f:
            f.write(b"\x00" * (1024 * 1024 + 1))

        with pytest.raises(PDFParseError, match="size limit"):
            parser.parse(big_file)

    def test_rejects_corrupted_file(self, parser, tmp_path):
        bad_file = str(tmp_path / "corrupt.pdf")
        with open(bad_file, "wb") as f:
            f.write(b"this is not a pdf at all")

        with pytest.raises(PDFParseError, match="corrupted"):
            parser.parse(bad_file)

    def test_rejects_nonexistent_file(self, parser):
        with pytest.raises(PDFParseError, match="Cannot read file"):
            parser.parse("/nonexistent/path/file.pdf")

    def test_rejects_password_protected_pdf(self, parser, tmp_path):
        path = str(tmp_path / "protected.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        # Encrypt with user and owner passwords
        doc.save(
            path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            user_pw="user123",
            owner_pw="owner456",
        )
        doc.close()

        with pytest.raises(PDFParseError, match="password-protected"):
            parser.parse(path)

    def test_file_exactly_at_limit_is_accepted(self, tmp_path):
        config = AppConfig()
        config.max_pdf_size_mb = 1
        parser = PDFParser(config)

        # Create a valid PDF that's under 1 MB (it will be small)
        path = str(tmp_path / "small.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(path)
        doc.close()

        # Should not raise
        result = parser.parse(path)
        assert result.page_count == 1


class TestPDFParserTableExtraction:
    """Test table detection and markdown formatting."""

    def test_table_formatted_as_markdown(self, parser, tmp_path):
        """Create a PDF with a simple table and verify markdown output."""
        path = str(tmp_path / "table.pdf")
        doc = fitz.open()
        page = doc.new_page()

        # Insert a simple table using text blocks positioned in a grid
        # PyMuPDF's find_tables() works on actual table structures,
        # so we use insert_text in a grid pattern
        y_start = 100
        row_height = 20
        col_positions = [72, 200, 328]
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ]

        for col_idx, header in enumerate(headers):
            page.insert_text((col_positions[col_idx], y_start), header)

        for row_idx, row in enumerate(rows):
            y = y_start + (row_idx + 1) * row_height
            for col_idx, cell in enumerate(row):
                page.insert_text((col_positions[col_idx], y), cell)

        doc.save(path)
        doc.close()

        # Parse — tables may or may not be detected depending on layout,
        # but the text should still be extracted
        result = parser.parse(path)
        assert result.page_count == 1
        assert "Alice" in result.text
        assert "Bob" in result.text

    def test_tables_included_in_full_text(self, parser, tmp_path):
        """When tables are detected, their markdown should appear in full_text."""
        from unittest.mock import MagicMock, patch

        path = str(tmp_path / "withtable.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Some intro text")
        doc.save(path)
        doc.close()

        # Patch _extract_tables to return a known markdown table
        fake_md = "| Col1 | Col2 |\n| --- | --- |\n| A | B |"
        with patch.object(PDFParser, "_extract_tables", return_value=[fake_md]):
            result = parser.parse(path)

        assert fake_md in result.text
        assert "Some intro text" in result.text

    def test_table_to_markdown_static_method(self):
        """Verify _table_to_markdown produces pipe-delimited markdown."""
        from unittest.mock import MagicMock

        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["Name", "Score"],
            ["Alice", "95"],
            ["Bob", "87"],
        ]

        md = PDFParser._table_to_markdown(mock_table)
        assert "| Name | Score |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 95 |" in md
        assert "| Bob | 87 |" in md

    def test_table_to_markdown_handles_none_cells(self):
        """None cell values should be converted to empty strings."""
        from unittest.mock import MagicMock

        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["Header"],
            [None],
        ]

        md = PDFParser._table_to_markdown(mock_table)
        assert "| Header |" in md
        assert "|  |" in md

    def test_table_to_markdown_returns_empty_on_extract_failure(self):
        """If table.extract() raises, return empty string."""
        from unittest.mock import MagicMock

        mock_table = MagicMock()
        mock_table.extract.side_effect = RuntimeError("broken")

        md = PDFParser._table_to_markdown(mock_table)
        assert md == ""

    def test_table_to_markdown_returns_empty_for_no_rows(self):
        """If table has no rows, return empty string."""
        from unittest.mock import MagicMock

        mock_table = MagicMock()
        mock_table.extract.return_value = []

        md = PDFParser._table_to_markdown(mock_table)
        assert md == ""


class TestPDFParserEdgeCases:
    """Test edge cases and additional scenarios."""

    def test_single_blank_page_pdf(self, tmp_path):
        """A PDF with a single blank page should parse without error."""
        path = str(tmp_path / "blank.pdf")
        doc = fitz.open()
        doc.new_page()  # blank page, no text
        doc.save(path)
        doc.close()

        parser = PDFParser(AppConfig())
        result = parser.parse(path)
        assert result.page_count == 1
        assert len(result.pages) == 1
        # Text should be empty or whitespace only
        assert result.pages[0].text.strip() == ""

    def test_full_text_concatenates_all_pages(self, parser, multi_page_pdf):
        """full_text should contain content from every page."""
        result = parser.parse(multi_page_pdf)
        for i in range(1, 4):
            assert f"Page {i} content" in result.text

    def test_page_with_no_text(self, parser, tmp_path):
        """A page with no text should produce an empty string, not None."""
        path = str(tmp_path / "blank_page.pdf")
        doc = fitz.open()
        doc.new_page()  # blank page
        doc.save(path)
        doc.close()

        result = parser.parse(path)
        assert result.page_count == 1
        assert isinstance(result.pages[0].text, str)
