# tests/test_extractors.py
import pytest
from pathlib import Path
from data_ai.providers.extractors import (
    extract_text,
    extract_txt,
    extract_pdf,
    extract_docx,
)


def test_extract_txt(fixtures_dir: Path):
    text = extract_txt(fixtures_dir / "sample.txt")
    assert "sample text file" in text


def test_extract_text_dispatches_to_txt(fixtures_dir: Path):
    text = extract_text(fixtures_dir / "sample.txt")
    assert "sample text file" in text


def test_extract_text_unsupported_extension(tmp_path: Path):
    unknown_file = tmp_path / "file.xyz"
    unknown_file.write_text("content")

    result = extract_text(unknown_file)
    assert result is None


def test_extract_pdf_returns_none_when_no_text(tmp_path: Path):
    # PDF extraction tested with real file in integration tests
    # Unit test verifies graceful handling of missing file
    result = extract_pdf(tmp_path / "nonexistent.pdf")
    assert result is None


def test_extract_docx_returns_none_when_no_text(tmp_path: Path):
    result = extract_docx(tmp_path / "nonexistent.docx")
    assert result is None
