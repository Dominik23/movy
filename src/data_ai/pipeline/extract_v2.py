from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".html", ".md", ".txt",
}


_converter: Optional["DocumentConverter"] = None


def _get_converter():
    """Lazy-load the DocumentConverter (heavy initialization)."""
    from docling.document_converter import DocumentConverter

    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def extract_text(file_path: Path) -> str | None:
    """
    Extract text from document using Docling.

    Supports: PDF, DOCX, PPTX, images, HTML, Markdown, TXT
    Returns None if extraction fails or format unsupported.
    """
    if not file_path.exists():
        return None

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    try:
        converter = _get_converter()
        result = converter.convert(str(file_path))
        text = result.document.export_to_markdown()
        return text.strip() if text and text.strip() else None
    except Exception:
        return None
