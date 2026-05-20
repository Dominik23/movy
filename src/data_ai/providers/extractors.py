# src/data_ai/providers/extractors.py
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document


def extract_txt(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return None


def extract_pdf(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        with pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None


def extract_docx(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        doc = Document(file_path)
        text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None


def extract_text(file_path: Path) -> Optional[str]:
    suffix = file_path.suffix.lower()

    extractors = {
        ".txt": extract_txt,
        ".md": extract_txt,
        ".pdf": extract_pdf,
        ".docx": extract_docx,
    }

    extractor = extractors.get(suffix)
    if extractor:
        return extractor(file_path)

    return None
