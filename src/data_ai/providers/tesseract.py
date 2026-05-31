# src/data_ai/providers/tesseract.py
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image


def extract_ocr(file_path: Path) -> Optional[str]:
    """Extract text from image using OCR."""
    if not file_path.exists():
        return None

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception:
        return None


def extract_ocr_from_pdf(file_path: Path) -> Optional[str]:
    """Extract text from scanned PDF using OCR on each page."""
    if not file_path.exists():
        return None

    try:
        from pdf2image import convert_from_path

        # Convert PDF pages to images
        images = convert_from_path(file_path, dpi=200)

        text_parts = []
        for image in images:
            page_text = pytesseract.image_to_string(image)
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

        return "\n".join(text_parts) if text_parts else None
    except ImportError:
        # pdf2image not installed
        return None
    except Exception:
        return None
