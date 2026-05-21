# src/data_ai/pipeline/extract.py
from pathlib import Path
from typing import Optional

from data_ai.providers.extractors import extract_text
from data_ai.providers.tesseract import extract_ocr
from data_ai.providers.ollama import describe_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def extract_stage(file_path: Path, vision_model: str = "llava") -> Optional[str]:
    suffix = file_path.suffix.lower()

    # Try text extraction first
    text = extract_text(file_path)
    if text and text.strip():
        return text

    # For images, try OCR
    if suffix in IMAGE_EXTENSIONS:
        text = extract_ocr(file_path)
        if text and text.strip():
            return text

    # For PDFs with no text (scanned), try OCR
    if suffix == ".pdf":
        # PDF was already tried, might be scanned
        # OCR on PDF pages would need pdf2image, skip for now
        pass

    # Last resort: vision model
    if suffix in IMAGE_EXTENSIONS:
        description = describe_image(str(file_path), model=vision_model)
        if description:
            return description

    return None
