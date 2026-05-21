# src/data_ai/providers/tesseract.py
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image


def extract_ocr(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception:
        return None
