# src/data_ai/pipeline/extract.py
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from data_ai.providers.extractors import extract_text
from data_ai.providers.tesseract import extract_ocr, extract_ocr_from_pdf
from data_ai.providers.ollama import describe_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}


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
        text = extract_ocr_from_pdf(file_path)
        if text and text.strip():
            return text

    # Last resort for images: vision model
    if suffix in IMAGE_EXTENSIONS:
        description = describe_image(str(file_path), model=vision_model)
        if description:
            return description

    return None


def scan_folder(
    folder: Path,
    trash_dir: Path | None = None,
) -> tuple[list[Path], list[dict]]:
    """
    Recursively scan folder for supported files.
    Moves unsupported files to trash_dir if provided.

    Returns: (supported_files, trash_log)
    """
    supported_files = []
    trash_log = []

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip hidden files and trash folder
        if file_path.name.startswith("."):
            continue
        if trash_dir and trash_dir in file_path.parents:
            continue

        suffix = file_path.suffix.lower()

        if suffix in SUPPORTED_EXTENSIONS:
            supported_files.append(file_path)
        elif trash_dir:
            # Move to trash
            trash_dir.mkdir(parents=True, exist_ok=True)
            target = trash_dir / file_path.name
            if target.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target = trash_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            shutil.copy2(file_path, target)
            trash_log.append({
                "source": str(file_path),
                "target": str(target),
                "reason": f"Unsupported file type: {suffix}",
                "timestamp": datetime.now().isoformat(),
            })

    return supported_files, trash_log
