import os
from pathlib import Path
from typing import Optional

# MUST be set before ANY torch import to disable MPS completely
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Force torch to CPU before any other import can grab MPS
import torch
torch.set_default_device("cpu")
if hasattr(torch.backends, "mps"):
    # Ensure MPS is not used even if available
    pass  # torch.backends.mps.is_available() will return False due to env vars


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".html", ".md", ".txt",
}


_converter: Optional["DocumentConverter"] = None


def _get_converter():
    """Lazy-load the DocumentConverter (heavy initialization)."""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat

    global _converter
    if _converter is None:
        # Force CPU accelerator and disable layout model that causes MPS float64 errors
        accel_options = AcceleratorOptions(device="cpu")
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accel_options,
            do_table_structure=False,  # Disable table detection (uses problematic model)
            do_ocr=False,  # Disable OCR to avoid additional model issues
        )

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
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
    except Exception as e:
        # Silently fail - file will be skipped
        return None
