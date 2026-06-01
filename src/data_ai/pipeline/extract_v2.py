import os
from pathlib import Path
from typing import Optional

# Force CPU for PyTorch to avoid MPS float64 issues on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".html", ".md", ".txt",
}


_converter: Optional["DocumentConverter"] = None


def _get_converter():
    """Lazy-load the DocumentConverter (heavy initialization)."""
    import torch
    # Force CPU before importing docling
    torch.set_default_device("cpu")

    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    global _converter
    if _converter is None:
        # Configure pipeline to use CPU
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options.device = "cpu"

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
        print(f"Extraction error for {file_path}: {e}")
        return None
