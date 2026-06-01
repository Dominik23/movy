import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.extract_v2 import extract_text, SUPPORTED_EXTENSIONS


def test_supported_extensions_include_common_formats():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".jpg" in SUPPORTED_EXTENSIONS


def test_extract_text_returns_none_for_unsupported():
    path = Path("/docs/file.xyz")
    assert extract_text(path) == None


def test_extract_text_calls_docling():
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "Extracted text content"

    mock_converter_class = MagicMock(return_value=mock_result.document.export_to_markdown.return_value)
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_converter_class.return_value = mock_converter

    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".pdf"
    mock_path.__str__.return_value = "/docs/test.pdf"

    with patch("data_ai.pipeline.extract_v2._get_converter", return_value=mock_converter):
        result = extract_text(mock_path)

    assert result == "Extracted text content"
    mock_converter.convert.assert_called_once_with("/docs/test.pdf")


def test_extract_text_returns_none_on_error():
    mock_converter = MagicMock()
    mock_converter.convert.side_effect = Exception("Conversion failed")

    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".pdf"
    mock_path.__str__.return_value = "/docs/corrupt.pdf"

    with patch("data_ai.pipeline.extract_v2._get_converter", return_value=mock_converter):
        result = extract_text(mock_path)

    assert result is None
