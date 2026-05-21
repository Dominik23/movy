# tests/test_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_stage_txt(fixtures_dir: Path):
    from data_ai.pipeline.extract import extract_stage

    result = extract_stage(fixtures_dir / "sample.txt")
    assert result is not None
    assert "sample text file" in result


def test_extract_stage_unsupported_falls_back_to_none(tmp_path: Path):
    from data_ai.pipeline.extract import extract_stage

    unknown = tmp_path / "file.xyz"
    unknown.write_text("content")

    # Without OCR or vision, unsupported files return None
    with patch("data_ai.pipeline.extract.extract_ocr", return_value=None):
        with patch("data_ai.pipeline.extract.describe_image", return_value=None):
            result = extract_stage(unknown)
            assert result is None


def test_embed_stage_calls_ollama():
    from data_ai.pipeline.embed import embed_stage

    with patch("data_ai.pipeline.embed.get_embedding") as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]

        result = embed_stage("test text", model="nomic-embed-text")

        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with("test text", model="nomic-embed-text")
