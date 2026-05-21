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


def test_match_stage_finds_best_category():
    from data_ai.pipeline.match import match_stage, CategoryEmbedding

    file_vector = [1.0, 0.0, 0.0]

    category_embeddings = [
        CategoryEmbedding(name="Category A", vector=[1.0, 0.0, 0.0]),  # Perfect match
        CategoryEmbedding(name="Category B", vector=[0.0, 1.0, 0.0]),  # Orthogonal
    ]

    result = match_stage(file_vector, category_embeddings, threshold=0.5)

    assert result is not None
    assert result.category == "Category A"
    assert result.confidence == pytest.approx(1.0)


def test_match_stage_returns_none_below_threshold():
    from data_ai.pipeline.match import match_stage, CategoryEmbedding

    file_vector = [1.0, 0.0, 0.0]

    category_embeddings = [
        CategoryEmbedding(name="Category A", vector=[0.0, 1.0, 0.0]),  # Orthogonal
    ]

    result = match_stage(file_vector, category_embeddings, threshold=0.5)

    assert result is None


def test_match_stage_returns_all_matches_sorted():
    from data_ai.pipeline.match import match_stage, CategoryEmbedding

    file_vector = [0.7, 0.7, 0.0]  # Between A and B

    category_embeddings = [
        CategoryEmbedding(name="Category A", vector=[1.0, 0.0, 0.0]),
        CategoryEmbedding(name="Category B", vector=[0.0, 1.0, 0.0]),
    ]

    result = match_stage(file_vector, category_embeddings, threshold=0.3)

    assert result is not None
    assert len(result.all_matches) == 2
    # Both should have similar scores
    assert result.all_matches[0][1] == pytest.approx(result.all_matches[1][1], abs=0.01)
