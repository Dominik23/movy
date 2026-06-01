import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.run import run_pipeline


@pytest.fixture
def test_files(tmp_path):
    """Create test input files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create some test files
    (input_dir / "rechnung_2024_001.txt").write_text("Rechnung über 100 Euro für Dienstleistungen")
    (input_dir / "rechnung_2024_002.txt").write_text("Rechnung Zahlung fällig bis Ende des Monats")
    (input_dir / "vertrag_2024.txt").write_text("Mietvertrag für die Wohnung in der Hauptstraße")
    (input_dir / "steuer_2025.txt").write_text("Steuererklärung für das Jahr 2025")

    return input_dir, tmp_path / "output"


def test_full_pipeline_creates_year_folders(test_files):
    input_dir, output_dir = test_files

    # Mock BERTopic to avoid heavy computation
    mock_model = MagicMock()
    mock_model.fit_transform.return_value = ([0, 0, 1, 2], None)
    mock_model.get_topic.return_value = [("rechnung", 0.5)]

    with patch("data_ai.pipeline.cluster_v2.BERTopic", return_value=mock_model):
        with patch("data_ai.pipeline.cluster_v2.SentenceTransformer"):
            with patch("data_ai.pipeline.naming.ollama.chat", return_value={"message": {"content": "Rechnungen"}}):
                run_pipeline(input_dir, output_dir, min_topic_size=2)

    assert output_dir.exists()
    assert (output_dir / "2024").exists()
    assert (output_dir / "2025").exists()


def test_full_pipeline_dry_run_no_copy(test_files):
    input_dir, output_dir = test_files

    mock_model = MagicMock()
    mock_model.fit_transform.return_value = ([0, 0, 0, 0], None)
    mock_model.get_topic.return_value = [("doc", 0.5)]

    with patch("data_ai.pipeline.cluster_v2.BERTopic", return_value=mock_model):
        with patch("data_ai.pipeline.cluster_v2.SentenceTransformer"):
            with patch("data_ai.pipeline.naming.ollama.chat", return_value={"message": {"content": "Dokumente"}}):
                run_pipeline(input_dir, output_dir, min_topic_size=2, dry_run=True)

    # Output should not be created in dry run
    assert not output_dir.exists()
