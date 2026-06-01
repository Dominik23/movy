import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from data_ai.pipeline.run import (
    scan_files,
    group_by_year,
    process_year_batch,
    run_pipeline,
)


def test_scan_files_finds_supported_files(tmp_path):
    # Create test files
    (tmp_path / "doc.pdf").write_text("")
    (tmp_path / "image.png").write_text("")
    (tmp_path / "skip.xyz").write_text("")
    (tmp_path / ".hidden.pdf").write_text("")

    with patch("data_ai.pipeline.run.SUPPORTED_EXTENSIONS", {".pdf", ".png"}):
        files = scan_files(tmp_path)

    assert len(files) == 2
    assert any(f.name == "doc.pdf" for f in files)
    assert any(f.name == "image.png" for f in files)


def test_group_by_year():
    files = [
        Path("/2024/a.pdf"),
        Path("/2024/b.pdf"),
        Path("/2025/c.pdf"),
    ]

    with patch("data_ai.pipeline.run.detect_year", side_effect=[2024, 2024, 2025]):
        groups = group_by_year(files)

    assert 2024 in groups
    assert 2025 in groups
    assert len(groups[2024]) == 2
    assert len(groups[2025]) == 1


def test_process_year_batch_returns_clusters():
    files = [Path("/a.pdf"), Path("/b.pdf")]

    mock_model = MagicMock()

    with patch("data_ai.pipeline.run.extract_text", return_value="text"):
        with patch("data_ai.pipeline.run.cluster_documents", return_value=({0: files}, mock_model)):
            with patch("data_ai.pipeline.run.get_topic_keywords", return_value=["keyword"]):
                with patch("data_ai.pipeline.run.generate_cluster_name", return_value="TestCluster"):
                    result = process_year_batch(files, min_topic_size=2)

    assert "TestCluster" in result
    assert result["TestCluster"] == files
