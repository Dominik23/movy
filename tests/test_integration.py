# tests/test_integration.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from data_ai.cli import app

runner = CliRunner()


def test_init_creates_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"

    result = runner.invoke(app, ["init", "--config", str(config_path)])

    assert result.exit_code == 0
    assert config_path.exists()
    assert "Created config" in result.stdout


def test_init_fails_if_exists(tmp_config: Path):
    result = runner.invoke(app, ["init", "--config", str(tmp_config)])

    assert result.exit_code == 1
    assert "already exists" in result.stdout


def test_config_shows_settings(tmp_config: Path):
    result = runner.invoke(app, ["config", "--config", str(tmp_config)])

    assert result.exit_code == 0
    assert "nomic-embed-text" in result.stdout


def test_categories_shows_table(tmp_config: Path):
    result = runner.invoke(app, ["categories", "--config", str(tmp_config)])

    assert result.exit_code == 0
    assert "TestCategory" in result.stdout


def test_full_sort_workflow(tmp_path: Path, tmp_config: Path):
    # Create inbox with test file
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    test_file = inbox / "document.txt"
    test_file.write_text("This is a test example document")

    target = tmp_path / "sorted"

    with patch("data_ai.pipeline.embed_stage") as mock_embed:
        with patch("data_ai.pipeline.build_category_embeddings") as mock_build:
            mock_embed.return_value = [1.0, 0.0, 0.0]

            from data_ai.pipeline.match import CategoryEmbedding
            mock_build.return_value = [
                CategoryEmbedding(name="TestCategory", vector=[1.0, 0.0, 0.0])
            ]

            result = runner.invoke(app, [
                "sort", str(inbox),
                "--config", str(tmp_config),
                "--target", str(target),
            ])

    assert result.exit_code == 0
    assert (target / "TestCategory" / "document.txt").exists()
    assert not test_file.exists()
