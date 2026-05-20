# tests/test_config.py
import pytest
from pathlib import Path
from data_ai.config import Config, load_config, Settings, Category


def test_load_config_from_yaml(tmp_config: Path):
    config = load_config(tmp_config)

    assert config.settings.ollama_model == "nomic-embed-text"
    assert config.settings.vision_model == "llava"
    assert config.settings.similarity_threshold == 0.6
    assert config.settings.inbox == "./inbox"

    assert "TestCategory" in config.categories
    assert config.categories["TestCategory"].keywords == ["test", "example"]
    assert config.categories["TestCategory"].examples == []


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_config_validation_missing_categories(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("settings:\n  ollama_model: test\n")

    with pytest.raises(ValueError):
        load_config(config_path)
