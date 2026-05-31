# tests/test_config.py
import pytest
from pathlib import Path
from data_ai.config import Config, load_config, Settings, Category, create_default_config


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


def test_create_default_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    create_default_config(config_path)

    assert config_path.exists()
    config = load_config(config_path)
    assert "Documents" in config.categories
    assert "Images" in config.categories


def test_config_has_qdrant_settings(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
settings:
  qdrant_url: "localhost:6333"
  min_clusters: 2
  max_clusters: 20
  variance_threshold: 0.4
categories:
  Test:
    keywords: ["test"]
""")
    from data_ai.config import load_config
    cfg = load_config(config_file)

    assert cfg.settings.qdrant_url == "localhost:6333"
    assert cfg.settings.variance_threshold == 0.4


def test_settings_has_umap_components():
    from data_ai.config import Settings

    settings = Settings()
    assert settings.umap_components == 10
    # Old settings should be removed
    assert not hasattr(settings, "min_clusters") or settings.min_clusters is None
    assert not hasattr(settings, "max_clusters") or settings.max_clusters is None