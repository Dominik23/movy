import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    config_content = """
settings:
  ollama_model: "nomic-embed-text"
  vision_model: "llava"
  similarity_threshold: 0.6
  inbox: "./inbox"

categories:
  TestCategory:
    keywords:
      - "test"
      - "example"
    examples: []
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path
