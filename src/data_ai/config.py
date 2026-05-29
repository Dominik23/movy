# src/data_ai/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Existing
    ollama_model: str = "nomic-embed-text"
    vision_model: str = "llava"
    chat_model: str = "llama3.2"
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    learning_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    auto_learn: bool = True
    inbox: str = "./inbox"

    # New: Qdrant
    qdrant_url: str = "localhost:6333"
    qdrant_collection_prefix: str = "data_ai"

    # New: Clustering
    min_clusters: int = Field(default=2, ge=2)
    max_clusters: int = Field(default=20, ge=2)
    variance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    min_cluster_size: int = Field(default=3, ge=1)

    # New: Processing
    batch_size: int = Field(default=100, ge=1)
    summary_length: int = Field(default=2000, ge=100)

    # New: Output
    trash_folder: str = ".trash"
    log_file: str = "data-ai.log"
    review_html: str = "/tmp/data-ai-review.html"


class Category(BaseModel):
    keywords: list[str] = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)


class Config(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    categories: dict[str, Category] = Field(min_length=1)


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data.get("categories"):
        raise ValueError("Config must contain 'categories' section")

    return Config(**data)


def get_default_config_path() -> Path:
    return Path.home() / ".config" / "data-ai" / "config.yaml"


def create_default_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = """\
settings:
  ollama_model: "nomic-embed-text"
  vision_model: "llava"
  chat_model: "llama3.2"
  similarity_threshold: 0.6
  learning_threshold: 0.85
  auto_learn: true
  inbox: "./inbox"

categories:
  Documents:
    keywords:
      - "document"
      - "report"
      - "letter"
    examples: []

  Images:
    keywords:
      - "photo"
      - "picture"
      - "image"
    examples: []
"""
    config_path.write_text(default_config)


def add_keywords_to_config(
    config_path: Path,
    category: str,
    new_keywords: list[str],
) -> None:
    """Add new keywords to a category in the config file."""
    if not new_keywords:
        return

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if category not in data.get("categories", {}):
        return

    existing = data["categories"][category].get("keywords", [])

    # Add only truly new keywords
    for kw in new_keywords:
        if kw.lower() not in [k.lower() for k in existing]:
            existing.append(kw)

    data["categories"][category]["keywords"] = existing

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
