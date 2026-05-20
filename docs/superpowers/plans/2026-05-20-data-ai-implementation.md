# data-ai Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that automatically organizes files into folders based on semantic similarity using local Ollama embeddings.

**Architecture:** Pipeline-based design with 4 stages (Extract → Embed → Match → Execute). Config-driven categories with keywords and optional example documents. Interactive prompts for low-confidence matches.

**Tech Stack:** Python 3.11+, Typer (CLI), Pydantic (config), Ollama (embeddings + vision), pdfplumber, python-docx, pytesseract, watchdog

---

## File Structure

```
data-ai/
├── pyproject.toml                    # Package config, dependencies, entry point
├── src/
│   └── data_ai/
│       ├── __init__.py               # Version export
│       ├── cli.py                    # Typer commands: init, sort, scan, apply, watch, config, categories, test
│       ├── config.py                 # Pydantic models, YAML loading, validation
│       │
│       ├── pipeline/
│       │   ├── __init__.py           # Pipeline orchestration
│       │   ├── extract.py            # Text extraction from files
│       │   ├── embed.py              # Text → vector via Ollama
│       │   ├── match.py              # Vector similarity → category
│       │   └── execute.py            # File move + interactive prompts
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── ollama.py             # Ollama API client (embed + vision)
│       │   ├── tesseract.py          # OCR wrapper
│       │   └── extractors.py         # PDF, DOCX, TXT extractors
│       │
│       └── utils/
│           ├── __init__.py
│           └── similarity.py         # Cosine similarity, vector averaging
│
└── tests/
    ├── conftest.py                   # Shared fixtures
    ├── test_config.py
    ├── test_extractors.py
    ├── test_similarity.py
    ├── test_pipeline.py
    └── fixtures/
        ├── sample.pdf
        ├── sample.docx
        ├── sample.txt
        └── sample.png
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/data_ai/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "data-ai"
version = "0.1.0"
description = "Intelligent file organizer using semantic similarity"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "ollama>=0.1.0",
    "pdfplumber>=0.9.0",
    "python-docx>=0.8.0",
    "pytesseract>=0.3.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "watchdog>=3.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
data-ai = "data_ai.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_ai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create package init**

```python
# src/data_ai/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create directory structure**

Run:
```bash
mkdir -p src/data_ai/pipeline src/data_ai/providers src/data_ai/utils tests/fixtures
touch src/data_ai/pipeline/__init__.py src/data_ai/providers/__init__.py src/data_ai/utils/__init__.py
```

- [ ] **Step 4: Create test conftest**

```python
# tests/conftest.py
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
```

- [ ] **Step 5: Install in dev mode and verify**

Run: `uv pip install -e ".[dev]"`
Expected: Package installs successfully

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: scaffold project structure"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/data_ai/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loading**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "No module named 'data_ai.config'"

- [ ] **Step 3: Implement config module**

```python
# src/data_ai/config.py
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    ollama_model: str = "nomic-embed-text"
    vision_model: str = "llava"
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    inbox: str = "./inbox"


class Category(BaseModel):
    keywords: list[str] = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)


class Config(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    categories: dict[str, Category] = Field(min_length=1)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: dict[str, Category]) -> dict[str, Category]:
        if not v:
            raise ValueError("At least one category must be defined")
        return v


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
  similarity_threshold: 0.6
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/config.py tests/test_config.py
git commit -m "feat: add config module with YAML loading and validation"
```

---

## Task 3: Similarity Utils

**Files:**
- Create: `src/data_ai/utils/similarity.py`
- Create: `tests/test_similarity.py`

- [ ] **Step 1: Write failing tests for similarity functions**

```python
# tests/test_similarity.py
import pytest
import numpy as np
from data_ai.utils.similarity import cosine_similarity, average_vectors


def test_cosine_similarity_identical():
    vec = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [-1.0, 0.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)


def test_average_vectors_single():
    vectors = [[1.0, 2.0, 3.0]]
    result = average_vectors(vectors)
    assert result == pytest.approx([1.0, 2.0, 3.0])


def test_average_vectors_multiple():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    result = average_vectors(vectors)
    assert result == pytest.approx([0.5, 0.5])


def test_average_vectors_empty():
    with pytest.raises(ValueError):
        average_vectors([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_similarity.py -v`
Expected: FAIL with "No module named 'data_ai.utils.similarity'"

- [ ] **Step 3: Implement similarity module**

```python
# src/data_ai/utils/similarity.py
import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average empty list of vectors")

    arr = np.array(vectors)
    return arr.mean(axis=0).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_similarity.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/utils/similarity.py tests/test_similarity.py
git commit -m "feat: add cosine similarity and vector averaging utils"
```

---

## Task 4: Text Extractors

**Files:**
- Create: `src/data_ai/providers/extractors.py`
- Create: `tests/test_extractors.py`
- Create: `tests/fixtures/sample.txt`

- [ ] **Step 1: Create test fixture**

```bash
echo "This is a sample text file for testing extraction." > tests/fixtures/sample.txt
```

- [ ] **Step 2: Write failing tests for extractors**

```python
# tests/test_extractors.py
import pytest
from pathlib import Path
from data_ai.providers.extractors import (
    extract_text,
    extract_txt,
    extract_pdf,
    extract_docx,
)


def test_extract_txt(fixtures_dir: Path):
    text = extract_txt(fixtures_dir / "sample.txt")
    assert "sample text file" in text


def test_extract_text_dispatches_to_txt(fixtures_dir: Path):
    text = extract_text(fixtures_dir / "sample.txt")
    assert "sample text file" in text


def test_extract_text_unsupported_extension(tmp_path: Path):
    unknown_file = tmp_path / "file.xyz"
    unknown_file.write_text("content")

    result = extract_text(unknown_file)
    assert result is None


def test_extract_pdf_returns_none_when_no_text(tmp_path: Path):
    # PDF extraction tested with real file in integration tests
    # Unit test verifies graceful handling of missing file
    result = extract_pdf(tmp_path / "nonexistent.pdf")
    assert result is None


def test_extract_docx_returns_none_when_no_text(tmp_path: Path):
    result = extract_docx(tmp_path / "nonexistent.docx")
    assert result is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_extractors.py -v`
Expected: FAIL with "No module named 'data_ai.providers.extractors'"

- [ ] **Step 4: Implement extractors module**

```python
# src/data_ai/providers/extractors.py
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document


def extract_txt(file_path: Path) -> Optional[str]:
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return None


def extract_pdf(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        with pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None


def extract_docx(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        doc = Document(file_path)
        text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None


def extract_text(file_path: Path) -> Optional[str]:
    suffix = file_path.suffix.lower()

    extractors = {
        ".txt": extract_txt,
        ".md": extract_txt,
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".doc": extract_docx,
    }

    extractor = extractors.get(suffix)
    if extractor:
        return extractor(file_path)

    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_extractors.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/data_ai/providers/extractors.py tests/test_extractors.py tests/fixtures/sample.txt
git commit -m "feat: add text extractors for TXT, PDF, and DOCX"
```

---

## Task 5: Tesseract OCR Provider

**Files:**
- Create: `src/data_ai/providers/tesseract.py`
- Add test in: `tests/test_extractors.py`

- [ ] **Step 1: Write failing test for OCR**

Add to `tests/test_extractors.py`:

```python
from data_ai.providers.tesseract import extract_ocr


def test_extract_ocr_returns_none_for_nonexistent(tmp_path: Path):
    result = extract_ocr(tmp_path / "nonexistent.png")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractors.py::test_extract_ocr_returns_none_for_nonexistent -v`
Expected: FAIL with "cannot import name 'extract_ocr'"

- [ ] **Step 3: Implement OCR provider**

```python
# src/data_ai/providers/tesseract.py
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image


def extract_ocr(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractors.py::test_extract_ocr_returns_none_for_nonexistent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/providers/tesseract.py tests/test_extractors.py
git commit -m "feat: add Tesseract OCR provider"
```

---

## Task 6: Ollama Provider

**Files:**
- Create: `src/data_ai/providers/ollama.py`

- [ ] **Step 1: Implement Ollama provider**

```python
# src/data_ai/providers/ollama.py
from typing import Optional

import ollama


def get_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def describe_image(image_path: str, model: str = "llava") -> Optional[str]:
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in detail. Focus on any text, documents, or content visible.",
                    "images": [image_path],
                }
            ],
        )
        return response["message"]["content"]
    except Exception:
        return None
```

- [ ] **Step 2: Commit**

```bash
git add src/data_ai/providers/ollama.py
git commit -m "feat: add Ollama provider for embeddings and vision"
```

---

## Task 7: Pipeline - Extract Stage

**Files:**
- Create: `src/data_ai/pipeline/extract.py`

- [ ] **Step 1: Write failing test for extract pipeline**

Add to `tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with "No module named 'data_ai.pipeline.extract'"

- [ ] **Step 3: Implement extract stage**

```python
# src/data_ai/pipeline/extract.py
from pathlib import Path
from typing import Optional

from data_ai.providers.extractors import extract_text
from data_ai.providers.tesseract import extract_ocr
from data_ai.providers.ollama import describe_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def extract_stage(file_path: Path, vision_model: str = "llava") -> Optional[str]:
    suffix = file_path.suffix.lower()

    # Try text extraction first
    text = extract_text(file_path)
    if text and text.strip():
        return text

    # For images, try OCR
    if suffix in IMAGE_EXTENSIONS:
        text = extract_ocr(file_path)
        if text and text.strip():
            return text

    # For PDFs with no text (scanned), try OCR
    if suffix == ".pdf":
        # PDF was already tried, might be scanned
        # OCR on PDF pages would need pdf2image, skip for now
        pass

    # Last resort: vision model
    if suffix in IMAGE_EXTENSIONS:
        description = describe_image(str(file_path), model=vision_model)
        if description:
            return description

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/extract.py tests/test_pipeline.py
git commit -m "feat: add extract stage with text, OCR, and vision fallback"
```

---

## Task 8: Pipeline - Embed Stage

**Files:**
- Create: `src/data_ai/pipeline/embed.py`

- [ ] **Step 1: Write failing test for embed stage**

Add to `tests/test_pipeline.py`:

```python
def test_embed_stage_calls_ollama():
    from data_ai.pipeline.embed import embed_stage

    with patch("data_ai.pipeline.embed.get_embedding") as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]

        result = embed_stage("test text", model="nomic-embed-text")

        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with("test text", model="nomic-embed-text")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_embed_stage_calls_ollama -v`
Expected: FAIL with "No module named 'data_ai.pipeline.embed'"

- [ ] **Step 3: Implement embed stage**

```python
# src/data_ai/pipeline/embed.py
from data_ai.providers.ollama import get_embedding


def embed_stage(text: str, model: str = "nomic-embed-text") -> list[float]:
    return get_embedding(text, model=model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py::test_embed_stage_calls_ollama -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/embed.py tests/test_pipeline.py
git commit -m "feat: add embed stage wrapping Ollama embeddings"
```

---

## Task 9: Pipeline - Match Stage

**Files:**
- Create: `src/data_ai/pipeline/match.py`

- [ ] **Step 1: Write failing tests for match stage**

Add to `tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::test_match_stage_finds_best_category -v`
Expected: FAIL with "No module named 'data_ai.pipeline.match'"

- [ ] **Step 3: Implement match stage**

```python
# src/data_ai/pipeline/match.py
from dataclasses import dataclass
from typing import Optional

from data_ai.utils.similarity import cosine_similarity


@dataclass
class CategoryEmbedding:
    name: str
    vector: list[float]


@dataclass
class MatchResult:
    category: str
    confidence: float
    all_matches: list[tuple[str, float]]  # All categories with scores, sorted desc


def match_stage(
    file_vector: list[float],
    category_embeddings: list[CategoryEmbedding],
    threshold: float = 0.6,
) -> Optional[MatchResult]:
    if not category_embeddings:
        return None

    # Calculate similarity for all categories
    scores: list[tuple[str, float]] = []
    for cat_emb in category_embeddings:
        score = cosine_similarity(file_vector, cat_emb.vector)
        scores.append((cat_emb.name, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    best_category, best_score = scores[0]

    if best_score < threshold:
        return None

    return MatchResult(
        category=best_category,
        confidence=best_score,
        all_matches=scores,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v -k match`
Expected: All match tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/match.py tests/test_pipeline.py
git commit -m "feat: add match stage with similarity scoring"
```

---

## Task 10: Pipeline - Execute Stage

**Files:**
- Create: `src/data_ai/pipeline/execute.py`

- [ ] **Step 1: Write failing tests for execute stage**

Add to `tests/test_pipeline.py`:

```python
def test_execute_stage_moves_file(tmp_path: Path):
    from data_ai.pipeline.execute import execute_move

    # Create source file
    source = tmp_path / "inbox" / "doc.txt"
    source.parent.mkdir()
    source.write_text("content")

    # Create target dir
    target_dir = tmp_path / "sorted" / "Category"

    result = execute_move(source, target_dir)

    assert result is True
    assert not source.exists()
    assert (target_dir / "doc.txt").exists()


def test_execute_stage_handles_duplicate_filename(tmp_path: Path):
    from data_ai.pipeline.execute import execute_move

    # Create source file
    source = tmp_path / "inbox" / "doc.txt"
    source.parent.mkdir()
    source.write_text("new content")

    # Create target with existing file
    target_dir = tmp_path / "sorted" / "Category"
    target_dir.mkdir(parents=True)
    (target_dir / "doc.txt").write_text("existing")

    result = execute_move(source, target_dir)

    assert result is True
    assert not source.exists()
    # Original still exists
    assert (target_dir / "doc.txt").exists()
    # New file has timestamp suffix
    files = list(target_dir.glob("doc_*.txt"))
    assert len(files) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::test_execute_stage_moves_file -v`
Expected: FAIL with "No module named 'data_ai.pipeline.execute'"

- [ ] **Step 3: Implement execute stage**

```python
# src/data_ai/pipeline/execute.py
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

console = Console()


def execute_move(source: Path, target_dir: Path) -> bool:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / source.name

        # Handle duplicate filenames
        if target_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = source.stem
            suffix = source.suffix
            target_path = target_dir / f"{stem}_{timestamp}{suffix}"

        shutil.move(str(source), str(target_path))
        return True
    except Exception as e:
        console.print(f"[red]Error moving file: {e}[/red]")
        return False


def prompt_for_category(
    file_path: Path,
    matches: list[tuple[str, float]],
) -> Optional[str]:
    console.print(f"\n[yellow]? {file_path.name}[/yellow] — uncertain")
    console.print()

    for i, (category, score) in enumerate(matches[:5], 1):
        console.print(f"  [{i}] {category} ({score:.0%})")

    console.print("  [s] Skip")
    console.print("  [q] Abort")
    console.print()

    choice = Prompt.ask("Selection")

    if choice.lower() == "s":
        return None
    if choice.lower() == "q":
        raise KeyboardInterrupt("User aborted")

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(matches):
            return matches[idx][0]
    except ValueError:
        pass

    console.print("[red]Invalid choice, skipping[/red]")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v -k execute`
Expected: All execute tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/execute.py tests/test_pipeline.py
git commit -m "feat: add execute stage with file moving and interactive prompts"
```

---

## Task 11: Pipeline Orchestration

**Files:**
- Modify: `src/data_ai/pipeline/__init__.py`

- [ ] **Step 1: Write failing test for full pipeline**

Add to `tests/test_pipeline.py`:

```python
def test_pipeline_process_file(tmp_path: Path, tmp_config: Path):
    from data_ai.pipeline import process_file
    from data_ai.config import load_config

    # Create test file
    source = tmp_path / "inbox" / "test_document.txt"
    source.parent.mkdir()
    source.write_text("This is a test example document")

    config = load_config(tmp_config)
    target_base = tmp_path / "sorted"

    with patch("data_ai.pipeline.embed_stage") as mock_embed:
        with patch("data_ai.pipeline.build_category_embeddings") as mock_build:
            # File embedding
            mock_embed.return_value = [1.0, 0.0, 0.0]

            # Category embedding matches "TestCategory"
            from data_ai.pipeline.match import CategoryEmbedding
            mock_build.return_value = [
                CategoryEmbedding(name="TestCategory", vector=[1.0, 0.0, 0.0])
            ]

            result = process_file(source, config, target_base)

    assert result is True
    assert (target_base / "TestCategory" / "test_document.txt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_pipeline_process_file -v`
Expected: FAIL with "cannot import name 'process_file'"

- [ ] **Step 3: Implement pipeline orchestration**

```python
# src/data_ai/pipeline/__init__.py
from pathlib import Path
from typing import Optional

from rich.console import Console

from data_ai.config import Config
from data_ai.pipeline.extract import extract_stage
from data_ai.pipeline.embed import embed_stage
from data_ai.pipeline.match import match_stage, MatchResult, CategoryEmbedding
from data_ai.pipeline.execute import execute_move, prompt_for_category
from data_ai.utils.similarity import average_vectors

console = Console()

# Cache for category embeddings
_category_embeddings_cache: dict[str, list[CategoryEmbedding]] = {}


def build_category_embeddings(
    config: Config,
    model: str = "nomic-embed-text",
) -> list[CategoryEmbedding]:
    cache_key = f"{id(config)}_{model}"

    if cache_key in _category_embeddings_cache:
        return _category_embeddings_cache[cache_key]

    embeddings = []

    for name, category in config.categories.items():
        vectors = []

        # Embed keywords
        for keyword in category.keywords:
            vec = embed_stage(keyword, model=model)
            vectors.append(vec)

        # Embed example documents
        for example_path in category.examples:
            path = Path(example_path)
            if path.exists():
                text = extract_stage(path)
                if text:
                    vec = embed_stage(text, model=model)
                    vectors.append(vec)

        if vectors:
            avg_vector = average_vectors(vectors)
            embeddings.append(CategoryEmbedding(name=name, vector=avg_vector))

    _category_embeddings_cache[cache_key] = embeddings
    return embeddings


def process_file(
    file_path: Path,
    config: Config,
    target_base: Path,
    dry_run: bool = False,
) -> bool:
    # Step 1: Extract
    text = extract_stage(file_path, vision_model=config.settings.vision_model)
    if not text:
        console.print(f"[yellow]Skipping {file_path.name}: no text extracted[/yellow]")
        return False

    # Step 2: Embed
    file_vector = embed_stage(text, model=config.settings.ollama_model)

    # Step 3: Match
    category_embeddings = build_category_embeddings(config, config.settings.ollama_model)
    match_result = match_stage(
        file_vector,
        category_embeddings,
        threshold=config.settings.similarity_threshold,
    )

    # Step 4: Execute
    if match_result:
        target_dir = target_base / match_result.category
        if dry_run:
            console.print(
                f"[green]{file_path.name}[/green] → "
                f"[blue]{match_result.category}[/blue] ({match_result.confidence:.0%})"
            )
            return True
        return execute_move(file_path, target_dir)
    else:
        # Below threshold - get all matches for prompt
        all_matches = []
        for cat_emb in category_embeddings:
            from data_ai.utils.similarity import cosine_similarity
            score = cosine_similarity(file_vector, cat_emb.vector)
            all_matches.append((cat_emb.name, score))
        all_matches.sort(key=lambda x: x[1], reverse=True)

        if dry_run:
            best = all_matches[0] if all_matches else ("unknown", 0.0)
            console.print(
                f"[yellow]{file_path.name}[/yellow] → "
                f"[dim]uncertain (best: {best[0]} {best[1]:.0%})[/dim]"
            )
            return False

        chosen = prompt_for_category(file_path, all_matches)
        if chosen:
            target_dir = target_base / chosen
            return execute_move(file_path, target_dir)
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py::test_pipeline_process_file -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/__init__.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestration with caching"
```

---

## Task 12: CLI - Basic Commands

**Files:**
- Create: `src/data_ai/cli.py`

- [ ] **Step 1: Implement CLI with init, config, categories commands**

```python
# src/data_ai/cli.py
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from data_ai.config import load_config, get_default_config_path, create_default_config

app = typer.Typer(
    name="data-ai",
    help="Intelligent file organizer using semantic similarity",
)
console = Console()


def get_config(config_path: Optional[Path]) -> "Config":
    from data_ai.config import Config

    path = config_path or get_default_config_path()
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        console.print("Run [green]data-ai init[/green] to create a config file")
        raise typer.Exit(1)

    return load_config(path)


@app.command()
def init(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Create a default config file."""
    path = config_path or get_default_config_path()

    if path.exists():
        console.print(f"[yellow]Config already exists: {path}[/yellow]")
        raise typer.Exit(1)

    create_default_config(path)
    console.print(f"[green]Created config at: {path}[/green]")
    console.print("Edit this file to define your categories and keywords.")


@app.command()
def config(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Show current config."""
    cfg = get_config(config_path)

    console.print("\n[bold]Settings:[/bold]")
    console.print(f"  Ollama model: {cfg.settings.ollama_model}")
    console.print(f"  Vision model: {cfg.settings.vision_model}")
    console.print(f"  Threshold: {cfg.settings.similarity_threshold}")
    console.print(f"  Inbox: {cfg.settings.inbox}")


@app.command()
def categories(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """List all categories with keywords."""
    cfg = get_config(config_path)

    table = Table(title="Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Keywords", style="green")
    table.add_column("Examples", style="yellow")

    for name, cat in cfg.categories.items():
        keywords = ", ".join(cat.keywords[:3])
        if len(cat.keywords) > 3:
            keywords += f" (+{len(cat.keywords) - 3})"

        examples = str(len(cat.examples))
        table.add_row(name, keywords, examples)

    console.print(table)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Test CLI manually**

Run: `uv run data-ai --help`
Expected: Shows help with init, config, categories commands

- [ ] **Step 3: Commit**

```bash
git add src/data_ai/cli.py
git commit -m "feat: add CLI with init, config, categories commands"
```

---

## Task 13: CLI - Sort Command

**Files:**
- Modify: `src/data_ai/cli.py`

- [ ] **Step 1: Add sort command to CLI**

Add to `src/data_ai/cli.py`:

```python
@app.command()
def sort(
    inbox: Optional[Path] = typer.Argument(
        None, help="Directory to sort (uses config inbox if not specified)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target base directory (default: parent of inbox)"
    ),
) -> None:
    """Sort files from inbox into categories."""
    from data_ai.pipeline import process_file

    cfg = get_config(config_path)

    source_dir = inbox or Path(cfg.settings.inbox)
    if not source_dir.exists():
        console.print(f"[red]Inbox not found: {source_dir}[/red]")
        raise typer.Exit(1)

    target_base = target or source_dir.parent

    files = [f for f in source_dir.iterdir() if f.is_file()]

    if not files:
        console.print("[yellow]No files to sort[/yellow]")
        return

    console.print(f"[bold]Sorting {len(files)} files...[/bold]\n")

    success = 0
    failed = 0

    for file_path in files:
        try:
            if process_file(file_path, cfg, target_base):
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Aborted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
            failed += 1

    console.print(f"\n[bold]Done:[/bold] {success} sorted, {failed} skipped/failed")
```

- [ ] **Step 2: Test sort command help**

Run: `uv run data-ai sort --help`
Expected: Shows help for sort command

- [ ] **Step 3: Commit**

```bash
git add src/data_ai/cli.py
git commit -m "feat: add sort command to CLI"
```

---

## Task 14: CLI - Scan and Apply Commands

**Files:**
- Modify: `src/data_ai/cli.py`

- [ ] **Step 1: Add scan and apply commands**

Add to `src/data_ai/cli.py`:

```python
import json

SCAN_RESULT_FILE = Path.home() / ".cache" / "data-ai" / "last_scan.json"


@app.command()
def scan(
    inbox: Optional[Path] = typer.Argument(
        None, help="Directory to scan (uses config inbox if not specified)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target base directory"
    ),
) -> None:
    """Scan files and show what would be sorted (dry run)."""
    from data_ai.pipeline import process_file

    cfg = get_config(config_path)

    source_dir = inbox or Path(cfg.settings.inbox)
    if not source_dir.exists():
        console.print(f"[red]Inbox not found: {source_dir}[/red]")
        raise typer.Exit(1)

    target_base = target or source_dir.parent

    files = [f for f in source_dir.iterdir() if f.is_file()]

    if not files:
        console.print("[yellow]No files to scan[/yellow]")
        return

    console.print(f"[bold]Scanning {len(files)} files...[/bold]\n")

    scan_results = []

    for file_path in files:
        try:
            process_file(file_path, cfg, target_base, dry_run=True)
            scan_results.append({
                "source": str(file_path),
                "target_base": str(target_base),
            })
        except Exception as e:
            console.print(f"[red]Error scanning {file_path.name}: {e}[/red]")

    # Save scan results
    SCAN_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_RESULT_FILE.write_text(json.dumps({
        "config_path": str(config_path or get_default_config_path()),
        "files": scan_results,
    }))

    console.print(f"\n[dim]Run [green]data-ai apply[/green] to execute[/dim]")


@app.command()
def apply() -> None:
    """Execute the last scan."""
    from data_ai.pipeline import process_file

    if not SCAN_RESULT_FILE.exists():
        console.print("[red]No scan results found. Run [green]data-ai scan[/green] first.[/red]")
        raise typer.Exit(1)

    data = json.loads(SCAN_RESULT_FILE.read_text())
    config_path = Path(data["config_path"])
    files = data["files"]

    if not files:
        console.print("[yellow]No files in scan results[/yellow]")
        return

    cfg = load_config(config_path)

    console.print(f"[bold]Applying to {len(files)} files...[/bold]\n")

    success = 0
    failed = 0

    for item in files:
        source = Path(item["source"])
        target_base = Path(item["target_base"])

        if not source.exists():
            console.print(f"[yellow]Skipping (not found): {source.name}[/yellow]")
            failed += 1
            continue

        try:
            if process_file(source, cfg, target_base):
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Aborted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            failed += 1

    # Clear scan results
    SCAN_RESULT_FILE.unlink(missing_ok=True)

    console.print(f"\n[bold]Done:[/bold] {success} sorted, {failed} skipped/failed")
```

- [ ] **Step 2: Test scan command help**

Run: `uv run data-ai scan --help`
Expected: Shows help for scan command

- [ ] **Step 3: Commit**

```bash
git add src/data_ai/cli.py
git commit -m "feat: add scan and apply commands for two-phase sorting"
```

---

## Task 15: CLI - Watch Command

**Files:**
- Modify: `src/data_ai/cli.py`

- [ ] **Step 1: Add watch command**

Add to `src/data_ai/cli.py`:

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent


class SortHandler(FileSystemEventHandler):
    def __init__(self, config: "Config", target_base: Path):
        self.config = config
        self.target_base = target_base
        self._processing = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return

        from data_ai.pipeline import process_file

        file_path = Path(event.src_path)

        # Avoid processing the same file twice
        if file_path in self._processing:
            return

        self._processing.add(file_path)

        # Wait a moment for file to be fully written
        time.sleep(0.5)

        try:
            console.print(f"\n[bold]New file:[/bold] {file_path.name}")
            process_file(file_path, self.config, self.target_base)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        finally:
            self._processing.discard(file_path)


@app.command()
def watch(
    inbox: Optional[Path] = typer.Argument(
        None, help="Directory to watch (uses config inbox if not specified)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target base directory"
    ),
) -> None:
    """Watch a directory and sort new files automatically."""
    cfg = get_config(config_path)

    source_dir = inbox or Path(cfg.settings.inbox)
    if not source_dir.exists():
        console.print(f"[red]Inbox not found: {source_dir}[/red]")
        raise typer.Exit(1)

    target_base = target or source_dir.parent

    console.print(f"[bold]Watching:[/bold] {source_dir}")
    console.print(f"[bold]Target:[/bold] {target_base}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    handler = SortHandler(cfg, target_base)
    observer = Observer()
    observer.schedule(handler, str(source_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Stopped watching[/yellow]")

    observer.join()
```

- [ ] **Step 2: Add Config import at top of file**

Ensure this import is at the top of `cli.py`:

```python
from data_ai.config import Config, load_config, get_default_config_path, create_default_config
```

- [ ] **Step 3: Test watch command help**

Run: `uv run data-ai watch --help`
Expected: Shows help for watch command

- [ ] **Step 4: Commit**

```bash
git add src/data_ai/cli.py
git commit -m "feat: add watch command for continuous file sorting"
```

---

## Task 16: CLI - Test Command

**Files:**
- Modify: `src/data_ai/cli.py`

- [ ] **Step 1: Add test command**

Add to `src/data_ai/cli.py`:

```python
@app.command("test")
def test_file(
    file_path: Path = typer.Argument(..., help="File to test"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Test classification of a single file without moving it."""
    from data_ai.pipeline import build_category_embeddings
    from data_ai.pipeline.extract import extract_stage
    from data_ai.pipeline.embed import embed_stage
    from data_ai.pipeline.match import match_stage

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    cfg = get_config(config_path)

    console.print(f"[bold]Testing:[/bold] {file_path.name}\n")

    # Extract
    console.print("[dim]Extracting text...[/dim]")
    text = extract_stage(file_path, vision_model=cfg.settings.vision_model)
    if not text:
        console.print("[red]Could not extract text from file[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Extracted {len(text)} characters[/green]")
    console.print(f"[dim]Preview: {text[:100]}...[/dim]\n")

    # Embed
    console.print("[dim]Creating embedding...[/dim]")
    file_vector = embed_stage(text, model=cfg.settings.ollama_model)
    console.print(f"[green]Created {len(file_vector)}-dim vector[/green]\n")

    # Match
    console.print("[dim]Matching against categories...[/dim]")
    category_embeddings = build_category_embeddings(cfg, cfg.settings.ollama_model)

    # Get all scores
    from data_ai.utils.similarity import cosine_similarity

    scores = []
    for cat_emb in category_embeddings:
        score = cosine_similarity(file_vector, cat_emb.vector)
        scores.append((cat_emb.name, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    console.print("\n[bold]Results:[/bold]")
    for category, score in scores:
        threshold = cfg.settings.similarity_threshold
        if score >= threshold:
            console.print(f"  [green]✓ {category}: {score:.1%}[/green]")
        else:
            console.print(f"  [dim]✗ {category}: {score:.1%}[/dim]")

    best = scores[0]
    threshold = cfg.settings.similarity_threshold

    console.print()
    if best[1] >= threshold:
        console.print(f"[bold green]Would sort to: {best[0]}[/bold green]")
    else:
        console.print(f"[bold yellow]Would prompt user (best: {best[0]} at {best[1]:.1%})[/bold yellow]")
```

- [ ] **Step 2: Test the test command help**

Run: `uv run data-ai test --help`
Expected: Shows help for test command

- [ ] **Step 3: Commit**

```bash
git add src/data_ai/cli.py
git commit -m "feat: add test command for single file classification"
```

---

## Task 17: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
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
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add CLI integration tests"
```

---

## Task 18: README and Final Polish

**Files:**
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Create README**

```markdown
# data-ai

Intelligent file organizer using semantic similarity. Define categories with keywords, and let AI sort your files automatically.

## Features

- **YAML Configuration** — Define categories with keywords and optional example documents
- **Local AI** — Uses Ollama for embeddings (nomic-embed-text) and vision (llava)
- **Multiple Modes** — Single sort, preview+apply, or continuous watch
- **Smart Matching** — Hybrid similarity using keywords and example documents
- **Interactive** — Prompts for uncertain classifications

## Installation

```bash
# Install with uv
uv tool install data-ai

# Or with pipx
pipx install data-ai
```

### Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Tesseract for OCR: `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)

Pull required models:
```bash
ollama pull nomic-embed-text
ollama pull llava  # For image description
```

## Quick Start

```bash
# Create config
data-ai init

# Edit ~/.config/data-ai/config.yaml with your categories

# Sort files
data-ai sort ./inbox

# Or preview first
data-ai scan ./inbox
data-ai apply

# Or watch continuously
data-ai watch ./inbox
```

## Configuration

```yaml
settings:
  ollama_model: "nomic-embed-text"
  vision_model: "llava"
  similarity_threshold: 0.6
  inbox: "./inbox"

categories:
  Invoices/Outgoing:
    keywords:
      - "Invoice"
      - "Rechnung"
      - "Amount due"
    examples:
      - "./examples/sample-invoice.pdf"

  Contracts:
    keywords:
      - "Contract"
      - "Agreement"
      - "Terms"
    examples: []
```

## Commands

| Command | Description |
|---------|-------------|
| `data-ai init` | Create default config |
| `data-ai sort [DIR]` | Sort files immediately |
| `data-ai scan [DIR]` | Preview what would be sorted |
| `data-ai apply` | Execute last scan |
| `data-ai watch [DIR]` | Watch and sort new files |
| `data-ai test FILE` | Test single file classification |
| `data-ai config` | Show current settings |
| `data-ai categories` | List categories and keywords |

## License

MIT
```

- [ ] **Step 2: Create LICENSE**

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: add README and LICENSE"
```

- [ ] **Step 4: Run all tests**

Run: `pytest -v --cov=data_ai`
Expected: All tests pass with good coverage

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final polish" --allow-empty
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project Scaffolding | pyproject.toml, __init__.py, conftest.py |
| 2 | Config Module | config.py, test_config.py |
| 3 | Similarity Utils | similarity.py, test_similarity.py |
| 4 | Text Extractors | extractors.py, test_extractors.py |
| 5 | OCR Provider | tesseract.py |
| 6 | Ollama Provider | ollama.py |
| 7 | Extract Stage | pipeline/extract.py |
| 8 | Embed Stage | pipeline/embed.py |
| 9 | Match Stage | pipeline/match.py |
| 10 | Execute Stage | pipeline/execute.py |
| 11 | Pipeline Orchestration | pipeline/__init__.py |
| 12 | CLI Basic | cli.py (init, config, categories) |
| 13 | CLI Sort | cli.py (sort) |
| 14 | CLI Scan/Apply | cli.py (scan, apply) |
| 15 | CLI Watch | cli.py (watch) |
| 16 | CLI Test | cli.py (test) |
| 17 | Integration Tests | test_integration.py |
| 18 | README & Polish | README.md, LICENSE |
