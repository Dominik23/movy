# data-ai v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace current extraction/clustering stack with Docling + BERTopic, organized by year.

**Architecture:** Year detection → Docling extraction → BERTopic clustering per year → Ollama naming → Copy to output

**Tech Stack:** Docling, BERTopic, sentence-transformers, Ollama, Typer

---

## File Structure

```
src/data_ai/
  cli_v2.py                    # MODIFY: Add `run` command
  pipeline/
    year_detect.py             # CREATE: Year detection logic
    extract_v2.py              # CREATE: Docling wrapper
    cluster_v2.py              # CREATE: BERTopic wrapper
    naming.py                  # CREATE: Ollama cluster naming
    run.py                     # CREATE: Main pipeline orchestration
```

---

### Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

```toml
[project]
name = "data-ai"
version = "0.2.0"
description = "Intelligent file organizer using semantic similarity"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "ollama>=0.1.0",
    "rich>=13.0.0",
    "docling>=2.0.0",
    "bertopic>=0.16.0",
    "sentence-transformers>=2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
data-ai = "data_ai.cli_v2:app"
data-ai-legacy = "data_ai.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_ai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Install new dependencies**

Run: `pip install -e .`
Expected: Successfully installed docling, bertopic, sentence-transformers

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: switch to docling + bertopic stack"
```

---

### Task 2: Year Detection Module

**Files:**
- Create: `src/data_ai/pipeline/year_detect.py`
- Create: `tests/pipeline/test_year_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_year_detect.py
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from data_ai.pipeline.year_detect import detect_year


def test_detect_year_from_filename_4digits():
    path = Path("/docs/rechnung_2024_001.pdf")
    assert detect_year(path) == 2024


def test_detect_year_from_filename_in_path():
    path = Path("/archive/2023/invoice.pdf")
    assert detect_year(path) == 2023


def test_detect_year_prefers_filename_over_path():
    path = Path("/archive/2020/rechnung_2024.pdf")
    assert detect_year(path) == 2024


def test_detect_year_ignores_invalid_years():
    path = Path("/docs/file_1899.pdf")
    with patch("data_ai.pipeline.year_detect._get_mtime_year", return_value=2025):
        assert detect_year(path) == 2025


def test_detect_year_falls_back_to_mtime():
    path = Path("/docs/random_file.pdf")
    with patch("data_ai.pipeline.year_detect._get_mtime_year", return_value=2022):
        assert detect_year(path) == 2022


def test_detect_year_falls_back_to_current_year():
    path = Path("/docs/random_file.pdf")
    with patch("data_ai.pipeline.year_detect._get_mtime_year", return_value=None):
        with patch("data_ai.pipeline.year_detect._get_current_year", return_value=2026):
            assert detect_year(path) == 2026
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_year_detect.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_ai/pipeline/year_detect.py
import re
from datetime import datetime
from pathlib import Path


YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _get_mtime_year(file_path: Path) -> int | None:
    """Get year from file modification time."""
    try:
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).year
    except (OSError, ValueError):
        return None


def _get_current_year() -> int:
    """Get current year."""
    return datetime.now().year


def detect_year(file_path: Path) -> int:
    """
    Detect year from file.

    Priority:
    1. Year in filename (e.g., rechnung_2024.pdf)
    2. Year in path (e.g., /archive/2024/file.pdf)
    3. File modification time
    4. Current year as fallback
    """
    # Try filename first
    filename_match = YEAR_PATTERN.search(file_path.name)
    if filename_match:
        return int(filename_match.group())

    # Try full path
    path_match = YEAR_PATTERN.search(str(file_path))
    if path_match:
        return int(path_match.group())

    # Try mtime
    mtime_year = _get_mtime_year(file_path)
    if mtime_year:
        return mtime_year

    # Fallback to current year
    return _get_current_year()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_year_detect.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/year_detect.py tests/pipeline/test_year_detect.py
git commit -m "feat: add year detection module"
```

---

### Task 3: Docling Extraction Module

**Files:**
- Create: `src/data_ai/pipeline/extract_v2.py`
- Create: `tests/pipeline/test_extract_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_extract_v2.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.extract_v2 import extract_text, SUPPORTED_EXTENSIONS


def test_supported_extensions_include_common_formats():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".jpg" in SUPPORTED_EXTENSIONS


def test_extract_text_returns_none_for_unsupported():
    path = Path("/docs/file.xyz")
    assert extract_text(path) == None


def test_extract_text_calls_docling():
    path = Path("/docs/test.pdf")

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "Extracted text content"

    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result

    with patch("data_ai.pipeline.extract_v2.DocumentConverter", return_value=mock_converter):
        with patch.object(path, "exists", return_value=True):
            with patch.object(path, "suffix", ".pdf"):
                result = extract_text(path)

    assert result == "Extracted text content"


def test_extract_text_returns_none_on_error():
    path = Path("/docs/corrupt.pdf")

    mock_converter = MagicMock()
    mock_converter.convert.side_effect = Exception("Conversion failed")

    with patch("data_ai.pipeline.extract_v2.DocumentConverter", return_value=mock_converter):
        with patch.object(path, "exists", return_value=True):
            result = extract_text(path)

    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_extract_v2.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_ai/pipeline/extract_v2.py
from pathlib import Path

from docling.document_converter import DocumentConverter


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".html", ".md", ".txt",
}


_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Lazy-load the DocumentConverter (heavy initialization)."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def extract_text(file_path: Path) -> str | None:
    """
    Extract text from document using Docling.

    Supports: PDF, DOCX, PPTX, images, HTML, Markdown, TXT
    Returns None if extraction fails or format unsupported.
    """
    if not file_path.exists():
        return None

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    try:
        converter = _get_converter()
        result = converter.convert(str(file_path))
        text = result.document.export_to_markdown()
        return text.strip() if text and text.strip() else None
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_extract_v2.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/extract_v2.py tests/pipeline/test_extract_v2.py
git commit -m "feat: add docling extraction module"
```

---

### Task 4: BERTopic Clustering Module

**Files:**
- Create: `src/data_ai/pipeline/cluster_v2.py`
- Create: `tests/pipeline/test_cluster_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_cluster_v2.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.cluster_v2 import cluster_documents, get_topic_keywords


def test_cluster_documents_returns_dict():
    docs = [
        (Path("/a.pdf"), "Invoice for services rendered"),
        (Path("/b.pdf"), "Invoice payment due"),
        (Path("/c.pdf"), "Contract agreement terms"),
        (Path("/d.pdf"), "Contract renewal notice"),
    ]

    mock_model = MagicMock()
    mock_model.fit_transform.return_value = ([0, 0, 1, 1], None)
    mock_model.get_topic_info.return_value = MagicMock()

    with patch("data_ai.pipeline.cluster_v2.BERTopic", return_value=mock_model):
        with patch("data_ai.pipeline.cluster_v2.SentenceTransformer"):
            result = cluster_documents(docs)

    assert isinstance(result, dict)
    assert 0 in result
    assert 1 in result
    assert result[0] == [Path("/a.pdf"), Path("/b.pdf")]
    assert result[1] == [Path("/c.pdf"), Path("/d.pdf")]


def test_cluster_documents_handles_outliers():
    docs = [
        (Path("/a.pdf"), "Some text"),
        (Path("/b.pdf"), "Other text"),
    ]

    mock_model = MagicMock()
    mock_model.fit_transform.return_value = ([-1, -1], None)  # All outliers
    mock_model.get_topic_info.return_value = MagicMock()

    with patch("data_ai.pipeline.cluster_v2.BERTopic", return_value=mock_model):
        with patch("data_ai.pipeline.cluster_v2.SentenceTransformer"):
            result = cluster_documents(docs)

    assert -1 in result
    assert len(result[-1]) == 2


def test_get_topic_keywords():
    mock_model = MagicMock()
    mock_model.get_topic.return_value = [
        ("invoice", 0.5),
        ("payment", 0.3),
        ("due", 0.2),
    ]

    keywords = get_topic_keywords(mock_model, topic_id=0, top_n=3)

    assert keywords == ["invoice", "payment", "due"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_cluster_v2.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_ai/pipeline/cluster_v2.py
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def cluster_documents(
    documents: list[tuple[Path, str]],
    min_topic_size: int = 10,
) -> tuple[dict[int, list[Path]], BERTopic]:
    """
    Cluster documents using BERTopic.

    Args:
        documents: List of (file_path, text) tuples
        min_topic_size: Minimum documents per topic

    Returns:
        Tuple of:
        - Dict mapping topic_id to list of file paths
        - The fitted BERTopic model (for keyword extraction)
    """
    if not documents:
        return {}, None

    paths = [doc[0] for doc in documents]
    texts = [doc[1] for doc in documents]

    embedding_model = _get_embedding_model()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(texts)

    # Group paths by topic
    result: dict[int, list[Path]] = {}
    for path, topic_id in zip(paths, topics):
        if topic_id not in result:
            result[topic_id] = []
        result[topic_id].append(path)

    return result, topic_model


def get_topic_keywords(
    model: BERTopic,
    topic_id: int,
    top_n: int = 5,
) -> list[str]:
    """
    Get top keywords for a topic.

    Args:
        model: Fitted BERTopic model
        topic_id: Topic ID
        top_n: Number of keywords to return

    Returns:
        List of keywords
    """
    if topic_id == -1:
        return ["sonstiges"]

    topic_words = model.get_topic(topic_id)
    if not topic_words:
        return []

    return [word for word, _ in topic_words[:top_n]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_cluster_v2.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/cluster_v2.py tests/pipeline/test_cluster_v2.py
git commit -m "feat: add bertopic clustering module"
```

---

### Task 5: Cluster Naming Module

**Files:**
- Create: `src/data_ai/pipeline/naming.py`
- Create: `tests/pipeline/test_naming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_naming.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.naming import generate_cluster_name, sanitize_folder_name


def test_sanitize_folder_name_removes_special_chars():
    assert sanitize_folder_name("Test/Name") == "Test_Name"
    assert sanitize_folder_name("A:B:C") == "A_B_C"
    assert sanitize_folder_name("Name!@#$") == "Name"


def test_sanitize_folder_name_handles_spaces():
    assert sanitize_folder_name("  Spaced Name  ") == "Spaced Name"


def test_sanitize_folder_name_limits_length():
    long_name = "A" * 100
    result = sanitize_folder_name(long_name)
    assert len(result) <= 50


def test_generate_cluster_name_for_outliers():
    result = generate_cluster_name(
        keywords=["sonstiges"],
        sample_filenames=[],
        model="llama3.2",
    )
    assert result == "_Sonstiges"


def test_generate_cluster_name_calls_ollama():
    mock_response = {"message": {"content": "Rechnungen"}}

    with patch("data_ai.pipeline.naming.ollama.chat", return_value=mock_response):
        result = generate_cluster_name(
            keywords=["rechnung", "euro", "zahlung"],
            sample_filenames=["rechnung_001.pdf", "invoice_002.pdf"],
            model="llama3.2",
        )

    assert result == "Rechnungen"


def test_generate_cluster_name_fallback_on_error():
    with patch("data_ai.pipeline.naming.ollama.chat", side_effect=Exception("API error")):
        result = generate_cluster_name(
            keywords=["rechnung", "euro"],
            sample_filenames=[],
            model="llama3.2",
        )

    # Falls back to first keyword, capitalized
    assert result == "Rechnung"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_naming.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_ai/pipeline/naming.py
import re

import ollama


def sanitize_folder_name(name: str) -> str:
    """
    Sanitize a string for use as folder name.

    - Removes/replaces special characters
    - Limits length to 50 chars
    - Strips whitespace
    """
    # Replace problematic characters with underscore
    name = re.sub(r'[/\\:*?"<>|]', '_', name)
    # Remove other special characters
    name = re.sub(r'[^\w\s\-]', '', name)
    # Clean up multiple underscores/spaces
    name = re.sub(r'[_\s]+', ' ', name)
    # Strip and limit length
    name = name.strip()[:50]
    return name


def generate_cluster_name(
    keywords: list[str],
    sample_filenames: list[str],
    model: str = "llama3.2",
) -> str:
    """
    Generate a nice folder name from topic keywords using Ollama.

    Args:
        keywords: Topic keywords from BERTopic
        sample_filenames: Example filenames from the cluster
        model: Ollama model to use

    Returns:
        Sanitized folder name
    """
    # Special case for outliers
    if keywords == ["sonstiges"] or not keywords:
        return "_Sonstiges"

    prompt = f"""Gegeben diese Keywords aus einem Dokumenten-Cluster: {', '.join(keywords)}

Beispiel-Dateinamen: {', '.join(sample_filenames[:5]) if sample_filenames else 'keine'}

Generiere einen kurzen, deutschen Ordnernamen (1-2 Wörter) der den Inhalt beschreibt.
Beispiele guter Namen: "Rechnungen", "Steuerunterlagen", "Verträge", "Kontoauszüge", "Versicherungen"

Antworte NUR mit dem Ordnernamen, nichts anderes."""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        name = response["message"]["content"].strip()
        return sanitize_folder_name(name)
    except Exception:
        # Fallback: capitalize first keyword
        fallback = keywords[0].capitalize() if keywords else "Dokumente"
        return sanitize_folder_name(fallback)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_naming.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/naming.py tests/pipeline/test_naming.py
git commit -m "feat: add ollama cluster naming module"
```

---

### Task 6: Main Pipeline Orchestration

**Files:**
- Create: `src/data_ai/pipeline/run.py`
- Create: `tests/pipeline/test_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_run.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_run.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_ai/pipeline/run.py
import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from data_ai.pipeline.year_detect import detect_year
from data_ai.pipeline.extract_v2 import extract_text, SUPPORTED_EXTENSIONS
from data_ai.pipeline.cluster_v2 import cluster_documents, get_topic_keywords
from data_ai.pipeline.naming import generate_cluster_name


console = Console()


def scan_files(input_dir: Path) -> list[Path]:
    """Scan directory for supported files."""
    files = []
    for file_path in input_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)
    return files


def group_by_year(files: list[Path]) -> dict[int, list[Path]]:
    """Group files by detected year."""
    groups: dict[int, list[Path]] = {}
    for file_path in files:
        year = detect_year(file_path)
        if year not in groups:
            groups[year] = []
        groups[year].append(file_path)
    return groups


def process_year_batch(
    files: list[Path],
    min_topic_size: int = 10,
    model: str = "llama3.2",
    progress: Progress | None = None,
    task_id: int | None = None,
) -> dict[str, list[Path]]:
    """
    Process a batch of files for one year.

    Returns: Dict mapping cluster_name to list of files
    """
    # Extract text from all files
    documents: list[tuple[Path, str]] = []

    for i, file_path in enumerate(files):
        if progress and task_id is not None:
            progress.update(task_id, advance=1)

        text = extract_text(file_path)
        if text:
            documents.append((file_path, text))

    if not documents:
        return {"_Sonstiges": files}

    # Cluster documents
    clusters, topic_model = cluster_documents(documents, min_topic_size=min_topic_size)

    if not clusters:
        return {"_Sonstiges": files}

    # Generate names for each cluster
    result: dict[str, list[Path]] = {}

    for topic_id, topic_files in clusters.items():
        keywords = get_topic_keywords(topic_model, topic_id)
        sample_names = [f.name for f in topic_files[:5]]

        name = generate_cluster_name(keywords, sample_names, model=model)

        # Handle duplicate names
        original_name = name
        counter = 1
        while name in result:
            name = f"{original_name}_{counter}"
            counter += 1

        result[name] = topic_files

    return result


def copy_files(
    clusters: dict[int, dict[str, list[Path]]],
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """
    Copy files to output directory structure.

    Args:
        clusters: Dict of year -> cluster_name -> files
        output_dir: Target directory
        dry_run: If True, don't actually copy

    Returns:
        Number of files copied
    """
    count = 0

    for year, year_clusters in sorted(clusters.items()):
        year_dir = output_dir / str(year)

        for cluster_name, files in year_clusters.items():
            cluster_dir = year_dir / cluster_name

            if not dry_run:
                cluster_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                target = cluster_dir / file_path.name

                # Handle duplicates
                if target.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    counter = 1
                    while target.exists():
                        target = cluster_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if not dry_run:
                    shutil.copy2(file_path, target)

                count += 1

    return count


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    min_topic_size: int = 10,
    model: str = "llama3.2",
    dry_run: bool = False,
) -> None:
    """
    Run the complete pipeline.

    1. Scan for files
    2. Group by year
    3. Process each year (extract, cluster, name)
    4. Copy to output
    """
    console.print(f"\n[bold]Scanning {input_dir}...[/bold]")
    files = scan_files(input_dir)
    console.print(f"Found [green]{len(files)}[/green] files\n")

    if not files:
        console.print("[yellow]No files to process.[/yellow]")
        return

    console.print("[bold]Detecting years...[/bold]")
    year_groups = group_by_year(files)

    for year in sorted(year_groups.keys()):
        console.print(f"  {year}: [cyan]{len(year_groups[year])}[/cyan] files")
    console.print()

    all_clusters: dict[int, dict[str, list[Path]]] = {}

    for year in sorted(year_groups.keys()):
        year_files = year_groups[year]
        console.print(f"[bold]Processing {year} ({len(year_files)} files)...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting & clustering", total=len(year_files))

            clusters = process_year_batch(
                year_files,
                min_topic_size=min_topic_size,
                model=model,
                progress=progress,
                task_id=task,
            )

        all_clusters[year] = clusters

        console.print(f"  Found [green]{len(clusters)}[/green] topics:")
        for name, topic_files in sorted(clusters.items(), key=lambda x: -len(x[1])):
            console.print(f"    - {name}: {len(topic_files)} files")
        console.print()

    console.print(f"[bold]{'Would copy' if dry_run else 'Copying'} files to {output_dir}...[/bold]")
    copied = copy_files(all_clusters, output_dir, dry_run=dry_run)

    if dry_run:
        console.print(f"\n[yellow]Dry run:[/yellow] Would copy {copied} files")
    else:
        console.print(f"\n[green]Done![/green] Copied {copied} files")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_run.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/run.py tests/pipeline/test_run.py
git commit -m "feat: add main pipeline orchestration"
```

---

### Task 7: CLI Integration

**Files:**
- Modify: `src/data_ai/cli_v2.py`
- Create: `tests/test_cli_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_run.py
import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from data_ai.cli_v2 import app


runner = CliRunner()


def test_run_command_exists():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "input" in result.output.lower()
    assert "output" in result.output.lower()


def test_run_command_requires_input():
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_run_command_calls_pipeline(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with patch("data_ai.cli_v2.run_pipeline") as mock_run:
        result = runner.invoke(app, [
            "run",
            str(input_dir),
            "--output", str(output_dir),
            "--dry-run",
        ])

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.kwargs["input_dir"] == input_dir
    assert call_args.kwargs["output_dir"] == output_dir
    assert call_args.kwargs["dry_run"] == True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_run.py -v`
Expected: FAIL with "No such command 'run'"

- [ ] **Step 3: Add run command to CLI**

Read `src/data_ai/cli_v2.py` first, then add the run command:

```python
# Add this import at the top
from data_ai.pipeline.run import run_pipeline

# Add this command
@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    min_topic_size: int = typer.Option(10, "--min-topic-size", help="Minimum documents per cluster"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model for naming"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without copying"),
) -> None:
    """
    Process documents: detect year, extract text, cluster, and organize.

    Example:
        data-ai run /input --output /output
    """
    if not input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory does not exist: {input_dir}")
        raise typer.Exit(1)

    output_dir = output or (input_dir.parent / "output")

    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        min_topic_size=min_topic_size,
        model=model,
        dry_run=dry_run,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_run.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/cli_v2.py tests/test_cli_run.py
git commit -m "feat: add run command to CLI"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/integration/test_full_pipeline.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_full_pipeline.py
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
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/integration/test_full_pipeline.py -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_full_pipeline.py
git commit -m "test: add integration tests for full pipeline"
```

---

### Task 9: Cleanup Old Code (Optional)

**Files:**
- Delete or deprecate old pipeline files

- [ ] **Step 1: Mark old files as deprecated**

Add deprecation notice to old files (don't delete yet to avoid breaking existing users):

```python
# Add to top of src/data_ai/pipeline/extract.py
"""
DEPRECATED: Use extract_v2.py with Docling instead.
This module will be removed in v0.3.0.
"""

# Add to top of src/data_ai/pipeline/cluster.py
"""
DEPRECATED: Use cluster_v2.py with BERTopic instead.
This module will be removed in v0.3.0.
"""
```

- [ ] **Step 2: Commit**

```bash
git add src/data_ai/pipeline/extract.py src/data_ai/pipeline/cluster.py
git commit -m "chore: mark old pipeline modules as deprecated"
```

---

### Task 10: Final Test with Real Data

- [ ] **Step 1: Test with small dataset**

Run: `data-ai run /path/to/test/folder --output /tmp/output --dry-run`
Expected: Shows year detection, clustering progress, and planned output structure

- [ ] **Step 2: Test with actual copy**

Run: `data-ai run /path/to/test/folder --output /tmp/output`
Expected: Files organized in `/tmp/output/YEAR/CLUSTER/` structure

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete data-ai v2 with docling + bertopic"
```
