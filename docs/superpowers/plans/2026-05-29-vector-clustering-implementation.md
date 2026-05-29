# Vector-Based Document Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace keyword-based file matching with vector DB clustering for automatic category discovery.

**Architecture:** Pipeline with 6 stages (scan, embed, cluster, name, review, apply). Qdrant stores embeddings and cluster state. KMeans with Elbow method finds optimal clusters. LLM generates cluster names. Pyvis creates interactive HTML review.

**Tech Stack:** Python 3.11+, Qdrant (Docker), scikit-learn, pyvis, Ollama (nomic-embed-text, llava, llama3.2)

---

## File Structure

### New Files
| File | Responsibility |
|------|----------------|
| `src/data_ai/storage/__init__.py` | Storage layer exports |
| `src/data_ai/storage/qdrant.py` | Qdrant client wrapper, collection management |
| `src/data_ai/storage/models.py` | Pydantic models for Document, Cluster |
| `src/data_ai/pipeline/cluster.py` | KMeans, Elbow method, split logic |
| `src/data_ai/pipeline/naming.py` | LLM-based cluster naming |
| `src/data_ai/pipeline/review.py` | HTML graph generation with pyvis |
| `src/data_ai/cli_v2.py` | New CLI commands (scan, cluster, review, apply) |
| `docker-compose.yaml` | Qdrant container setup |
| `tests/test_storage.py` | Storage layer tests |
| `tests/test_cluster.py` | Clustering tests |
| `tests/test_naming.py` | Naming tests |
| `tests/test_review.py` | Review HTML tests |
| `tests/test_cli_v2.py` | New CLI tests |

### Modified Files
| File | Changes |
|------|---------|
| `pyproject.toml` | Add qdrant-client, scikit-learn, pyvis deps |
| `src/data_ai/config.py` | Add Qdrant + clustering settings |
| `src/data_ai/pipeline/extract.py` | Add recursive scan, trash handling |
| `src/data_ai/pipeline/embed.py` | Batch processing with progress |
| `src/data_ai/pipeline/execute.py` | Change move→copy, add logging |
| `src/data_ai/utils/similarity.py` | Add cosine_distance, variance calc |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:8-21`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

```toml
dependencies = [
    "typer>=0.9.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "ollama>=0.1.0",
    "pdfplumber>=0.9.0",
    "python-docx>=0.8.0",
    "python-pptx>=0.6.0",
    "pytesseract>=0.3.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "watchdog>=3.0.0",
    "rich>=13.0.0",
    "qdrant-client>=1.7.0",
    "scikit-learn>=1.3.0",
    "pyvis>=0.3.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

- [ ] **Step 3: Verify imports work**

Run: `python -c "from qdrant_client import QdrantClient; from sklearn.cluster import KMeans; from pyvis.network import Network; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add qdrant-client, scikit-learn, pyvis"
```

---

## Task 2: Add Docker Compose for Qdrant

**Files:**
- Create: `docker-compose.yaml`

- [ ] **Step 1: Create docker-compose.yaml**

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

volumes:
  qdrant_data:
```

- [ ] **Step 2: Start Qdrant container**

Run: `docker-compose up -d`
Expected: Container starts without errors

- [ ] **Step 3: Verify Qdrant is running**

Run: `curl -s http://localhost:6333/collections | python -m json.tool`
Expected: `{"result": {"collections": []}, "status": "ok", "time": ...}`

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yaml
git commit -m "infra: add docker-compose for Qdrant"
```

---

## Task 3: Update Config with New Settings

**Files:**
- Modify: `src/data_ai/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test for new config fields**

Add to `tests/test_config.py`:

```python
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
    assert cfg.settings.min_clusters == 2
    assert cfg.settings.max_clusters == 20
    assert cfg.settings.variance_threshold == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_has_qdrant_settings -v`
Expected: FAIL with "AttributeError: 'Settings' object has no attribute 'qdrant_url'"

- [ ] **Step 3: Add new settings fields**

Edit `src/data_ai/config.py`, update the `Settings` class:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_config_has_qdrant_settings -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/config.py tests/test_config.py
git commit -m "feat(config): add Qdrant and clustering settings"
```

---

## Task 4: Create Storage Models

**Files:**
- Create: `src/data_ai/storage/__init__.py`
- Create: `src/data_ai/storage/models.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Create storage package init**

```python
# src/data_ai/storage/__init__.py
from data_ai.storage.models import Document, Cluster, DocumentStatus, ClusterStatus

__all__ = ["Document", "Cluster", "DocumentStatus", "ClusterStatus"]
```

- [ ] **Step 2: Write failing test for models**

Create `tests/test_storage.py`:

```python
import pytest
from datetime import datetime


def test_document_model_creation():
    from data_ai.storage.models import Document, DocumentStatus

    doc = Document(
        id="test-uuid",
        source_path="/path/to/file.pdf",
        file_type="pdf",
        file_size=1024,
        summary="Test summary",
        status=DocumentStatus.PENDING,
    )

    assert doc.id == "test-uuid"
    assert doc.status == DocumentStatus.PENDING
    assert doc.cluster_id is None
    assert doc.created_at is not None


def test_cluster_model_creation():
    from data_ai.storage.models import Cluster, ClusterStatus

    cluster = Cluster(
        id="cluster-uuid",
        name="Rechnungen",
        doc_count=10,
        variance=0.25,
        centroid=[0.1] * 768,
        status=ClusterStatus.PROPOSED,
    )

    assert cluster.name == "Rechnungen"
    assert cluster.status == ClusterStatus.PROPOSED
    assert len(cluster.centroid) == 768
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_ai.storage'"

- [ ] **Step 4: Create models.py**

```python
# src/data_ai/storage/models.py
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    EMBEDDED = "embedded"
    CLUSTERED = "clustered"
    APPLIED = "applied"


class ClusterStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"


class Document(BaseModel):
    id: str
    source_path: str
    file_type: str
    file_size: int
    summary: str
    status: DocumentStatus = DocumentStatus.PENDING
    cluster_id: Optional[str] = None
    vector: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Cluster(BaseModel):
    id: str
    name: str
    doc_count: int
    variance: float
    centroid: list[float]
    status: ClusterStatus = ClusterStatus.PROPOSED
    parent_cluster: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/data_ai/storage/ tests/test_storage.py
git commit -m "feat(storage): add Document and Cluster models"
```

---

## Task 5: Create Qdrant Client Wrapper

**Files:**
- Create: `src/data_ai/storage/qdrant.py`
- Modify: `src/data_ai/storage/__init__.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing test for Qdrant wrapper**

Add to `tests/test_storage.py`:

```python
from unittest.mock import MagicMock, patch


def test_qdrant_store_init_creates_collections():
    with patch("data_ai.storage.qdrant.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.collection_exists.return_value = False

        from data_ai.storage.qdrant import QdrantStore

        store = QdrantStore(url="localhost:6333", prefix="test")

        assert mock_instance.create_collection.call_count == 2


def test_qdrant_store_upsert_document():
    with patch("data_ai.storage.qdrant.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.collection_exists.return_value = True

        from data_ai.storage.qdrant import QdrantStore
        from data_ai.storage.models import Document, DocumentStatus

        store = QdrantStore(url="localhost:6333", prefix="test")

        doc = Document(
            id="doc-1",
            source_path="/path/file.pdf",
            file_type="pdf",
            file_size=1024,
            summary="Test",
            status=DocumentStatus.PENDING,
            vector=[0.1] * 768,
        )

        store.upsert_document(doc)

        mock_instance.upsert.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py::test_qdrant_store_init_creates_collections -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_ai.storage.qdrant'"

- [ ] **Step 3: Create qdrant.py**

```python
# src/data_ai/storage/qdrant.py
from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from data_ai.storage.models import Document, Cluster, DocumentStatus, ClusterStatus

VECTOR_SIZE = 768  # nomic-embed-text dimension


class QdrantStore:
    def __init__(self, url: str = "localhost:6333", prefix: str = "data_ai"):
        self.client = QdrantClient(url=url)
        self.docs_collection = f"{prefix}_documents"
        self.clusters_collection = f"{prefix}_clusters"
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        if not self.client.collection_exists(self.docs_collection):
            self.client.create_collection(
                collection_name=self.docs_collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

        if not self.client.collection_exists(self.clusters_collection):
            self.client.create_collection(
                collection_name=self.clusters_collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def upsert_document(self, doc: Document) -> None:
        vector = doc.vector if doc.vector else [0.0] * VECTOR_SIZE
        self.client.upsert(
            collection_name=self.docs_collection,
            points=[
                PointStruct(
                    id=doc.id,
                    vector=vector,
                    payload={
                        "source_path": doc.source_path,
                        "file_type": doc.file_type,
                        "file_size": doc.file_size,
                        "summary": doc.summary,
                        "status": doc.status.value,
                        "cluster_id": doc.cluster_id,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat(),
                    },
                )
            ],
        )

    def upsert_cluster(self, cluster: Cluster) -> None:
        self.client.upsert(
            collection_name=self.clusters_collection,
            points=[
                PointStruct(
                    id=cluster.id,
                    vector=cluster.centroid,
                    payload={
                        "name": cluster.name,
                        "doc_count": cluster.doc_count,
                        "variance": cluster.variance,
                        "status": cluster.status.value,
                        "parent_cluster": cluster.parent_cluster,
                        "created_at": cluster.created_at.isoformat(),
                        "updated_at": cluster.updated_at.isoformat(),
                    },
                )
            ],
        )

    def get_documents_by_status(self, status: DocumentStatus) -> list[Document]:
        results = self.client.scroll(
            collection_name=self.docs_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="status", match=MatchValue(value=status.value))]
            ),
            with_vectors=True,
            limit=10000,
        )

        documents = []
        for point in results[0]:
            documents.append(Document(
                id=point.id,
                vector=point.vector,
                **point.payload,
            ))
        return documents

    def get_all_documents(self) -> list[Document]:
        results = self.client.scroll(
            collection_name=self.docs_collection,
            with_vectors=True,
            limit=10000,
        )

        documents = []
        for point in results[0]:
            documents.append(Document(
                id=point.id,
                vector=point.vector,
                **point.payload,
            ))
        return documents

    def get_all_clusters(self) -> list[Cluster]:
        results = self.client.scroll(
            collection_name=self.clusters_collection,
            with_vectors=True,
            limit=1000,
        )

        clusters = []
        for point in results[0]:
            clusters.append(Cluster(
                id=point.id,
                centroid=point.vector,
                **point.payload,
            ))
        return clusters

    def get_documents_by_cluster(self, cluster_id: str) -> list[Document]:
        results = self.client.scroll(
            collection_name=self.docs_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]
            ),
            with_vectors=True,
            limit=10000,
        )

        documents = []
        for point in results[0]:
            documents.append(Document(
                id=point.id,
                vector=point.vector,
                **point.payload,
            ))
        return documents

    def update_document_status(self, doc_id: str, status: DocumentStatus) -> None:
        self.client.set_payload(
            collection_name=self.docs_collection,
            payload={"status": status.value},
            points=[doc_id],
        )

    def update_document_cluster(self, doc_id: str, cluster_id: str) -> None:
        self.client.set_payload(
            collection_name=self.docs_collection,
            payload={"cluster_id": cluster_id, "status": DocumentStatus.CLUSTERED.value},
            points=[doc_id],
        )

    def update_cluster_status(self, cluster_id: str, status: ClusterStatus) -> None:
        self.client.set_payload(
            collection_name=self.clusters_collection,
            payload={"status": status.value},
            points=[cluster_id],
        )

    def update_cluster_name(self, cluster_id: str, name: str) -> None:
        self.client.set_payload(
            collection_name=self.clusters_collection,
            payload={"name": name},
            points=[cluster_id],
        )

    def delete_all_clusters(self) -> None:
        clusters = self.get_all_clusters()
        if clusters:
            self.client.delete(
                collection_name=self.clusters_collection,
                points_selector=[c.id for c in clusters],
            )

        # Reset cluster assignments on documents
        docs = self.get_all_documents()
        for doc in docs:
            if doc.cluster_id:
                self.client.set_payload(
                    collection_name=self.docs_collection,
                    payload={"cluster_id": None, "status": DocumentStatus.EMBEDDED.value},
                    points=[doc.id],
                )

    def reset(self) -> None:
        self.client.delete_collection(self.docs_collection)
        self.client.delete_collection(self.clusters_collection)
        self._ensure_collections()

    def generate_id(self) -> str:
        return str(uuid4())
```

- [ ] **Step 4: Update storage __init__.py**

```python
# src/data_ai/storage/__init__.py
from data_ai.storage.models import Document, Cluster, DocumentStatus, ClusterStatus
from data_ai.storage.qdrant import QdrantStore

__all__ = ["Document", "Cluster", "DocumentStatus", "ClusterStatus", "QdrantStore"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/data_ai/storage/ tests/test_storage.py
git commit -m "feat(storage): add Qdrant client wrapper"
```

---

## Task 6: Add Utility Functions for Clustering

**Files:**
- Modify: `src/data_ai/utils/similarity.py`
- Test: `tests/test_similarity.py`

- [ ] **Step 1: Write failing tests for new utility functions**

Add to `tests/test_similarity.py`:

```python
def test_cosine_distance():
    from data_ai.utils.similarity import cosine_distance

    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]

    assert cosine_distance(vec_a, vec_b) == pytest.approx(0.0)

    vec_c = [0.0, 1.0, 0.0]
    assert cosine_distance(vec_a, vec_c) == pytest.approx(1.0)


def test_compute_variance():
    from data_ai.utils.similarity import compute_variance

    vectors = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    # All same vectors -> variance should be 0
    assert compute_variance(vectors) == pytest.approx(0.0)


def test_compute_variance_different_vectors():
    from data_ai.utils.similarity import compute_variance

    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    # Orthogonal vectors -> high variance
    variance = compute_variance(vectors)
    assert variance > 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_similarity.py::test_cosine_distance -v`
Expected: FAIL with "cannot import name 'cosine_distance'"

- [ ] **Step 3: Add new utility functions**

Add to `src/data_ai/utils/similarity.py`:

```python
def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Returns 1 - cosine_similarity (0 = identical, 2 = opposite)."""
    return 1.0 - cosine_similarity(vec_a, vec_b)


def compute_variance(vectors: list[list[float]]) -> float:
    """Compute variance of distances from centroid."""
    if len(vectors) < 2:
        return 0.0

    centroid = average_vectors(vectors)
    distances = [cosine_distance(v, centroid) for v in vectors]
    return float(np.var(distances))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_similarity.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/utils/similarity.py tests/test_similarity.py
git commit -m "feat(utils): add cosine_distance and compute_variance"
```

---

## Task 7: Update Extract Stage with Recursive Scan and Trash

**Files:**
- Modify: `src/data_ai/pipeline/extract.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for recursive scan**

Add to `tests/test_pipeline.py`:

```python
def test_scan_folder_recursive(tmp_path: Path):
    from data_ai.pipeline.extract import scan_folder

    # Create nested structure
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub1" / "sub2").mkdir()

    (tmp_path / "file1.txt").write_text("content 1")
    (tmp_path / "sub1" / "file2.txt").write_text("content 2")
    (tmp_path / "sub1" / "sub2" / "file3.txt").write_text("content 3")

    files = scan_folder(tmp_path)

    assert len(files) == 3
    assert all(f.suffix == ".txt" for f in files)


def test_scan_folder_moves_unsupported_to_trash(tmp_path: Path):
    from data_ai.pipeline.extract import scan_folder

    (tmp_path / "good.txt").write_text("content")
    (tmp_path / "bad.xyz").write_text("unsupported")

    trash_dir = tmp_path / ".trash"
    files, trash_log = scan_folder(tmp_path, trash_dir=trash_dir)

    assert len(files) == 1
    assert files[0].name == "good.txt"
    assert (trash_dir / "bad.xyz").exists()
    assert len(trash_log) == 1
    assert "unsupported" in trash_log[0]["reason"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_scan_folder_recursive -v`
Expected: FAIL with "cannot import name 'scan_folder'"

- [ ] **Step 3: Add scan_folder function**

Add to `src/data_ai/pipeline/extract.py`:

```python
import shutil
from datetime import datetime

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}


def scan_folder(
    folder: Path,
    trash_dir: Path | None = None,
) -> tuple[list[Path], list[dict]]:
    """
    Recursively scan folder for supported files.
    Moves unsupported files to trash_dir if provided.

    Returns: (supported_files, trash_log)
    """
    supported_files = []
    trash_log = []

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip hidden files and trash folder
        if file_path.name.startswith("."):
            continue
        if trash_dir and trash_dir in file_path.parents:
            continue

        suffix = file_path.suffix.lower()

        if suffix in SUPPORTED_EXTENSIONS:
            supported_files.append(file_path)
        elif trash_dir:
            # Move to trash
            trash_dir.mkdir(parents=True, exist_ok=True)
            target = trash_dir / file_path.name
            if target.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target = trash_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            shutil.copy2(file_path, target)
            trash_log.append({
                "source": str(file_path),
                "target": str(target),
                "reason": f"Unsupported file type: {suffix}",
                "timestamp": datetime.now().isoformat(),
            })

    return supported_files, trash_log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py::test_scan_folder_recursive tests/test_pipeline.py::test_scan_folder_moves_unsupported_to_trash -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/extract.py tests/test_pipeline.py
git commit -m "feat(extract): add recursive scan with trash handling"
```

---

## Task 8: Create Clustering Stage

**Files:**
- Create: `src/data_ai/pipeline/cluster.py`
- Test: `tests/test_cluster.py`

- [ ] **Step 1: Write failing tests for clustering**

Create `tests/test_cluster.py`:

```python
import pytest
import numpy as np


def test_find_optimal_k_returns_reasonable_k():
    from data_ai.pipeline.cluster import find_optimal_k

    # Create 3 distinct clusters
    np.random.seed(42)
    cluster1 = np.random.randn(20, 10) + [5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cluster2 = np.random.randn(20, 10) + [0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    cluster3 = np.random.randn(20, 10) + [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]

    vectors = np.vstack([cluster1, cluster2, cluster3]).tolist()

    k = find_optimal_k(vectors, min_k=2, max_k=10)

    assert 2 <= k <= 5  # Should find approximately 3


def test_cluster_documents_returns_assignments():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    vectors = np.random.randn(30, 10).tolist()

    assignments, centroids = cluster_documents(vectors, k=3)

    assert len(assignments) == 30
    assert len(centroids) == 3
    assert all(0 <= a < 3 for a in assignments)


def test_should_split_returns_true_for_high_variance():
    from data_ai.pipeline.cluster import should_split

    # Very different vectors
    vectors = [
        [1.0] + [0.0] * 9,
        [0.0, 1.0] + [0.0] * 8,
        [0.0, 0.0, 1.0] + [0.0] * 7,
    ]

    assert should_split(vectors, threshold=0.1) is True


def test_should_split_returns_false_for_low_variance():
    from data_ai.pipeline.cluster import should_split

    # Very similar vectors
    vectors = [
        [1.0, 0.1, 0.0],
        [1.0, 0.0, 0.1],
        [1.0, 0.05, 0.05],
    ]

    assert should_split(vectors, threshold=0.5) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cluster.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_ai.pipeline.cluster'"

- [ ] **Step 3: Create cluster.py**

```python
# src/data_ai/pipeline/cluster.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_ai.utils.similarity import compute_variance, average_vectors


def find_optimal_k(
    vectors: list[list[float]],
    min_k: int = 2,
    max_k: int = 20,
) -> int:
    """
    Find optimal number of clusters using Elbow method + Silhouette score.
    """
    n_samples = len(vectors)
    if n_samples < min_k:
        return min_k

    max_k = min(max_k, n_samples // 2, n_samples - 1)
    if max_k < min_k:
        return min_k

    X = np.array(vectors)

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_documents(
    vectors: list[list[float]],
    k: int,
) -> tuple[list[int], list[list[float]]]:
    """
    Cluster vectors using KMeans.

    Returns: (cluster_assignments, centroids)
    """
    X = np.array(vectors)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_.tolist()

    return labels.tolist(), centroids


def should_split(
    vectors: list[list[float]],
    threshold: float = 0.4,
) -> bool:
    """
    Check if a cluster should be split based on variance.
    """
    if len(vectors) < 4:  # Need at least 4 to split into 2+2
        return False

    variance = compute_variance(vectors)
    return variance > threshold


def split_cluster(
    vectors: list[list[float]],
    doc_ids: list[str],
) -> tuple[list[tuple[str, int]], list[list[float]]]:
    """
    Split a cluster into 2 sub-clusters.

    Returns: ([(doc_id, sub_cluster_idx), ...], [centroid1, centroid2])
    """
    assignments, centroids = cluster_documents(vectors, k=2)

    result = [(doc_id, assignment) for doc_id, assignment in zip(doc_ids, assignments)]

    return result, centroids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cluster.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/cluster.py tests/test_cluster.py
git commit -m "feat(cluster): add KMeans clustering with elbow method"
```

---

## Task 9: Create Naming Stage

**Files:**
- Create: `src/data_ai/pipeline/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Write failing tests for naming**

Create `tests/test_naming.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


def test_generate_cluster_name_calls_ollama():
    with patch("data_ai.pipeline.naming.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {
            "message": {"content": "Rechnungen"}
        }

        from data_ai.pipeline.naming import generate_cluster_name

        summaries = [
            "Rechnung Nr. 12345 über 500 EUR",
            "Invoice for services rendered",
            "Zahlungsaufforderung vom 01.01.2026",
        ]

        name = generate_cluster_name(summaries, model="llama3.2")

        assert name == "Rechnungen"
        mock_ollama.chat.assert_called_once()


def test_generate_cluster_name_cleans_response():
    with patch("data_ai.pipeline.naming.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {
            "message": {"content": "  Verträge und Dokumente  \n"}
        }

        from data_ai.pipeline.naming import generate_cluster_name

        name = generate_cluster_name(["test"], model="llama3.2")

        assert name == "Verträge und Dokumente"


def test_generate_cluster_name_fallback_on_error():
    with patch("data_ai.pipeline.naming.ollama") as mock_ollama:
        mock_ollama.chat.side_effect = Exception("Connection error")

        from data_ai.pipeline.naming import generate_cluster_name

        name = generate_cluster_name(["test"], model="llama3.2")

        assert name.startswith("Cluster_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_naming.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_ai.pipeline.naming'"

- [ ] **Step 3: Create naming.py**

```python
# src/data_ai/pipeline/naming.py
from datetime import datetime
from typing import Optional

import ollama


def generate_cluster_name(
    summaries: list[str],
    model: str = "llama3.2",
    max_summaries: int = 5,
) -> str:
    """
    Generate a cluster name using LLM based on document summaries.
    """
    if not summaries:
        return _fallback_name()

    # Take only first N summaries
    selected = summaries[:max_summaries]

    # Truncate each summary
    truncated = [s[:500] for s in selected]

    docs_text = "\n\n".join(
        f"Dokument {i+1}: {summary}"
        for i, summary in enumerate(truncated)
    )

    prompt = f"""Analysiere diese Dokument-Zusammenfassungen und gib einen kurzen, beschreibenden Kategorie-Namen (1-3 Wörter, deutsch).

{docs_text}

Antworte NUR mit dem Kategorie-Namen, ohne Erklärung:"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        name = response["message"]["content"].strip()

        # Clean up common artifacts
        name = name.strip('"\'')
        name = name.split("\n")[0]  # Take only first line

        if len(name) > 50:
            name = name[:50]

        if not name:
            return _fallback_name()

        return name

    except Exception:
        return _fallback_name()


def _fallback_name() -> str:
    """Generate fallback cluster name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"Cluster_{timestamp}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_naming.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/naming.py tests/test_naming.py
git commit -m "feat(naming): add LLM-based cluster naming"
```

---

## Task 10: Create Review Stage (HTML Generation)

**Files:**
- Create: `src/data_ai/pipeline/review.py`
- Test: `tests/test_review.py`

- [ ] **Step 1: Write failing tests for review**

Create `tests/test_review.py`:

```python
import pytest
from pathlib import Path


def test_generate_review_html_creates_file(tmp_path: Path):
    from data_ai.pipeline.review import generate_review_html
    from data_ai.storage.models import Cluster, ClusterStatus

    clusters = [
        Cluster(
            id="c1",
            name="Rechnungen",
            doc_count=10,
            variance=0.2,
            centroid=[0.1] * 768,
            status=ClusterStatus.PROPOSED,
        ),
        Cluster(
            id="c2",
            name="Verträge",
            doc_count=5,
            variance=0.3,
            centroid=[0.2] * 768,
            status=ClusterStatus.PROPOSED,
        ),
    ]

    cluster_docs = {
        "c1": ["doc1.pdf", "doc2.pdf"],
        "c2": ["doc3.pdf"],
    }

    output_path = tmp_path / "review.html"

    generate_review_html(clusters, cluster_docs, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Rechnungen" in content
    assert "Verträge" in content


def test_generate_review_html_contains_graph():
    from data_ai.pipeline.review import generate_review_html
    from data_ai.storage.models import Cluster, ClusterStatus
    import tempfile

    clusters = [
        Cluster(
            id="c1",
            name="Test",
            doc_count=5,
            variance=0.1,
            centroid=[0.1] * 768,
            status=ClusterStatus.PROPOSED,
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output_path = Path(f.name)

    generate_review_html(clusters, {"c1": ["doc.pdf"]}, output_path)

    content = output_path.read_text()
    assert "vis-network" in content or "pyvis" in content.lower() or "nodes" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_review.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_ai.pipeline.review'"

- [ ] **Step 3: Create review.py**

```python
# src/data_ai/pipeline/review.py
from pathlib import Path

from pyvis.network import Network

from data_ai.storage.models import Cluster
from data_ai.utils.similarity import cosine_similarity


def generate_review_html(
    clusters: list[Cluster],
    cluster_docs: dict[str, list[str]],
    output_path: Path,
    edge_threshold: float = 0.3,
) -> None:
    """
    Generate interactive HTML visualization of clusters.
    """
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
    )

    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)

    # Add nodes (clusters)
    for cluster in clusters:
        # Size based on doc count (min 20, max 100)
        size = min(100, max(20, cluster.doc_count * 5))

        # Color based on variance (green=low, red=high)
        variance_normalized = min(1.0, cluster.variance / 0.5)
        r = int(255 * variance_normalized)
        g = int(255 * (1 - variance_normalized))
        color = f"rgb({r},{g},100)"

        # Hover title with doc list
        docs = cluster_docs.get(cluster.id, [])
        docs_preview = docs[:10]
        if len(docs) > 10:
            docs_preview.append(f"... und {len(docs) - 10} weitere")

        title = f"""
<b>{cluster.name}</b><br>
Dokumente: {cluster.doc_count}<br>
Varianz: {cluster.variance:.2f}<br>
Status: {cluster.status.value}<br>
<hr>
{'<br>'.join(docs_preview)}
"""

        net.add_node(
            cluster.id,
            label=f"{cluster.name}\n({cluster.doc_count})",
            title=title,
            size=size,
            color=color,
        )

    # Add edges based on centroid similarity
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            similarity = cosine_similarity(c1.centroid, c2.centroid)

            if similarity > edge_threshold:
                width = similarity * 5
                net.add_edge(
                    c1.id,
                    c2.id,
                    value=width,
                    title=f"Similarity: {similarity:.2f}",
                )

    # Generate HTML with custom additions
    net.save_graph(str(output_path))

    # Add summary table
    _add_summary_table(output_path, clusters, cluster_docs)


def _add_summary_table(
    html_path: Path,
    clusters: list[Cluster],
    cluster_docs: dict[str, list[str]],
) -> None:
    """Add a summary table to the HTML file."""
    content = html_path.read_text()

    table_html = """
<div style="margin: 20px; font-family: Arial, sans-serif;">
    <h2>Cluster-Übersicht</h2>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f0f0f0;">
            <th style="padding: 10px;">Name</th>
            <th style="padding: 10px;">Dokumente</th>
            <th style="padding: 10px;">Varianz</th>
            <th style="padding: 10px;">Status</th>
        </tr>
"""

    for cluster in sorted(clusters, key=lambda c: c.doc_count, reverse=True):
        status_color = {
            "proposed": "#ffd700",
            "approved": "#90ee90",
            "applied": "#add8e6",
        }.get(cluster.status.value, "#ffffff")

        table_html += f"""
        <tr>
            <td style="padding: 10px;"><b>{cluster.name}</b></td>
            <td style="padding: 10px; text-align: center;">{cluster.doc_count}</td>
            <td style="padding: 10px; text-align: center;">{cluster.variance:.2f}</td>
            <td style="padding: 10px; text-align: center; background-color: {status_color};">
                {cluster.status.value}
            </td>
        </tr>
"""

    table_html += """
    </table>
</div>
"""

    # Insert before closing body tag
    content = content.replace("</body>", f"{table_html}</body>")
    html_path.write_text(content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_review.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/review.py tests/test_review.py
git commit -m "feat(review): add HTML graph generation with pyvis"
```

---

## Task 11: Update Execute Stage (Copy Instead of Move)

**Files:**
- Modify: `src/data_ai/pipeline/execute.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for copy function**

Add to `tests/test_pipeline.py`:

```python
def test_execute_copy_preserves_original(tmp_path: Path):
    from data_ai.pipeline.execute import execute_copy

    source = tmp_path / "inbox" / "doc.txt"
    source.parent.mkdir()
    source.write_text("content")

    target_dir = tmp_path / "sorted" / "Category"

    result = execute_copy(source, target_dir)

    assert result is not None
    assert source.exists()  # Original preserved
    assert (target_dir / "doc.txt").exists()


def test_execute_copy_writes_log(tmp_path: Path):
    from data_ai.pipeline.execute import execute_copy

    source = tmp_path / "doc.txt"
    source.write_text("content")

    target_dir = tmp_path / "sorted"
    log_file = tmp_path / "copy.log"

    execute_copy(source, target_dir, log_file=log_file)

    assert log_file.exists()
    import json
    log_entry = json.loads(log_file.read_text().strip().split("\n")[-1])
    assert log_entry["source"] == str(source)
    assert "target" in log_entry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::test_execute_copy_preserves_original -v`
Expected: FAIL with "cannot import name 'execute_copy'"

- [ ] **Step 3: Add execute_copy function**

Add to `src/data_ai/pipeline/execute.py`:

```python
import json
import re


def execute_copy(
    source: Path,
    target_dir: Path,
    log_file: Optional[Path] = None,
) -> Optional[Path]:
    """
    Copy file to target directory (preserves original).
    Returns target path on success, None on failure.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / source.name

        # Handle duplicate filenames
        if target_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = source.stem
            suffix = source.suffix
            target_path = target_dir / f"{stem}_{timestamp}{suffix}"

        shutil.copy2(str(source), str(target_path))

        # Write log entry
        if log_file:
            log_entry = {
                "source": str(source),
                "target": str(target_path),
                "timestamp": datetime.now().isoformat(),
            }
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return target_path

    except Exception as e:
        console.print(f"[red]Error copying file: {e}[/red]")
        return None


def sanitize_folder_name(name: str) -> str:
    """
    Sanitize a string to be a valid folder name.
    """
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    # Fallback if empty
    if not sanitized:
        sanitized = "Unnamed"
    return sanitized
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py::test_execute_copy_preserves_original tests/test_pipeline.py::test_execute_copy_writes_log -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/execute.py tests/test_pipeline.py
git commit -m "feat(execute): add copy function with logging"
```

---

## Task 12: Create New CLI Commands

**Files:**
- Create: `src/data_ai/cli_v2.py`
- Test: `tests/test_cli_v2.py`

- [ ] **Step 1: Write failing tests for CLI**

Create `tests/test_cli_v2.py`:

```python
import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

runner = CliRunner()


def test_cli_status_shows_counts():
    with patch("data_ai.cli_v2.QdrantStore") as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.get_all_documents.return_value = []
        mock_store.get_all_clusters.return_value = []

        from data_ai.cli_v2 import app

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Documents:" in result.stdout or "documents" in result.stdout.lower()


def test_cli_reset_requires_confirm():
    from data_ai.cli_v2 import app

    result = runner.invoke(app, ["reset"])

    # Should fail without --confirm flag
    assert result.exit_code != 0 or "confirm" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_v2.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_ai.cli_v2'"

- [ ] **Step 3: Create cli_v2.py with basic commands**

```python
# src/data_ai/cli_v2.py
import json
import webbrowser
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from data_ai.config import load_config, get_default_config_path, create_default_config
from data_ai.storage import QdrantStore, Document, Cluster, DocumentStatus, ClusterStatus
from data_ai.pipeline.extract import scan_folder, extract_stage
from data_ai.pipeline.embed import embed_stage
from data_ai.pipeline.cluster import find_optimal_k, cluster_documents, should_split, split_cluster
from data_ai.pipeline.naming import generate_cluster_name
from data_ai.pipeline.review import generate_review_html
from data_ai.pipeline.execute import execute_copy, sanitize_folder_name

app = typer.Typer(
    name="data-ai",
    help="Intelligent file organizer using vector clustering",
)
console = Console()


def get_store(config_path: Optional[Path] = None) -> QdrantStore:
    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
        return QdrantStore(url=cfg.settings.qdrant_url, prefix=cfg.settings.qdrant_collection_prefix)
    return QdrantStore()


@app.command()
def init(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    db_url: str = typer.Option("localhost:6333", "--db-url"),
) -> None:
    """Initialize config and verify Qdrant connection."""
    path = config_path or get_default_config_path()

    if not path.exists():
        create_default_config(path)
        console.print(f"[green]Created config at: {path}[/green]")

    # Test Qdrant connection
    try:
        store = QdrantStore(url=db_url)
        console.print(f"[green]Connected to Qdrant at {db_url}[/green]")
    except Exception as e:
        console.print(f"[red]Could not connect to Qdrant: {e}[/red]")
        console.print("Make sure Qdrant is running: docker-compose up -d")
        raise typer.Exit(1)


@app.command()
def scan(
    folder: Path = typer.Argument(..., help="Folder to scan"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Scan folder, extract text, and create embeddings."""
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)

    path = config_path or get_default_config_path()
    cfg = load_config(path)
    store = get_store(config_path)

    trash_dir = folder / cfg.settings.trash_folder

    console.print(f"[bold]Scanning {folder}...[/bold]")

    files, trash_log = scan_folder(folder, trash_dir=trash_dir)

    if trash_log:
        console.print(f"[yellow]Moved {len(trash_log)} unsupported files to {trash_dir}[/yellow]")

    console.print(f"[green]Found {len(files)} supported files[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Processing {file_path.name}...")

            # Extract text
            text = extract_stage(file_path, vision_model=cfg.settings.vision_model)
            if not text:
                progress.advance(task)
                continue

            # Truncate for summary
            summary = text[:cfg.settings.summary_length]

            # Create embedding
            vector = embed_stage(summary, model=cfg.settings.ollama_model)

            # Store in Qdrant
            doc = Document(
                id=store.generate_id(),
                source_path=str(file_path.absolute()),
                file_type=file_path.suffix.lower().lstrip("."),
                file_size=file_path.stat().st_size,
                summary=summary,
                status=DocumentStatus.EMBEDDED,
                vector=vector,
            )
            store.upsert_document(doc)

            progress.advance(task)

    docs = store.get_all_documents()
    console.print(f"\n[bold green]Done![/bold green] {len(docs)} documents in database")


@app.command()
def cluster(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Cluster documents and generate names."""
    path = config_path or get_default_config_path()
    cfg = load_config(path)
    store = get_store(config_path)

    docs = store.get_documents_by_status(DocumentStatus.EMBEDDED)
    if not docs:
        # Also include already clustered docs for re-clustering
        docs = [d for d in store.get_all_documents() if d.vector]

    if len(docs) < cfg.settings.min_clusters:
        console.print(f"[red]Need at least {cfg.settings.min_clusters} documents to cluster[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Clustering {len(docs)} documents...[/bold]")

    # Clear existing clusters
    store.delete_all_clusters()

    vectors = [d.vector for d in docs]
    doc_ids = [d.id for d in docs]

    # Find optimal K
    max_k = min(cfg.settings.max_clusters, len(docs) // cfg.settings.min_cluster_size)
    optimal_k = find_optimal_k(vectors, min_k=cfg.settings.min_clusters, max_k=max_k)
    console.print(f"[green]Optimal cluster count: {optimal_k}[/green]")

    # Cluster
    assignments, centroids = cluster_documents(vectors, k=optimal_k)

    # Group docs by cluster
    cluster_groups: dict[int, list[tuple[str, list[float]]]] = {}
    for doc_id, vector, assignment in zip(doc_ids, vectors, assignments):
        if assignment not in cluster_groups:
            cluster_groups[assignment] = []
        cluster_groups[assignment].append((doc_id, vector))

    # Check for splits and create clusters
    final_clusters = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating clusters...", total=len(cluster_groups))

        for cluster_idx, group in cluster_groups.items():
            progress.update(task, description=f"Processing cluster {cluster_idx + 1}...")

            group_doc_ids = [g[0] for g in group]
            group_vectors = [g[1] for g in group]

            # Check if split needed
            if should_split(group_vectors, threshold=cfg.settings.variance_threshold):
                console.print(f"[yellow]Splitting cluster {cluster_idx + 1} (high variance)[/yellow]")
                split_assignments, split_centroids = split_cluster(group_vectors, group_doc_ids)

                for sub_idx, centroid in enumerate(split_centroids):
                    sub_doc_ids = [d for d, a in split_assignments if a == sub_idx]
                    sub_vectors = [group_vectors[group_doc_ids.index(d)] for d in sub_doc_ids]

                    final_clusters.append({
                        "doc_ids": sub_doc_ids,
                        "vectors": sub_vectors,
                        "centroid": centroid,
                        "parent": cluster_idx,
                    })
            else:
                from data_ai.utils.similarity import average_vectors
                final_clusters.append({
                    "doc_ids": group_doc_ids,
                    "vectors": group_vectors,
                    "centroid": average_vectors(group_vectors),
                    "parent": None,
                })

            progress.advance(task)

    # Generate names and save clusters
    console.print("[bold]Generating cluster names...[/bold]")

    for cluster_data in final_clusters:
        # Get summaries for naming
        summaries = []
        for doc_id in cluster_data["doc_ids"][:5]:
            doc = next((d for d in docs if d.id == doc_id), None)
            if doc:
                summaries.append(doc.summary)

        name = generate_cluster_name(summaries, model=cfg.settings.chat_model)

        from data_ai.utils.similarity import compute_variance
        variance = compute_variance(cluster_data["vectors"])

        cluster = Cluster(
            id=store.generate_id(),
            name=name,
            doc_count=len(cluster_data["doc_ids"]),
            variance=variance,
            centroid=cluster_data["centroid"],
            status=ClusterStatus.PROPOSED,
            parent_cluster=str(cluster_data["parent"]) if cluster_data["parent"] is not None else None,
        )
        store.upsert_cluster(cluster)

        # Update document assignments
        for doc_id in cluster_data["doc_ids"]:
            store.update_document_cluster(doc_id, cluster.id)

        console.print(f"  [green]✓[/green] {name} ({cluster.doc_count} docs)")

    console.print(f"\n[bold green]Created {len(final_clusters)} clusters[/bold green]")
    console.print("Run [green]data-ai review[/green] to visualize")


@app.command()
def review(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    no_open: bool = typer.Option(False, "--no-open"),
) -> None:
    """Generate and open review HTML."""
    path = config_path or get_default_config_path()
    cfg = load_config(path)
    store = get_store(config_path)

    clusters = store.get_all_clusters()
    if not clusters:
        console.print("[red]No clusters found. Run [green]data-ai cluster[/green] first.[/red]")
        raise typer.Exit(1)

    # Build cluster -> docs mapping
    cluster_docs = {}
    for cluster in clusters:
        docs = store.get_documents_by_cluster(cluster.id)
        cluster_docs[cluster.id] = [Path(d.source_path).name for d in docs]

    output_path = output or Path(cfg.settings.review_html)

    console.print(f"[bold]Generating review HTML...[/bold]")
    generate_review_html(clusters, cluster_docs, output_path)

    console.print(f"[green]Saved to: {output_path}[/green]")

    if not no_open:
        webbrowser.open(f"file://{output_path.absolute()}")


@app.command()
def apply(
    target: Path = typer.Argument(..., help="Target directory for organized files"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    all_clusters: bool = typer.Option(False, "--all", help="Apply all clusters, not just approved"),
) -> None:
    """Copy files to target directory based on clusters."""
    path = config_path or get_default_config_path()
    cfg = load_config(path)
    store = get_store(config_path)

    clusters = store.get_all_clusters()

    if not all_clusters:
        clusters = [c for c in clusters if c.status == ClusterStatus.APPROVED]

    if not clusters:
        console.print("[yellow]No approved clusters. Use --all to apply all, or approve clusters first.[/yellow]")
        raise typer.Exit(1)

    log_file = target / cfg.settings.log_file

    console.print(f"[bold]Copying files to {target}...[/bold]")

    total_copied = 0

    for cluster in clusters:
        folder_name = sanitize_folder_name(cluster.name)
        target_dir = target / folder_name

        docs = store.get_documents_by_cluster(cluster.id)

        console.print(f"\n[bold]{cluster.name}[/bold] → {folder_name}/")

        for doc in docs:
            source = Path(doc.source_path)
            if not source.exists():
                console.print(f"  [yellow]Skip (not found): {source.name}[/yellow]")
                continue

            result = execute_copy(source, target_dir, log_file=log_file)
            if result:
                console.print(f"  [green]✓[/green] {source.name}")
                total_copied += 1
            else:
                console.print(f"  [red]✗[/red] {source.name}")

        store.update_cluster_status(cluster.id, ClusterStatus.APPLIED)

    console.print(f"\n[bold green]Done![/bold green] {total_copied} files copied")
    console.print(f"Log: {log_file}")


@app.command()
def status(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show current pipeline status."""
    store = get_store(config_path)

    docs = store.get_all_documents()
    clusters = store.get_all_clusters()

    # Count by status
    doc_status_counts = {}
    for doc in docs:
        status = doc.status.value if hasattr(doc.status, 'value') else doc.status
        doc_status_counts[status] = doc_status_counts.get(status, 0) + 1

    cluster_status_counts = {}
    for cluster in clusters:
        status = cluster.status.value if hasattr(cluster.status, 'value') else cluster.status
        cluster_status_counts[status] = cluster_status_counts.get(status, 0) + 1

    console.print("\n[bold]Documents:[/bold]")
    table = Table()
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for status, count in sorted(doc_status_counts.items()):
        table.add_row(status, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(docs)}[/bold]")

    console.print(table)

    console.print("\n[bold]Clusters:[/bold]")
    table = Table()
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for status, count in sorted(cluster_status_counts.items()):
        table.add_row(status, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(clusters)}[/bold]")

    console.print(table)


@app.command()
def reset(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm reset"),
) -> None:
    """Reset all data in Qdrant."""
    if not confirm:
        console.print("[red]This will delete all documents and clusters![/red]")
        console.print("Run with [green]--confirm[/green] to proceed")
        raise typer.Exit(1)

    store = get_store(config_path)
    store.reset()

    console.print("[green]Database reset complete[/green]")


@app.command()
def run(
    folder: Path = typer.Argument(..., help="Folder to scan"),
    target: Path = typer.Option(..., "--target", "-t", help="Target directory"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Skip review step"),
) -> None:
    """Run complete pipeline: scan → cluster → [review] → apply."""
    # Import and run each step
    from typer import Context

    ctx = Context(scan)
    scan(folder=folder, config_path=config_path)

    cluster(config_path=config_path)

    if not auto_approve:
        review(config_path=config_path)
        console.print("\n[yellow]Review the clusters, then run:[/yellow]")
        console.print(f"  data-ai apply {target} --all")
    else:
        apply(target=target, config_path=config_path, all_clusters=True)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_v2.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/cli_v2.py tests/test_cli_v2.py
git commit -m "feat(cli): add new CLI commands (scan, cluster, review, apply)"
```

---

## Task 13: Update pyproject.toml Entry Point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update entry point to use new CLI**

Edit `pyproject.toml`:

```toml
[project.scripts]
data-ai = "data_ai.cli_v2:app"
data-ai-legacy = "data_ai.cli:app"
```

- [ ] **Step 2: Reinstall package**

Run: `uv sync`
Expected: Success

- [ ] **Step 3: Verify CLI works**

Run: `data-ai --help`
Expected: Shows new commands (scan, cluster, review, apply, status, reset, run)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(cli): switch to new CLI, keep legacy as data-ai-legacy"
```

---

## Task 14: Integration Test

**Files:**
- Create: `tests/test_integration_v2.py`

- [ ] **Step 1: Create integration test**

```python
# tests/test_integration_v2.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant to avoid needing running server."""
    with patch("data_ai.storage.qdrant.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.collection_exists.return_value = True

        # Track upserted points
        mock_instance._docs = []
        mock_instance._clusters = []

        def mock_upsert(collection_name, points):
            if "documents" in collection_name:
                mock_instance._docs.extend(points)
            else:
                mock_instance._clusters.extend(points)

        def mock_scroll(collection_name, **kwargs):
            if "documents" in collection_name:
                return (mock_instance._docs, None)
            return (mock_instance._clusters, None)

        mock_instance.upsert.side_effect = mock_upsert
        mock_instance.scroll.side_effect = mock_scroll

        yield mock_instance


@pytest.fixture
def mock_ollama():
    """Mock Ollama for embeddings and chat."""
    with patch("data_ai.pipeline.embed.get_embedding") as mock_embed:
        with patch("data_ai.pipeline.naming.ollama") as mock_chat:
            # Return random embeddings
            mock_embed.side_effect = lambda text, model: np.random.randn(768).tolist()

            # Return generic cluster name
            mock_chat.chat.return_value = {"message": {"content": "Dokumente"}}

            yield mock_embed, mock_chat


def test_full_pipeline_with_mocks(tmp_path: Path, mock_qdrant, mock_ollama):
    """Test complete pipeline with mocked external services."""
    from data_ai.pipeline.extract import scan_folder

    # Create test files
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    for i in range(5):
        (inbox / f"doc{i}.txt").write_text(f"Test document content {i}")

    (inbox / "unsupported.xyz").write_text("unsupported")

    # Test scan
    files, trash_log = scan_folder(inbox, trash_dir=inbox / ".trash")

    assert len(files) == 5
    assert len(trash_log) == 1
    assert (inbox / ".trash" / "unsupported.xyz").exists()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration_v2.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_v2.py
git commit -m "test: add integration test for new pipeline"
```

---

## Task 15: Final Verification

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All tests pass

- [ ] **Step 2: Verify Qdrant connection (if running)**

Run: `docker-compose up -d && sleep 3 && data-ai status`
Expected: Shows "Documents: 0, Clusters: 0" or similar

- [ ] **Step 3: Create final commit**

```bash
git add -A
git status
git commit -m "feat: complete vector clustering pipeline implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add dependencies | pyproject.toml |
| 2 | Docker Compose for Qdrant | docker-compose.yaml |
| 3 | Update config | config.py |
| 4 | Storage models | storage/models.py |
| 5 | Qdrant wrapper | storage/qdrant.py |
| 6 | Utility functions | utils/similarity.py |
| 7 | Recursive scan + trash | pipeline/extract.py |
| 8 | Clustering stage | pipeline/cluster.py |
| 9 | Naming stage | pipeline/naming.py |
| 10 | Review HTML | pipeline/review.py |
| 11 | Copy execution | pipeline/execute.py |
| 12 | New CLI commands | cli_v2.py |
| 13 | Entry point update | pyproject.toml |
| 14 | Integration test | test_integration_v2.py |
| 15 | Final verification | - |

Total: ~15 tasks, ~75 steps
