# UMAP + HDBSCAN Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace KMeans + Silhouette scoring with UMAP + HDBSCAN for better clustering of homogeneous document collections.

**Architecture:** UMAP reduces 768-dim embeddings to 10-dim, HDBSCAN clusters based on density (auto-determines cluster count), outliers (label=-1) go to "_nicht_zuordenbar" folder.

**Tech Stack:** umap-learn, hdbscan, numpy, existing Qdrant storage

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add umap-learn, hdbscan dependencies |
| `src/data_ai/storage/models.py` | Modify | Add ClusterStatus.OUTLIER |
| `src/data_ai/config.py` | Modify | Replace min/max_clusters with umap_components |
| `src/data_ai/pipeline/cluster.py` | Rewrite | New UMAP+HDBSCAN cluster_documents function |
| `src/data_ai/cli_v2.py` | Modify | Update cluster command, apply command for outliers |
| `tests/test_cluster.py` | Rewrite | New tests for UMAP+HDBSCAN |

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:8-24`

- [ ] **Step 1: Add umap-learn and hdbscan to dependencies**

Edit `pyproject.toml` to add the new dependencies after scikit-learn:

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
    "umap-learn>=0.5.0",
    "hdbscan>=0.8.33",
]
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -e .`
Expected: Dependencies install successfully

- [ ] **Step 3: Verify imports work**

Run: `python -c "import umap; import hdbscan; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
deps: add umap-learn and hdbscan for density-based clustering
EOF
)"
```

---

### Task 2: Add ClusterStatus.OUTLIER

**Files:**
- Modify: `src/data_ai/storage/models.py:16-19`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_storage.py`:

```python
def test_cluster_status_has_outlier():
    from data_ai.storage.models import ClusterStatus

    assert hasattr(ClusterStatus, "OUTLIER")
    assert ClusterStatus.OUTLIER.value == "outlier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py::test_cluster_status_has_outlier -v`
Expected: FAIL with `AttributeError: OUTLIER`

- [ ] **Step 3: Add OUTLIER to ClusterStatus enum**

Edit `src/data_ai/storage/models.py` to change the ClusterStatus enum:

```python
class ClusterStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    OUTLIER = "outlier"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py::test_cluster_status_has_outlier -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/storage/models.py tests/test_storage.py
git commit -m "$(cat <<'EOF'
feat(storage): add OUTLIER status for unclassifiable documents
EOF
)"
```

---

### Task 3: Update Config Settings

**Files:**
- Modify: `src/data_ai/config.py:22-26`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_settings_has_umap_components():
    from data_ai.config import Settings

    settings = Settings()
    assert settings.umap_components == 10
    # Old settings should be removed
    assert not hasattr(settings, "min_clusters") or settings.min_clusters is None
    assert not hasattr(settings, "max_clusters") or settings.max_clusters is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_settings_has_umap_components -v`
Expected: FAIL with `AttributeError: umap_components`

- [ ] **Step 3: Update Settings class**

Edit `src/data_ai/config.py` to replace the clustering settings:

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

    # New: Clustering (UMAP + HDBSCAN)
    min_cluster_size: int = Field(default=15, ge=2)
    umap_components: int = Field(default=10, ge=2)
    variance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # New: Processing
    batch_size: int = Field(default=100, ge=1)
    summary_length: int = Field(default=2000, ge=100)

    # New: Output
    trash_folder: str = ".trash"
    log_file: str = "data-ai.log"
    review_html: str = "/tmp/data-ai-review.html"
```

Note: Remove `min_clusters` and `max_clusters` entirely.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_settings_has_umap_components -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
feat(config): replace min/max_clusters with umap_components setting
EOF
)"
```

---

### Task 4: Rewrite Cluster Module

**Files:**
- Rewrite: `src/data_ai/pipeline/cluster.py`
- Rewrite: `tests/test_cluster.py`

- [ ] **Step 1: Write the failing test for basic clustering**

Replace `tests/test_cluster.py` entirely:

```python
import pytest
import numpy as np


def test_cluster_documents_returns_labels_and_outliers():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    vectors = np.random.randn(50, 768).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=5,
    )

    assert len(labels) == 50
    assert all(isinstance(label, int) for label in labels)
    assert isinstance(outlier_indices, list)
    assert all(isinstance(idx, int) for idx in outlier_indices)
    # Outliers should have label -1
    for idx in outlier_indices:
        assert labels[idx] == -1


def test_cluster_documents_finds_distinct_groups():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    # Two clearly separated groups
    group1 = np.random.randn(30, 768) + np.array([10.0] * 768)
    group2 = np.random.randn(30, 768) + np.array([-10.0] * 768)
    vectors = np.vstack([group1, group2]).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=10,
    )

    # Should find at least 2 clusters
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 2
    assert len(centroids) >= 2


def test_cluster_documents_centroids_in_original_space():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    vectors = np.random.randn(50, 768).tolist()

    labels, centroids, _ = cluster_documents(vectors, min_cluster_size=5)

    # Centroids should be 768-dimensional (original space)
    for centroid in centroids:
        assert len(centroid) == 768


def test_cluster_documents_handles_small_dataset():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    # Very small dataset - should still work
    vectors = np.random.randn(10, 768).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=3,
    )

    assert len(labels) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cluster.py -v`
Expected: FAIL (old cluster_documents has different signature)

- [ ] **Step 3: Rewrite cluster.py with UMAP + HDBSCAN**

Replace `src/data_ai/pipeline/cluster.py` entirely:

```python
# src/data_ai/pipeline/cluster.py
import numpy as np
import umap
import hdbscan


def cluster_documents(
    vectors: list[list[float]],
    min_cluster_size: int = 15,
    umap_n_components: int = 10,
) -> tuple[list[int], list[list[float]], list[int]]:
    """
    Cluster vectors using UMAP + HDBSCAN.

    Args:
        vectors: List of embedding vectors (768-dimensional)
        min_cluster_size: Minimum cluster size for HDBSCAN
        umap_n_components: Target dimensions for UMAP reduction

    Returns:
        labels: Cluster assignment per document (-1 = outlier)
        centroids: Centroid of each cluster (in original 768-dim space)
        outlier_indices: Indices of outlier documents
    """
    X = np.array(vectors)
    n_samples = len(vectors)

    # Handle edge cases
    if n_samples < min_cluster_size:
        # Too few samples - everything is an outlier
        return [-1] * n_samples, [], list(range(n_samples))

    # Adjust UMAP components if necessary
    effective_components = min(umap_n_components, n_samples - 1, X.shape[1])

    # Dimensionality reduction with UMAP
    reducer = umap.UMAP(
        n_components=effective_components,
        metric="cosine",
        random_state=42,
        n_neighbors=min(15, n_samples - 1),
    )
    reduced = reducer.fit_transform(X)

    # Clustering with HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # Extract outlier indices
    outlier_indices = [i for i, label in enumerate(labels) if label == -1]

    # Compute centroids in original space
    unique_labels = sorted(set(labels) - {-1})
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = X[mask].mean(axis=0).tolist()
        centroids.append(centroid)

    return labels.tolist(), centroids, outlier_indices
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cluster.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/pipeline/cluster.py tests/test_cluster.py
git commit -m "$(cat <<'EOF'
feat(cluster): replace KMeans with UMAP+HDBSCAN for density-based clustering

- HDBSCAN automatically determines cluster count based on density
- UMAP reduces dimensionality before clustering for better results
- Outliers (label=-1) are now explicitly identified
- Centroids computed in original embedding space
EOF
)"
```

---

### Task 5: Update CLI Cluster Command

**Files:**
- Modify: `src/data_ai/cli_v2.py:208-323`

- [ ] **Step 1: Update imports in cli_v2.py**

At the top of the file, change the cluster import from:
```python
from data_ai.pipeline.cluster import find_optimal_k, cluster_documents
```
to:
```python
from data_ai.pipeline.cluster import cluster_documents
```

- [ ] **Step 2: Rewrite the cluster command**

Replace the entire `cluster` function (lines 208-323) with:

```python
@app.command()
def cluster(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    min_cluster_size: Optional[int] = typer.Option(None, "--min-cluster-size", help="Minimum cluster size"),
    recluster: bool = typer.Option(False, "--recluster", help="Delete existing clusters and recluster"),
) -> None:
    """Run clustering on embedded documents and generate names."""
    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
    else:
        from data_ai.config import Settings
        cfg = type('Config', (), {'settings': Settings()})()

    store = get_store(config_path)

    # Delete existing clusters if reclustering
    if recluster:
        console.print("[yellow]Deleting existing clusters...[/yellow]")
        store.delete_all_clusters()

    # Get embedded documents
    docs = store.get_documents_by_status(DocumentStatus.EMBEDDED)

    if not docs:
        console.print("[yellow]No embedded documents found. Run 'scan' first.[/yellow]")
        return

    console.print(f"[blue]Clustering {len(docs)} documents...[/blue]")

    # Extract vectors
    vectors = [doc.vector for doc in docs if doc.vector]
    doc_ids = [doc.id for doc in docs if doc.vector]

    if len(vectors) < 2:
        console.print("[yellow]Need at least 2 documents for clustering[/yellow]")
        return

    # Get clustering parameters
    effective_min_cluster_size = min_cluster_size or cfg.settings.min_cluster_size
    umap_components = cfg.settings.umap_components

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running UMAP + HDBSCAN clustering...", total=None)

        # Run clustering
        labels, centroids, outlier_indices = cluster_documents(
            vectors,
            min_cluster_size=effective_min_cluster_size,
            umap_n_components=umap_components,
        )

        progress.update(task, description="Creating cluster records...")

        # Group documents by cluster
        cluster_docs: dict[int, list[tuple[str, Document]]] = {}
        outlier_docs: list[tuple[str, Document]] = []

        for idx, (doc_id, label) in enumerate(zip(doc_ids, labels)):
            doc = next(d for d in docs if d.id == doc_id)
            if label == -1:
                outlier_docs.append((doc_id, doc))
            else:
                if label not in cluster_docs:
                    cluster_docs[label] = []
                cluster_docs[label].append((doc_id, doc))

        progress.update(task, description="Generating cluster names...")

        # Create regular clusters
        for cluster_idx, doc_list in cluster_docs.items():
            cluster_id = store.generate_id()

            # Assign documents to cluster
            for doc_id, _ in doc_list:
                store.update_document_cluster(doc_id, cluster_id)

            # Generate name from summaries
            summaries = [doc.summary for _, doc in doc_list]
            name = generate_cluster_name(summaries, model=cfg.settings.chat_model)

            # Get centroid
            centroid = centroids[cluster_idx] if cluster_idx < len(centroids) else []

            # Calculate variance
            vectors_in_cluster = [doc.vector for _, doc in doc_list if doc.vector]
            variance = compute_variance(vectors_in_cluster) if vectors_in_cluster else 0.0

            cluster_record = Cluster(
                id=cluster_id,
                name=name,
                doc_count=len(doc_list),
                variance=variance,
                centroid=centroid,
                status=ClusterStatus.PROPOSED,
            )

            store.upsert_cluster(cluster_record)

        # Create outlier cluster if there are outliers
        if outlier_docs:
            outlier_cluster_id = store.generate_id()

            for doc_id, _ in outlier_docs:
                store.update_document_cluster(doc_id, outlier_cluster_id)

            outlier_cluster = Cluster(
                id=outlier_cluster_id,
                name="Nicht zuordenbar",
                doc_count=len(outlier_docs),
                variance=0.0,
                centroid=[],
                status=ClusterStatus.OUTLIER,
            )

            store.upsert_cluster(outlier_cluster)

    # Show results
    clusters = store.get_all_clusters()
    regular_clusters = [c for c in clusters if c.status != ClusterStatus.OUTLIER]
    outlier_cluster = next((c for c in clusters if c.status == ClusterStatus.OUTLIER), None)

    console.print(f"\n[green]Created {len(regular_clusters)} clusters:[/green]\n")

    table = Table()
    table.add_column("Name")
    table.add_column("Documents", justify="right")
    table.add_column("Variance", justify="right")

    for c in sorted(regular_clusters, key=lambda x: x.doc_count, reverse=True):
        table.add_row(c.name, str(c.doc_count), f"{c.variance:.2f}")

    console.print(table)

    if outlier_cluster:
        console.print(f"\n[yellow]Outliers: {outlier_cluster.doc_count} documents not assignable[/yellow]")
```

- [ ] **Step 3: Remove old imports that are no longer used**

Remove `find_optimal_k` from imports if still present.

- [ ] **Step 4: Run the CLI to verify it works**

Run: `data-ai cluster --help`
Expected: Shows help with `--min-cluster-size` option (not `--min-k` or `--max-k`)

- [ ] **Step 5: Commit**

```bash
git add src/data_ai/cli_v2.py
git commit -m "$(cat <<'EOF'
feat(cli): update cluster command for UMAP+HDBSCAN

- Replace --min-k/--max-k with --min-cluster-size
- Create outlier cluster for unassignable documents
- Show outlier count in results summary
EOF
)"
```

---

### Task 6: Update Apply Command for Outliers

**Files:**
- Modify: `src/data_ai/cli_v2.py:487-565`

- [ ] **Step 1: Update apply command to handle outlier cluster**

In the `apply` function, after getting approved_clusters, add handling for the outlier cluster. Find the section around line 529-534 and update the folder name logic:

```python
        for cluster in approved_clusters:
            progress.update(task, description=f"Processing {cluster.name}...")

            docs = store.get_documents_by_cluster(cluster.id)

            # Special folder for outliers
            if cluster.status == ClusterStatus.OUTLIER:
                folder_name = "_nicht_zuordenbar"
            else:
                folder_name = sanitize_folder_name(cluster.name)

            target_dir = target / folder_name
```

- [ ] **Step 2: Include outlier cluster in apply logic**

Update the cluster selection to also include outlier clusters:

```python
    clusters = store.get_all_clusters()
    approved_clusters = [c for c in clusters if c.status == ClusterStatus.APPROVED]
    outlier_clusters = [c for c in clusters if c.status == ClusterStatus.OUTLIER]

    if not approved_clusters:
        # If no approved clusters, use all proposed ones
        approved_clusters = [c for c in clusters if c.status == ClusterStatus.PROPOSED]
        if not approved_clusters and not outlier_clusters:
            console.print("[yellow]No clusters found. Run 'cluster' first.[/yellow]")
            return
        if not approved_clusters:
            console.print("[yellow]No approved clusters. Using proposed clusters.[/yellow]")

    # Add outlier clusters to the list to process
    all_clusters_to_apply = approved_clusters + outlier_clusters
```

Then use `all_clusters_to_apply` in the for loop instead of `approved_clusters`.

- [ ] **Step 3: Run a quick manual test**

Run: `data-ai apply --help`
Expected: Command help shows correctly

- [ ] **Step 4: Commit**

```bash
git add src/data_ai/cli_v2.py
git commit -m "$(cat <<'EOF'
feat(cli): apply command handles outlier cluster

- Outliers go to _nicht_zuordenbar folder
- Include outlier cluster in apply processing
EOF
)"
```

---

### Task 7: Run Full Test Suite

**Files:**
- None (verification only)

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Fix any failing tests**

If tests fail due to the removed `find_optimal_k`, `should_split`, or `split_cluster` functions, update the tests to remove those references.

Check `tests/test_cli_v2.py` for any tests that use the old cluster function signature.

- [ ] **Step 3: Run tests again after fixes**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit any test fixes**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: update tests for new UMAP+HDBSCAN clustering
EOF
)"
```

---

### Task 8: Integration Test

**Files:**
- None (verification only)

- [ ] **Step 1: Create a test folder with sample files**

Run: `mkdir -p /tmp/cluster-test && echo "This is a test document about machine learning" > /tmp/cluster-test/ml.txt && echo "This is about cooking recipes" > /tmp/cluster-test/cooking.txt`

- [ ] **Step 2: Run the full pipeline**

Run: `data-ai scan /tmp/cluster-test && data-ai cluster --min-cluster-size 2`
Expected: Clustering completes, may show outliers due to small dataset

- [ ] **Step 3: Verify clustering output**

Check that the output shows clusters with document counts and optional outlier information.

- [ ] **Step 4: Clean up**

Run: `rm -rf /tmp/cluster-test`

---

## Summary

After completing all tasks, the clustering system will:

1. Use UMAP to reduce 768-dim embeddings to 10-dim
2. Use HDBSCAN to cluster based on density (automatic k)
3. Identify outliers (documents that don't fit any cluster)
4. Place outliers in `_nicht_zuordenbar` folder on apply
5. Be configurable via `--min-cluster-size` CLI option
