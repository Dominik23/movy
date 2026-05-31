# UMAP + HDBSCAN Clustering Design

## Problem

Das aktuelle KMeans-Clustering mit Silhouette-Score zur k-Bestimmung versagt bei homogenen Dokumentensammlungen. Bei 3006 Elektro-Dokumenten wurden nur 2 Cluster gefunden, weil der Silhouette-Score bei thematisch ähnlichen Dokumenten keine guten Trennlinien findet.

## Lösung

Umstellung auf UMAP + HDBSCAN - den Industry-Standard für Document Clustering (verwendet von BERTopic, Top2Vec, etc.).

**Vorteile:**
- HDBSCAN bestimmt Cluster-Anzahl automatisch basierend auf Dichte
- Erkennt Outliers (Dokumente die nirgends passen)
- Funktioniert besser bei homogenen Daten
- UMAP-Dimensionsreduktion verbessert Clustering-Qualität

## Architektur

### Neuer Clustering-Flow

```
Embeddings (768-dim)
       ↓
    UMAP (→ 10-dim)
       ↓
   HDBSCAN
       ↓
  ┌────┴────┐
  ↓         ↓
Cluster   Outliers
  ↓         ↓
Ordner  "_nicht_zuordenbar/"
```

### Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `pyproject.toml` | + `umap-learn>=0.5.0`, `hdbscan>=0.8.33` |
| `src/data_ai/pipeline/cluster.py` | Komplett neu geschrieben |
| `src/data_ai/cli_v2.py` | `cluster` Command anpassen |
| `src/data_ai/config.py` | Neue Settings |
| `src/data_ai/storage.py` | + `ClusterStatus.OUTLIER` |
| `tests/test_pipeline.py` | Tests anpassen |

## Detailliertes Design

### 1. Dependencies (`pyproject.toml`)

```toml
dependencies = [
    ...
    "umap-learn>=0.5.0",
    "hdbscan>=0.8.33",
]
```

### 2. Cluster Module (`cluster.py`)

**Entfernte Funktionen:**
- `find_optimal_k()` - nicht mehr nötig
- `should_split()` - nicht mehr nötig
- `split_cluster()` - nicht mehr nötig

**Neue Funktion:**

```python
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
        vectors: Liste von Embedding-Vektoren
        min_cluster_size: Minimale Cluster-Größe für HDBSCAN
        umap_n_components: Ziel-Dimensionen für UMAP

    Returns:
        labels: Cluster-Zuordnung pro Dokument (-1 = Outlier)
        centroids: Mittelpunkt jedes Clusters (im Original-Raum)
        outlier_indices: Indizes der Outlier-Dokumente
    """
    X = np.array(vectors)

    # Dimensionsreduktion mit UMAP
    reducer = umap.UMAP(
        n_components=umap_n_components,
        metric='cosine',
        random_state=42,
    )
    reduced = reducer.fit_transform(X)

    # Clustering mit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(reduced)

    # Outliers extrahieren
    outlier_indices = [i for i, label in enumerate(labels) if label == -1]

    # Centroids berechnen (im Original-Raum)
    unique_labels = set(labels) - {-1}
    centroids = []
    for label in sorted(unique_labels):
        mask = labels == label
        centroid = X[mask].mean(axis=0).tolist()
        centroids.append(centroid)

    return labels.tolist(), centroids, outlier_indices
```

### 3. CLI Änderungen (`cli_v2.py`)

**Entfernte Parameter:**
- `--min-k`
- `--max-k`

**Neuer Parameter:**
- `--min-cluster-size` (Default: 15)

**Cluster Command Änderungen:**

```python
@app.command()
def cluster(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    min_cluster_size: Optional[int] = typer.Option(None, "--min-cluster-size"),
    recluster: bool = typer.Option(False, "--recluster"),
) -> None:
    ...

    # Clustering
    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=min_cluster_size or cfg.settings.min_cluster_size,
        umap_n_components=cfg.settings.umap_components,
    )

    # Outliers als eigenen Cluster behandeln
    outlier_doc_ids = [doc_ids[i] for i in outlier_indices]

    # Outlier-Cluster erstellen
    if outlier_doc_ids:
        outlier_cluster = Cluster(
            id=store.generate_id(),
            name="Nicht zuordenbar",
            doc_count=len(outlier_doc_ids),
            variance=0.0,
            centroid=[],
            status=ClusterStatus.OUTLIER,
        )
        store.upsert_cluster(outlier_cluster)
        for doc_id in outlier_doc_ids:
            store.update_document_cluster(doc_id, outlier_cluster.id)
```

### 4. Config Änderungen (`config.py`)

**Entfernte Settings:**
- `min_clusters: int = 2`
- `max_clusters: int = 20`

**Neue Settings:**
```python
class Settings(BaseModel):
    ...
    min_cluster_size: int = 15
    umap_components: int = 10
```

### 5. Storage Änderungen (`storage.py`)

```python
class ClusterStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    OUTLIER = "outlier"  # NEU
```

### 6. Apply Command Änderungen

Outlier-Cluster werden zu `_nicht_zuordenbar/` Ordner:

```python
# In apply command
folder_name = sanitize_folder_name(cluster.name)
if cluster.status == ClusterStatus.OUTLIER:
    folder_name = "_nicht_zuordenbar"
```

## Tests

### Angepasste Tests

```python
def test_cluster_documents_returns_labels_and_outliers():
    vectors = np.random.randn(50, 768).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=5,
    )

    assert len(labels) == 50
    assert all(isinstance(l, int) for l in labels)
    assert isinstance(outlier_indices, list)

def test_cluster_documents_finds_multiple_clusters():
    # Zwei deutlich unterschiedliche Gruppen
    group1 = np.random.randn(30, 768) + np.array([5] * 768)
    group2 = np.random.randn(30, 768) + np.array([-5] * 768)
    vectors = np.vstack([group1, group2]).tolist()

    labels, centroids, _ = cluster_documents(vectors, min_cluster_size=10)

    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 2
```

### Entfernte Tests

- Tests für `find_optimal_k`
- Tests für `should_split`
- Tests für `split_cluster`

## Migration

Keine Datenbank-Migration nötig. Bestehende Cluster können mit `--recluster` neu berechnet werden.

## Risiken

1. **UMAP/HDBSCAN sind langsamer als KMeans** - Für 3000 Docs akzeptabel (<1 Min)
2. **Neue Dependencies** - umap-learn und hdbscan sind etablierte Pakete
3. **Outlier-Anteil unklar** - Bei sehr homogenen Daten könnten viele Outliers entstehen → `min_cluster_size` anpassen
