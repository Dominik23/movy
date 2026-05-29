# Vector-Based Document Clustering Redesign

**Date:** 2026-05-29
**Status:** Approved

## Overview

Kompletter Umbau von data-ai: Weg vom keyword-basierten Matching hin zu einem Vector-DB + Clustering Ansatz mit automatischer Kategorie-Discovery.

### Kernänderungen

1. **Vector DB statt Keywords** - Qdrant speichert Dokument-Embeddings
2. **Auto-Discovery** - KMeans Clustering statt vordefinierter Kategorien
3. **LLM-Naming** - Cluster-Namen werden automatisch generiert
4. **Varianz-basiertes Splitting** - Zu heterogene Cluster werden automatisch gesplittet
5. **Copy statt Move** - Originale bleiben erhalten
6. **Graph-Visualisierung** - Interaktives HTML zur Review

## Architektur

```
┌─────────────────────────────────────────────────────────────────────┐
│                         data-ai CLI                                  │
├─────────────────────────────────────────────────────────────────────┤
│  scan       │  cluster    │  review     │  apply      │  visualize │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬─────┘
       │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Stages                               │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│   Extract   │   Embed     │  Cluster    │   Name      │   Execute   │
│   (text)    │  (vectors)  │  (KMeans)   │  (LLM)      │   (copy)    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                                │
├───────────────────────────────┬─────────────────────────────────────┤
│         Qdrant (Docker)       │        Filesystem (Copy)            │
│  - document vectors           │  - source files (readonly)          │
│  - cluster assignments        │  - target folders (organized)       │
│  - metadata (path, summary)   │  - trash folder (unsupported)       │
└───────────────────────────────┴─────────────────────────────────────┘
```

## Pipeline Stages

### Stage 1: Scan & Extract

**Input:** Folder-Pfad
**Output:** Dokumente in Qdrant mit Text-Summary

**Ablauf:**
1. Alle Files im Folder rekursiv iterieren
2. Pro File:
   - Dateityp prüfen
   - Supported → Text extrahieren
   - Unsupported → Trash-Folder + Log-Eintrag
3. Text truncaten auf ~2000 Zeichen für Summary
4. In Qdrant speichern mit Status "pending"

**Supported Dateitypen:**
| Typ | Extractor | Fallback |
|-----|-----------|----------|
| PDF | pdfplumber | OCR (Tesseract) für Scans |
| TXT, MD | Direct read | - |
| PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP | OCR (Tesseract) | Vision Model (llava) |
| DOCX | python-docx | - |
| PPTX | python-pptx | - |

**Unsupported → Trash:**
- Alle anderen Dateitypen
- Log-Entry mit Grund

### Stage 2: Embed

**Input:** Dokumente mit status="pending" aus Qdrant
**Output:** Vektoren in Qdrant

**Ablauf:**
1. Batch-weise Dokumente laden (100er Batches)
2. Ollama `nomic-embed-text` für jeden Text
3. Vektor in Qdrant upserten
4. Status → "embedded"

**Batch-Processing:**
- Parallelisierung möglich (Ollama unterstützt concurrent requests)
- Progress-Bar für User-Feedback
- Resume bei Abbruch (nur "pending" werden verarbeitet)

### Stage 3: Cluster

**Input:** Alle embedded Dokumente
**Output:** Cluster-Assignments + Centroid-Vektoren

**Ablauf:**
1. Alle Vektoren aus Qdrant laden
2. Optimale K finden:
   - Elbow-Methode (Inertia-Kurve)
   - Silhouette-Score zur Validierung
   - Range: 2 bis min(20, n_docs/5)
3. KMeans ausführen mit optimalem K
4. Pro Cluster: Intra-Cluster-Varianz berechnen
5. Splitting-Check:
   - Wenn Varianz > Threshold (default 0.4)
   - Rekursiv KMeans(k=2) auf diesen Cluster
   - Neuer Name wird in Stage 4 generiert
6. Cluster-Assignment in Dokument-Payload speichern
7. Centroid-Vektoren in "clusters" Collection speichern

**Splitting-Logik:**
```python
def should_split(cluster_vectors, threshold=0.4):
    centroid = np.mean(cluster_vectors, axis=0)
    distances = [cosine_distance(v, centroid) for v in cluster_vectors]
    variance = np.var(distances)
    return variance > threshold
```

### Stage 4: Name (LLM)

**Input:** Cluster mit zugeordneten Dokumenten
**Output:** Cluster-Namen

**Ablauf:**
1. Pro Cluster: 3-5 repräsentative Dokumente wählen
   - Sortiert nach Distanz zum Centroid (nächste zuerst)
2. LLM-Prompt an Ollama (llama3.2):
   ```
   Analysiere diese Dokument-Zusammenfassungen und gib einen
   kurzen, beschreibenden Kategorie-Namen (1-3 Wörter, deutsch).

   Dokument 1: [summary]
   Dokument 2: [summary]
   ...

   Kategorie-Name:
   ```
3. Namen in Cluster-Payload speichern
4. Status → "proposed"

### Stage 5: Review (HTML)

**Input:** Cluster mit Namen und Dokumenten
**Output:** Interaktives HTML zur Bestätigung

**Graph-Generierung (pyvis):**
- Knoten = Cluster
  - Grösse proportional zu doc_count
  - Farbe nach Varianz (grün=homogen, rot=heterogen)
  - Label = Cluster-Name
- Edges = Cosine-Similarity zwischen Centroids
  - Nur wenn Similarity > 0.3
  - Dicke proportional zu Similarity
- Hover-Info: Liste der ersten 10 Dokumente

**HTML-Struktur:**
```html
<div id="graph"><!-- pyvis network --></div>
<div id="sidebar">
  <h2>Cluster: [name]</h2>
  <p>Dokumente: [count]</p>
  <p>Varianz: [variance]</p>
  <ul><!-- document list --></ul>
  <button>Approve</button>
  <button>Rename</button>
</div>
<div id="summary">
  <table><!-- all clusters overview --></table>
</div>
```

**Interaktion:**
- Click auf Cluster → Details im Sidebar
- Approve-Button markiert Cluster als "approved" in Qdrant
- Rename öffnet Prompt → Update in Qdrant
- Page-Refresh lädt aktuellen Stand

### Stage 6: Apply (Copy)

**Input:** Cluster mit status="approved"
**Output:** Kopierte Dateien in Zielstruktur

**Ablauf:**
1. Nur Cluster mit status="approved" verarbeiten
2. Pro Cluster:
   - Zielordner erstellen: `{target}/{cluster_name}/`
   - Ordner-Name sanitizen (keine Sonderzeichen)
3. Pro Dokument im Cluster:
   - `shutil.copy2()` (erhält Metadaten)
   - Bei Duplikat-Filename: `{name}_{timestamp}.{ext}`
4. Log schreiben:
   - JSON-Format
   - source_path, target_path, cluster_name, timestamp
5. Status in Qdrant → "applied"

**Duplikat-Handling:**
```python
def safe_copy(source: Path, target_dir: Path) -> Path:
    target = target_dir / source.name
    if target.exists():
        stem = source.stem
        suffix = source.suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = target_dir / f"{stem}_{timestamp}{suffix}"
    shutil.copy2(source, target)
    return target
```

## Datenmodell

### Qdrant Collections

**Collection: documents**
```python
{
    "id": "uuid-v4",
    "vector": [float] * 768,  # nomic-embed-text dimension
    "payload": {
        "source_path": str,      # Absoluter Pfad zur Originaldatei
        "file_type": str,        # "pdf", "txt", "png", etc.
        "file_size": int,        # Bytes
        "summary": str,          # Erste 2000 Zeichen
        "status": str,           # "pending" | "embedded" | "clustered" | "applied"
        "cluster_id": str | None,
        "created_at": str,       # ISO 8601
        "updated_at": str
    }
}
```

**Collection: clusters**
```python
{
    "id": "uuid-v4",
    "vector": [float] * 768,  # Centroid-Vektor
    "payload": {
        "name": str,             # LLM-generierter Name
        "doc_count": int,
        "variance": float,       # Intra-Cluster-Varianz
        "status": str,           # "proposed" | "approved" | "applied"
        "parent_cluster": str | None,  # Bei Splits
        "created_at": str,
        "updated_at": str
    }
}
```

## CLI Commands

```bash
# Setup
data-ai init                    # Config erstellen
data-ai init --db-url HOST:PORT # Mit custom Qdrant URL

# Pipeline
data-ai scan /path/to/folder    # Stage 1+2: Extract & Embed
data-ai cluster                 # Stage 3+4: Cluster & Name
data-ai review                  # Stage 5: HTML generieren/öffnen
data-ai apply --target /output  # Stage 6: Dateien kopieren

# All-in-one
data-ai run /input --target /output [--auto-approve]

# Utilities
data-ai status                  # Pipeline-Status anzeigen
data-ai visualize [--output FILE]  # Nur Graph generieren
data-ai reset [--confirm]       # Qdrant clearen
data-ai logs                    # Copy-Log anzeigen
```

## Config

**Pfad:** `~/.config/data-ai/config.yaml`

```yaml
settings:
  # Qdrant Connection
  qdrant_url: "localhost:6333"
  qdrant_collection_prefix: "data_ai"

  # Ollama Models
  embed_model: "nomic-embed-text"
  vision_model: "llava"
  chat_model: "llama3.2"

  # Clustering Parameters
  min_clusters: 2
  max_clusters: 20
  variance_threshold: 0.4
  min_cluster_size: 3

  # Processing
  batch_size: 100
  summary_length: 2000

  # Output
  trash_folder: ".trash"
  log_file: "data-ai.log"
  review_html: "/tmp/data-ai-review.html"
```

## Docker Setup

**docker-compose.yaml:**
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_data:
```

**Startup:**
```bash
docker-compose up -d
data-ai init --db-url localhost:6333
```

## Testframework

### Unit Tests

| Modul | Tests |
|-------|-------|
| `extractors` | Pro Dateityp: Extraction funktioniert, Fehler handled |
| `embed` | Ollama-Mock, Batch-Processing, Truncation |
| `cluster` | KMeans korrekt, Elbow-Methode, Split-Logik |
| `naming` | LLM-Mock, Prompt-Format, Fallback |
| `review` | HTML-Generierung, Graph-Struktur |
| `apply` | Copy funktioniert, Duplikate, Permissions |

### Integration Tests

| Test | Beschreibung |
|------|--------------|
| `test_full_pipeline` | 10 Test-Dateien durch alle Stages |
| `test_resume_after_crash` | Abbruch simulieren, Resume prüfen |
| `test_large_batch` | 1000+ Mock-Dokumente |
| `test_split_triggered` | Heterogene Dokumente → Split |

### E2E Tests

| Test | Beschreibung |
|------|--------------|
| `test_cli_scan` | CLI scan Command mit echten Dateien |
| `test_cli_full_run` | Kompletter Durchlauf mit --auto-approve |

## Migration von bestehender Lösung

1. Alte Config bleibt kompatibel (wird ignoriert wenn Qdrant aktiv)
2. Alte CLI-Commands (`sort`, `watch`) deprecated aber funktional
3. Neue Commands parallel verfügbar
4. Nach Übergangszeit: alte Commands entfernen

## Offene Punkte

Keine - alle Anforderungen geklärt.

## Dependencies (neu)

```
qdrant-client>=1.7.0
scikit-learn>=1.3.0
pyvis>=0.3.0
```
