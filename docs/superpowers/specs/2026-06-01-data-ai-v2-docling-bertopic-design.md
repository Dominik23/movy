# data-ai v2: Docling + BERTopic Design Spec

> **Ziel:** Dokumente nach Jahr sortieren, dann per BERTopic clustern, mit schönen Cluster-Namen via Ollama LLM.

## Architektur

```
Input Folder
     │
     ▼
┌─────────────────┐
│  Year Detector  │  ← Filename-Regex → mtime → 2026
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Docling      │  ← PDF/DOCX/Images → Text
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Per-Year BERTopic Clustering   │
│  ├─ sentence-transformers       │
│  ├─ UMAP (intern)               │
│  └─ HDBSCAN (intern)            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Ollama LLM     │  ← Topic-Keywords → Schöner Name
└────────┬────────┘
         │
         ▼
/output/2024/Rechnungen/
/output/2024/Verträge/
/output/2025/Steuer/
```

## Komponenten

### 1. Year Detector

**Zweck:** Jahr aus Datei extrahieren für Batching.

**Logik (in Reihenfolge):**
1. Regex auf Filename: `(19|20)\d{2}` - erstes Match gewinnt
2. Fallback: `mtime` der Datei → Jahr extrahieren
3. Letzter Fallback: Aktuelles Jahr (2026)

**Interface:**
```python
def detect_year(file_path: Path) -> int:
    """Returns year as integer (e.g., 2024)."""
```

### 2. Docling Extraction

**Zweck:** Text aus allen Dokumenttypen extrahieren.

**Ersetzt:**
- pdfplumber
- python-docx
- python-pptx
- pytesseract
- pdf2image

**Supported Formats:**
- PDF (nativ + OCR für gescannte)
- DOCX
- PPTX
- Images (PNG, JPG, etc.)

**Interface:**
```python
def extract_text(file_path: Path) -> str | None:
    """Extract text using Docling. Returns None if extraction fails."""
```

### 3. BERTopic Clustering (pro Jahr)

**Zweck:** Dokumente eines Jahres semantisch clustern.

**Stack:**
- `sentence-transformers` für Embeddings
  - Model: `paraphrase-multilingual-MiniLM-L12-v2` (schnell, multilingual)
- BERTopic managed intern:
  - UMAP für Dimensionsreduktion
  - HDBSCAN für Clustering

**Interface:**
```python
def cluster_year_batch(
    documents: list[tuple[Path, str]],  # (file_path, text)
    min_topic_size: int = 10,
) -> dict[int, list[Path]]:
    """
    Cluster documents for one year.
    Returns: {topic_id: [file_paths]}
    Topic -1 = Outliers
    """
```

**BERTopic liefert auch:**
- Keywords pro Topic (für Naming)
- Topic-Repräsentationen

### 4. Cluster Naming via Ollama

**Zweck:** Aus BERTopic-Keywords schöne deutsche Ordnernamen machen.

**Input:** BERTopic Keywords (z.B. `["rechnung", "euro", "zahlung", "betrag"]`)

**Output:** Schöner Name (z.B. `"Rechnungen"`)

**Interface:**
```python
def generate_cluster_name(keywords: list[str], sample_filenames: list[str]) -> str:
    """Generate a nice folder name from topic keywords."""
```

**Prompt-Strategie:**
```
Gegeben diese Keywords: {keywords}
Und diese Beispiel-Dateinamen: {filenames}

Generiere einen kurzen, deutschen Ordnernamen (1-2 Wörter).
Beispiele: "Rechnungen", "Steuerunterlagen", "Verträge", "Kontoauszüge"
```

### 5. Output Structure

```
/output/
  2024/
    Rechnungen/
      rechnung_001.pdf
      rechnung_002.pdf
    Verträge/
      vertrag_miete.pdf
    _Sonstiges/          # Outliers (Topic -1)
      random_doc.pdf
  2025/
    Steuer/
    Versicherungen/
  2026/
    _Sonstiges/          # Falls nur Outliers
```

**Regeln:**
- Ein Ordner pro Jahr
- Ein Unterordner pro Cluster
- Outliers (Topic -1) → `_Sonstiges/`
- Dateien werden kopiert (nicht verschoben)

## Dependencies

### Neu hinzufügen

```toml
[project.dependencies]
docling = ">=2.0.0"
bertopic = ">=0.16.0"
sentence-transformers = ">=2.2.0"
```

### Entfernen

```toml
# Diese werden nicht mehr gebraucht:
# pdfplumber
# python-docx
# python-pptx
# pytesseract
# pdf2image
# qdrant-client
# umap-learn
# hdbscan
# scikit-learn
# pyvis
```

## CLI

### Neuer Command

```bash
data-ai run /input --output /output
```

**Optionen:**
- `--min-topic-size INT` - Minimum Dokumente pro Cluster (default: 10)
- `--dry-run` - Zeigt was passieren würde, kopiert nichts
- `--model TEXT` - Ollama Model für Naming (default: llama3.2)

### Ablauf

1. Scan `/input` für supported files
2. Year Detection für jede Datei
3. Gruppiere nach Jahr
4. Pro Jahr:
   a. Docling Extraction
   b. BERTopic Clustering
   c. Ollama Naming
5. Kopiere Dateien nach `/output/YEAR/CLUSTER/`

### Progress Output

```
Scanning /input... 10,247 files found

Year detection:
  2024: 3,421 files
  2025: 4,102 files
  2026: 2,724 files

Processing 2024 (3,421 files):
  Extracting text... ████████████████████ 100%
  Clustering... done (12 topics found)
  Naming clusters... done

  Topics:
    - Rechnungen (847 files)
    - Verträge (523 files)
    - Kontoauszüge (412 files)
    ...
    - _Sonstiges (89 outliers)

Processing 2025...
...

Copying files to /output... ████████████████████ 100%

Done! 10,247 files organized into 35 categories.
```

## File Structure

```
src/data_ai/
  __init__.py
  cli_v2.py              # Behalten, run command hinzufügen

  pipeline/
    __init__.py
    year_detect.py       # NEU: Year detection
    extract_v2.py        # NEU: Docling wrapper
    cluster_v2.py        # NEU: BERTopic wrapper
    naming.py            # NEU: Ollama cluster naming

  # Diese Ordner werden nicht mehr gebraucht:
  # storage/             # Kein Qdrant mehr
  # providers/           # Docling ersetzt alles
```

## Migration

### Was bleibt
- `cli_v2.py` - Basis CLI (wird erweitert)
- `providers/ollama.py` - Für Cluster Naming

### Was wird ersetzt
- `pipeline/extract.py` → `pipeline/extract_v2.py` (Docling)
- `pipeline/cluster.py` → `pipeline/cluster_v2.py` (BERTopic)
- `pipeline/embed.py` → entfällt (BERTopic macht das intern)

### Was wird gelöscht
- `storage/` - Kein Qdrant mehr nötig
- `providers/tesseract.py` - Docling macht OCR
- `providers/extractors.py` - Docling ersetzt alles

## Nicht im Scope

- Watch-Mode (später)
- Inkrementelles Processing (später)
- Web UI (später)
- Qdrant Integration (nicht mehr nötig)
