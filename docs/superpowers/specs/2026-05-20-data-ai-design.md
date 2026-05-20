# data-ai Design Spec

**Date:** 2026-05-20
**Status:** Approved

## Overview

data-ai is an intelligent CLI tool that automatically organizes files into predefined folders based on semantic similarity. Users define categories with keywords in a YAML config, and the tool uses local embeddings (Ollama) to match files to the most appropriate category.

## Core Requirements

| Aspect | Decision |
|--------|----------|
| Configuration | YAML file |
| Embeddings | Local with Ollama (nomic-embed-text) |
| Unclear files | Interactive prompt |
| File extraction | All types: Text, OCR, Vision-LLM |
| CLI modes | Single, Preview+Apply, Watch |
| Language | Python |
| Similarity | Hybrid (Keywords + example documents) |
| Installation | uv (uvx/uv tool install) |
| Action | Move only |

## Config Format

```yaml
# ~/.config/data-ai/config.yaml

settings:
  ollama_model: "nomic-embed-text"
  vision_model: "llava"
  similarity_threshold: 0.6
  inbox: "./inbox"

categories:
  Rechnungen/Ausgang:
    keywords:
      - "Rechnung"
      - "Invoice"
      - "Rechnungsnummer"
      - "Netto"
      - "Brutto"
    examples: []

  Rechnungen/Eingang:
    keywords:
      - "Lieferant"
      - "Bestellung"
      - "Zahlung fällig"
      - "Mahnung"
    examples:
      - "./examples/beispiel-eingangsrechnung.pdf"

  Verträge/Arbeit:
    keywords:
      - "Arbeitsvertrag"
      - "Gehalt"
      - "Kündigungsfrist"
    examples: []
```

**Behavior:**
- Config is searched at `~/.config/data-ai/config.yaml` or via `--config` flag
- `examples` are optional — when present, their embeddings are combined with keyword embeddings
- Relative paths in `examples` are relative to the config file

## CLI Interface

```bash
# Initialization
data-ai init                      # Creates config template at ~/.config/data-ai/

# Single Command — scan and sort in one pass
data-ai sort ./inbox              # Sorts all files from ./inbox
data-ai sort                      # Uses inbox from config

# Two-phase — Preview then Apply
data-ai scan ./inbox              # Shows plan: "dokument.pdf → Rechnungen/Ausgang (87%)"
data-ai apply                     # Executes last scan

# Watch mode — continuous
data-ai watch ./inbox             # Monitors folder, sorts new files automatically
data-ai watch --daemon            # As background process

# Helper commands
data-ai config                    # Shows current config
data-ai categories                # Lists all categories with keywords
data-ai test ./file.pdf           # Tests a file without moving
```

**Interactive prompt (low confidence):**

```
? dokument.pdf — uncertain (best: Rechnungen/Eingang 34%)

  [1] Rechnungen/Eingang (34%)
  [2] Rechnungen/Ausgang (28%)
  [3] Verträge/Arbeit (12%)
  [s] Skip
  [q] Abort

  Selection: _
```

## Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   EXTRACT   │───▶│    EMBED    │───▶│    MATCH    │───▶│   EXECUTE   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
 PDF, DOCX, IMG     Ollama API        Cosine Sim         Move/Prompt
 → plain text       → vector          → category         → filesystem
```

### Step 1: Extract (`pipeline/extract.py`)
- **Input:** File path
- **Output:** String (extracted text)
- **Logic:**
  - `.pdf` → pdfplumber, if no text → OCR
  - `.docx/.doc` → python-docx
  - `.txt/.md` → read directly
  - `.png/.jpg/.jpeg` → Tesseract OCR
  - No text found → Vision-LLM (llava) describes image

### Step 2: Embed (`pipeline/embed.py`)
- **Input:** String
- **Output:** Vector (list[float])
- **Logic:** Ollama API with `nomic-embed-text`

### Step 3: Match (`pipeline/match.py`)
- **Input:** Vector + Categories (with their embeddings)
- **Output:** `(category, confidence)` or `None` if below threshold
- **Logic:**
  - Load/cache category embeddings (keywords + examples averaged)
  - Cosine similarity against all categories
  - Highest score wins

### Step 4: Execute (`pipeline/execute.py`)
- **Input:** File path + match result
- **Output:** Success/error
- **Logic:**
  - Confidence >= threshold → Move
  - Confidence < threshold → Interactive prompt
  - User chooses category or skip

## Project Structure

```
data-ai/
├── pyproject.toml              # uv/pip config, dependencies, entry points
├── README.md
├── LICENSE                     # MIT
│
├── src/
│   └── data_ai/
│       ├── __init__.py
│       ├── cli.py              # Typer CLI (init, sort, scan, apply, watch)
│       ├── config.py           # YAML loading, validation, Pydantic models
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── extract.py      # Text extraction (PDF, DOCX, OCR, Vision)
│       │   ├── embed.py        # Ollama embedding
│       │   ├── match.py        # Similarity calculation
│       │   └── execute.py      # Move + interactive prompts
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── ollama.py       # Ollama API client (embed + vision)
│       │   ├── tesseract.py    # OCR wrapper
│       │   └── extractors.py   # PDF, DOCX, TXT extractors
│       │
│       └── utils/
│           ├── __init__.py
│           └── similarity.py   # Cosine similarity, vector averaging
│
└── tests/
    ├── test_config.py
    ├── test_extract.py
    ├── test_embed.py
    ├── test_match.py
    └── fixtures/               # Test PDFs, images etc.
```

## Dependencies

```toml
[project]
dependencies = [
    "typer>=0.9.0",           # CLI framework
    "pydantic>=2.0",          # Config validation
    "pyyaml>=6.0",            # YAML parsing
    "ollama>=0.1.0",          # Ollama Python client
    "pdfplumber>=0.9.0",      # PDF text extraction
    "python-docx>=0.8.0",     # DOCX extraction
    "pytesseract>=0.3.0",     # OCR
    "Pillow>=10.0.0",         # Image processing
    "numpy>=1.24.0",          # Vector operations
    "watchdog>=3.0.0",        # Watch mode (filesystem events)
    "rich>=13.0.0",           # Beautiful CLI output
]
```

## System Requirements

- Python 3.11+
- Ollama installed and running locally
- Tesseract installed for OCR (`brew install tesseract` / `apt install tesseract-ocr`)

## Future Considerations (Out of Scope)

- Copy/symlink modes (currently move only)
- Cloud embedding providers (OpenAI, etc.)
- Web UI
- Undo functionality
