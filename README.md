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
