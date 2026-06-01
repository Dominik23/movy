# src/data_ai/pipeline/__init__.py
"""
Pipeline modules for data-ai.

V2 Pipeline (new):
- year_detect: Detect year from filename/path/mtime
- extract_v2: Text extraction using Docling
- cluster_v2: Clustering using BERTopic
- naming: Cluster naming using Ollama
- run: Main pipeline orchestration

Legacy Pipeline (requires Qdrant, pdfplumber, etc.):
- extract: Text extraction using pdfplumber/tesseract
- embed: Embedding using Ollama
- cluster: Clustering using UMAP+HDBSCAN
- match: Category matching
- execute: File operations
- review: HTML review generation
"""
