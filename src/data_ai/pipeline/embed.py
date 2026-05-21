# src/data_ai/pipeline/embed.py
from data_ai.providers.ollama import get_embedding


def embed_stage(text: str, model: str = "nomic-embed-text") -> list[float]:
    return get_embedding(text, model=model)
