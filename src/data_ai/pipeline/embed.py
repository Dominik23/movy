# src/data_ai/pipeline/embed.py
from data_ai.providers.ollama import get_embedding

# nomic-embed-text has 8192 token context, ~4 chars per token = ~32k chars
MAX_TEXT_LENGTH = 30000


def embed_stage(text: str, model: str = "nomic-embed-text") -> list[float]:
    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    return get_embedding(text, model=model)
