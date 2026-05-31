# src/data_ai/pipeline/embed.py
from data_ai.providers.ollama import get_embedding

# nomic-embed-text has 8192 token context, ~4 chars per token = ~32k chars
MAX_TEXT_LENGTH = 30000
TRUNCATE_FACTOR = 0.5  # Cut in half on each retry


def embed_stage(text: str, model: str = "nomic-embed-text") -> list[float]:
    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    while len(text) > 100:
        try:
            return get_embedding(text, model=model)
        except Exception:
            # Truncate and retry
            text = text[: int(len(text) * TRUNCATE_FACTOR)]

    raise ValueError("Text too short to embed after truncation")
