# src/data_ai/providers/ollama.py
from typing import Optional

import ollama


def get_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def describe_image(image_path: str, model: str = "llava") -> Optional[str]:
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in detail. Focus on any text, documents, or content visible.",
                    "images": [image_path],
                }
            ],
        )
        return response["message"]["content"]
    except Exception:
        return None
