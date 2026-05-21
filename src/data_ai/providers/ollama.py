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


def extract_keywords(
    text: str,
    category: str,
    existing_keywords: list[str],
    model: str = "llama3.2",
) -> list[str]:
    """Use LLM to extract category-specific keywords from document text."""
    try:
        prompt = f"""Analysiere diesen Dokumenttext und extrahiere 3-5 spezifische Keywords die typisch für die Kategorie "{category}" sind.

Regeln:
- NUR kategorie-spezifische Begriffe (keine generischen Wörter wie "der", "und", "ist", "werden")
- NUR Substantive oder Fachbegriffe
- KEINE Eigennamen, Daten oder Zahlen
- Keywords die schon existieren NICHT wiederholen: {existing_keywords}

Dokumenttext (Auszug):
{text[:1500]}

Antworte NUR mit den Keywords, eins pro Zeile, ohne Nummerierung oder Erklärung:"""

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response["message"]["content"]

        # Parse keywords from response
        keywords = []
        for line in content.strip().split("\n"):
            kw = line.strip().strip("-").strip("•").strip()
            if kw and len(kw) > 2 and kw.lower() not in [k.lower() for k in existing_keywords]:
                keywords.append(kw)

        return keywords[:5]  # Max 5 new keywords
    except Exception:
        return []
