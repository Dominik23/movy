# src/data_ai/pipeline/naming.py
from datetime import datetime
from typing import Optional

import ollama


def generate_cluster_name(
    summaries: list[str],
    model: str = "llama3.2",
    max_summaries: int = 5,
) -> str:
    """
    Generate a cluster name using LLM based on document summaries.
    """
    if not summaries:
        return _fallback_name()

    # Take only first N summaries
    selected = summaries[:max_summaries]

    # Truncate each summary
    truncated = [s[:500] for s in selected]

    docs_text = "\n\n".join(
        f"Dokument {i+1}: {summary}"
        for i, summary in enumerate(truncated)
    )

    prompt = f"""Analysiere diese Dokument-Zusammenfassungen und gib einen kurzen, beschreibenden Kategorie-Namen (1-3 Wörter, deutsch).

{docs_text}

Antworte NUR mit dem Kategorie-Namen, ohne Erklärung:"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        name = response["message"]["content"].strip()

        # Clean up common artifacts
        name = name.strip('"\'')
        name = name.split("\n")[0]  # Take only first line

        if len(name) > 50:
            name = name[:50]

        if not name:
            return _fallback_name()

        return name

    except Exception:
        return _fallback_name()


def _fallback_name() -> str:
    """Generate fallback cluster name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"Cluster_{timestamp}"
