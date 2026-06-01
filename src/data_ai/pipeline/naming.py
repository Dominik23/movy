import re

import ollama


def sanitize_folder_name(name: str) -> str:
    """
    Sanitize a string for use as folder name.

    - Removes/replaces special characters
    - Limits length to 50 chars
    - Strips whitespace
    """
    # Strip leading/trailing whitespace first
    name = name.strip()
    # Replace problematic characters with underscore
    name = re.sub(r'[/\\:*?"<>|!@#$%^&]', '_', name)
    # Remove any remaining special characters
    name = re.sub(r'[^\w\s\-_]', '', name)
    # Collapse multiple consecutive underscores into single underscore
    name = re.sub(r'_+', '_', name)
    # Collapse multiple spaces into single space
    name = re.sub(r'\s+', ' ', name)
    # Strip leading/trailing underscores and spaces
    name = name.strip('_ ')
    # Limit length
    name = name[:50]
    return name


def generate_cluster_name(
    keywords: list[str],
    sample_filenames: list[str],
    model: str = "llama3.2",
) -> str:
    """
    Generate a nice folder name from topic keywords using Ollama.

    Args:
        keywords: Topic keywords from BERTopic
        sample_filenames: Example filenames from the cluster
        model: Ollama model to use

    Returns:
        Sanitized folder name
    """
    # Special case for outliers
    if keywords == ["sonstiges"] or not keywords:
        return "_Sonstiges"

    prompt = f"""Gegeben diese Keywords aus einem Dokumenten-Cluster: {', '.join(keywords)}

Beispiel-Dateinamen: {', '.join(sample_filenames[:5]) if sample_filenames else 'keine'}

Generiere einen kurzen, deutschen Ordnernamen (1-2 Wörter) der den Inhalt beschreibt.
Beispiele guter Namen: "Rechnungen", "Steuerunterlagen", "Verträge", "Kontoauszüge", "Versicherungen"

Antworte NUR mit dem Ordnernamen, nichts anderes."""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        name = response["message"]["content"].strip()
        return sanitize_folder_name(name)
    except Exception:
        # Fallback: capitalize first keyword
        fallback = keywords[0].capitalize() if keywords else "Dokumente"
        return sanitize_folder_name(fallback)
