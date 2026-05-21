# src/data_ai/pipeline/match.py
from dataclasses import dataclass
from typing import Optional

from data_ai.utils.similarity import cosine_similarity


@dataclass
class CategoryEmbedding:
    name: str
    vector: list[float]


@dataclass
class MatchResult:
    category: str
    confidence: float
    all_matches: list[tuple[str, float]]  # All categories with scores, sorted desc


def match_stage(
    file_vector: list[float],
    category_embeddings: list[CategoryEmbedding],
    threshold: float = 0.6,
) -> Optional[MatchResult]:
    if not category_embeddings:
        return None

    # Calculate similarity for all categories
    scores: list[tuple[str, float]] = []
    for cat_emb in category_embeddings:
        score = cosine_similarity(file_vector, cat_emb.vector)
        scores.append((cat_emb.name, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    best_category, best_score = scores[0]

    if best_score < threshold:
        return None

    return MatchResult(
        category=best_category,
        confidence=best_score,
        all_matches=scores,
    )
