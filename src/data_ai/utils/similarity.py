# src/data_ai/utils/similarity.py
import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average empty list of vectors")

    arr = np.array(vectors)
    return arr.mean(axis=0).tolist()


def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Returns 1 - cosine_similarity (0 = identical, 2 = opposite)."""
    return 1.0 - cosine_similarity(vec_a, vec_b)


def compute_variance(vectors: list[list[float]]) -> float:
    """Compute variance of distances from centroid."""
    if len(vectors) < 2:
        return 0.0

    centroid = average_vectors(vectors)
    distances = [cosine_distance(v, centroid) for v in vectors]
    return float(np.var(distances))
