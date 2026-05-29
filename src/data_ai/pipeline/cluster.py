# src/data_ai/pipeline/cluster.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_ai.utils.similarity import compute_variance, average_vectors


def find_optimal_k(
    vectors: list[list[float]],
    min_k: int = 2,
    max_k: int = 20,
) -> int:
    """
    Find optimal number of clusters using Elbow method + Silhouette score.
    """
    n_samples = len(vectors)
    if n_samples < min_k:
        return min_k

    max_k = min(max_k, n_samples // 2, n_samples - 1)
    if max_k < min_k:
        return min_k

    X = np.array(vectors)

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_documents(
    vectors: list[list[float]],
    k: int,
) -> tuple[list[int], list[list[float]]]:
    """
    Cluster vectors using KMeans.

    Returns: (cluster_assignments, centroids)
    """
    X = np.array(vectors)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_.tolist()

    return labels.tolist(), centroids


def should_split(
    vectors: list[list[float]],
    threshold: float = 0.4,
) -> bool:
    """
    Check if a cluster should be split based on variance.
    """
    if len(vectors) < 3:  # Need at least 3 to split into 2+1
        return False

    # Use mean distance from centroid as a measure of spread
    # (variance of distances can be 0 even for very different vectors)
    from data_ai.utils.similarity import cosine_distance
    centroid = average_vectors(vectors)
    distances = [cosine_distance(v, centroid) for v in vectors]
    mean_distance = np.mean(distances)

    return float(mean_distance) > threshold


def split_cluster(
    vectors: list[list[float]],
    doc_ids: list[str],
) -> tuple[list[tuple[str, int]], list[list[float]]]:
    """
    Split a cluster into 2 sub-clusters.

    Returns: ([(doc_id, sub_cluster_idx), ...], [centroid1, centroid2])
    """
    assignments, centroids = cluster_documents(vectors, k=2)

    result = [(doc_id, assignment) for doc_id, assignment in zip(doc_ids, assignments)]

    return result, centroids
