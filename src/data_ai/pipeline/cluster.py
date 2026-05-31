# src/data_ai/pipeline/cluster.py
import numpy as np
import umap
import hdbscan


def cluster_documents(
    vectors: list[list[float]],
    min_cluster_size: int = 15,
    umap_n_components: int = 10,
) -> tuple[list[int], list[list[float]], list[int]]:
    """
    Cluster vectors using UMAP + HDBSCAN.

    Args:
        vectors: List of embedding vectors (768-dimensional)
        min_cluster_size: Minimum cluster size for HDBSCAN
        umap_n_components: Target dimensions for UMAP reduction

    Returns:
        labels: Cluster assignment per document (-1 = outlier)
        centroids: Centroid of each cluster (in original 768-dim space)
        outlier_indices: Indices of outlier documents
    """
    X = np.array(vectors)
    n_samples = len(vectors)

    # Handle edge cases
    if n_samples < min_cluster_size:
        # Too few samples - everything is an outlier
        return [-1] * n_samples, [], list(range(n_samples))

    # Adjust UMAP components if necessary
    # UMAP needs n_components < n_samples, and we need room for spectral embedding
    effective_components = min(umap_n_components, n_samples - 2, X.shape[1])
    effective_components = max(2, effective_components)  # At least 2 dimensions

    # Effective neighbors must be less than n_samples
    effective_neighbors = min(15, n_samples - 1)

    # Dimensionality reduction with UMAP
    reducer = umap.UMAP(
        n_components=effective_components,
        metric="cosine",
        random_state=42,
        n_neighbors=effective_neighbors,
        init="random",  # Use random init for small datasets to avoid spectral issues
    )
    reduced = reducer.fit_transform(X)

    # Clustering with HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # Extract outlier indices
    outlier_indices = [i for i, label in enumerate(labels) if label == -1]

    # Compute centroids in original space
    unique_labels = sorted(set(labels) - {-1})
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = X[mask].mean(axis=0).tolist()
        centroids.append(centroid)

    return labels.tolist(), centroids, outlier_indices
