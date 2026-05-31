import pytest
import numpy as np


def test_cluster_documents_returns_labels_and_outliers():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    vectors = np.random.randn(50, 768).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=5,
    )

    assert len(labels) == 50
    assert all(isinstance(label, int) for label in labels)
    assert isinstance(outlier_indices, list)
    assert all(isinstance(idx, int) for idx in outlier_indices)
    # Outliers should have label -1
    for idx in outlier_indices:
        assert labels[idx] == -1


def test_cluster_documents_finds_distinct_groups():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    # Two clearly separated groups
    group1 = np.random.randn(30, 768) + np.array([10.0] * 768)
    group2 = np.random.randn(30, 768) + np.array([-10.0] * 768)
    vectors = np.vstack([group1, group2]).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=10,
    )

    # Should find at least 2 clusters
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 2
    assert len(centroids) >= 2


def test_cluster_documents_centroids_in_original_space():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    vectors = np.random.randn(50, 768).tolist()

    labels, centroids, _ = cluster_documents(vectors, min_cluster_size=5)

    # Centroids should be 768-dimensional (original space)
    for centroid in centroids:
        assert len(centroid) == 768


def test_cluster_documents_handles_small_dataset():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    # Very small dataset - should still work
    vectors = np.random.randn(10, 768).tolist()

    labels, centroids, outlier_indices = cluster_documents(
        vectors,
        min_cluster_size=3,
    )

    assert len(labels) == 10
