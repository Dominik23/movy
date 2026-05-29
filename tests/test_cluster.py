import pytest
import numpy as np


def test_find_optimal_k_returns_reasonable_k():
    from data_ai.pipeline.cluster import find_optimal_k

    # Create 3 distinct clusters
    np.random.seed(42)
    cluster1 = np.random.randn(20, 10) + [5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cluster2 = np.random.randn(20, 10) + [0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    cluster3 = np.random.randn(20, 10) + [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]

    vectors = np.vstack([cluster1, cluster2, cluster3]).tolist()

    k = find_optimal_k(vectors, min_k=2, max_k=10)

    assert 2 <= k <= 5  # Should find approximately 3


def test_cluster_documents_returns_assignments():
    from data_ai.pipeline.cluster import cluster_documents

    np.random.seed(42)
    vectors = np.random.randn(30, 10).tolist()

    assignments, centroids = cluster_documents(vectors, k=3)

    assert len(assignments) == 30
    assert len(centroids) == 3
    assert all(0 <= a < 3 for a in assignments)


def test_should_split_returns_true_for_high_variance():
    from data_ai.pipeline.cluster import should_split

    # Very different vectors
    vectors = [
        [1.0] + [0.0] * 9,
        [0.0, 1.0] + [0.0] * 8,
        [0.0, 0.0, 1.0] + [0.0] * 7,
    ]

    assert should_split(vectors, threshold=0.1) is True


def test_should_split_returns_false_for_low_variance():
    from data_ai.pipeline.cluster import should_split

    # Very similar vectors
    vectors = [
        [1.0, 0.1, 0.0],
        [1.0, 0.0, 0.1],
        [1.0, 0.05, 0.05],
    ]

    assert should_split(vectors, threshold=0.5) is False
