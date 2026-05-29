# tests/test_similarity.py
import pytest
import numpy as np
from data_ai.utils.similarity import cosine_similarity, average_vectors


def test_cosine_similarity_identical():
    vec = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [-1.0, 0.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)


def test_average_vectors_single():
    vectors = [[1.0, 2.0, 3.0]]
    result = average_vectors(vectors)
    assert result == pytest.approx([1.0, 2.0, 3.0])


def test_average_vectors_multiple():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    result = average_vectors(vectors)
    assert result == pytest.approx([0.5, 0.5])


def test_average_vectors_empty():
    with pytest.raises(ValueError):
        average_vectors([])


def test_cosine_distance():
    from data_ai.utils.similarity import cosine_distance

    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]

    assert cosine_distance(vec_a, vec_b) == pytest.approx(0.0)

    vec_c = [0.0, 1.0, 0.0]
    assert cosine_distance(vec_a, vec_c) == pytest.approx(1.0)


def test_compute_variance():
    from data_ai.utils.similarity import compute_variance

    vectors = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    # All same vectors -> variance should be 0
    assert compute_variance(vectors) == pytest.approx(0.0)


def test_compute_variance_different_vectors():
    from data_ai.utils.similarity import compute_variance

    vectors = [
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.0],
    ]

    # Different vectors -> should have non-zero variance
    variance = compute_variance(vectors)
    assert variance > 0.01
