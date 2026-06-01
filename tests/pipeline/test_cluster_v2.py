import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.cluster_v2 import cluster_documents, get_topic_keywords


def test_cluster_documents_returns_dict():
    docs = [
        (Path("/a.pdf"), "Invoice for services rendered"),
        (Path("/b.pdf"), "Invoice payment due"),
        (Path("/c.pdf"), "Contract agreement terms"),
        (Path("/d.pdf"), "Contract renewal notice"),
    ]

    mock_model = MagicMock()
    mock_model.fit_transform.return_value = ([0, 0, 1, 1], None)
    mock_model.get_topic_info.return_value = MagicMock()

    with patch("data_ai.pipeline.cluster_v2.BERTopic", return_value=mock_model):
        with patch("data_ai.pipeline.cluster_v2.SentenceTransformer"):
            # Use min_topic_size=2 so 4 docs can form clusters
            result, model = cluster_documents(docs, min_topic_size=2)

    assert isinstance(result, dict)
    assert 0 in result
    assert 1 in result
    assert result[0] == [Path("/a.pdf"), Path("/b.pdf")]
    assert result[1] == [Path("/c.pdf"), Path("/d.pdf")]


def test_cluster_documents_handles_outliers():
    docs = [
        (Path("/a.pdf"), "Some text"),
        (Path("/b.pdf"), "Other text"),
    ]

    mock_model = MagicMock()
    mock_model.fit_transform.return_value = ([-1, -1], None)  # All outliers
    mock_model.get_topic_info.return_value = MagicMock()

    with patch("data_ai.pipeline.cluster_v2.BERTopic", return_value=mock_model):
        with patch("data_ai.pipeline.cluster_v2.SentenceTransformer"):
            result, model = cluster_documents(docs)

    assert -1 in result
    assert len(result[-1]) == 2


def test_get_topic_keywords():
    mock_model = MagicMock()
    mock_model.get_topic.return_value = [
        ("invoice", 0.5),
        ("payment", 0.3),
        ("due", 0.2),
    ]

    keywords = get_topic_keywords(mock_model, topic_id=0, top_n=3)

    assert keywords == ["invoice", "payment", "due"]
