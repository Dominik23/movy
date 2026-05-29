import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


def test_document_model_creation():
    from data_ai.storage.models import Document, DocumentStatus

    doc = Document(
        id="test-uuid",
        source_path="/path/to/file.pdf",
        file_type="pdf",
        file_size=1024,
        summary="Test summary",
        status=DocumentStatus.PENDING,
    )

    assert doc.id == "test-uuid"
    assert doc.status == DocumentStatus.PENDING
    assert doc.cluster_id is None
    assert doc.created_at is not None


def test_cluster_model_creation():
    from data_ai.storage.models import Cluster, ClusterStatus

    cluster = Cluster(
        id="cluster-uuid",
        name="Rechnungen",
        doc_count=10,
        variance=0.25,
        centroid=[0.1] * 768,
        status=ClusterStatus.PROPOSED,
    )

    assert cluster.name == "Rechnungen"
    assert cluster.status == ClusterStatus.PROPOSED
    assert len(cluster.centroid) == 768


def test_qdrant_store_init_creates_collections():
    with patch("data_ai.storage.qdrant.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.collection_exists.return_value = False

        from data_ai.storage.qdrant import QdrantStore

        store = QdrantStore(url="localhost:6333", prefix="test")

        assert mock_instance.create_collection.call_count == 2


def test_qdrant_store_upsert_document():
    with patch("data_ai.storage.qdrant.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.collection_exists.return_value = True

        from data_ai.storage.qdrant import QdrantStore
        from data_ai.storage.models import Document, DocumentStatus

        store = QdrantStore(url="localhost:6333", prefix="test")

        doc = Document(
            id="doc-1",
            source_path="/path/file.pdf",
            file_type="pdf",
            file_size=1024,
            summary="Test",
            status=DocumentStatus.PENDING,
            vector=[0.1] * 768,
        )

        store.upsert_document(doc)

        mock_instance.upsert.assert_called_once()

