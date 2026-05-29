import pytest
from datetime import datetime


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
