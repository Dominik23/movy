import pytest
from pathlib import Path


def test_generate_review_html_creates_file(tmp_path: Path):
    from data_ai.pipeline.review import generate_review_html
    from data_ai.storage.models import Cluster, ClusterStatus

    clusters = [
        Cluster(
            id="c1",
            name="Rechnungen",
            doc_count=10,
            variance=0.2,
            centroid=[0.1] * 768,
            status=ClusterStatus.PROPOSED,
        ),
        Cluster(
            id="c2",
            name="Verträge",
            doc_count=5,
            variance=0.3,
            centroid=[0.2] * 768,
            status=ClusterStatus.PROPOSED,
        ),
    ]

    cluster_docs = {
        "c1": ["doc1.pdf", "doc2.pdf"],
        "c2": ["doc3.pdf"],
    }

    output_path = tmp_path / "review.html"

    generate_review_html(clusters, cluster_docs, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Rechnungen" in content
    assert "Verträge" in content


def test_generate_review_html_contains_graph():
    from data_ai.pipeline.review import generate_review_html
    from data_ai.storage.models import Cluster, ClusterStatus
    import tempfile

    clusters = [
        Cluster(
            id="c1",
            name="Test",
            doc_count=5,
            variance=0.1,
            centroid=[0.1] * 768,
            status=ClusterStatus.PROPOSED,
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output_path = Path(f.name)

    generate_review_html(clusters, {"c1": ["doc.pdf"]}, output_path)

    content = output_path.read_text()
    assert "vis-network" in content or "pyvis" in content.lower() or "nodes" in content
