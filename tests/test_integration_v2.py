import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant to avoid needing running server."""
    with patch("data_ai.storage.qdrant.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.collection_exists.return_value = True

        # Track upserted points
        mock_instance._docs = []
        mock_instance._clusters = []

        def mock_upsert(collection_name, points):
            if "documents" in collection_name:
                mock_instance._docs.extend(points)
            else:
                mock_instance._clusters.extend(points)

        def mock_scroll(collection_name, **kwargs):
            if "documents" in collection_name:
                return (mock_instance._docs, None)
            return (mock_instance._clusters, None)

        mock_instance.upsert.side_effect = mock_upsert
        mock_instance.scroll.side_effect = mock_scroll

        yield mock_instance


@pytest.fixture
def mock_ollama():
    """Mock Ollama for embeddings and chat."""
    with patch("data_ai.pipeline.embed.get_embedding") as mock_embed:
        with patch("data_ai.pipeline.naming.ollama") as mock_chat:
            # Return random embeddings
            mock_embed.side_effect = lambda text, model: np.random.randn(768).tolist()

            # Return generic cluster name
            mock_chat.chat.return_value = {"message": {"content": "Dokumente"}}

            yield mock_embed, mock_chat


def test_full_pipeline_with_mocks(tmp_path: Path, mock_qdrant, mock_ollama):
    """Test complete pipeline with mocked external services."""
    from data_ai.pipeline.extract import scan_folder

    # Create test files
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    for i in range(5):
        (inbox / f"doc{i}.txt").write_text(f"Test document content {i}")

    (inbox / "unsupported.xyz").write_text("unsupported")

    # Test scan
    files, trash_log = scan_folder(inbox, trash_dir=inbox / ".trash")

    assert len(files) == 5
    assert len(trash_log) == 1
    assert (inbox / ".trash" / "unsupported.xyz").exists()
