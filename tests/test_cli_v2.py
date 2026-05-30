import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

runner = CliRunner()


def test_cli_status_shows_counts():
    with patch("data_ai.cli_v2.QdrantStore") as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.get_all_documents.return_value = []
        mock_store.get_all_clusters.return_value = []

        from data_ai.cli_v2 import app

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Documents:" in result.stdout or "documents" in result.stdout.lower()


def test_cli_reset_requires_confirm():
    from data_ai.cli_v2 import app

    result = runner.invoke(app, ["reset"])

    # Should fail without --confirm flag
    assert result.exit_code != 0 or "confirm" in result.stdout.lower()


def test_cli_review_interactive_flag():
    """Test that --interactive flag is recognized."""
    from data_ai.cli_v2 import app

    result = runner.invoke(app, ["review", "--help"])

    assert result.exit_code == 0
    assert "--interactive" in result.stdout or "-i" in result.stdout


def test_cli_review_interactive_no_clusters():
    """Test interactive mode shows message when no clusters exist."""
    with patch("data_ai.cli_v2.QdrantStore") as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.get_all_clusters.return_value = []

        from data_ai.cli_v2 import app

        result = runner.invoke(app, ["review", "--interactive"])

        assert result.exit_code == 0
        assert "No clusters found" in result.stdout


def test_cli_review_interactive_quit():
    """Test interactive mode can quit immediately."""
    from data_ai.storage.models import Cluster, ClusterStatus
    from datetime import datetime

    with patch("data_ai.cli_v2.QdrantStore") as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        test_cluster = Cluster(
            id="test-1",
            name="Test Cluster",
            doc_count=5,
            variance=0.1,
            centroid=[0.0] * 768,
            status=ClusterStatus.PROPOSED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_store.get_all_clusters.return_value = [test_cluster]

        from data_ai.cli_v2 import app

        # Simulate user typing 'q' to quit
        result = runner.invoke(app, ["review", "--interactive"], input="q\n")

        assert result.exit_code == 0
        assert "Changes saved" in result.stdout


def test_cli_review_interactive_approve_all():
    """Test interactive mode can approve all clusters."""
    from data_ai.storage.models import Cluster, ClusterStatus
    from datetime import datetime

    with patch("data_ai.cli_v2.QdrantStore") as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        test_cluster = Cluster(
            id="test-1",
            name="Test Cluster",
            doc_count=5,
            variance=0.1,
            centroid=[0.0] * 768,
            status=ClusterStatus.PROPOSED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_store.get_all_clusters.return_value = [test_cluster]

        from data_ai.cli_v2 import app

        # Simulate user typing 'a' to approve all, then 'q' to quit
        result = runner.invoke(app, ["review", "--interactive"], input="a\nq\n")

        assert result.exit_code == 0
        mock_store.update_cluster_status.assert_called_once_with(
            "test-1", ClusterStatus.APPROVED
        )
