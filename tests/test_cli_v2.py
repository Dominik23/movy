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
