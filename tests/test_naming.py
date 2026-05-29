import pytest
from unittest.mock import patch, MagicMock


def test_generate_cluster_name_calls_ollama():
    with patch("data_ai.pipeline.naming.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {
            "message": {"content": "Rechnungen"}
        }

        from data_ai.pipeline.naming import generate_cluster_name

        summaries = [
            "Rechnung Nr. 12345 über 500 EUR",
            "Invoice for services rendered",
            "Zahlungsaufforderung vom 01.01.2026",
        ]

        name = generate_cluster_name(summaries, model="llama3.2")

        assert name == "Rechnungen"
        mock_ollama.chat.assert_called_once()


def test_generate_cluster_name_cleans_response():
    with patch("data_ai.pipeline.naming.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {
            "message": {"content": "  Verträge und Dokumente  \n"}
        }

        from data_ai.pipeline.naming import generate_cluster_name

        name = generate_cluster_name(["test"], model="llama3.2")

        assert name == "Verträge und Dokumente"


def test_generate_cluster_name_fallback_on_error():
    with patch("data_ai.pipeline.naming.ollama") as mock_ollama:
        mock_ollama.chat.side_effect = Exception("Connection error")

        from data_ai.pipeline.naming import generate_cluster_name

        name = generate_cluster_name(["test"], model="llama3.2")

        assert name.startswith("Cluster_")
