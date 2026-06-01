import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_ai.pipeline.naming import generate_cluster_name, sanitize_folder_name


def test_sanitize_folder_name_removes_special_chars():
    assert sanitize_folder_name("Test/Name") == "Test_Name"
    assert sanitize_folder_name("A:B:C") == "A_B_C"
    assert sanitize_folder_name("Name!@#$") == "Name"


def test_sanitize_folder_name_handles_spaces():
    assert sanitize_folder_name("  Spaced Name  ") == "Spaced Name"


def test_sanitize_folder_name_limits_length():
    long_name = "A" * 100
    result = sanitize_folder_name(long_name)
    assert len(result) <= 50


def test_generate_cluster_name_for_outliers():
    result = generate_cluster_name(
        keywords=["sonstiges"],
        sample_filenames=[],
        model="llama3.2",
    )
    assert result == "_Sonstiges"


def test_generate_cluster_name_calls_ollama():
    mock_response = {"message": {"content": "Rechnungen"}}

    with patch("data_ai.pipeline.naming.ollama.chat", return_value=mock_response):
        result = generate_cluster_name(
            keywords=["rechnung", "euro", "zahlung"],
            sample_filenames=["rechnung_001.pdf", "invoice_002.pdf"],
            model="llama3.2",
        )

    assert result == "Rechnungen"


def test_generate_cluster_name_fallback_on_error():
    with patch("data_ai.pipeline.naming.ollama.chat", side_effect=Exception("API error")):
        result = generate_cluster_name(
            keywords=["rechnung", "euro"],
            sample_filenames=[],
            model="llama3.2",
        )

    # Falls back to first keyword, capitalized
    assert result == "Rechnung"
