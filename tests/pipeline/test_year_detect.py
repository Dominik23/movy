import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from data_ai.pipeline.year_detect import detect_year


def test_detect_year_from_filename_4digits():
    path = Path("/docs/rechnung_2024_001.pdf")
    assert detect_year(path) == 2024


def test_detect_year_from_filename_in_path():
    path = Path("/archive/2023/invoice.pdf")
    assert detect_year(path) == 2023


def test_detect_year_prefers_filename_over_path():
    path = Path("/archive/2020/rechnung_2024.pdf")
    assert detect_year(path) == 2024


def test_detect_year_ignores_invalid_years():
    path = Path("/docs/file_1899.pdf")
    with patch("data_ai.pipeline.year_detect._get_mtime_year", return_value=2025):
        assert detect_year(path) == 2025


def test_detect_year_falls_back_to_mtime():
    path = Path("/docs/random_file.pdf")
    with patch("data_ai.pipeline.year_detect._get_mtime_year", return_value=2022):
        assert detect_year(path) == 2022


def test_detect_year_falls_back_to_current_year():
    path = Path("/docs/random_file.pdf")
    with patch("data_ai.pipeline.year_detect._get_mtime_year", return_value=None):
        with patch("data_ai.pipeline.year_detect._get_current_year", return_value=2026):
            assert detect_year(path) == 2026
