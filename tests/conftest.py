"""Set up mocks at the very beginning"""
import sys
from types import ModuleType
import importlib.machinery
from unittest.mock import MagicMock

# Create proper mock modules that won't fail during import
def create_mock_module(name):
    mock_module = ModuleType(name)
    # Use a proper ModuleSpec instead of None
    mock_module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mock_module.__loader__ = None
    return mock_module

# Set up mocks IMMEDIATELY before pytest does any imports
# We need to add attributes that extractors.py will try to import
docx_mock = create_mock_module('docx')
docx_mock.Document = MagicMock()
sys.modules['docx'] = docx_mock

pptx_mock = create_mock_module('pptx')
pptx_mock.Presentation = MagicMock()
sys.modules['pptx'] = pptx_mock

pdfplumber_mock = create_mock_module('pdfplumber')
sys.modules['pdfplumber'] = pdfplumber_mock

pytesseract_mock = create_mock_module('pytesseract')
sys.modules['pytesseract'] = pytesseract_mock

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    config_content = """
settings:
  ollama_model: "nomic-embed-text"
  vision_model: "llava"
  similarity_threshold: 0.6
  inbox: "./inbox"

categories:
  TestCategory:
    keywords:
      - "test"
      - "example"
    examples: []
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


