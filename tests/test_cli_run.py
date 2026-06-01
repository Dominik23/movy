# tests/test_cli_run.py
import sys
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock


runner = CliRunner()


@pytest.fixture(autouse=True, scope="module")
def mock_qdrant_dependency():
    """Mock qdrant_client and other old pipeline dependencies to avoid installation requirement."""
    sys.modules["qdrant_client"] = MagicMock()
    sys.modules["qdrant_client.models"] = MagicMock()
    sys.modules["pyvis"] = MagicMock()
    sys.modules["pyvis.network"] = MagicMock()
    yield
    sys.modules.pop("qdrant_client", None)
    sys.modules.pop("qdrant_client.models", None)
    sys.modules.pop("pyvis", None)
    sys.modules.pop("pyvis.network", None)


def test_run_command_exists():
    from data_ai.cli_v2 import app

    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "input" in result.output.lower()
    assert "output" in result.output.lower()


def test_run_command_requires_input():
    from data_ai.cli_v2 import app

    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_run_command_calls_pipeline(tmp_path):
    from data_ai.cli_v2 import app

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with patch("data_ai.cli_v2.run_pipeline") as mock_run:
        result = runner.invoke(app, [
            "run",
            str(input_dir),
            "--output", str(output_dir),
            "--dry-run",
        ])

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.kwargs["input_dir"] == input_dir
    assert call_args.kwargs["output_dir"] == output_dir
    assert call_args.kwargs["dry_run"] == True
