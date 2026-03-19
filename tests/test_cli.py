from pathlib import Path
from typer.testing import CliRunner
import pytest
from onesec.cli import app

runner = CliRunner()


def test_cli_run_requires_source():
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_cli_run_dry_run(synthetic_video, tmp_path):
    result = runner.invoke(app, [
        "run", str(synthetic_video.parent),
        "--dry-run", "--format", "table",
    ])
    assert result.exit_code == 0


def test_cli_run_dry_run_json(synthetic_video, tmp_path):
    import json
    result = runner.invoke(app, [
        "run", str(synthetic_video.parent),
        "--dry-run", "--format", "json",
    ])
    assert result.exit_code == 0
    output = result.output.strip()
    if output.startswith("["):
        data = json.loads(output)
        assert isinstance(data, list)
        if data:
            assert "video_path" in data[0]


def test_cli_analyzers_list():
    result = runner.invoke(app, ["analyzers", "list"])
    assert result.exit_code == 0
    assert "scene" in result.output


def test_cli_run_with_output(synthetic_video, tmp_path):
    output = tmp_path / "out.mp4"
    result = runner.invoke(app, [
        "run", str(synthetic_video.parent),
        "-o", str(output),
        "--clip-duration", "1.0",
        "--max-duration", "5.0",
    ])
    assert result.exit_code == 0
