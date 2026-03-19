from pathlib import Path
import pytest
from onesec.config import load_config
from onesec.models import Config


def test_load_config_from_toml(tmp_path):
    toml = tmp_path / "onesec.toml"
    toml.write_text("""
[output]
clip_duration = 2.0
transition = "crossfade"

[device]
device = "cpu"
""")
    cfg = load_config(toml)
    assert cfg.clip_duration == 2.0
    assert cfg.transition == "crossfade"
    assert cfg.device == "cpu"


def test_load_config_analyzers(tmp_path):
    toml = tmp_path / "onesec.toml"
    toml.write_text("""
[analyzers]
scene = { enabled = true, weight = 0.4 }
audio = { enabled = false, weight = 0.6 }
""")
    cfg = load_config(toml)
    assert cfg.analyzers["scene"].weight == 0.4
    assert cfg.analyzers["audio"].enabled is False


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/onesec.toml"))


def test_load_config_defaults(tmp_path):
    toml = tmp_path / "onesec.toml"
    toml.write_text("")
    cfg = load_config(toml)
    assert cfg.clip_duration == 1.0
