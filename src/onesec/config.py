from __future__ import annotations

import tomllib
from pathlib import Path

from onesec.models import AnalyzerConfig, Config


def load_config(path: str | Path) -> Config:
    """Load Config from a TOML file.

    TOML structure:
      [output]      → clip_duration, transition, max_duration, top_n, merge_gap_threshold
      [device]      → device key (e.g. device = "cpu")
      [parallelism] → workers
      [analyzers]   → name = { enabled, weight, options }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    output = data.get("output", {})
    device_section = data.get("device", {})
    parallelism = data.get("parallelism", {})
    analyzers_raw = data.get("analyzers", {})

    analyzers = {
        name: AnalyzerConfig(**cfg)
        for name, cfg in analyzers_raw.items()
    }

    return Config(
        clip_duration=output.get("clip_duration", 1.0),
        segment_duration=output.get("segment_duration"),
        max_duration=output.get("max_duration", 60.0),
        top_n=output.get("top_n"),
        transition=output.get("transition", "cut"),
        merge_gap_threshold=output.get("merge_gap_threshold", 0.5),
        device=device_section.get("device", "auto"),
        workers=parallelism.get("workers"),
        analyzers=analyzers,
    )
