from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class VideoFile:
    path: Path
    duration: float
    fps: float
    has_audio: bool


@dataclass
class ScoredSegment:
    video_path: Path
    start: float
    end: float
    score: float       # 0.0–1.0, raw score from analyzer
    analyzer: str      # Analyzer.name value; "merged" for multi-analyzer segments


class AnalyzerConfig(BaseModel):
    enabled: bool = True
    weight: float = 1.0
    options: dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    # [output]
    clip_duration: float = 1.0
    segment_duration: float | None = None   # None → same as clip_duration
    max_duration: float = 60.0
    top_n: int | None = None
    transition: str = "cut"
    merge_gap_threshold: float = 0.5

    # [device]
    device: str = "auto"

    # [parallelism]
    workers: int | None = None

    # [analyzers]
    analyzers: dict[str, AnalyzerConfig] = Field(default_factory=dict)
