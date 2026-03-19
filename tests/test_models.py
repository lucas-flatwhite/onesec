from pathlib import Path
import pytest
from onesec.models import VideoFile, ScoredSegment, Config, AnalyzerConfig


def test_video_file_fields():
    vf = VideoFile(path=Path("x.mp4"), duration=10.0, fps=30.0, has_audio=True)
    assert vf.path == Path("x.mp4")
    assert vf.duration == 10.0
    assert vf.fps == 30.0
    assert vf.has_audio is True


def test_scored_segment_fields():
    seg = ScoredSegment(
        video_path=Path("x.mp4"),
        start=1.0,
        end=2.0,
        score=0.8,
        analyzer="scene",
    )
    assert seg.score == 0.8
    assert seg.analyzer == "scene"


def test_config_defaults():
    cfg = Config()
    assert cfg.clip_duration == 1.0
    assert cfg.transition == "cut"
    assert cfg.device == "auto"
    assert cfg.merge_gap_threshold == 0.5
    assert cfg.workers is None


def test_config_segment_duration_defaults_to_clip_duration():
    cfg = Config(clip_duration=2.0)
    # When segment_duration is None, callers should use clip_duration
    assert cfg.segment_duration is None


def test_analyzer_config_defaults():
    ac = AnalyzerConfig()
    assert ac.enabled is True
    assert ac.weight == 1.0
    assert ac.options == {}


def test_config_analyzers_dict():
    cfg = Config(analyzers={"scene": AnalyzerConfig(weight=0.5)})
    assert cfg.analyzers["scene"].weight == 0.5
