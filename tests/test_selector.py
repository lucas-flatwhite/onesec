from pathlib import Path
import pytest
from onesec.models import AnalyzerConfig, Config, ScoredSegment
from onesec.selector import Selector

VIDEO = Path("v.mp4")


def seg(start, end, score, analyzer="scene"):
    return ScoredSegment(video_path=VIDEO, start=start, end=end, score=score, analyzer=analyzer)


def test_selector_returns_top_n():
    # Use non-adjacent segments (gap=1.0 > merge_gap_threshold=0.5) so they don't merge
    segs = [seg(0, 1, 0.9), seg(2, 3, 0.3), seg(4, 5, 0.7)]
    cfg = Config(clip_duration=1.0, top_n=2)
    result = Selector().select(segs, cfg)
    assert len(result) == 2
    assert result[0].start < result[1].start


def test_selector_respects_max_duration():
    segs = [seg(i, i + 1, 0.9) for i in range(10)]
    cfg = Config(clip_duration=1.0, max_duration=3.0)
    result = Selector().select(segs, cfg)
    assert len(result) <= 3


def test_selector_merges_adjacent_segments():
    segs = [
        seg(0.0, 1.0, 0.8, "scene"),
        seg(1.2, 2.2, 0.6, "audio"),
    ]
    cfg = Config(clip_duration=1.0, merge_gap_threshold=0.5)
    result = Selector().select(segs, cfg)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 2.2
    assert result[0].analyzer == "merged"


def test_selector_does_not_merge_far_segments():
    segs = [seg(0.0, 1.0, 0.8), seg(3.0, 4.0, 0.6)]
    cfg = Config(clip_duration=1.0, merge_gap_threshold=0.5)
    result = Selector().select(segs, cfg)
    assert len(result) == 2


def test_selector_normalizes_weights():
    segs = [
        ScoredSegment(VIDEO, 0, 1, 0.5, "a"),
        ScoredSegment(VIDEO, 0, 1, 0.5, "b"),
    ]
    cfg = Config(
        clip_duration=1.0,
        top_n=1,
        analyzers={
            "a": AnalyzerConfig(weight=2.0),
            "b": AnalyzerConfig(weight=2.0),
        },
    )
    result = Selector().select(segs, cfg)
    assert len(result) == 1


def test_selector_empty_input():
    result = Selector().select([], Config())
    assert result == []


def test_selector_single_analyzer_not_merged():
    # gap = 2.0 - 1.0 = 1.0 > merge_gap_threshold(0.5) → NOT merged
    segs = [seg(0, 1, 0.9, "scene"), seg(2, 3, 0.5, "scene")]
    cfg = Config(clip_duration=1.0, top_n=2)
    result = Selector().select(segs, cfg)
    assert len(result) == 2
    assert all(s.analyzer == "scene" for s in result)
