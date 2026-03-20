# tests/test_analyzer_clip.py
import pytest
from onesec.analyzer.clip_scorer import ClipScorer
from onesec.models import VideoFile


def test_clip_scorer_name():
    assert ClipScorer().name == "clip"


def test_clip_scorer_uses_gpu():
    assert ClipScorer().uses_gpu is True


def test_clip_scorer_is_available():
    result = ClipScorer().is_available()
    assert isinstance(result, bool)


def test_clip_scorer_returns_empty_if_unavailable(synthetic_video):
    """If open_clip is not installed, score_segments returns []."""
    scorer = ClipScorer()
    if scorer.is_available():
        pytest.skip("open_clip is installed — skip unavailability test")
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    assert scorer.score_segments(vf, segment_duration=1.0) == []


def test_clip_scorer_scores_if_available(synthetic_video):
    """If open_clip is installed, score_segments returns N segments with scores in [0,1]."""
    scorer = ClipScorer()
    if not scorer.is_available():
        pytest.skip("open_clip not installed")
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    segs = scorer.score_segments(vf, segment_duration=1.0)
    assert len(segs) == 3
    for s in segs:
        assert 0.0 <= s.score <= 1.0
        assert s.analyzer == "clip"


def test_clip_scorer_close_is_safe():
    """close() must not raise even before model is loaded."""
    ClipScorer().close()


def test_clip_scorer_short_video(synthetic_video):
    """Video shorter than segment_duration should return 1 segment."""
    scorer = ClipScorer()
    if not scorer.is_available():
        pytest.skip("open_clip not installed")
    vf = VideoFile(path=synthetic_video, duration=0.5, fps=10.0, has_audio=False)
    segs = scorer.score_segments(vf, segment_duration=1.0)
    assert len(segs) == 1


def test_clip_scorer_multiple_calls(synthetic_video):
    """Calling score_segments() twice on same instance should work (lazy load is idempotent)."""
    scorer = ClipScorer()
    if not scorer.is_available():
        pytest.skip("open_clip not installed")
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    segs1 = scorer.score_segments(vf, segment_duration=1.0)
    segs2 = scorer.score_segments(vf, segment_duration=1.0)
    assert len(segs1) == len(segs2)


def test_clip_scorer_custom_prompt(synthetic_video):
    """Custom prompt should be used for scoring (no error)."""
    scorer = ClipScorer(prompt="action scene")
    if not scorer.is_available():
        pytest.skip("open_clip not installed")
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    segs = scorer.score_segments(vf, segment_duration=1.0)
    assert len(segs) == 3
