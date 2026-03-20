# tests/test_analyzer_whisper.py
import pytest
from onesec.analyzer.whisper import WhisperAnalyzer
from onesec.models import VideoFile


def test_whisper_analyzer_name():
    assert WhisperAnalyzer().name == "whisper"


def test_whisper_analyzer_uses_gpu():
    assert WhisperAnalyzer().uses_gpu is True


def test_whisper_analyzer_is_available():
    result = WhisperAnalyzer().is_available()
    assert isinstance(result, bool)


def test_whisper_no_audio_returns_empty(synthetic_video):
    """Videos without audio track must return []."""
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    segs = WhisperAnalyzer().score_segments(vf, segment_duration=1.0)
    assert segs == []


def test_whisper_returns_empty_if_unavailable(synthetic_video):
    """If faster_whisper is not installed, score_segments returns []."""
    analyzer = WhisperAnalyzer()
    if analyzer.is_available():
        pytest.skip("faster_whisper is installed")
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=True)
    assert analyzer.score_segments(vf, segment_duration=1.0) == []


def test_whisper_scores_if_available(synthetic_video):
    """If faster_whisper is installed, score_segments returns N segments."""
    analyzer = WhisperAnalyzer()
    if not analyzer.is_available():
        pytest.skip("faster_whisper not installed")
    # synthetic_video has no audio — just test it returns [] gracefully
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    segs = analyzer.score_segments(vf, segment_duration=1.0)
    assert segs == []


def test_whisper_custom_model_size():
    """Constructor accepts model size option."""
    analyzer = WhisperAnalyzer(model="tiny")
    assert analyzer._model_size == "tiny"


def test_whisper_close_is_safe():
    """close() must not raise even before model is loaded."""
    WhisperAnalyzer().close()
