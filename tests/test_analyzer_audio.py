import pytest
from onesec.analyzer.audio import AudioAnalyzer
from onesec.models import VideoFile


@pytest.fixture
def audio_video_file(synthetic_video):
    # synthetic_video has no audio track — use for no-audio test
    return VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)


def test_audio_analyzer_name():
    assert AudioAnalyzer().name == "audio"


def test_audio_no_audio_returns_empty(audio_video_file):
    segs = AudioAnalyzer().score_segments(audio_video_file, segment_duration=1.0)
    assert segs == []


def test_audio_is_available():
    result = AudioAnalyzer().is_available()
    assert isinstance(result, bool)
