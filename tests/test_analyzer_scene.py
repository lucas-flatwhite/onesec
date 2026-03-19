from pathlib import Path
import pytest
from onesec.analyzer.scene import SceneAnalyzer
from onesec.models import VideoFile


@pytest.fixture
def video_file(synthetic_video) -> VideoFile:
    return VideoFile(
        path=synthetic_video,
        duration=3.0,
        fps=10.0,
        has_audio=False,
    )


def test_scene_analyzer_name():
    assert SceneAnalyzer().name == "scene"


def test_scene_analyzer_is_available():
    assert SceneAnalyzer().is_available() is True


def test_scene_returns_segments(video_file):
    analyzer = SceneAnalyzer()
    segments = analyzer.score_segments(video_file, segment_duration=1.0)
    assert len(segments) == 3
    for seg in segments:
        assert 0.0 <= seg.score <= 1.0
        assert seg.analyzer == "scene"
        assert seg.video_path == video_file.path


def test_scene_scores_higher_at_changes(video_file):
    analyzer = SceneAnalyzer()
    segments = analyzer.score_segments(video_file, segment_duration=1.0)
    scores = [s.score for s in segments]
    assert max(scores) > 0.0


def test_scene_short_video(video_file):
    vf = VideoFile(path=video_file.path, duration=0.5, fps=10.0, has_audio=False)
    analyzer = SceneAnalyzer()
    segments = analyzer.score_segments(vf, segment_duration=1.0)
    assert len(segments) == 1
