from onesec.analyzer.motion import MotionAnalyzer
from onesec.models import VideoFile


def test_motion_analyzer_name():
    assert MotionAnalyzer().name == "motion"


def test_motion_is_available():
    assert MotionAnalyzer().is_available() is True


def test_motion_returns_segments(synthetic_video):
    vf = VideoFile(path=synthetic_video, duration=3.0, fps=10.0, has_audio=False)
    segs = MotionAnalyzer().score_segments(vf, segment_duration=1.0)
    assert len(segs) == 3
    for s in segs:
        assert 0.0 <= s.score <= 1.0
        assert s.analyzer == "motion"
