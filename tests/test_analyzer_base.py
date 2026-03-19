from pathlib import Path
import pytest
from onesec.analyzer.base import Analyzer
from onesec.models import VideoFile, ScoredSegment


class ConcreteAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "test"

    def score_segments(self, video: VideoFile, segment_duration: float) -> list[ScoredSegment]:
        return []

    def is_available(self) -> bool:
        return True


def test_analyzer_default_weight():
    a = ConcreteAnalyzer()
    assert a.weight == 1.0


def test_analyzer_custom_weight():
    a = ConcreteAnalyzer(weight=0.5)
    assert a.weight == 0.5


def test_analyzer_uses_gpu_default():
    a = ConcreteAnalyzer()
    assert a.uses_gpu is False


def test_abstract_name_enforced():
    class NoName(Analyzer):
        def score_segments(self, video, segment_duration):
            return []
        def is_available(self):
            return True

    with pytest.raises(TypeError):
        NoName()


def test_abstract_score_segments_enforced():
    class NoScore(Analyzer):
        @property
        def name(self):
            return "x"
        def is_available(self):
            return True

    with pytest.raises(TypeError):
        NoScore()
