from pathlib import Path
import pytest
import av
from onesec.editor.extractor import Extractor
from onesec.models import ScoredSegment


@pytest.fixture
def video_path(synthetic_video):
    return synthetic_video


def test_extractor_creates_clip(video_path, tmp_path):
    seg = ScoredSegment(
        video_path=video_path,
        start=0.0,
        end=1.0,
        score=0.9,
        analyzer="scene",
    )
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    clip = Extractor(clip_duration=1.0).extract(seg, out_dir)
    assert clip.exists()
    assert clip.suffix == ".mp4"


def test_extractor_clip_duration(video_path, tmp_path):
    seg = ScoredSegment(
        video_path=video_path, start=0.0, end=1.0, score=0.9, analyzer="scene"
    )
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    clip = Extractor(clip_duration=1.0).extract(seg, out_dir)
    with av.open(str(clip)) as c:
        duration = float(c.duration / av.time_base)
    assert abs(duration - 1.0) < 0.2


def test_extractor_clamps_start(video_path, tmp_path):
    seg = ScoredSegment(
        video_path=video_path, start=2.8, end=3.0, score=0.5, analyzer="scene"
    )
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    clip = Extractor(clip_duration=1.0).extract(seg, out_dir)
    assert clip.exists()
