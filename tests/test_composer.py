from pathlib import Path
import pytest
import av
from onesec.editor.composer import Composer
from onesec.editor.extractor import Extractor
from onesec.models import ScoredSegment


@pytest.fixture
def two_clips(synthetic_video, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    extractor = Extractor(clip_duration=1.0)
    segs = [
        ScoredSegment(synthetic_video, 0.0, 1.0, 0.9, "scene"),
        ScoredSegment(synthetic_video, 1.0, 2.0, 0.7, "scene"),
    ]
    return [extractor.extract(s, clips_dir) for s in segs]


def test_composer_creates_output(two_clips, tmp_path):
    output = tmp_path / "out.mp4"
    Composer(transition="cut").compose(two_clips, output)
    assert output.exists()


def test_composer_output_has_content(two_clips, tmp_path):
    output = tmp_path / "out.mp4"
    Composer(transition="cut").compose(two_clips, output)
    with av.open(str(output)) as c:
        duration = float(c.duration / av.time_base)
    assert duration > 1.0


def test_composer_single_clip(synthetic_video, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    seg = ScoredSegment(synthetic_video, 0.0, 1.0, 0.9, "scene")
    clip = Extractor(clip_duration=1.0).extract(seg, clips_dir)
    output = tmp_path / "out.mp4"
    Composer().compose([clip], output)
    assert output.exists()
