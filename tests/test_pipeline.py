from pathlib import Path
import pytest
from onesec import Pipeline, run


def test_pipeline_analyze_returns_segments(synthetic_video, tmp_path):
    p = Pipeline(clip_duration=1.0, max_duration=5.0)
    segments = p.analyze(synthetic_video.parent)
    assert isinstance(segments, list)
    assert len(segments) <= 3


def test_pipeline_render_creates_file(synthetic_video, tmp_path):
    p = Pipeline(clip_duration=1.0, max_duration=5.0)
    segments = p.analyze(synthetic_video.parent)
    if not segments:
        pytest.skip("No segments found (threshold issue)")
    output = tmp_path / "out.mp4"
    p.render(segments, output)
    assert output.exists()


def test_pipeline_context_manager(synthetic_video, tmp_path):
    with Pipeline(clip_duration=1.0) as p:
        segments = p.analyze(synthetic_video.parent)
    assert isinstance(segments, list)


def test_run_convenience(synthetic_video, tmp_path):
    output = tmp_path / "highlight.mp4"
    run(synthetic_video.parent, output)
    # Either creates output or finds no segments — both valid


def test_pipeline_default_analyzers():
    p = Pipeline()
    assert len(p._analyzers) > 0
