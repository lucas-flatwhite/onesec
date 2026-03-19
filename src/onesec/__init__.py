from __future__ import annotations

import tempfile
from pathlib import Path

from onesec.analyzer.base import Analyzer
from onesec.analyzer.audio import AudioAnalyzer
from onesec.analyzer.motion import MotionAnalyzer
from onesec.analyzer.scene import SceneAnalyzer
from onesec.editor.composer import Composer
from onesec.editor.extractor import Extractor
from onesec.models import Config, ScoredSegment
from onesec.scanner import scan
from onesec.selector import Selector

# v0.1: sequential analysis only.
# v0.3 will add ProcessPoolExecutor for CPU analyzers and GPU Semaphore for ML analyzers.
DEFAULT_ANALYZERS = ["scene", "audio", "motion"]


def _build_default_analyzers(config: Config) -> list[Analyzer]:
    available = {
        "scene": SceneAnalyzer,
        "audio": AudioAnalyzer,
        "motion": MotionAnalyzer,
    }
    result = []
    for name, cls in available.items():
        ac = config.analyzers.get(name)
        if ac is not None and not ac.enabled:
            continue
        weight = ac.weight if ac else 1.0
        a = cls(weight=weight)
        if a.is_available():
            result.append(a)
    return result


class Pipeline:
    def __init__(
        self,
        analyzers: list[Analyzer] | None = None,
        *,
        config: Config | None = None,
        clip_duration: float = 1.0,
        segment_duration: float | None = None,
        max_duration: float = 60.0,
        top_n: int | None = None,
        transition: str = "cut",
        device: str = "auto",
        merge_gap_threshold: float = 0.5,
        workers: int | None = None,
    ) -> None:
        if config is None:
            config = Config(
                clip_duration=clip_duration,
                segment_duration=segment_duration,
                max_duration=max_duration,
                top_n=top_n,
                transition=transition,
                device=device,
                merge_gap_threshold=merge_gap_threshold,
                workers=workers,
            )
        self._config = config
        self._analyzers = analyzers if analyzers is not None else _build_default_analyzers(config)
        self._selector = Selector()

    def analyze(self, source: str | Path) -> list[ScoredSegment]:
        """Scan source, run analyzers, return Top-N selected segments."""
        import av
        from onesec.models import VideoFile

        source = Path(source)
        video_paths = scan(source)
        seg_dur = self._config.segment_duration or self._config.clip_duration

        all_segments: list[ScoredSegment] = []
        for video_path in video_paths:
            try:
                with av.open(str(video_path)) as container:
                    duration = float(container.duration / av.time_base)
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate or 25)
                    has_audio = any(s.type == "audio" for s in container.streams)
                vf = VideoFile(path=video_path, duration=duration, fps=fps, has_audio=has_audio)
                for analyzer in self._analyzers:
                    if not analyzer.is_available():
                        continue
                    all_segments.extend(analyzer.score_segments(vf, seg_dur))
            except Exception as e:
                import warnings
                warnings.warn(f"Skipping {video_path}: {e}")

        return self._selector.select(all_segments, self._config)

    def render(self, segments: list[ScoredSegment], output: str | Path) -> Path:
        """Extract clips for each segment and compose into output file."""
        output = Path(output)
        with tempfile.TemporaryDirectory() as tmp:
            clips_dir = Path(tmp)
            extractor = Extractor(clip_duration=self._config.clip_duration)
            clips = []
            for seg in segments:
                try:
                    clips.append(extractor.extract(seg, clips_dir))
                except ValueError as e:
                    import warnings
                    warnings.warn(str(e))
            if not clips:
                raise RuntimeError("No clips extracted — nothing to compose")
            Composer(transition=self._config.transition).compose(clips, output)
        return output

    def close(self) -> None:
        """Release GPU model memory (no-op for CPU analyzers)."""
        pass

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def run(
    source: str | Path,
    output: str | Path,
    analyzers: list[Analyzer] | None = None,
    **kwargs,
) -> Path:
    """Convenience: create Pipeline and run analyze → render in one call."""
    with Pipeline(analyzers, **kwargs) as p:
        segments = p.analyze(source)
        if not segments:
            import warnings
            warnings.warn("No segments found — try lowering merge_gap_threshold")
            return Path(output)
        return p.render(segments, output)
