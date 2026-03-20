from __future__ import annotations

import tempfile
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from onesec.analyzer.base import Analyzer
from onesec.analyzer.scene import SceneAnalyzer
from onesec.analyzer.audio import AudioAnalyzer
from onesec.analyzer.motion import MotionAnalyzer
from onesec.editor.composer import Composer
from onesec.editor.extractor import Extractor
from onesec.models import Config, ScoredSegment, VideoFile
from onesec.scanner import scan
from onesec.selector import Selector

DEFAULT_ANALYZERS = ["scene", "audio", "motion"]

# Global semaphore: only one GPU analyzer runs at a time across all Pipeline instances.
_GPU_SEMAPHORE = threading.Semaphore(1)


def _probe_video(video_path: Path) -> VideoFile:
    """Extract VideoFile metadata from a video path. Runs in main process."""
    import av
    with av.open(str(video_path)) as container:
        duration = float(container.duration / av.time_base)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        has_audio = any(s.type == "audio" for s in container.streams)
    return VideoFile(path=video_path, duration=duration, fps=fps, has_audio=has_audio)


def _analyze_video_cpu(args: tuple) -> list[ScoredSegment]:
    """Top-level function required by ProcessPoolExecutor (must be module-level for pickling).

    args: (analyzer, video_file, segment_duration)
    Returns scored segments or [] on error.
    """
    analyzer, video_file, segment_duration = args
    try:
        return analyzer.score_segments(video_file, segment_duration)
    except Exception as e:
        import warnings
        warnings.warn(f"{analyzer.name} failed on {video_file.path}: {e}")
        return []


def _build_default_analyzers(config: Config) -> list[Analyzer]:
    available: dict[str, type[Analyzer]] = {
        "scene": SceneAnalyzer,
        "audio": AudioAnalyzer,
        "motion": MotionAnalyzer,
    }
    # Lazy-import GPU analyzers to avoid hard dependency at import time
    try:
        from onesec.analyzer.clip_scorer import ClipScorer
        available["clip"] = ClipScorer
    except ImportError:
        pass
    try:
        from onesec.analyzer.whisper import WhisperAnalyzer
        available["whisper"] = WhisperAnalyzer
    except ImportError:
        pass

    result = []
    for name, cls in available.items():
        ac = config.analyzers.get(name)
        if ac is not None and not ac.enabled:
            continue
        weight = ac.weight if ac else 1.0
        options = ac.options if ac else {}
        try:
            a = cls(weight=weight, **options)
        except TypeError:
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
        """Scan source, run CPU analyzers in parallel + GPU analyzers under semaphore,
        return Top-N selected segments."""
        source = Path(source)
        video_paths = scan(source)
        seg_dur = self._config.segment_duration or self._config.clip_duration

        # Step 1: probe all videos in main process
        video_files: list[VideoFile] = []
        for vp in video_paths:
            try:
                video_files.append(_probe_video(vp))
            except Exception as e:
                warnings.warn(f"Skipping {vp}: {e}")

        cpu_analyzers = [a for a in self._analyzers if not a.uses_gpu and a.is_available()]
        gpu_analyzers = [a for a in self._analyzers if a.uses_gpu and a.is_available()]

        all_segments: list[ScoredSegment] = []

        # Step 2: CPU analyzers — parallel across (video, analyzer) pairs
        if cpu_analyzers and video_files:
            tasks = [
                (analyzer, vf, seg_dur)
                for vf in video_files
                for analyzer in cpu_analyzers
            ]
            with ProcessPoolExecutor(max_workers=self._config.workers) as executor:
                for result in executor.map(_analyze_video_cpu, tasks):
                    all_segments.extend(result)

        # Step 3: GPU analyzers — sequential under global semaphore
        for vf in video_files:
            for analyzer in gpu_analyzers:
                with _GPU_SEMAPHORE:
                    try:
                        all_segments.extend(analyzer.score_segments(vf, seg_dur))
                    except Exception as e:
                        warnings.warn(f"GPU analyzer {analyzer.name} failed on {vf.path}: {e}")

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
                    warnings.warn(str(e))
            if not clips:
                raise RuntimeError("No clips extracted — nothing to compose")
            Composer(transition=self._config.transition).compose(clips, output)
        return output

    def close(self) -> None:
        """Release GPU model memory for all analyzers that support it."""
        for analyzer in self._analyzers:
            if hasattr(analyzer, "close"):
                analyzer.close()

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
            warnings.warn("No segments found — try lowering merge_gap_threshold")
            return Path(output)
        return p.render(segments, output)
