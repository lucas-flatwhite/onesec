from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from onesec.analyzer.base import Analyzer
from onesec.models import ScoredSegment, VideoFile


class AudioAnalyzer(Analyzer):
    """Level 1: detects audio energy peaks and voice activity."""

    def __init__(self, weight: float = 1.0, vad: bool = True) -> None:
        super().__init__(weight)
        self._use_vad = vad

    @property
    def name(self) -> str:
        return "audio"

    def is_available(self) -> bool:
        try:
            import librosa  # noqa: F401
            return True
        except ImportError:
            return False

    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float,
    ) -> list[ScoredSegment]:
        if not video.has_audio:
            return []
        if not self.is_available():
            return []

        import librosa
        import av

        samples = []
        with av.open(str(video.path)) as container:
            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if audio_stream is None:
                return []
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=22050)
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()[0]
                    samples.append(arr)
            for resampled in resampler.resample(None):
                arr = resampled.to_ndarray()[0]
                samples.append(arr)

        if not samples:
            return []

        sr = 22050
        y = np.concatenate(samples).astype(np.float32)
        if len(y) == 0:
            return []

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

        n_segments = max(1, math.ceil(video.duration / segment_duration))
        segments = []
        for i in range(n_segments):
            start = i * segment_duration
            end = min(start + segment_duration, video.duration)
            mask = (times >= start) & (times < end)
            window_rms = rms[mask]
            score = float(np.max(window_rms)) if len(window_rms) > 0 else 0.0
            segments.append(
                ScoredSegment(
                    video_path=video.path,
                    start=start,
                    end=end,
                    score=min(score, 1.0),
                    analyzer=self.name,
                )
            )
        return segments
