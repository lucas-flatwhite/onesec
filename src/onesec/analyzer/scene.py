from __future__ import annotations

import math
from pathlib import Path

import av
import cv2
import numpy as np

from onesec.analyzer.base import Analyzer
from onesec.models import ScoredSegment, VideoFile


def _histogram_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute normalized histogram difference between two BGR frames (0.0–1.0)."""
    diff = 0.0
    for ch in range(3):
        hist_a = cv2.calcHist([frame_a], [ch], None, [64], [0, 256])
        hist_b = cv2.calcHist([frame_b], [ch], None, [64], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        diff += cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
    return min(diff / 3.0, 1.0)


class SceneAnalyzer(Analyzer):
    """Level 1: detects scene changes via histogram differences between frames."""

    def __init__(self, weight: float = 1.0, threshold: float = 0.3) -> None:
        super().__init__(weight)
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "scene"

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import av    # noqa: F401
            return True
        except ImportError:
            return False

    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float,
    ) -> list[ScoredSegment]:
        n_segments = max(1, math.ceil(video.duration / segment_duration))
        segments: list[ScoredSegment] = []

        with av.open(str(video.path)) as container:
            stream = container.streams.video[0]
            prev_frame: np.ndarray | None = None
            frame_scores: list[tuple[float, float]] = []

            for frame in container.decode(video=0):
                ts = float(frame.pts * stream.time_base)
                img = frame.to_ndarray(format="bgr24")
                if prev_frame is not None:
                    diff = _histogram_diff(prev_frame, img)
                    frame_scores.append((ts, diff))
                prev_frame = img

        for i in range(n_segments):
            start = i * segment_duration
            end = min(start + segment_duration, video.duration)
            window = [d for ts, d in frame_scores if start <= ts < end]
            score = max(window) if window else 0.0
            segments.append(
                ScoredSegment(
                    video_path=video.path,
                    start=start,
                    end=end,
                    score=float(np.clip(score, 0.0, 1.0)),
                    analyzer=self.name,
                )
            )

        return segments
