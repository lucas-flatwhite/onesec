from __future__ import annotations

import math
import numpy as np

from onesec.analyzer.base import Analyzer
from onesec.models import ScoredSegment, VideoFile


class MotionAnalyzer(Analyzer):
    """Level 1: detects motion using optical flow magnitude."""

    def __init__(self, weight: float = 1.0, threshold: float = 0.5) -> None:
        super().__init__(weight)
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "motion"

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
        import av
        import cv2

        n_segments = max(1, math.ceil(video.duration / segment_duration))
        frame_scores: list[tuple[float, float]] = []
        prev_gray = None

        with av.open(str(video.path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(video=0):
                ts = float(frame.pts * stream.time_base)
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    score = float(np.mean(mag)) / 10.0
                    frame_scores.append((ts, min(score, 1.0)))
                prev_gray = gray

        segments = []
        for i in range(n_segments):
            start = i * segment_duration
            end = min(start + segment_duration, video.duration)
            window = [d for ts, d in frame_scores if start <= ts < end]
            score = float(np.mean(window)) if window else 0.0
            segments.append(
                ScoredSegment(
                    video_path=video.path,
                    start=start,
                    end=end,
                    score=score,
                    analyzer=self.name,
                )
            )
        return segments
