from __future__ import annotations

from abc import ABC, abstractmethod

from onesec.models import ScoredSegment, VideoFile


class Analyzer(ABC):
    """Base class for all segment scorers.

    Subclasses must implement: name (property), score_segments, is_available.
    """

    uses_gpu: bool = False

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier — must match Config.analyzers key."""
        ...

    @abstractmethod
    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float,
    ) -> list[ScoredSegment]:
        """Score each segment_duration window of the video.

        Rules:
        - If video.duration < segment_duration, treat the whole video as one segment.
        - If video.has_audio is False and analyzer needs audio, return [].
        - Scores must be in [0.0, 1.0].
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all required dependencies are installed."""
        ...
