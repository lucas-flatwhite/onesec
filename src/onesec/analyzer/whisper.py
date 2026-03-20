# src/onesec/analyzer/whisper.py
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from onesec.analyzer.base import Analyzer
from onesec.models import ScoredSegment, VideoFile


class WhisperAnalyzer(Analyzer):
    """Level 2: scores segments by speech density using faster-whisper STT.

    Lazy-loads the Whisper model on first call. Model is cached per instance.
    Score = fraction of the segment window covered by transcribed speech (0.0–1.0).
    Requires: pip install onesec[whisper]
    """

    uses_gpu: bool = True

    def __init__(self, weight: float = 1.0, model: str = "base") -> None:
        super().__init__(weight)
        self._model_size = model
        self._model = None

    @property
    def name(self) -> str:
        return "whisper"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            self._model_size, device=device, compute_type=compute_type
        )

    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float,
    ) -> list[ScoredSegment]:
        if not video.has_audio:
            return []
        if not self.is_available():
            return []

        self._ensure_loaded()

        n_segs = max(1, math.ceil(video.duration / segment_duration))
        speech_time = [0.0] * n_segs

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)

        try:
            # Extract mono 16kHz WAV for Whisper
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video.path),
                    "-ac", "1", "-ar", "16000",
                    str(wav_path),
                ],
                capture_output=True,
                check=True,
            )

            segments_gen, _ = self._model.transcribe(str(wav_path), beam_size=1)
            for seg in segments_gen:
                for i in range(n_segs):
                    win_start = i * segment_duration
                    win_end = min(win_start + segment_duration, video.duration)
                    overlap = min(seg.end, win_end) - max(seg.start, win_start)
                    if overlap > 0:
                        speech_time[i] += overlap
        finally:
            wav_path.unlink(missing_ok=True)

        return [
            ScoredSegment(
                video_path=video.path,
                start=i * segment_duration,
                end=min((i + 1) * segment_duration, video.duration),
                score=min(speech_time[i] / segment_duration, 1.0),
                analyzer=self.name,
            )
            for i in range(n_segs)
        ]

    def close(self) -> None:
        """Release Whisper model memory."""
        self._model = None
