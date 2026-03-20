# src/onesec/analyzer/clip_scorer.py
from __future__ import annotations

import math
from pathlib import Path

from onesec.analyzer.base import Analyzer
from onesec.models import ScoredSegment, VideoFile


class ClipScorer(Analyzer):
    """Level 2: scores segments by CLIP cosine similarity to a text prompt.

    Lazy-loads the ViT-B/32 model on first call. Model is cached per instance.
    Requires: pip install onesec[clip]
    """

    uses_gpu: bool = True

    def __init__(self, weight: float = 1.0, prompt: str = "interesting moment") -> None:
        super().__init__(weight)
        self._prompt = prompt
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "clip"

    def is_available(self) -> bool:
        try:
            import open_clip  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._model = self._model.to(self._device).eval()

    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float,
    ) -> list[ScoredSegment]:
        if not self.is_available():
            return []

        import torch
        import av

        self._ensure_loaded()

        # Pre-encode text prompt (once per call)
        with torch.no_grad():
            text_tokens = self._tokenizer([self._prompt]).to(self._device)
            text_features = self._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        n_segments = max(1, math.ceil(video.duration / segment_duration))
        # seg_idx → best similarity score seen so far
        best_sim: dict[int, float] = {}

        with av.open(str(video.path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(video=0):
                ts = float(frame.pts * stream.time_base)
                seg_idx = min(int(ts / segment_duration), n_segments - 1)
                if seg_idx in best_sim:
                    continue  # use only first frame per segment

                img = frame.to_image()  # PIL Image
                img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    img_features = self._model.encode_image(img_tensor)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    sim = (img_features @ text_features.T).squeeze().item()

                best_sim[seg_idx] = sim

        if not best_sim:
            return []

        raw = [best_sim.get(i, 0.0) for i in range(n_segments)]

        # Normalize to [0.0, 1.0] across this video's segments
        lo, hi = min(raw), max(raw)
        if hi > lo:
            normalized = [(s - lo) / (hi - lo) for s in raw]
        else:
            normalized = [0.0] * n_segments

        return [
            ScoredSegment(
                video_path=video.path,
                start=i * segment_duration,
                end=min((i + 1) * segment_duration, video.duration),
                score=float(normalized[i]),
                analyzer=self.name,
            )
            for i in range(n_segments)
        ]

    def close(self) -> None:
        """Release GPU model memory."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
