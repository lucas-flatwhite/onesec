from __future__ import annotations

import subprocess
from pathlib import Path

from onesec.models import ScoredSegment

MIN_CLIP_DURATION = 0.1  # seconds


def _probe_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


class Extractor:
    def __init__(self, clip_duration: float = 1.0) -> None:
        self.clip_duration = clip_duration

    def extract(self, segment: ScoredSegment, output_dir: Path) -> Path:
        """Extract a clip_duration clip from segment.start and save to output_dir."""
        video_duration = _probe_duration(segment.video_path)
        max_start = max(0.0, video_duration - self.clip_duration)
        extract_start = min(segment.start, max_start)
        extract_duration = min(self.clip_duration, video_duration - extract_start)

        if extract_duration < MIN_CLIP_DURATION:
            raise ValueError(
                f"Clip too short ({extract_duration:.3f}s) for segment {segment}"
            )

        stem = f"{segment.video_path.stem}_{extract_start:.3f}"
        output = output_dir / f"{stem}.mp4"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(extract_start),
                "-i", str(segment.video_path),
                "-t", str(extract_duration),
                "-c:v", "libx264", "-c:a", "aac",
                "-movflags", "+faststart",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
        return output
