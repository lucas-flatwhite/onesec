from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class Composer:
    def __init__(self, transition: str = "cut") -> None:
        if transition not in ("cut", "crossfade", "dip-to-black"):
            raise ValueError(f"Unknown transition: {transition!r}")
        self.transition = transition

    def compose(self, clips: list[Path], output: Path) -> Path:
        """Concatenate clips into a single video file."""
        if not clips:
            raise ValueError("No clips to compose")
        if len(clips) == 1:
            import shutil
            shutil.copy2(clips[0], output)
            return output

        if self.transition == "cut":
            return self._concat_cut(clips, output)
        else:
            return self._concat_with_transition(clips, output)

    def _concat_cut(self, clips: list[Path], output: Path) -> Path:
        """Use FFmpeg concat demuxer with re-encoding for compatibility.

        Note: Spec allows -c copy when all codecs match. v0.1 always re-encodes
        for simplicity. Codec-detection optimization is deferred to v0.2.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for clip in clips:
                f.write(f"file '{clip.resolve()}'\n")
            list_file = Path(f.name)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c:v", "libx264", "-c:a", "aac",
                "-movflags", "+faststart",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
        list_file.unlink(missing_ok=True)
        return output

    def _concat_with_transition(self, clips: list[Path], output: Path) -> Path:
        """Apply xfade/fade transitions between clips via FFmpeg filter graph."""
        import av as _av

        xfade_dur = 0.3
        n = len(clips)

        clip_durations: list[float] = []
        for c in clips:
            with _av.open(str(c)) as cont:
                clip_durations.append(float(cont.duration / _av.time_base))

        inputs = []
        for clip in clips:
            inputs.extend(["-i", str(clip)])

        filter_parts = []
        prev = "[0:v]"
        cumulative = 0.0
        for i in range(1, n):
            cumulative += clip_durations[i - 1]
            offset = cumulative - xfade_dur
            out = f"[v{i}]" if i < n - 1 else "[vout]"
            filter_parts.append(
                f"{prev}[{i}:v]xfade=transition=fade:duration={xfade_dur}:offset={offset:.3f}{out}"
            )
            prev = f"[v{i}]"

        filtergraph = ";".join(filter_parts)

        subprocess.run(
            ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filtergraph,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-movflags", "+faststart",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
        return output
