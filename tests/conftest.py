"""Shared test fixtures."""
from pathlib import Path
import numpy as np
import pytest
import av


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """Create a 3-second synthetic MP4 with alternating black/white frames."""
    output = tmp_path / "test.mp4"
    container = av.open(str(output), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"

    for i in range(30):  # 3 seconds at 10fps
        color = 255 if (i // 10) % 2 == 0 else 0
        frame_data = np.full((64, 64, 3), color, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame = frame.reformat(format=stream.pix_fmt)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # Remux with faststart so ffprobe can read duration reliably
    import subprocess, shutil
    fast = tmp_path / "test_fast.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(output), "-c", "copy", "-movflags", "+faststart", str(fast)],
        capture_output=True, check=True,
    )
    shutil.move(str(fast), str(output))
    return output
