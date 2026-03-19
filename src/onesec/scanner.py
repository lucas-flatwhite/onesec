from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


def scan(source: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted list of video file paths found in source folder (or single file)."""
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in source.glob(pattern)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
