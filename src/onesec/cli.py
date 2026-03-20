from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from onesec import Pipeline
from onesec.analyzer.base import Analyzer
from onesec.analyzer.audio import AudioAnalyzer
from onesec.analyzer.motion import MotionAnalyzer
from onesec.analyzer.scene import SceneAnalyzer
from onesec.config import load_config

try:
    from onesec.analyzer.clip_scorer import ClipScorer
    _HAVE_CLIP = True
except ImportError:
    _HAVE_CLIP = False

try:
    from onesec.analyzer.whisper import WhisperAnalyzer
    _HAVE_WHISPER = True
except ImportError:
    _HAVE_WHISPER = False

app = typer.Typer(help="onesec — automatic video highlight editor")
console = Console()

_BUILTIN_ANALYZERS: dict[str, type[Analyzer]] = {
    "scene": SceneAnalyzer,
    "audio": AudioAnalyzer,
    "motion": MotionAnalyzer,
    **({"clip": ClipScorer} if _HAVE_CLIP else {}),
    **({"whisper": WhisperAnalyzer} if _HAVE_WHISPER else {}),
}


def _parse_analyzers(value: str) -> list[Analyzer] | None:
    """Parse 'scene:0.3,audio:0.4' into Analyzer instances."""
    result = []
    for part in value.split(","):
        part = part.strip()
        if ":" in part:
            name, weight_str = part.split(":", 1)
            weight = float(weight_str)
        else:
            name, weight = part, 1.0
        if name in _BUILTIN_ANALYZERS:
            result.append(_BUILTIN_ANALYZERS[name](weight=weight))
        else:
            console.print(f"[yellow]Warning:[/] Unknown analyzer '{name}' — skipping")
    return result or None


@app.command()
def run(
    source: Path = typer.Argument(..., help="Folder or file to analyze"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output video path"),
    clip_duration: float = typer.Option(1.0, "--clip-duration"),
    max_duration: float = typer.Option(60.0, "--max-duration"),
    top_n: Optional[int] = typer.Option(None, "--top-n"),
    analyzers: Optional[str] = typer.Option(None, "--analyzers"),
    transition: str = typer.Option("cut", "--transition"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format: str = typer.Option("table", "--format"),
    config: Optional[Path] = typer.Option(None, "-c", "--config"),
) -> None:
    """Analyze a video folder and create a highlight montage."""
    parsed_analyzers = _parse_analyzers(analyzers) if analyzers else None

    if config:
        cfg = load_config(config)
        pipeline = Pipeline(analyzers=parsed_analyzers, config=cfg)
    else:
        pipeline = Pipeline(
            analyzers=parsed_analyzers,
            clip_duration=clip_duration,
            max_duration=max_duration,
            top_n=top_n,
            transition=transition,
        )

    with console.status("Analyzing videos..."):
        segments = pipeline.analyze(source)

    if not segments:
        console.print("[yellow]No segments found.[/] Try lowering --merge-gap-threshold.")
        raise typer.Exit(0)

    if dry_run:
        if format == "json":
            data = [
                {
                    "video_path": str(s.video_path),
                    "start": s.start,
                    "end": s.end,
                    "score": s.score,
                    "analyzer": s.analyzer,
                }
                for s in segments
            ]
            typer.echo(json.dumps(data, indent=2))
        else:
            table = Table(title="Selected Segments")
            for col in ["video", "start", "end", "score", "analyzer"]:
                table.add_column(col)
            for s in segments:
                table.add_row(
                    s.video_path.name,
                    f"{s.start:.2f}",
                    f"{s.end:.2f}",
                    f"{s.score:.3f}",
                    s.analyzer,
                )
            console.print(table)
        return

    if output is None:
        console.print("[red]Error:[/] -o/--output is required unless --dry-run is set.")
        raise typer.Exit(1)

    with console.status("Rendering..."):
        pipeline.render(segments, output)

    console.print(f"[green]Done![/] {len(segments)} clips → {output}")


@app.command(name="list")
def analyzers_list() -> None:
    """List available analyzers."""
    table = Table(title="Available Analyzers")
    for col in ["name", "level", "gpu", "available", "description"]:
        table.add_column(col)
    table.add_row("scene",  "1", "No", "✓", "Histogram-based scene change")
    table.add_row("audio",  "1", "No", "?" if not AudioAnalyzer().is_available() else "✓", "Librosa energy + VAD")
    table.add_row("motion", "1", "No", "✓", "Optical flow magnitude")
    clip_avail = "✓" if (_HAVE_CLIP and ClipScorer().is_available()) else "✗"
    table.add_row("clip",   "2", "Yes", clip_avail, "CLIP embedding scorer (pip install onesec[clip])")
    wh_avail = "✓" if (_HAVE_WHISPER and WhisperAnalyzer().is_available()) else "✗"
    table.add_row("whisper","2", "Yes", wh_avail,   "faster-whisper STT (pip install onesec[whisper])")
    console.print(table)


analyzers_app = typer.Typer()
analyzers_app.command("list")(analyzers_list)
app.add_typer(analyzers_app, name="analyzers")
