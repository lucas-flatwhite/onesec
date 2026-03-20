<div align="right">
  🇺🇸 English | <a href="README.ko.md">🇰🇷 한국어</a>
</div>

<div align="center">
  <h1>onesec</h1>
  <p><em>Automatically find the best moments in your videos and stitch them into a highlight reel.</em></p>

  <a href="https://pypi.org/project/onesec/"><img src="https://img.shields.io/pypi/v/onesec" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/onesec" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/ffmpeg-required-orange" alt="FFmpeg required">

</div>

---

🎬 Zero manual editing · ⚡ Parallel analysis · 🔌 Plugin-friendly analyzers · 🤖 Optional ML (CLIP + Whisper)

## Features

- 🎞 **Scene detection** — histogram-based cut detection, no GPU required
- 🔊 **Audio energy** — librosa RMS energy + silence detection
- 🌊 **Motion scoring** — optical flow magnitude via OpenCV
- 🧠 **CLIP scoring** — CLIP ViT-B/32 embedding similarity to a custom text prompt (optional)
- 🗣 **Speech density** — faster-whisper STT scores segments by how much speech they contain (optional)
- ⚡ **Parallel pipeline** — CPU analyzers run in parallel via `ProcessPoolExecutor`; GPU analyzers are serialized automatically
- 🔌 **Extensible** — bring your own `Analyzer` subclass

## Requirements

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/download.html) on your `PATH`

## Installation

```bash
# Core (scene + motion analyzers)
pip install onesec

# With audio analyzer
pip install onesec[audio]

# With CLIP scoring (GPU recommended)
pip install onesec[clip]

# With Whisper speech scoring (GPU recommended)
pip install onesec[whisper]

# Everything
pip install "onesec[audio,clip,whisper]"
```

## Quick Start

```bash
# Analyze a folder and render a 60-second highlight reel
onesec run ./footage -o highlight.mp4

# Preview which segments would be selected (no render)
onesec run ./footage --dry-run

# Limit to 30 seconds, use fade transitions
onesec run ./footage -o highlight.mp4 --max-duration 30 --transition fade

# Use specific analyzers with custom weights
onesec run ./footage -o highlight.mp4 --analyzers scene:0.5,audio:1.0,motion:0.3

# See all available analyzers
onesec analyzers list
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <PATH>` | — | Output video path (required unless `--dry-run`) |
| `--clip-duration <SEC>` | `1.0` | Duration of each extracted clip |
| `--max-duration <SEC>` | `60.0` | Maximum total output duration |
| `--top-n <N>` | auto | Number of segments to include |
| `--analyzers <LIST>` | all available | Comma-separated list, e.g. `scene:0.5,audio:1.0` |
| `--transition <TYPE>` | `cut` | Transition between clips: `cut` or `fade` |
| `--dry-run` | off | Print selected segments without rendering |
| `--format <FMT>` | `table` | Dry-run output format: `table` or `json` |
| `-c, --config <PATH>` | — | Path to a TOML config file |

## Config File

For repeatable runs, use a TOML config file:

```toml
[output]
clip_duration = 1.0
max_duration  = 45.0
transition    = "fade"

[parallelism]
workers = 4

[analyzers.scene]
weight = 0.5

[analyzers.audio]
weight = 1.5

[analyzers.clip]
weight = 2.0
options = { prompt = "exciting action moment" }
```

```bash
onesec run ./footage -o highlight.mp4 -c onesec.toml
```

## Python API

```python
from onesec import Pipeline, run

# One-liner
run("./footage", "highlight.mp4", max_duration=30.0)

# Full control
with Pipeline(
    clip_duration=1.0,
    max_duration=60.0,
    transition="fade",
) as p:
    segments = p.analyze("./footage")
    output = p.render(segments, "highlight.mp4")

# Custom analyzers
from onesec.analyzer.scene import SceneAnalyzer
from onesec.analyzer.clip_scorer import ClipScorer

with Pipeline(
    analyzers=[
        SceneAnalyzer(weight=0.5),
        ClipScorer(weight=2.0, prompt="dramatic landscape"),
    ],
    max_duration=30.0,
) as p:
    segments = p.analyze("./footage")
    p.render(segments, "highlight.mp4")
```

## Analyzers

| Name | Level | GPU | Extra | Description |
|------|-------|-----|-------|-------------|
| `scene` | 1 | No | — | Histogram-based scene change detection |
| `audio` | 1 | No | `[audio]` | Librosa RMS energy + VAD |
| `motion` | 1 | No | — | Optical flow magnitude |
| `clip` | 2 | Yes | `[clip]` | CLIP ViT-B/32 cosine similarity to a text prompt |
| `whisper` | 2 | Yes | `[whisper]` | faster-whisper speech density scoring |

## Custom Analyzer

```python
from onesec.analyzer.base import Analyzer
from onesec.models import ScoredSegment, VideoFile

class MyAnalyzer(Analyzer):
    uses_gpu = False

    @property
    def name(self) -> str:
        return "my_analyzer"

    def score_segments(self, video: VideoFile, segment_duration: float) -> list[ScoredSegment]:
        n = max(1, int(video.duration / segment_duration))
        return [
            ScoredSegment(
                video_path=video.path,
                start=i * segment_duration,
                end=min((i + 1) * segment_duration, video.duration),
                score=0.5,  # your scoring logic here
                analyzer=self.name,
            )
            for i in range(n)
        ]

# Use it
with Pipeline(analyzers=[MyAnalyzer()]) as p:
    segments = p.analyze("./footage")
    p.render(segments, "output.mp4")
```

## Roadmap

- **v0.1–v0.2** ✅ Core pipeline, scene/audio/motion analyzers, CLI
- **v0.3** ✅ Parallel pipeline, CLIP scoring, Whisper speech scoring
- **v0.4** Rich progress UI, `onesec preview` command
- **v0.5** LLM scoring protocol, BGM sync

## Contributing

Contributions are welcome! Bug reports, feature requests, and pull requests are all appreciated.

For design decisions and architecture, see [`docs/superpowers/specs/`](docs/superpowers/specs/).

```bash
# Setup
pip install -e ".[dev,audio]"

# Test
pytest
```

## License

[MIT](LICENSE)
