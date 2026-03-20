<div align="right">
  <a href="README.md">🇺🇸 English</a> | 🇰🇷 한국어
</div>

<div align="center">
  <h1>onesec</h1>
  <p><em>영상에서 가장 흥미로운 순간을 자동으로 찾아 하이라이트 영상으로 만들어 드립니다.</em></p>

  <a href="https://pypi.org/project/onesec/"><img src="https://img.shields.io/pypi/v/onesec" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/onesec" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/ffmpeg-required-orange" alt="FFmpeg 필요">

</div>

---

🎬 수동 편집 불필요 · ⚡ 병렬 분석 · 🔌 확장 가능한 분석기 · 🤖 선택적 ML (CLIP + Whisper)

## 기능

- 🎞 **장면 감지** — 히스토그램 기반 컷 감지, GPU 불필요
- 🔊 **오디오 에너지** — librosa RMS 에너지 + 묵음 구간 감지
- 🌊 **동작 점수** — OpenCV optical flow 기반 움직임 강도 측정
- 🧠 **CLIP 점수** — 사용자 텍스트 프롬프트와의 CLIP ViT-B/32 임베딩 유사도 (선택)
- 🗣 **발화 밀도** — faster-whisper STT로 구간 내 말하는 비율 측정 (선택)
- ⚡ **병렬 파이프라인** — CPU 분석기는 `ProcessPoolExecutor`로 병렬 실행, GPU 분석기는 자동 직렬화
- 🔌 **확장성** — `Analyzer` 서브클래스로 나만의 분석기 추가 가능

## 요구 사항

- Python 3.12+
- `PATH`에 등록된 [FFmpeg](https://ffmpeg.org/download.html)

## 설치

```bash
# 기본 (장면 감지 + 움직임 분석기)
pip install onesec

# 오디오 분석기 포함
pip install onesec[audio]

# CLIP 점수 포함 (GPU 권장)
pip install onesec[clip]

# Whisper 발화 점수 포함 (GPU 권장)
pip install onesec[whisper]

# 전체
pip install "onesec[audio,clip,whisper]"
```

## 빠른 시작

```bash
# 폴더를 분석해 60초 하이라이트 영상 생성
onesec run ./footage -o highlight.mp4

# 렌더링 없이 선택될 구간만 미리 보기
onesec run ./footage --dry-run

# 30초로 제한, 페이드 전환 사용
onesec run ./footage -o highlight.mp4 --max-duration 30 --transition fade

# 분석기와 가중치 직접 지정
onesec run ./footage -o highlight.mp4 --analyzers scene:0.5,audio:1.0,motion:0.3

# 사용 가능한 분석기 목록 보기
onesec analyzers list
```

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-o, --output <PATH>` | — | 출력 영상 경로 (`--dry-run` 없이는 필수) |
| `--clip-duration <초>` | `1.0` | 각 클립의 길이 |
| `--max-duration <초>` | `60.0` | 출력 영상 최대 길이 |
| `--top-n <N>` | 자동 | 포함할 구간 수 |
| `--analyzers <목록>` | 전체 | 콤마 구분, 예: `scene:0.5,audio:1.0` |
| `--transition <종류>` | `cut` | 클립 간 전환: `cut` 또는 `fade` |
| `--dry-run` | 꺼짐 | 렌더링 없이 선택 구간만 출력 |
| `--format <형식>` | `table` | dry-run 출력 형식: `table` 또는 `json` |
| `-c, --config <PATH>` | — | TOML 설정 파일 경로 |

## 설정 파일

반복 실행에는 TOML 설정 파일을 활용하세요:

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

# 한 줄로 끝내기
run("./footage", "highlight.mp4", max_duration=30.0)

# 세밀한 제어
with Pipeline(
    clip_duration=1.0,
    max_duration=60.0,
    transition="fade",
) as p:
    segments = p.analyze("./footage")
    output = p.render(segments, "highlight.mp4")

# 커스텀 분석기 조합
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

## 분석기

| 이름 | 레벨 | GPU | 추가 설치 | 설명 |
|------|------|-----|-----------|------|
| `scene` | 1 | 아니오 | — | 히스토그램 기반 장면 변화 감지 |
| `audio` | 1 | 아니오 | `[audio]` | Librosa RMS 에너지 + VAD |
| `motion` | 1 | 아니오 | — | Optical flow 기반 움직임 강도 |
| `clip` | 2 | 예 | `[clip]` | CLIP ViT-B/32 텍스트 프롬프트 유사도 |
| `whisper` | 2 | 예 | `[whisper]` | faster-whisper 발화 밀도 측정 |

## 커스텀 분석기

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
                score=0.5,  # 여기에 점수 계산 로직 작성
                analyzer=self.name,
            )
            for i in range(n)
        ]

# 사용하기
with Pipeline(analyzers=[MyAnalyzer()]) as p:
    segments = p.analyze("./footage")
    p.render(segments, "output.mp4")
```

## 로드맵

- **v0.1–v0.2** ✅ 핵심 파이프라인, 장면·오디오·동작 분석기, CLI
- **v0.3** ✅ 병렬 파이프라인, CLIP 점수, Whisper 발화 점수
- **v0.4** Rich 진행 UI, `onesec preview` 명령어
- **v0.5** LLM 점수 프로토콜, BGM 싱크

## 기여하기

기여는 언제나 환영합니다! 버그 리포트, 기능 요청, 풀 리퀘스트 모두 소중히 여깁니다.

설계 결정과 아키텍처는 [`docs/superpowers/specs/`](docs/superpowers/specs/)를 참고하세요.

```bash
# 개발 환경 설정
pip install -e ".[dev,audio]"

# 테스트 실행
pytest
```

## 라이선스

[MIT](LICENSE)
