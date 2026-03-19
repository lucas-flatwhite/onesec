# onesec — 설계 문서

**날짜:** 2026-03-19
**상태:** 승인됨

---

## 개요

`onesec`는 영상 폴더를 스캔해 "의미 있는" 구간을 자동으로 감지하고, 1초 내외의 클립으로 잘라 이어붙이는 오픈소스 하이라이트 편집 도구다.
CLI와 Python 라이브러리 API를 모두 지원하며, macOS + Linux를 타겟 플랫폼으로 한다.

---

## 요구사항 요약

| 항목 | 결정 |
|------|------|
| 구현 범위 | v0.1 ~ v0.5 전체 로드맵 |
| 지원 플랫폼 | macOS + Linux |
| 주요 영상 유형 | 범용 (특정 장르 없음) |
| 인터페이스 | CLI + Python API |
| LLM 분석기 | 추상 Protocol만 정의, 구체 구현은 플러그인으로 |

---

## 아키텍처

```
[Scanner] → [Analyzer Pool] → [Selector] → [Editor] → [Output]
                ↑
         (Analyzer Protocol)
         scene / audio / motion        ← Level 1 (rule-based)
         clip_scorer / whisper         ← Level 2 (ML)
         llm (abstract Protocol)       ← Level 3 (plugin)
```

### 데이터 흐름

1. `Scanner`가 폴더를 탐색해 `VideoFile` 목록 반환
2. 각 `VideoFile`을 `ProcessPoolExecutor`로 병렬 분석 — 활성화된 분석기들이 `list[ScoredSegment]` 반환
3. `Selector`가 분석기별 가중치를 적용해 합산, Top-N 구간 선택
4. `Editor`가 FFmpeg로 클립 추출 → 트랜지션 적용 → 최종 concat

---

## 프로젝트 구조

```
onesec/
├── pyproject.toml
├── onesec.toml.example
├── src/onesec/
│   ├── __init__.py          # 공개 Python API
│   ├── cli.py               # Typer CLI 엔트리포인트
│   ├── config.py            # TOML 설정 로더 (Pydantic)
│   ├── models.py            # VideoFile, ScoredSegment, Config 등
│   ├── scanner.py           # 폴더 → VideoFile 목록
│   ├── analyzer/
│   │   ├── base.py          # Analyzer Protocol 정의
│   │   ├── scene.py         # Level 1: 히스토그램 scene change
│   │   ├── audio.py         # Level 1: librosa 에너지/VAD
│   │   ├── motion.py        # Level 1: optical flow
│   │   ├── clip_scorer.py   # Level 2: CLIP 임베딩 스코어링
│   │   ├── whisper.py       # Level 2: faster-whisper STT
│   │   └── llm.py           # Level 3: 추상 LLM Analyzer Protocol
│   ├── selector.py          # 가중 합산 → Top-N 구간 선택
│   └── editor/
│       ├── extractor.py     # FFmpeg 클립 추출
│       ├── transition.py    # crossfade / cut / dip-to-black
│       └── composer.py      # 최종 영상 조립
└── tests/
```

---

## 핵심 데이터 모델

```python
@dataclass
class ScoredSegment:
    video_path: Path
    start: float      # 초
    end: float        # 초
    score: float      # 0.0 ~ 1.0
    analyzer: str     # 생성한 분석기 이름

@dataclass
class VideoFile:
    path: Path
    duration: float
    fps: float
    has_audio: bool
```

---

## Analyzer Protocol & 플러그인 인터페이스

```python
# analyzer/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Analyzer(Protocol):
    name: str
    weight: float  # 기본 가중치 (0.0 ~ 1.0)

    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float = 1.0,
    ) -> list[ScoredSegment]: ...

    def is_available(self) -> bool: ...
```

### 분석기 레벨별 의존성

| 레벨 | 분석기 | 추가 의존성 | 기본 활성화 |
|------|--------|------------|------------|
| 1 | scene, audio, motion | opencv-python, librosa | ✅ |
| 2 | clip_scorer, whisper | openai-clip, faster-whisper | ❌ (opt-in) |
| 3 | llm | 사용자 구현 플러그인 | ❌ (opt-in) |

### LLM 플러그인 등록 (entry_points)

외부 패키지가 `onesec.analyzers` entry point를 선언하면 자동으로 등록됨:

```toml
# 서드파티 pyproject.toml
[project.entry-points."onesec.analyzers"]
my_claude_analyzer = "my_package:ClaudeAnalyzer"
```

---

## CLI UX

```bash
# 기본 실행
onesec run ./videos -o highlight.mp4

# 분석기 지정 + 가중치
onesec run ./videos -o out.mp4 --analyzers scene:0.3,audio:0.4,motion:0.3

# 클립 길이, 총 길이 조절
onesec run ./videos -o out.mp4 --clip-duration 1.5 --max-duration 60

# 드라이런 (분석 결과만 출력)
onesec run ./videos --dry-run --format json
onesec run ./videos --dry-run --format table

# 설정 파일 기반
onesec run ./videos -c onesec.toml

# 분석기 목록 확인 (플러그인 포함)
onesec analyzers list
```

---

## Python API

```python
import onesec

# 간단한 사용
onesec.run("./videos", output="highlight.mp4")

# 세밀한 제어
from onesec import Pipeline
from onesec.analyzer import SceneAnalyzer, AudioAnalyzer

pipeline = Pipeline(
    analyzers=[
        SceneAnalyzer(weight=0.4),
        AudioAnalyzer(weight=0.6),
    ],
    clip_duration=1.0,
    max_duration=60,
)

segments = pipeline.analyze("./videos")   # list[ScoredSegment]
pipeline.render(segments, "highlight.mp4")
```

---

## 설정 파일 (onesec.toml)

```toml
[output]
clip_duration = 1.0
max_duration  = 60
transition    = "crossfade"  # cut | crossfade | dip-to-black

[analyzers]
scene   = { enabled = true,  weight = 0.3 }
audio   = { enabled = true,  weight = 0.4 }
motion  = { enabled = true,  weight = 0.3 }
clip    = { enabled = false, weight = 0.5 }
whisper = { enabled = false, weight = 0.4, model = "base" }

[device]
prefer = "cuda"  # auto | cpu | cuda | mps
```

---

## Editor 동작 방식

### FFmpeg concat 전략
1. 모든 클립 코덱 확인
2. 동일 코덱 → `-c copy` concat (재인코딩 없음, 빠름)
3. 코덱 불일치 → libx264/aac 재인코딩 후 concat

### 트랜지션
- `cut`: 직접 이어붙이기 (기본값)
- `crossfade`: FFmpeg `xfade` 필터
- `dip-to-black`: FFmpeg `fade` 필터

---

## 에러 처리

| 상황 | 동작 |
|------|------|
| 영상 파일 손상/읽기 불가 | 경고 출력 후 해당 파일 건너뜀 |
| 분석기 의존성 미설치 | `is_available()` 체크 → 설치 안내 메시지 |
| FFmpeg 미설치 | 즉시 에러 + 설치 가이드 링크 |
| 선택된 구간이 0개 | 임계값 낮추기 제안 메시지 |
| GPU 없음 | CPU로 자동 폴백 |

---

## Rich 프로그레스 UI

```
Scanning videos...        ████████████ 12 files found
Analyzing [scene+audio]   ████████░░░░ 8/12 videos
Extracting clips          ██████████░░ 23/28 segments
Composing highlight.mp4   ████████████ Done ✓

Total: 28 clips → 42s highlight from 12 videos (3m 21s)
```

---

## 기술 스택

| 레이어 | 선택 | 이유 |
|--------|------|------|
| Language | Python 3.12+ | ML 생태계 |
| 영상 처리 | FFmpeg (ffmpeg-python) | 업계 표준, 재인코딩·concat 모두 지원 |
| 프레임 분석 | decord / PyAV | GPU 디코딩 지원, OpenCV보다 빠름 |
| 오디오 분석 | librosa / silero-vad | 에너지, VAD, 비트 감지 |
| ML 추론 | ONNX Runtime / openai-clip | 경량화 + 크로스플랫폼 |
| STT | faster-whisper | whisper 대비 4~8배 빠름 |
| CLI | Typer + Rich | 모던 Python CLI 표준 |
| Config | TOML + Pydantic | pyproject.toml과 통합 |
| 패키징 | uv | pip/poetry 대체 |
| 병렬처리 | concurrent.futures | 가벼운 멀티프로세싱 |
| 테스트 | pytest + hypothesis | property-based testing |

---

## 로드맵

| 버전 | 범위 |
|------|------|
| v0.1 | Scanner + SceneAnalyzer + FFmpeg concat + CLI 기본 |
| v0.2 | AudioAnalyzer + MotionAnalyzer + 트랜지션 옵션 |
| v0.3 | CLIP scorer + faster-whisper + GPU 가속 |
| v0.4 | Rich UI + preview 명령 + TOML 설정 파일 |
| v0.5 | LLM Analyzer Protocol + BGM 싱크 + 자막 오버레이 |
