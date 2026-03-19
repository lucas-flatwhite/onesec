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
         (Analyzer ABC)
         scene / audio / motion          ← Level 1 (rule-based)
         clip (clip_scorer.py) / whisper ← Level 2 (ML)
         llm (abstract ABC)              ← Level 3 (plugin)
```

### 데이터 흐름

1. `Scanner`가 폴더를 탐색해 `VideoFile` 목록 반환
2. 각 `VideoFile`을 병렬 분석:
   - **CPU 분석기** (scene, audio, motion): `ProcessPoolExecutor`로 병렬 실행
     - 워커 수 기본값: CPU 코어 수. `Config.workers`로 재정의 가능
   - **GPU 분석기** (`clip`, `whisper`, llm 플러그인 등 `uses_gpu=True`):
     전역 `threading.Semaphore(1)` Lock 하에 순차 실행 (CUDA/MPS 공유 충돌 방지)
3. 활성화된 모든 분석기의 결과(`list[ScoredSegment]`)를 수집
4. `Selector`가 분析기별 가중치를 적용해 구간별 점수를 합산, 인접·겹치는 구간을 병합한 뒤 Top-N 선택
5. `Editor`가 FFmpeg로 클립 추출 → 트랜지션 적용 → 최종 concat

---

## 프로젝트 구조

```
onesec/
├── pyproject.toml
├── onesec.toml.example
├── src/onesec/
│   ├── __init__.py          # 공개 Python API (run, Pipeline)
│   ├── cli.py               # Typer CLI 엔트리포인트
│   ├── config.py            # TOML 설정 로더 (Pydantic)
│   ├── models.py            # VideoFile, ScoredSegment, Config
│   ├── scanner.py           # 폴더 → VideoFile 목록
│   ├── analyzer/
│   │   ├── base.py          # Analyzer ABC 정의
│   │   ├── scene.py         # Level 1: 히스토그램 scene change
│   │   ├── audio.py         # Level 1: librosa 에너지/VAD
│   │   ├── motion.py        # Level 1: optical flow
│   │   ├── clip_scorer.py   # Level 2: CLIP 임베딩 스코어링 (Analyzer.name = "clip")
│   │   ├── whisper.py       # Level 2: faster-whisper STT
│   │   └── llm.py           # Level 3: 추상 LLM Analyzer ABC
│   ├── selector.py          # 가중 합산 → 인접 구간 병합 → Top-N 선택
│   └── editor/
│       ├── extractor.py     # FFmpeg 클립 추출
│       ├── transition.py    # crossfade / cut / dip-to-black
│       └── composer.py      # 최종 영상 조립
└── tests/
```

---

## 핵심 데이터 모델

```python
# models.py

@dataclass
class VideoFile:
    path: Path
    duration: float   # 초
    fps: float
    has_audio: bool

@dataclass
class ScoredSegment:
    video_path: Path
    start: float      # 초 (분析 단위 구간의 시작)
    end: float        # 초 (분析 단위 구간의 끝)
    score: float      # 0.0 ~ 1.0 (각 분析기가 반환하는 원시 점수)
    analyzer: str     # 생성한 분析기 이름 — Analyzer.name 값과 일치해야 함

class AnalyzerConfig(BaseModel):
    enabled: bool = True
    weight: float = 1.0
    options: dict[str, Any] = Field(default_factory=dict)

class Config(BaseModel):
    # [output]
    clip_duration: float = 1.0            # 출력 클립 길이 (초)
    segment_duration: float | None = None # 분析 창 크기 (None이면 clip_duration과 동일)
                                          # segment_duration < clip_duration도 허용:
                                          # 예) 0.5초 창으로 분析 후 1.0초 클립 추출
    max_duration: float = 60.0            # 최종 영상 최대 길이 (초)
    top_n: int | None = None              # 선택할 클립 수 (None이면 max_duration 기준)
    transition: str = "cut"               # cut | crossfade | dip-to-black
    merge_gap_threshold: float = 0.5      # 두 구간 사이 허용 최대 gap (초) — 이하면 병합

    # [device]
    device: str = "auto"                  # auto | cpu | cuda | mps

    # [parallelism]
    workers: int | None = None            # CPU 분析기 ProcessPoolExecutor 워커 수
                                          # None이면 os.cpu_count() 사용

    # [analyzers] — 분析기 name을 키로 사용
    analyzers: dict[str, AnalyzerConfig] = Field(default_factory=dict)
```

---

## Analyzer 인터페이스

플러그인 유효성 검사의 신뢰성을 위해 `Protocol` 대신 **ABC(Abstract Base Class)**를 사용한다.
(`@runtime_checkable Protocol`은 메서드만 검사하고 `name`, `weight` 같은 데이터 속성은 검사하지 않으므로 플러그인 검증에 부적합하다.)
`name`은 추상 프로퍼티로 선언해 서브클래스가 반드시 구현하도록 강제한다.

```python
# analyzer/base.py
from abc import ABC, abstractmethod

class Analyzer(ABC):
    uses_gpu: bool = False  # GPU 사용 여부 — 병렬 전략 결정에 사용

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @property
    @abstractmethod
    def name(self) -> str:
        """분析기 식별자. Config.analyzers 딕셔너리 키 및 ScoredSegment.analyzer 값과 일치해야 함."""
        ...

    @abstractmethod
    def score_segments(
        self,
        video: VideoFile,
        segment_duration: float,  # 분析 창 크기 (초); clip_duration과 독립적
    ) -> list[ScoredSegment]:
        """영상을 segment_duration 단위로 나눠 각 구간의 점수 반환.

        - segment_duration: 분析 단위 창 크기. clip_duration(출력 클립 길이)과 별개.
          segment_duration < clip_duration도 유효한 사용 패턴임.
        - video.duration < segment_duration인 경우: 영상 전체를 단일 구간으로 처리.
        - video.has_audio == False인 오디오 分析기: 빈 리스트 반환.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """필요한 의존성(패키지, 모델 파일 등)이 설치/접근 가능한지 확인."""
        ...
```

### 분析기 레벨별 의존성

분析기 `name` 값은 `Config.analyzers` 키 및 `ScoredSegment.analyzer` 값과 동일해야 한다.

| 레벨 | `name` (Config 키) | 추가 의존성 | 기본 활성화 | GPU |
|------|--------|------------|------------|-----|
| 1 | `scene` | opencv-python, decord | ✅ | ❌ |
| 1 | `audio` | librosa, silero-vad | ✅ | ❌ |
| 1 | `motion` | opencv-python, decord | ✅ | ❌ |
| 2 | `clip` | openai-clip, torch | ❌ (opt-in) | ✅ |
| 2 | `whisper` | faster-whisper | ❌ (opt-in) | ✅ |
| 3 | 플러그인이 정의 | 사용자 구현 플러그인 | ❌ (opt-in) | 구현체에 따라 다름 |

### 분析기별 지원 options 키

| 분析기 | options 키 | 타입 | 기본값 | 설명 |
|--------|-----------|------|--------|------|
| `whisper` | `model` | str | `"base"` | Whisper 모델 크기 (tiny/base/small/medium/large) |
| `clip` | `prompt` | str | `"interesting moment"` | CLIP 유사도 비교 텍스트 프롬프트 |
| `audio` | `vad` | bool | `true` | silero-vad 사용 여부 (false면 librosa 에너지만 사용) |
| `scene` | `threshold` | float | `0.3` | 히스토그램 차이 임계값 |
| `motion` | `threshold` | float | `0.5` | Optical flow 크기 임계값 |

플러그인 분析기는 자신이 지원하는 options 키를 문서화할 책임이 있다. 알 수 없는 키는 무시한다.

### LLM 플러그인 등록 (entry_points)

외부 패키지가 `onesec.analyzers` entry point를 선언하면 자동으로 등록됨:

```toml
# 서드파티 pyproject.toml
[project.entry-points."onesec.analyzers"]
my_claude_analyzer = "my_package:ClaudeAnalyzer"
```

**플러그인 로딩 시점 및 실패 처리:**
- 로딩 시점: `Pipeline` 인스턴스 생성 시 (CLI에서는 명령 파싱 단계) entry_point를 열거하고 임포트
- 임포트 오류(패키지 미설치 등): 경고 로그 출력 후 해당 플러그인 건너뜀 — 나머지 분析기는 정상 동작
- ABC 미구현(추상 메서드 누락): `TypeError` 발생 → 경고 출력 후 해당 플러그인 건너뜀
- 유효하게 로드된 플러그인은 `onesec analyzers list` 출력에 포함됨

---

## Selector 인터페이스

```python
# selector.py

class Selector:
    """여러 분析기 결과를 합산해 최종 구간 목록을 선택."""

    def select(
        self,
        segments: list[ScoredSegment],   # 모든 분析기의 원시 결과 (중복·겹침 포함)
        config: Config,
    ) -> list[ScoredSegment]:
        """
        단계 1. [가중치 정규화] — 병합 전에 수행
           w_norm[a] = w[a] / sum(w) for each active analyzer a
           각 ScoredSegment에 대해 내부 임시 변수 ws[seg] = seg.score * w_norm[seg.analyzer]를 계산한다.
           (ScoredSegment.score는 원본 그대로 유지 — dry-run 출력에 원시 점수 표시)

        단계 2. [구간 병합] — 단계 1 완료 후 수행
           동일 video_path 내에서 시간 축으로 정렬 후:
             gap = start_j - end_i  (i, j는 인접 구간)
           gap <= merge_gap_threshold 이면 두 구간 병합
             (gap <= 0이면 실제 겹침; 0 < gap <= threshold이면 인접 구간)
           병합된 구간:
             start    = min(start_i...)
             end      = max(end_j...)
             score    = sum( max(ws[s] for s in group if s.analyzer == a) for each analyzer a )
                        → 결과 score는 1.0을 초과할 수 있음 (상대 비교에만 사용)
             analyzer = "merged"  ← 여러 분析기가 기여한 구간을 나타내는 sentinel 값
                        단일 분析기만 기여한 구간은 해당 분析기의 name을 그대로 유지

        단계 3. [Top-N 선택]
           top_n이 설정된 경우: merged_score 내림차순으로 top_n개 선택
             단, top_n 선택이 완료된 후 누적 clip_duration이 max_duration을 초과해도 허용
             (max_duration은 top_n이 None일 때만 상한선으로 적용됨)
           top_n이 None인 경우: 내림차순으로 누적 clip_duration <= max_duration인 동안 선택

        단계 4. 선택된 구간을 (video_path, start) 기준으로 정렬해 반환
        """
        ...
```

---

## Pipeline (Python API 핵심 클래스)

**`analyzers` vs `config.analyzers` 충돌 규칙:**
- `Pipeline(analyzers=[...])`: 전달된 `Analyzer` 인스턴스 목록을 그대로 사용. 각 인스턴스의 `self.weight`가 Selector 가중치 합산에 사용됨.
- `Pipeline(config=config)` (`analyzers=None`): `config.analyzers`의 enabled 분析기만 기본 인스턴스로 생성, `config.analyzers[name].weight`로 인스턴스의 `self.weight` 설정.
- `Pipeline(analyzers=[...], config=config)`: `analyzers` 인스턴스 목록이 우선. 각 인스턴스의 `self.weight` 사용. `config.analyzers`의 `enabled`/`weight`는 완전히 무시됨.

`analyze()`는 Selector를 포함한 전체 파이프라인을 실행해 **이미 선택이 완료된** `list[ScoredSegment]`를 반환한다.
`render()`는 해당 결과를 그대로 입력으로 받아 영상을 조립한다. 사용자가 `analyze()` 결과를 직접 필터링한 후 `render()`에 전달하는 것도 허용된다.

```python
# __init__.py

DEFAULT_ANALYZERS = ["scene", "audio", "motion"]  # 기본 활성화 분析기 이름 목록

class Pipeline:
    def __init__(
        self,
        analyzers: list[Analyzer] | None = None,
        # None이면 DEFAULT_ANALYZERS의 기본 인스턴스를 사용
        *,
        config: Config | None = None,
        # config가 None일 때 아래 인수로 Config 자동 생성
        clip_duration: float = 1.0,
        segment_duration: float | None = None,
        max_duration: float = 60.0,
        top_n: int | None = None,
        transition: str = "cut",
        device: str = "auto",
        merge_gap_threshold: float = 0.5,
        workers: int | None = None,
    ): ...

    def analyze(self, source: str | Path) -> list[ScoredSegment]:
        """폴더 또는 단일 영상 파일을 분析 → Selector 실행 → Top-N 구간 반환.
        반환값은 이미 선택이 완료된 구간 목록이다.
        """
        ...

    def render(
        self,
        segments: list[ScoredSegment],
        output: str | Path,
    ) -> Path:
        """analyze()가 반환한 (또는 사용자가 필터링한) 구간을 잘라 이어붙인 영상 생성.
        각 구간에서 clip_duration 길이만큼 추출 — Editor 섹션의 'clip_duration 추출 규칙' 따름.
        """
        ...

    def __enter__(self) -> "Pipeline": ...
    def __exit__(self, *args) -> None: ...
    def close(self) -> None:
        """GPU 모델 메모리 해제."""
        ...

def run(
    source: str | Path,
    output: str | Path,
    analyzers: list[Analyzer] | None = None,
    **kwargs,  # Pipeline 생성자의 Config 관련 키워드 인수와 동일
) -> Path:
    """편의 함수: Pipeline(analyzers, **kwargs)을 구성하고 analyze → render를 한 번에 실행."""
    ...
```

---

## CLI UX

### --analyzers 파싱

`--analyzers scene:0.3,audio:0.4,motion:0.3` 형식은 Typer 커스텀 파서로 처리:
- `name:weight` 쌍을 쉼표로 분리
- 가중치를 명시하지 않으면 기본값(`1.0`) 사용
- 가중치 합이 1.0을 초과해도 오류 없음 — Selector 내부에서 정규화
- **options 키 전달 미지원:** `--analyzers`는 이름:가중치만 허용. options(예: Whisper 모델 크기)는 `onesec.toml`을 통해서만 설정 가능

```bash
# 기본 실행 (-o 필수)
onesec run ./videos -o highlight.mp4

# 분析기 지정 + 가중치
onesec run ./videos -o out.mp4 --analyzers scene:0.3,audio:0.4,motion:0.3

# 클립 길이, 최대 길이, 클립 수 조절
onesec run ./videos -o out.mp4 --clip-duration 1.5 --max-duration 60
onesec run ./videos -o out.mp4 --top-n 20

# 드라이런 — 분析 결과만 출력, 렌더링 없음
# --dry-run 모드에서 -o 인수는 선택적 (지정해도 무시됨)
# --format json: list[ScoredSegment] JSON 배열 (video_path, start, end, score, analyzer 필드)
# --format table: Rich 테이블 (video | start | end | score | analyzer 컬럼)
onesec run ./videos --dry-run --format json
onesec run ./videos --dry-run --format table

# 설정 파일 기반
onesec run ./videos -c onesec.toml

# 분析기 목록 확인 (플러그인 포함)
# 출력 컬럼: name | level | gpu | available | description
onesec analyzers list

# onesec preview 명령은 v0.4에서 추가됨.
# v0.1~v0.3에서는 이 명령이 존재하지 않음 (Typer 미등록 명령 오류 반환).
# 상세 명세는 v0.4 설계 문서에서 별도 정의. 현재 문서 범위 밖.
```

---

## Python API 예시

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

segments = pipeline.analyze("./videos")   # list[ScoredSegment] (이미 Top-N 선택 완료)
pipeline.render(segments, "highlight.mp4")

# context manager 사용 (GPU 모델 메모리 자동 해제)
with Pipeline(clip_duration=1.5) as p:
    p.render(p.analyze("./videos"), "out.mp4")
```

---

## 설정 파일 (onesec.toml)

TOML 키는 `Config` 필드명과 1:1 대응한다.

```toml
[output]
clip_duration           = 1.0
segment_duration        = null   # null이면 clip_duration과 동일
max_duration            = 60
top_n                   = null   # null이면 max_duration 기준
transition              = "cut"  # cut | crossfade | dip-to-black  ← 기본값: cut
merge_gap_threshold     = 0.5    # 두 구간 사이 최대 허용 gap (초)

[device]
device = "auto"  # auto | cpu | cuda | mps

[parallelism]
workers = null  # null이면 os.cpu_count()

[analyzers]
scene   = { enabled = true,  weight = 0.3 }
audio   = { enabled = true,  weight = 0.4 }
motion  = { enabled = true,  weight = 0.3 }
clip    = { enabled = false, weight = 0.5 }
whisper = { enabled = false, weight = 0.4, options = { model = "base" } }
```

---

## Editor 동작 방식

### clip_duration 추출 규칙

`Extractor`는 `ScoredSegment`에서 다음 규칙으로 클립을 추출한다:

- `extract_start = segment.start`
- `extract_start`가 `video.duration - clip_duration`을 초과하면 `video.duration - clip_duration`으로 클램핑
- `segment_len = segment.end - segment.start`
  - `segment_len >= clip_duration`: `extract_start`부터 `clip_duration`만큼 추출 (고정 길이)
  - `segment_len < clip_duration`: `extract_start`부터 `min(clip_duration, video.duration - extract_start)`만큼 추출
    → 짧은 클립은 그대로 concat에 포함 (padding 없음, drop 없음)
    → 추출 길이 < 0.1초이면 경고 후 skip

**참고:** `segment_duration < clip_duration`으로 설정하면 모든 구간이 `segment_len < clip_duration` 케이스에 해당한다. 이는 의도된 사용 패턴이다 (세밀한 분析 창 + 넉넉한 출력 클립).

### FFmpeg concat 전략

트랜지션 여부에 따라 전략이 달라진다:

```
transition == "cut"?
  모든 클립 코덱 동일?
    YES → -c copy concat (재인코딩 없음, 가장 빠름)
    NO  → libx264/aac 재인코딩 후 concat

transition == "crossfade" | "dip-to-black"?
  항상 libx264/aac 재인코딩 필요 (FFmpeg 필터 그래프 사용)
  → xfade / fade 필터 적용 후 concat
```

**이유:** `-c copy`는 FFmpeg 필터 그래프를 사용할 수 없으므로 `xfade`/`fade` 트랜지션과 양립하지 않는다.

### 트랜지션
- `cut`: 직접 이어붙이기 (기본값, 가장 빠름)
- `crossfade`: FFmpeg `xfade` 필터 (재인코딩 필요)
- `dip-to-black`: FFmpeg `fade` 필터 (재인코딩 필요)

---

## 에러 처리

| 상황 | 동작 |
|------|------|
| 영상 파일 손상/읽기 불가 | 경고 출력 후 해당 파일 건너뜀 |
| 분析기 의존성 미설치 | `is_available()` 체크 → 설치 안내 메시지 출력 후 해당 분析기 비활성화 |
| FFmpeg 미설치 | 즉시 에러 + 설치 가이드 링크 |
| 선택된 구간이 0개 | `merge_gap_threshold` 낮추기 또는 분析기 추가 제안 메시지 |
| GPU 없음 | `device=auto` 시 CPU로 자동 폴백, 경고 출력 |
| `video.duration < segment_duration` | 영상 전체를 단일 구간으로 처리 |
| `has_audio=False`인 영상에 오디오 분析기 | 해당 영상에 대해 빈 결과 반환 (분析기 비활성화 아님) |
| `transition != "cut"` | 재인코딩 필요 — 경고 후 진행 |
| 추출 클립 길이 < 0.1초 | 경고 후 해당 클립 skip |

---

## GPU 분析기 모델 메모리 관리

- GPU 분析기(`uses_gpu=True`)는 `Pipeline` 초기화 시 모델을 한 번만 로드하고 인스턴스에 캐싱한다.
- 동일 `Pipeline`으로 여러 영상을 분析할 때 모델을 재로드하지 않는다.
- **GPU 전역 직렬화:** 모든 GPU 분析기가 공유하는 단일 전역 `threading.Semaphore(1)` Lock으로 동시에 하나의 GPU 분析기만 실행되도록 보장한다.
- `Pipeline` 소멸 시 또는 `pipeline.close()`를 명시적으로 호출할 때 모델 메모리를 해제한다.
- `Pipeline`은 context manager(`with Pipeline(...) as p:`)를 지원한다.

---

## Rich 프로그레스 UI

```
Scanning videos...        ████████████ 12 files found
Analyzing [scene+audio]   ████████░░░░ 8/12 videos
Extracting clips          ██████████░░ 23/28 segments
Composing highlight.mp4   ████████████ Done ✓

Total: 28 clips → 42s highlight from 12 videos (3m 21s)
```

`onesec analyzers list` 출력 예시:
```
name     level  gpu  available  description
scene    1      No   ✓          Histogram-based scene change
audio    1      No   ✓          Librosa energy + silero-vad
motion   1      No   ✓          Optical flow magnitude
clip     2      Yes  ✗          CLIP embedding scorer (requires: pip install onesec[clip])
whisper  2      Yes  ✗          faster-whisper STT (requires: pip install onesec[whisper])
```

---

## 기술 스택

| 레이어 | 선택 | 이유 |
|--------|------|------|
| Language | Python 3.12+ | ML 생태계 |
| 영상 처리 | FFmpeg (ffmpeg-python) | 업계 표준, 재인코딩·concat 모두 지원 |
| 프레임 디코딩 | PyAV (기본) / decord (선택) | PyAV는 Python 3.12 호환 안정적. decord는 GPU 디코딩 지원하나 2022년 이후 유지보수 중단 — 선택적 의존성으로 제공 |
| scene/motion 분析 | opencv-python | 히스토그램 비교, optical flow |
| 오디오 분析 | librosa (에너지/비트) + silero-vad (음성 감지) | 각각 다른 역할 |
| ML 추론 | ONNX Runtime / openai-clip | 경량화 + 크로스플랫폼 |
| STT | faster-whisper | whisper 대비 4~8배 빠름 |
| CLI | Typer + Rich | 모던 Python CLI 표준 |
| Config | TOML + Pydantic | pyproject.toml과 통합, 타입 안전 |
| 패키징 | uv | pip/poetry 대체 |
| 병렬처리 | concurrent.futures (CPU: ProcessPool, GPU: global Semaphore) | 가벼운 멀티프로세싱 |
| 테스트 | pytest + hypothesis | property-based testing |

---

## 로드맵

| 버전 | 범위 |
|------|------|
| v0.1 | Scanner + SceneAnalyzer + FFmpeg concat + CLI 기본 |
| v0.2 | AudioAnalyzer + MotionAnalyzer + 트랜지션 옵션 |
| v0.3 | CLIP scorer + faster-whisper + GPU 가속 (`uses_gpu` 기반 병렬 전략) |
| v0.4 | Rich UI + `onesec preview` 명령 (별도 설계 문서) + TOML 설정 파일 완전 지원 |
| v0.5 | LLM Analyzer Protocol + BGM 싱크 + 자막 오버레이 |
