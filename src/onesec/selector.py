from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from onesec.models import Config, ScoredSegment


class Selector:
    """Combine analyzer outputs into a ranked list of segments."""

    def select(
        self,
        segments: list[ScoredSegment],
        config: Config,
    ) -> list[ScoredSegment]:
        if not segments:
            return []

        # Step 1: Weight normalization (internal ranking key, does NOT mutate score)
        active_analyzers = {s.analyzer for s in segments}
        raw_weights = {
            a: config.analyzers[a].weight if a in config.analyzers else 1.0
            for a in active_analyzers
        }
        total = sum(raw_weights.values()) or 1.0
        w_norm = {a: w / total for a, w in raw_weights.items()}

        def weighted(seg: ScoredSegment) -> float:
            """Internal weighted score for ranking — does NOT overwrite seg.score."""
            return seg.score * w_norm.get(seg.analyzer, 1.0)

        # Step 2: Merge adjacent/overlapping segments per video
        by_video: dict[Path, list[ScoredSegment]] = {}
        for s in segments:
            by_video.setdefault(s.video_path, []).append(s)

        merged: list[ScoredSegment] = []
        for video_path, segs in by_video.items():
            merged.extend(_merge_segments(segs, config.merge_gap_threshold, weighted))

        # Step 3: Select top-N
        merged.sort(key=lambda s: s.score, reverse=True)
        if config.top_n is not None:
            selected = merged[: config.top_n]
        else:
            selected = []
            total_dur = 0.0
            for s in merged:
                if total_dur + config.clip_duration > config.max_duration:
                    break
                selected.append(s)
                total_dur += config.clip_duration

        # Step 4: Sort by (video_path, start)
        selected.sort(key=lambda s: (str(s.video_path), s.start))
        return selected


def _merge_segments(
    segs: list[ScoredSegment],
    gap_threshold: float,
    weighted: Callable[[ScoredSegment], float],
) -> list[ScoredSegment]:
    """Merge segments closer than gap_threshold seconds.

    For single-segment groups: score = raw seg.score (original preserved).
    For multi-segment groups: score = sum of per-analyzer max weighted scores.
    """
    segs = sorted(segs, key=lambda s: s.start)
    groups: list[list[ScoredSegment]] = []
    for s in segs:
        if groups and s.start - groups[-1][-1].end <= gap_threshold:
            groups[-1].append(s)
        else:
            groups.append([s])

    result = []
    for group in groups:
        if len(group) == 1:
            s = group[0]
            result.append(
                ScoredSegment(
                    video_path=s.video_path,
                    start=s.start,
                    end=s.end,
                    score=s.score,
                    analyzer=s.analyzer,
                )
            )
        else:
            analyzer_scores: dict[str, float] = {}
            for s in group:
                ws = weighted(s)
                if s.analyzer not in analyzer_scores or ws > analyzer_scores[s.analyzer]:
                    analyzer_scores[s.analyzer] = ws
            merged_score = sum(analyzer_scores.values())
            unique_analyzers = set(analyzer_scores)
            result.append(
                ScoredSegment(
                    video_path=group[0].video_path,
                    start=group[0].start,
                    end=group[-1].end,
                    score=merged_score,
                    analyzer="merged" if len(unique_analyzers) > 1 else list(unique_analyzers)[0],
                )
            )
    return result
