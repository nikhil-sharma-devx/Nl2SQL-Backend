"""In-process cache hit/miss counters, split by layer (L1 exact / L2 semantic).

The orchestrator records one event per query; the analytics API exposes the
snapshot so the dashboard can show per-layer hit rates. Counters are
process-local (reset on restart) — good enough for operational visibility
without adding storage.
"""
import threading
from typing import Any, Literal

CacheLayer = Literal["exact", "semantic", "miss"]


class CacheMetrics:
    """Thread-safe counters for two-layer cache lookups."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

    def record(self, layer: CacheLayer) -> None:
        with self._lock:
            if layer == "exact":
                self.exact_hits += 1
            elif layer == "semantic":
                self.semantic_hits += 1
            else:
                self.misses += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            total = self.exact_hits + self.semantic_hits + self.misses
            return {
                "exact_hits": self.exact_hits,
                "semantic_hits": self.semantic_hits,
                "misses": self.misses,
                "total_lookups": total,
                "exact_hit_rate": round(self.exact_hits / total, 4) if total else 0.0,
                "semantic_hit_rate": round(self.semantic_hits / total, 4) if total else 0.0,
                "overall_hit_rate": round(
                    (self.exact_hits + self.semantic_hits) / total, 4
                ) if total else 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self.exact_hits = 0
            self.semantic_hits = 0
            self.misses = 0


_metrics = CacheMetrics()


def get_cache_metrics() -> CacheMetrics:
    """Return the process-wide cache metrics singleton."""
    return _metrics


class StageTimingMetrics:
    """Thread-safe rolling averages of per-stage pipeline latency (ms).

    Process-local, like CacheMetrics — surfaces the latency breakdown
    (retrieval vs generation vs execution) so regressions are visible on the
    analytics dashboard instead of anecdotal.
    """

    _STAGES = ("retrieval", "table_selection", "generation", "validation", "execution")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sum: dict[str, int] = dict.fromkeys(self._STAGES, 0)
        self._count: dict[str, int] = dict.fromkeys(self._STAGES, 0)
        self._samples = 0

    def record(self, timings: dict[str, int]) -> None:
        with self._lock:
            self._samples += 1
            for stage, ms in timings.items():
                if stage in self._sum:
                    self._sum[stage] += int(ms)
                    self._count[stage] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg = {
                stage: round(self._sum[stage] / self._count[stage], 1)
                if self._count[stage]
                else 0.0
                for stage in self._STAGES
            }
            return {"samples": self._samples, "avg_stage_ms": avg}

    def reset(self) -> None:
        with self._lock:
            self._sum = dict.fromkeys(self._STAGES, 0)
            self._count = dict.fromkeys(self._STAGES, 0)
            self._samples = 0


_stage_metrics = StageTimingMetrics()


def get_stage_metrics() -> StageTimingMetrics:
    """Return the process-wide stage-timing metrics singleton."""
    return _stage_metrics
