from __future__ import annotations

from dataclasses import dataclass


RECALL_DROP_MAX = 0.01
FDE16_RATIO_MAX = 1.02
LATENCY_RATIO_MAX = 1.20
PARAMETER_RATIO_MAX = 1.20
MIN_F2_GAIN = 0.01


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    fde16: float
    cpu_p95_ms: float
    parameters: int
    ade16: float
    tp: int
    fp: int
    fn: int

    @property
    def f2(self) -> float:
        return f2_score(self.precision, self.recall)


@dataclass(frozen=True)
class GuardReport:
    passed: bool
    failures: tuple[str, ...]


def f2_score(precision: float, recall: float) -> float:
    denominator = 4 * precision + recall
    return 0.0 if denominator == 0 else 5 * precision * recall / denominator


def evaluate_guards(candidate: Metrics, baseline: Metrics) -> GuardReport:
    failures = []
    if candidate.recall < baseline.recall - RECALL_DROP_MAX:
        failures.append("recall")
    if candidate.fde16 > baseline.fde16 * FDE16_RATIO_MAX:
        failures.append("fde16")
    if candidate.cpu_p95_ms > baseline.cpu_p95_ms * LATENCY_RATIO_MAX:
        failures.append("cpu_p95_ms")
    if candidate.parameters > baseline.parameters * PARAMETER_RATIO_MAX:
        failures.append("parameters")
    return GuardReport(not failures, tuple(failures))


def rank_key(metrics: Metrics) -> tuple[float, float, float, float]:
    return (metrics.f2, metrics.recall, -metrics.fde16, -metrics.cpu_p95_ms)
