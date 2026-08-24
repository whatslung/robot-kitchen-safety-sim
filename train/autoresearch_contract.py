from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from trajectory.sim_traj import load_windows
from trajectory.traj_v2 import validate_manifest


RECALL_DROP_MAX = 0.01
FDE16_RATIO_MAX = 1.02
LATENCY_RATIO_MAX = 1.20
PARAMETER_RATIO_MAX = 1.20
MIN_F2_GAIN = 0.01

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "dataset" / "trajectories_v2"
V2_MANIFEST = ROOT / "docs" / "chanwoo" / "results" / "traj-v2-manifest.json"


class TestSplitLockedError(ValueError):
    __test__ = False

    pass


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


def development_windows(
    split: str,
    dataset_dir: Path = V2_DIR,
    manifest_path: Path = V2_MANIFEST,
):
    if split not in {"train", "val"}:
        raise TestSplitLockedError(f"자동 실험은 train/val만 허용: {split!r}")
    dataset_dir = Path(dataset_dir)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(dataset_dir, manifest)
    return load_windows(
        split,
        traj_dir=dataset_dir,
        manifest_path=manifest_path,
    )
