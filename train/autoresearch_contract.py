from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from trajectory.bootstrap import scene_bootstrap_ci
from trajectory.evaluator import enters_radius, entry_confusion
from trajectory.risk import track_risk
from trajectory.sim_traj import load_windows
from trajectory.traj_v2 import validate_manifest_splits


RECALL_DROP_MAX = 0.01
FDE16_RATIO_MAX = 1.02
LATENCY_RATIO_MAX = 1.20
PARAMETER_RATIO_MAX = 1.20
MIN_F2_GAIN = 0.01
STOP_R = 3.10
SLOW_R = 3.90
HORIZON = 1.6
KSIG = 1.0
TAU = 0.1
HORIZON_STEPS = 4

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


@dataclass(frozen=True)
class Evaluation:
    metrics: Metrics
    ci: dict[str, tuple[float, float, float]]
    by_scene: dict[str, dict[str, object]]


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
    validate_manifest_splits(dataset_dir, manifest, (split,))
    return load_windows(
        split,
        traj_dir=dataset_dir,
        manifest_path=manifest_path,
    )


def _predicted_entry(modes, robot) -> bool:
    risk = track_risk(modes, robot, STOP_R, SLOW_R, HORIZON, KSIG, TAU)
    return risk["tEntryStop"] is not None


def _actual_entry(window) -> bool:
    gt = [(x, z) for _, x, z in window.gt[:HORIZON_STEPS]]
    return enters_radius(gt, window.robot, STOP_R)


def _evaluation_from_scene_rows(
    by_scene,
    cpu_p95_ms,
    parameters,
    bootstrap_samples,
):
    rows = list(by_scene.values())
    tp = sum(row["confusion"][0] for row in rows)
    fp = sum(row["confusion"][1] for row in rows)
    fn = sum(row["confusion"][2] for row in rows)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    ades = [value for row in rows for value in row["ade16"]]
    fdes = [value for row in rows for value in row["fde16"]]
    metrics = Metrics(
        precision=precision,
        recall=recall,
        fde16=sum(fdes) / len(fdes),
        cpu_p95_ms=cpu_p95_ms,
        parameters=parameters,
        ade16=sum(ades) / len(ades),
        tp=tp,
        fp=fp,
        fn=fn,
    )

    def confusion_stat(sample, kind):
        sample_tp = sum(row["confusion"][0] for row in sample)
        sample_fp = sum(row["confusion"][1] for row in sample)
        sample_fn = sum(row["confusion"][2] for row in sample)
        sample_recall = (
            sample_tp / (sample_tp + sample_fn)
            if sample_tp + sample_fn
            else 0.0
        )
        sample_precision = (
            sample_tp / (sample_tp + sample_fp)
            if sample_tp + sample_fp
            else 0.0
        )
        return {
            "recall": sample_recall,
            "precision": sample_precision,
            "f2": f2_score(sample_precision, sample_recall),
        }[kind]

    def mean_field(sample, field):
        values = [value for row in sample for value in row[field]]
        return sum(values) / len(values)

    ci = {
        kind: scene_bootstrap_ci(
            rows,
            lambda sample, selected=kind: confusion_stat(sample, selected),
            B=bootstrap_samples,
            seed=0,
        )
        for kind in ("recall", "precision", "f2")
    }
    ci["ade16"] = scene_bootstrap_ci(
        rows,
        lambda sample: mean_field(sample, "ade16"),
        B=bootstrap_samples,
        seed=0,
    )
    ci["fde16"] = scene_bootstrap_ci(
        rows,
        lambda sample: mean_field(sample, "fde16"),
        B=bootstrap_samples,
        seed=0,
    )
    return Evaluation(metrics, ci, dict(by_scene))


def evaluate_windows(
    predictor,
    windows,
    cpu_p95_ms=0.0,
    parameters=0,
    bootstrap_samples=2000,
) -> Evaluation:
    hists = [
        [(x, z) for _, x, z in window.scene.agents[0].history]
        for window in windows
    ]
    predicted = predictor.predict_batch(hists)
    by_scene = defaultdict(
        lambda: {"confusion": [0, 0, 0], "ade16": [], "fde16": []}
    )
    for window, modes in zip(windows, predicted):
        x, z = window.scene.agents[0].history[-1][1:3]
        if math.hypot(x - window.robot[0], z - window.robot[1]) < STOP_R:
            continue
        actual = _actual_entry(window)
        alert = _predicted_entry(modes, window.robot)
        cell = entry_confusion(STOP_R + 1.0, actual, alert, STOP_R)
        index = {"TP": 0, "FP": 1, "FN": 2}.get(cell)
        if index is not None:
            by_scene[window.scene_id]["confusion"][index] += 1
        pred = modes[0]["path"][:HORIZON_STEPS]
        gt = [(gx, gz) for _, gx, gz in window.gt[:HORIZON_STEPS]]
        errors = [
            math.hypot(px - gx, pz - gz)
            for (px, pz), (gx, gz) in zip(pred, gt)
        ]
        by_scene[window.scene_id]["ade16"].append(sum(errors) / len(errors))
        by_scene[window.scene_id]["fde16"].append(errors[-1])
    return _evaluation_from_scene_rows(
        by_scene,
        cpu_p95_ms,
        parameters,
        bootstrap_samples,
    )


def measure_cpu_p95(
    predictor,
    hist,
    warmups=100,
    repeats=1000,
    clock_ns=time.perf_counter_ns,
) -> float:
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(warmups):
            predictor.predict_batch([hist])
        elapsed = []
        for _ in range(repeats):
            start = clock_ns()
            predictor.predict_batch([hist])
            elapsed.append(clock_ns() - start)
        return float(
            np.percentile(np.asarray(elapsed, dtype=np.float64), 95) / 1_000_000
        )
    finally:
        torch.set_num_threads(previous_threads)


def count_trainable_parameters(net) -> int:
    return sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad)
