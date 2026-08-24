"""Explicit one-time evaluator for the locked trajectory-v2 test split."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from trajectory.autoresearch_candidate import build_candidate
from trajectory.learned_predictor import LearnedPredictor
from trajectory.sim_traj import load_windows
from trajectory.traj_v2 import sha256_file, validate_manifest
from train.autoresearch_contract import (
    V2_DIR,
    V2_MANIFEST,
    count_trainable_parameters,
    evaluate_windows,
    measure_cpu_p95,
)


DEFAULT_WINNER = ROOT / "training" / "autoresearch" / "winner.json"
DEFAULT_OUTPUT = ROOT / "docs" / "chanwoo" / "results" / "autoresearch-final.json"
CONFIRM = "RUN_LOCKED_TEST_ONCE"


class LockedTestError(RuntimeError):
    pass


def _winner_weights(winner: dict) -> Path:
    weights = Path(winner["weights"])
    return weights if weights.is_absolute() else ROOT / weights


def validate_gate(winner, confirmation, output, head_commit):
    output = Path(output)
    if confirmation != CONFIRM:
        raise LockedTestError(f"확인 문자열은 {CONFIRM!r} 이어야 함")
    if not winner.get("minimum_success"):
        raise LockedTestError("validation 최소 성공 기준을 통과하지 못함")
    if winner.get("selected_commit") != head_commit:
        raise LockedTestError("승자 후보 커밋과 현재 HEAD 커밋이 다름")
    if output.exists():
        raise LockedTestError(f"locked-test 결과가 이미 존재: {output}")
    weights = _winner_weights(winner)
    if not weights.is_file() or sha256_file(weights) != winner.get("weights_sha256"):
        raise LockedTestError("winner weights SHA-256 불일치")


def _locked_test_windows():
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(V2_DIR, manifest)
    return load_windows("test", traj_dir=V2_DIR, manifest_path=V2_MANIFEST)


def _head_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _require_candidate_clean() -> None:
    status = subprocess.check_output(
        ["git", "status", "--short", "--", "trajectory/autoresearch_candidate.py"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise LockedTestError("후보 파일에 커밋되지 않은 변경이 있음")


def _write_exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args(argv)

    winner = json.loads(args.winner.read_text(encoding="utf-8"))
    head_commit = _head_commit()
    validate_gate(winner, args.confirm, args.output, head_commit)
    _require_candidate_clean()

    weights = _winner_weights(winner)
    net = build_candidate()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    predictor = LearnedPredictor(net=net, device="cpu")
    windows = _locked_test_windows()
    hist = [
        (x, z)
        for _, x, z in windows[0].scene.agents[0].history
    ]
    cpu_p95_ms = measure_cpu_p95(predictor, hist)
    parameters = count_trainable_parameters(net)
    evaluation = evaluate_windows(
        predictor,
        windows,
        cpu_p95_ms=cpu_p95_ms,
        parameters=parameters,
    )
    metrics = asdict(evaluation.metrics)
    metrics["f2"] = evaluation.metrics.f2
    result = {
        "selected_commit": head_commit,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "manifest_sha256": sha256_file(V2_MANIFEST),
        "metrics": metrics,
        "ci": evaluation.ci,
    }
    _write_exclusive_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
