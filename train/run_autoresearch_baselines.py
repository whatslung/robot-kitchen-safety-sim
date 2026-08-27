"""Run the three fixed validation baselines for trajectory autoresearch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.run_autoresearch_experiment import (
    AUTORESEARCH_DIR,
    DEFAULT_RESULTS,
)


MODELS = ("lstm", "transformer", "cvae")
DEFAULT_OUTPUT = AUTORESEARCH_DIR / "baselines.json"
RUNNER = ROOT / "train" / "run_autoresearch_experiment.py"


class BaselineError(RuntimeError):
    pass


def validate_baselines(rows):
    missing = [name for name in MODELS if name not in rows]
    if missing:
        raise BaselineError("기준 모델 누락: " + ", ".join(missing))
    return {"guard_reference": "transformer", "models": rows}


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_trial(path: Path, trial_id: str) -> dict:
    matches = [row for row in _read_rows(path) if row.get("trial_id") == trial_id]
    if not matches:
        raise BaselineError(f"기준 모델 record 누락: {trial_id}")
    return matches[-1]


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.budget_seconds != 300.0:
        parser.error("기준 측정의 --budget-seconds는 300이어야 함")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise BaselineError(f"기준 결과가 이미 존재: {args.output}")
    rows = {}
    for model in MODELS:
        trial_id = f"baseline-{model}-seed0"
        print(f"baseline start: {model}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--trial-id",
                trial_id,
                "--model",
                model,
                "--seed",
                "0",
                "--budget-seconds",
                str(args.budget_seconds),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--results",
                str(args.results),
            ],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            raise BaselineError(f"기준 runner 종료 실패: {model}")
        rows[model] = _latest_trial(args.results, trial_id)
        if rows[model].get("status") != "ok":
            raise BaselineError(
                f"기준 모델 실패: {model} ({rows[model].get('failure')})"
            )
        print(f"baseline done: {model}", flush=True)
    result = validate_baselines(rows)
    _write_atomic(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
