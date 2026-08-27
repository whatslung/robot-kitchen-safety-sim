"""Build or verify the immutable-file hash lock for autoresearch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory.traj_v2 import sha256_file


LOCK_PATH = ROOT / "docs" / "chanwoo" / "results" / "autoresearch-contract-lock.json"
FIXED_FILES = (
    "docs/chanwoo/results/traj-v2-manifest.json",
    "trajectory/traj_v2.py",
    "trajectory/sim_traj.py",
    "trajectory/risk.py",
    "trajectory/evaluator.py",
    "trajectory/bootstrap.py",
    "trajectory/learned_predictor.py",
    "train/autoresearch_contract.py",
    "train/autoresearch_training.py",
    "train/autoresearch_worker.py",
    "train/run_autoresearch_experiment.py",
    "tests/test_autoresearch_candidate.py",
    "tests/test_autoresearch_contract.py",
)


class ContractLockError(RuntimeError):
    pass


def build_lock(root: Path, files=FIXED_FILES) -> dict:
    root = Path(root)
    return {
        "schema": 1,
        "files": {
            Path(path).as_posix(): sha256_file(root / path)
            for path in files
        },
    }


def verify_lock(root: Path, lock: dict) -> None:
    root = Path(root)
    mismatches = [
        path
        for path, expected in lock["files"].items()
        if not (root / path).is_file()
        or sha256_file(root / path) != expected
    ]
    if mismatches:
        raise ContractLockError("고정 계약 변경: " + ", ".join(mismatches))


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args(argv)
    if args.write:
        lock = build_lock(ROOT)
        _write_atomic(args.lock, lock)
        print(args.lock)
    else:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        verify_lock(ROOT, lock)
        print("autoresearch contract lock: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
