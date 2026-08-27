"""Build or verify the immutable paired fixed-step experiment contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.lock_autoresearch_contract import (
    ContractLockError,
    build_lock,
    verify_lock,
)


LOCK_PATH = (
    ROOT
    / "docs"
    / "chanwoo"
    / "results"
    / "autoresearch-paired-v2-contract-lock.json"
)
FIXED_FILES = (
    "docs/chanwoo/results/traj-v2-manifest.json",
    "docs/chanwoo/autoresearch-paired-v2-program.md",
    "trajectory/traj_v2.py",
    "trajectory/sim_traj.py",
    "trajectory/risk.py",
    "trajectory/evaluator.py",
    "trajectory/bootstrap.py",
    "trajectory/learned_predictor.py",
    "train/autoresearch_contract.py",
    "train/autoresearch_worker.py",
    "train/paired_autoresearch_training.py",
    "train/paired_autoresearch_contract.py",
    "train/paired_autoresearch_worker.py",
    "train/run_paired_autoresearch.py",
    "train/lock_paired_autoresearch_contract.py",
    "tests/test_paired_autoresearch_training.py",
    "tests/test_paired_autoresearch_contract.py",
    "tests/test_paired_autoresearch_worker.py",
    "tests/test_paired_autoresearch_runner.py",
    "tests/test_paired_autoresearch_lock.py",
)


def build_paired_lock(root: Path, files=FIXED_FILES) -> dict:
    return build_lock(root, files=files)


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
        _write_atomic(args.lock, build_paired_lock(ROOT))
        print(args.lock)
    else:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        verify_lock(ROOT, lock)
        print("paired autoresearch contract lock: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
