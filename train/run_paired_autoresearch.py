"""Run the isolated paired-seed fixed-step experiment generation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.paired_autoresearch_contract import (
    SEEDS,
    TRAINING_STEPS,
    VARIANTS,
    select_paired_winner,
)


DEFAULT_OUTPUT_DIR = ROOT / "training" / "autoresearch-paired-v2"
DEFAULT_SMOKE_DIR = ROOT / "training" / "autoresearch-paired-v2-smoke"


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int
    output_json: Path
    weights_path: Path


def build_jobs(variants, seeds, output_dir: Path) -> list[Job]:
    output_dir = Path(output_dir)
    return [
        Job(
            variant,
            seed,
            output_dir / f"{variant}-s{seed}.json",
            output_dir / f"{variant}-s{seed}.pt",
        )
        for variant in variants
        for seed in seeds
    ]


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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


def _run_job(job: Job, steps: int, timeout_seconds: int) -> dict:
    command = [
        sys.executable,
        "-m",
        "train.paired_autoresearch_worker",
        "--variant",
        job.variant,
        "--seed",
        str(job.seed),
        "--steps",
        str(steps),
        "--output-json",
        str(job.output_json),
        "--weights-path",
        str(job.weights_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{job.variant} seed {job.seed} 실패\n{completed.stderr.strip()}"
        )
    return json.loads(job.output_json.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS))
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke and args.steps != TRAINING_STEPS:
        parser.error(f"정식 실험의 --steps는 {TRAINING_STEPS}이어야 함")
    if args.steps <= 0:
        parser.error("--steps는 양수여야 함")
    if args.output_dir is None:
        args.output_dir = DEFAULT_SMOKE_DIR if args.smoke else DEFAULT_OUTPUT_DIR
    args.variants = tuple(args.variant or VARIANTS)
    args.seeds = tuple(args.seed or SEEDS)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    existing = _read_rows(results_path)
    completed_keys = {
        (row.get("variant"), row.get("seed"))
        for row in existing
        if row.get("status") == "ok"
    }
    jobs = build_jobs(args.variants, args.seeds, args.output_dir)
    for index, job in enumerate(jobs, start=1):
        key = (job.variant, job.seed)
        if key in completed_keys:
            print(f"[{index}/{len(jobs)}] skip {job.variant} seed={job.seed}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] start {job.variant} seed={job.seed}", flush=True)
        record = _run_job(job, args.steps, args.timeout_seconds)
        _append_jsonl(results_path, record)
        existing.append(record)
        print(
            f"[{index}/{len(jobs)}] done {job.variant} seed={job.seed} "
            f"F2={record['metrics']['f2']:.6f} ",
            flush=True,
        )

    if not args.smoke and set(args.variants) == set(VARIANTS) and set(args.seeds) == set(SEEDS):
        winner = select_paired_winner(existing)
        _write_atomic(args.output_dir / "winner.json", winner)
        print(json.dumps(winner, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
