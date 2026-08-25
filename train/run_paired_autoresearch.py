"""Run the isolated paired-seed fixed-step experiment generation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.paired_autoresearch_contract import (
    SEEDS,
    TRAINING_STEPS,
    VARIANTS,
    select_paired_winner,
)
from train.paired_autoresearch_worker import state_dict_sha256
from train.lock_paired_autoresearch_contract import (
    LOCK_PATH,
    verify_lock,
)
from train.autoresearch_contract import V2_MANIFEST
from trajectory.traj_v2 import sha256_file


DEFAULT_OUTPUT_DIR = ROOT / "training" / "autoresearch-paired-v2"
DEFAULT_SMOKE_DIR = ROOT / "training" / "autoresearch-paired-v2-smoke"


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int
    output_json: Path
    weights_path: Path


@dataclass(frozen=True)
class ContractEvidence:
    contract_sha256: str
    manifest_sha256: str


class RunContractError(ValueError):
    pass


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


def load_contract_evidence(
    root: Path = ROOT,
    lock_path: Path = LOCK_PATH,
    manifest_path: Path = V2_MANIFEST,
) -> ContractEvidence:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    verify_lock(Path(root), lock)
    return ContractEvidence(
        contract_sha256=sha256_file(Path(lock_path)),
        manifest_sha256=sha256_file(Path(manifest_path)),
    )


def validate_formal_record(
    record: dict,
    job: Job,
    evidence: ContractEvidence,
) -> None:
    spec = VARIANTS[job.variant]
    if record.get("status") != "ok":
        raise RunContractError(f"status 불일치: {job.variant} seed {job.seed}")
    if record.get("variant") != job.variant or record.get("seed") != job.seed:
        raise RunContractError(f"variant/seed 불일치: {job.variant} seed {job.seed}")
    if record.get("model") != spec.model:
        raise RunContractError(f"model 불일치: {job.variant} seed {job.seed}")
    if record.get("training", {}).get("steps") != TRAINING_STEPS:
        raise RunContractError(f"steps 불일치: {job.variant} seed {job.seed}")
    if record.get("hyperparameters") != asdict(spec):
        raise RunContractError(f"hyperparameters 불일치: {job.variant} seed {job.seed}")
    if record.get("contract_sha256") != evidence.contract_sha256:
        raise RunContractError(f"contract SHA-256 불일치: {job.variant} seed {job.seed}")
    if record.get("manifest_sha256") != evidence.manifest_sha256:
        raise RunContractError(f"manifest SHA-256 불일치: {job.variant} seed {job.seed}")
    if not record.get("environment", {}).get("deterministic_algorithms"):
        raise RunContractError(f"결정론 설정 누락: {job.variant} seed {job.seed}")
    recorded_weights = Path(record.get("weights", ""))
    if recorded_weights.resolve() != job.weights_path.resolve():
        raise RunContractError(f"weights 경로 불일치: {job.variant} seed {job.seed}")
    if not job.weights_path.is_file():
        raise RunContractError(f"weights 파일 누락: {job.variant} seed {job.seed}")
    state_dict = torch.load(job.weights_path, map_location="cpu", weights_only=True)
    if state_dict_sha256(state_dict) != record.get("weights_sha256"):
        raise RunContractError(f"weights SHA-256 불일치: {job.variant} seed {job.seed}")


def validated_completed_keys(
    records: list[dict],
    jobs: dict[tuple[str, int], Job],
    evidence: ContractEvidence,
) -> set[tuple[str, int]]:
    completed = set()
    for record in records:
        key = (record.get("variant"), record.get("seed"))
        if key in completed:
            raise RunContractError(f"중복 기존 결과: {key}")
        if key not in jobs:
            raise RunContractError(f"등록되지 않은 기존 결과: {key}")
        validate_formal_record(record, jobs[key], evidence)
        completed.add(key)
    return completed


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
    if args.smoke and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        parser.error("smoke 결과는 정식 실험 디렉터리에 쓸 수 없음")
    args.variants = tuple(args.variant or VARIANTS)
    args.seeds = tuple(args.seed or SEEDS)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    existing = _read_rows(results_path)
    jobs = build_jobs(args.variants, args.seeds, args.output_dir)
    evidence = None if args.smoke else load_contract_evidence()
    all_jobs = {
        (job.variant, job.seed): job
        for job in build_jobs(VARIANTS, SEEDS, args.output_dir)
    }
    if evidence is not None:
        completed_keys = validated_completed_keys(existing, all_jobs, evidence)
    else:
        completed_keys = {
            (row.get("variant"), row.get("seed"))
            for row in existing
            if row.get("status") == "ok"
        }
    for index, job in enumerate(jobs, start=1):
        key = (job.variant, job.seed)
        if key in completed_keys:
            print(f"[{index}/{len(jobs)}] skip {job.variant} seed={job.seed}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] start {job.variant} seed={job.seed}", flush=True)
        record = _run_job(job, args.steps, args.timeout_seconds)
        if evidence is not None:
            record["contract_sha256"] = evidence.contract_sha256
            record["manifest_sha256"] = evidence.manifest_sha256
            _write_atomic(job.output_json, record)
            validate_formal_record(record, job, evidence)
        _append_jsonl(results_path, record)
        existing.append(record)
        print(
            f"[{index}/{len(jobs)}] done {job.variant} seed={job.seed} "
            f"F2={record['metrics']['f2']:.6f} ",
            flush=True,
        )

    if not args.smoke and set(args.variants) == set(VARIANTS) and set(args.seeds) == set(SEEDS):
        validator = (
            None
            if evidence is None
            else lambda row: validate_formal_record(
                row,
                all_jobs[(row["variant"], row["seed"])],
                evidence,
            )
        )
        winner = select_paired_winner(existing, record_validator=validator)
        winner["contract_sha256"] = evidence.contract_sha256
        winner["manifest_sha256"] = evidence.manifest_sha256
        _write_atomic(args.output_dir / "winner.json", winner)
        print(json.dumps(winner, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
