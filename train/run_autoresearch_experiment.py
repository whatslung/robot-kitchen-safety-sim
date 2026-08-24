"""Parent process that supervises and records one autoresearch child trial."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.autoresearch_contract import Metrics, evaluate_guards


AUTORESEARCH_DIR = ROOT / "training" / "autoresearch"
DEFAULT_RESULTS = AUTORESEARCH_DIR / "results.jsonl"
DEFAULT_BASELINES = AUTORESEARCH_DIR / "baselines.json"
WORKER = ROOT / "train" / "autoresearch_worker.py"
TRIAL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def classify_child(
    trial_id,
    model,
    seed,
    returncode,
    timed_out,
    child_result,
    stderr,
):
    if timed_out:
        return {
            "trial_id": trial_id,
            "model": model,
            "seed": seed,
            "status": "failed",
            "failure": "timeout",
        }
    if returncode != 0:
        reason = "oom" if "out of memory" in stderr.lower() else "child_exit"
        return {
            "trial_id": trial_id,
            "model": model,
            "seed": seed,
            "status": "failed",
            "failure": reason,
            "returncode": returncode,
            "stderr_tail": stderr[-4000:],
        }
    if not isinstance(child_result, dict):
        return {
            "trial_id": trial_id,
            "model": model,
            "seed": seed,
            "status": "failed",
            "failure": "missing_child_result",
        }
    return {"trial_id": trial_id, **child_result}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metric_object(record: dict) -> Metrics:
    fields = {
        "precision",
        "recall",
        "fde16",
        "cpu_p95_ms",
        "parameters",
        "ade16",
        "tp",
        "fp",
        "fn",
    }
    return Metrics(**{name: record["metrics"][name] for name in fields})


def _load_transformer_baseline(path: Path) -> dict:
    baselines = json.loads(path.read_text(encoding="utf-8"))
    return baselines["models"]["transformer"]


def _decorate_candidate(
    record: dict,
    baseline: dict,
    previous_rows: list[dict],
    rerun: bool,
    smoke: bool,
) -> dict:
    if record.get("status") != "ok":
        return record
    if record.get("environment") != baseline.get("environment"):
        return {
            **record,
            "status": "failed",
            "failure": "environment_mismatch",
        }
    guards = evaluate_guards(_metric_object(record), _metric_object(baseline))
    baseline_f2 = float(baseline["metrics"]["f2"])
    candidate_f2 = float(record["metrics"]["f2"])
    record = {
        **record,
        "guards": asdict(guards),
        "f2_gain": candidate_f2 - baseline_f2,
    }
    if rerun or smoke:
        return record
    kept_f2 = [
        float(row["metrics"]["f2"])
        for row in previous_rows
        if row.get("status") == "ok"
        and row.get("verdict") == "keep"
        and not row.get("rerun")
        and not row.get("smoke")
    ]
    best_f2 = max(kept_f2, default=baseline_f2)
    record["verdict"] = (
        "keep" if guards.passed and candidate_f2 > best_f2 else "revert"
    )
    return record


def _validate_trial_id(trial_id: str, seed: int, rerun: bool) -> str | None:
    if not TRIAL_ID_PATTERN.fullmatch(trial_id) or ".." in trial_id:
        return "invalid_trial_id"
    if rerun and not trial_id.endswith(f"-rerun-s{seed}"):
        return "invalid_rerun_id"
    return None


def _run_child(
    model: str,
    seed: int,
    budget_seconds: float,
    timeout_seconds: float,
    output_json: Path,
    weights_path: Path,
):
    command = [
        sys.executable,
        str(WORKER),
        "--model",
        model,
        "--seed",
        str(seed),
        "--budget-seconds",
        str(budget_seconds),
        "--output-json",
        str(output_json),
        "--weights-path",
        str(weights_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        child_result = None
        if completed.returncode == 0 and output_json.is_file():
            try:
                child_result = json.loads(output_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                child_result = None
        return completed.returncode, False, child_result, completed.stderr
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return None, True, None, stderr


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=("lstm", "transformer", "cvae", "candidate"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.rerun and args.smoke:
        parser.error("--rerun과 --smoke는 함께 쓸 수 없음")
    if not args.smoke and args.budget_seconds != 300.0:
        parser.error("정식 실험의 --budget-seconds는 300이어야 함")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    previous_rows = _read_jsonl(args.results)
    failure = _validate_trial_id(args.trial_id, args.seed, args.rerun)
    if any(row.get("trial_id") == args.trial_id for row in previous_rows):
        failure = "duplicate_trial_id"

    output_json = AUTORESEARCH_DIR / f"{args.trial_id}.json"
    weights_path = AUTORESEARCH_DIR / f"{args.trial_id}.pt"
    if failure is None and (output_json.exists() or weights_path.exists()):
        failure = "trial_artifact_exists"

    if failure is not None:
        record = {
            "trial_id": args.trial_id,
            "model": args.model,
            "seed": args.seed,
            "status": "failed",
            "failure": failure,
        }
    else:
        returncode, timed_out, child_result, stderr = _run_child(
            args.model,
            args.seed,
            args.budget_seconds,
            args.timeout_seconds,
            output_json,
            weights_path,
        )
        record = classify_child(
            args.trial_id,
            args.model,
            args.seed,
            returncode,
            timed_out,
            child_result,
            stderr,
        )

    if args.rerun:
        record["rerun"] = True
        record["base_trial_id"] = args.trial_id.rsplit("-rerun-s", 1)[0]
    if args.smoke:
        record["smoke"] = True
    if args.model == "candidate" and record.get("status") == "ok":
        try:
            baseline = _load_transformer_baseline(args.baselines)
            record = _decorate_candidate(
                record,
                baseline,
                previous_rows,
                rerun=args.rerun,
                smoke=args.smoke,
            )
        except (KeyError, FileNotFoundError, json.JSONDecodeError) as error:
            record = {
                **record,
                "status": "failed",
                "failure": "baseline_unavailable",
                "detail": str(error),
            }
    append_jsonl(args.results, record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
