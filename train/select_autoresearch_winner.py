"""Select autoresearch candidates and a three-seed validation winner."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory.traj_v2 import sha256_file
from train.run_autoresearch_experiment import DEFAULT_RESULTS


DEFAULT_WINNER = ROOT / "training" / "autoresearch" / "winner.json"
CANDIDATE_PATH = ROOT / "trajectory" / "autoresearch_candidate.py"
KEEP_PATTERN = re.compile(r"^experiment: keep (.+) transformer candidate$")


class SelectionError(RuntimeError):
    pass


def top_candidates(rows, commits, limit=3):
    candidates = []
    for row in rows:
        if (
            row.get("status") != "ok"
            or row.get("verdict") != "keep"
            or row.get("rerun")
            or row.get("smoke")
        ):
            continue
        trial_id = row["trial_id"]
        if trial_id not in commits:
            raise SelectionError(f"keep commit 누락: {trial_id}")
        candidates.append(
            {
                "trial_id": trial_id,
                "commit": commits[trial_id],
                "f2": float(row["metrics"]["f2"]),
                "candidate_sha256": row.get("candidate_sha256"),
            }
        )
    return sorted(
        candidates,
        key=lambda row: (-row["f2"], row["trial_id"]),
    )[:limit]


def select_winner(rows, commits):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("rerun"):
            grouped[row["base_trial_id"]].append(row)
    eligible = []
    for trial_id, runs in grouped.items():
        seeds = {run.get("seed") for run in runs if run.get("status") == "ok"}
        if (
            len(runs) != 3
            or seeds != {0, 1, 2}
            or any(not run.get("guards", {}).get("passed") for run in runs)
        ):
            continue
        if trial_id not in commits:
            raise SelectionError(f"재검증 commit 누락: {trial_id}")
        medians = {
            name: median(float(run["metrics"][name]) for run in runs)
            for name in ("f2", "recall", "fde16", "cpu_p95_ms")
        }
        eligible.append(
            {
                "trial_id": trial_id,
                "commit": commits[trial_id],
                "median": medians,
            }
        )
    baseline = next(
        (row for row in eligible if row["trial_id"] == "baseline-transformer"),
        None,
    )
    candidates = [
        row for row in eligible if row["trial_id"] != "baseline-transformer"
    ]
    if baseline is None:
        raise SelectionError("3-seed Transformer baseline 누락")
    winner = (
        max(
            candidates,
            key=lambda row: (
                row["median"]["f2"],
                row["median"]["recall"],
                -row["median"]["fde16"],
                -row["median"]["cpu_p95_ms"],
                row["trial_id"],
            ),
        )
        if candidates
        else dict(baseline)
    )
    winner["baseline_median"] = baseline["median"]
    winner["f2_gain"] = winner["median"]["f2"] - baseline["median"]["f2"]
    winner["minimum_success"] = winner["f2_gain"] >= 0.01
    winner["source_commit"] = (
        winner["commit"] if winner["minimum_success"] else baseline["commit"]
    )
    selected_trial = (
        winner["trial_id"]
        if winner["minimum_success"]
        else "baseline-transformer"
    )
    seed_zero = next(
        run for run in grouped[selected_trial] if run["seed"] == 0
    )
    winner["weights"] = seed_zero["weights"]
    winner["weights_sha256"] = seed_zero["weights_sha256"]
    winner["candidate_sha256"] = seed_zero.get("candidate_sha256")
    return winner


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        encoding="utf-8",
    ).strip()


def commit_map() -> dict[str, str]:
    mapping = {}
    log = _git_output("log", "--format=%H%x09%s")
    for line in log.splitlines():
        commit, _, subject = line.partition("\t")
        match = KEEP_PATTERN.fullmatch(subject)
        if match and match.group(1) not in mapping:
            mapping[match.group(1)] = commit
    additions = _git_output(
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--",
        "trajectory/autoresearch_candidate.py",
    ).splitlines()
    if not additions:
        raise SelectionError("최초 candidate commit 누락")
    mapping["baseline-transformer"] = additions[-1]
    return mapping


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _finalize(path: Path, selected_commit: str) -> dict:
    winner = json.loads(path.read_text(encoding="utf-8"))
    if "selected_commit" in winner:
        raise SelectionError("winner selected_commit이 이미 고정됨")
    head = _git_output("rev-parse", "HEAD")
    if selected_commit != head:
        raise SelectionError("선택 commit과 현재 HEAD가 다름")
    expected_hash = winner.get("candidate_sha256")
    if expected_hash and sha256_file(CANDIDATE_PATH) != expected_hash:
        raise SelectionError("승자 candidate SHA-256 불일치")
    winner["selected_commit"] = selected_commit
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(winner, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return winner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-top", action="store_true")
    modes.add_argument("--print-baseline", action="store_true")
    modes.add_argument("--write-winner", action="store_true")
    modes.add_argument("--finalize-commit")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    args = parser.parse_args(argv)

    commits = commit_map()
    if args.print_baseline:
        output = {
            "trial_id": "baseline-transformer",
            "commit": commits["baseline-transformer"],
        }
    elif args.finalize_commit:
        output = _finalize(args.winner, args.finalize_commit)
    else:
        rows = _read_rows(args.results)
        if args.print_top:
            output = top_candidates(rows, commits)
        else:
            output = select_winner(rows, commits)
            _write_exclusive(args.winner, output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
