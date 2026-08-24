"""Create validation-only JSON and Markdown reports for autoresearch."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.run_autoresearch_experiment import DEFAULT_BASELINES, DEFAULT_RESULTS
from train.select_autoresearch_winner import DEFAULT_WINNER


DEFAULT_JSON = ROOT / "docs" / "chanwoo" / "results" / "autoresearch-validation-summary.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "chanwoo" / "autoresearch-validation-report.md"
BOUNDARY = "validation-only simulator trajectory result; locked test not run"


def _portable_record(record):
    value = deepcopy(record)
    if isinstance(value, dict):
        if "weights" in value:
            value["weights"] = str(
                Path("training") / "autoresearch" / Path(value["weights"]).name
            ).replace("\\", "/")
        return {key: _portable_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_record(item) for item in value]
    return value


def _rerun_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("rerun"):
            grouped[row["base_trial_id"]].append(row)
    output = []
    for trial_id in sorted(grouped):
        runs = grouped[trial_id]
        ok_runs = [run for run in runs if run.get("status") == "ok"]
        is_reference = trial_id == "baseline-transformer"
        metrics = {}
        if ok_runs:
            metrics = {
                name: median(float(run["metrics"][name]) for run in ok_runs)
                for name in ("f2", "recall", "fde16", "cpu_p95_ms")
            }
        output.append(
            {
                "trial_id": trial_id,
                "seeds": sorted(run.get("seed") for run in runs),
                "all_successful": len(ok_runs) == 3,
                "all_guards_passed": None
                if is_reference
                else bool(runs)
                and all(run.get("guards", {}).get("passed") for run in runs),
                "median": metrics,
            }
        )
    return output


def summarize(rows, winner, baselines=None):
    failed_by = Counter(
        row.get("failure")
        for row in rows
        if row.get("status") == "failed"
    )
    counts = {
        "total": len(rows),
        "ok": sum(row.get("status") == "ok" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "keep": sum(row.get("verdict") == "keep" for row in rows),
    }
    fast_trials = [
        row
        for row in rows
        if row.get("trial_id", "").startswith("trial-")
        and not row.get("rerun")
    ]
    guard_failures = Counter(
        failure
        for row in fast_trials
        for failure in row.get("guards", {}).get("failures", [])
    )
    return {
        "schema": 1,
        "counts": counts,
        "failure_counts": {
            key: failed_by[key]
            for key in sorted(failed_by)
            if key is not None
        },
        "guard_failure_counts": {
            key: guard_failures[key] for key in sorted(guard_failures)
        },
        "baselines": _portable_record(baselines or {}),
        "fast_trials": _portable_record(fast_trials),
        "keep_trials": _portable_record(
            [row for row in rows if row.get("verdict") == "keep"]
        ),
        "reruns": _portable_record(
            [row for row in rows if row.get("rerun")]
        ),
        "rerun_groups": _rerun_groups(rows),
        "winner": _portable_record(winner),
        "boundary": BOUNDARY,
    }


def _number(value, digits=4):
    return "-" if value is None else f"{float(value):.{digits}f}"


def render_markdown(summary: dict) -> str:
    lines = [
        "# Transformer 자동실험 validation 결과",
        "",
        "> 이 결과는 v2 validation에서 고른 시뮬레이터 궤적 예측 성능이다.  ",
        "> locked test는 아직 실행하지 않았고 실제 급식실 안전 성능을 뜻하지 않는다.",
        "",
        "## 실행 요약",
        "",
    ]
    counts = summary["counts"]
    lines.extend(
        [
            f"- 기록 {counts['total']}개: 성공 {counts['ok']}, 실패 {counts['failed']}, keep {counts['keep']}",
            "- 빠른 탐색: warmup 10 step 제외 300초, seed 0, 최대 20개 또는 2시간",
            "- 실제 종료: 12개 후보, keep 이후 6회 정체와 gated residual 비교 후 종료",
            "- 선택: 후보와 기존 Transformer를 seed 0·1·2로 재검증",
            "",
            "## 300초 기준 모델",
            "",
            "| 모델 | F2 | 재현율 | 정밀도 | FDE@1.6s(m) | CPU p95(ms) | 파라미터 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    models = summary.get("baselines", {}).get("models", {})
    for name in ("lstm", "transformer", "cvae"):
        row = models.get(name, {})
        metrics = row.get("metrics", {})
        if not metrics:
            continue
        lines.append(
            "| {name} | {f2} | {recall} | {precision} | {fde} | {latency} | {parameters} |".format(
                name=name,
                f2=_number(metrics.get("f2")),
                recall=_number(metrics.get("recall")),
                precision=_number(metrics.get("precision")),
                fde=_number(metrics.get("fde16")),
                latency=_number(metrics.get("cpu_p95_ms")),
                parameters=metrics.get("parameters", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## 빠른 탐색",
            "",
            "| 후보 | F2 | 재현율 | FDE@1.6s(m) | 보호 조건 | 판정 |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in summary.get("fast_trials", []):
        metrics = row.get("metrics", {})
        guards = row.get("guards", {})
        failures = ", ".join(guards.get("failures", [])) or "통과"
        lines.append(
            f"| {row['trial_id']} | {_number(metrics.get('f2'))} | "
            f"{_number(metrics.get('recall'))} | {_number(metrics.get('fde16'))} | "
            f"{failures} | {row.get('verdict', '-')} |"
        )
    lines.extend(
        [
            "",
            "## 3-seed 재검증",
            "",
            "| 대상 | 중앙 F2 | 중앙 재현율 | 중앙 FDE@1.6s(m) | 중앙 CPU p95(ms) | 모든 seed 보호 통과 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary.get("rerun_groups", []):
        metrics = row.get("median", {})
        guard_status = (
            "기준"
            if row["all_guards_passed"] is None
            else str(row["all_guards_passed"])
        )
        lines.append(
            f"| {row['trial_id']} | {_number(metrics.get('f2'))} | "
            f"{_number(metrics.get('recall'))} | {_number(metrics.get('fde16'))} | "
            f"{_number(metrics.get('cpu_p95_ms'))} | {guard_status} |"
        )
    winner = summary.get("winner", {})
    lines.extend(
        [
            "",
            "## 결론",
            "",
            f"- 선택: `{winner.get('trial_id', '-')}`",
            f"- 최소 개선(F2 +0.01): `{winner.get('minimum_success', False)}`",
            f"- 3-seed 중앙 F2 개선: {_number(winner.get('f2_gain'))}",
            "- 빠른 탐색 keep 후보는 단일 seed에서 개선됐지만 3-seed 모두 보호 조건을 통과하지 못했다.",
            "- 따라서 기존 Transformer로 되돌렸고 locked test를 실행하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    winner = json.loads(args.winner.read_text(encoding="utf-8"))
    output = summarize(rows, winner, baselines)
    _write_atomic(
        args.json_output,
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_atomic(args.markdown_output, render_markdown(output))
    print(args.json_output)
    print(args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
