"""Create validation-only reports for paired fixed-step autoresearch."""
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

from train.paired_autoresearch_contract import VARIANTS
from train.run_paired_autoresearch import DEFAULT_OUTPUT_DIR


DEFAULT_RESULTS = DEFAULT_OUTPUT_DIR / "results.jsonl"
DEFAULT_WINNER = DEFAULT_OUTPUT_DIR / "winner.json"
DEFAULT_JSON = (
    ROOT
    / "docs"
    / "chanwoo"
    / "results"
    / "autoresearch-paired-v2-summary.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "docs" / "chanwoo" / "autoresearch-paired-v2-validation-report.md"
)
BOUNDARY = (
    "validation-only simulator trajectory result; "
    "locked test not run; not real-world safety performance"
)


def _portable(value):
    value = deepcopy(value)
    if isinstance(value, dict):
        output = {key: _portable(item) for key, item in value.items()}
        if "weights" in output:
            output["weights"] = str(
                Path("training")
                / "autoresearch-paired-v2"
                / Path(output["weights"]).name
            ).replace("\\", "/")
        return output
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def summarize(rows: list[dict], winner: dict) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[row["variant"]].append(row)
    decisions = {row["variant"]: row for row in winner.get("candidates", [])}
    variants = []
    for variant in VARIANTS:
        runs = sorted(grouped.get(variant, []), key=lambda row: row["seed"])
        if not runs:
            continue
        decision = decisions.get(variant, {})
        failures = Counter(
            failure
            for pair in decision.get("paired", [])
            for failure in pair["guards"]["failures"]
        )
        variants.append(
            {
                "variant": variant,
                "seeds": [row["seed"] for row in runs],
                "median": {
                    name: median(float(row["metrics"][name]) for row in runs)
                    for name in (
                        "f2",
                        "recall",
                        "precision",
                        "fde16",
                        "cpu_p95_ms",
                        "parameters",
                    )
                },
                "median_train_seconds": median(
                    float(row["training"]["train_seconds"]) for row in runs
                ),
                "f2_gain": float(decision.get("f2_gain", 0.0)),
                "all_guards_pass": (
                    True
                    if variant == "transformer-lr1e3"
                    else bool(decision.get("all_guards_pass", False))
                ),
                "guard_failures": dict(sorted(failures.items())),
            }
        )
    return {
        "schema": 1,
        "run_count": len(rows),
        "training_steps": 70_000,
        "variants": variants,
        "winner": _portable(winner),
        "boundary": BOUNDARY,
    }


def _number(value, digits=6):
    return f"{float(value):.{digits}f}"


def render_markdown(summary: dict) -> str:
    lines = [
        "# Paired fixed-step 자동실험 validation 결과",
        "",
        "> v2 validation 30개에서 모델을 비교한 시뮬레이터 궤적 결과다.",
        "> locked test는 실행하지 않았다. 실제 급식실 안전 성능을 뜻하지 않는다.",
        "",
        "## 실행 조건",
        "",
        "- train 90개, validation 30개, locked test 30개",
        "- 모델마다 70,000 step, seed 0·1·2",
        "- 같은 seed의 `transformer-lr1e3`과 보호 조건 비교",
        "",
        "## 3-seed 중앙값",
        "",
        "| 설정 | F2 | 재현율 | 정밀도 | FDE@1.6s(m) | CPU p95(ms) | F2 개선 | 보호 조건 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["variants"]:
        metrics = row["median"]
        failures = ", ".join(
            f"{name}×{count}" for name, count in row["guard_failures"].items()
        )
        guard = "통과" if row["all_guards_pass"] else failures or "실패"
        lines.append(
            f"| `{row['variant']}` | {_number(metrics['f2'])} | "
            f"{_number(metrics['recall'])} | {_number(metrics['precision'])} | "
            f"{_number(metrics['fde16'])} | {_number(metrics['cpu_p95_ms'])} | "
            f"{_number(row['f2_gain'])} | {guard} |"
        )
    winner = summary["winner"]
    best = max(
        summary["variants"],
        key=lambda row: row["median"]["f2"],
        default=None,
    )
    lines.extend(
        [
            "",
            "## 결론",
            "",
            f"- 자동 선택: `{winner['selected_variant']}`",
            f"- 최소 개선 기준 통과: `{winner['minimum_success']}`",
        ]
    )
    if best is not None:
        lines.append(
            f"- 중앙 F2 최고: `{best['variant']}` {_number(best['median']['f2'])}; "
            f"보호 조건: {'통과' if best['all_guards_pass'] else '실패'}"
        )
    lines.extend(
        [
            "- 보호 조건을 모두 통과하면서 중앙 F2가 0.01 이상 오른 후보가 없어 기준 Transformer를 유지한다.",
            "- 이 결론은 validation 전용이며 locked test 평가나 실제 카메라 성능 주장이 아니다.",
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
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    winner = json.loads(args.winner.read_text(encoding="utf-8"))
    output = summarize(rows, winner)
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
