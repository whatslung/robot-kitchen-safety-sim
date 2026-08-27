"""Pre-registered variants and paired-seed selection rules."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from train.autoresearch_contract import (
    FDE16_RATIO_MAX,
    LATENCY_RATIO_MAX,
    MIN_F2_GAIN,
    PARAMETER_RATIO_MAX,
    RECALL_DROP_MAX,
)


BASELINE_VARIANT = "transformer-lr1e3"
SEEDS = (0, 1, 2)
TRAINING_STEPS = 70_000


@dataclass(frozen=True)
class Variant:
    model: str
    learning_rate: float
    weight_decay: float = 0.0
    batch_size: int = 512


VARIANTS = {
    "lstm-lr1e3": Variant("lstm", 1e-3),
    "lstm-lr6e4": Variant("lstm", 6e-4),
    "transformer-lr1e3": Variant("transformer", 1e-3),
    "transformer-lr6e4": Variant("transformer", 6e-4),
    "cvae-lr1e3": Variant("cvae", 1e-3),
    "cvae-lr6e4": Variant("cvae", 6e-4),
}


class PairedSelectionError(ValueError):
    pass


def _guards(candidate: dict, baseline: dict) -> dict:
    failures = []
    if candidate["recall"] < baseline["recall"] - RECALL_DROP_MAX:
        failures.append("recall")
    if candidate["fde16"] > baseline["fde16"] * FDE16_RATIO_MAX:
        failures.append("fde16")
    if candidate["cpu_p95_ms"] > baseline["cpu_p95_ms"] * LATENCY_RATIO_MAX:
        failures.append("cpu_p95_ms")
    if candidate["parameters"] > baseline["parameters"] * PARAMETER_RATIO_MAX:
        failures.append("parameters")
    return {"passed": not failures, "failures": failures}


def _median_metrics(rows: list[dict]) -> dict:
    return {
        name: median(float(row["metrics"][name]) for row in rows)
        for name in ("f2", "recall", "fde16", "cpu_p95_ms", "parameters")
    }


def _index_complete(rows: list[dict], variant: str) -> dict[int, dict] | None:
    selected = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("variant") == variant
    ]
    by_seed = {int(row["seed"]): row for row in selected}
    if len(selected) != len(by_seed):
        raise PairedSelectionError(f"중복 seed 결과: {variant}")
    return by_seed if set(by_seed) == set(SEEDS) else None


def _paired_summary(
    variant: str,
    candidate_by_seed: dict[int, dict],
    baseline_by_seed: dict[int, dict],
    baseline_median: dict,
) -> dict:
    runs = [candidate_by_seed[seed] for seed in SEEDS]
    medians = _median_metrics(runs)
    pairs = []
    for seed in SEEDS:
        candidate = candidate_by_seed[seed]
        baseline = baseline_by_seed[seed]
        pairs.append(
            {
                "seed": seed,
                "f2_delta": (
                    float(candidate["metrics"]["f2"])
                    - float(baseline["metrics"]["f2"])
                ),
                "guards": _guards(candidate["metrics"], baseline["metrics"]),
            }
        )
    return {
        "variant": variant,
        "median": medians,
        "f2_gain": medians["f2"] - baseline_median["f2"],
        "paired": pairs,
        "all_guards_pass": all(pair["guards"]["passed"] for pair in pairs),
    }


def select_paired_winner(rows: list[dict], record_validator=None) -> dict:
    if record_validator is not None:
        for row in rows:
            record_validator(row)
    baseline_by_seed = _index_complete(rows, BASELINE_VARIANT)
    if baseline_by_seed is None:
        raise PairedSelectionError("Transformer baseline은 seed 0, 1, 2가 모두 필요")
    baseline_rows = [baseline_by_seed[seed] for seed in SEEDS]
    baseline_median = _median_metrics(baseline_rows)

    candidates = []
    for variant in VARIANTS:
        if variant == BASELINE_VARIANT:
            continue
        candidate_by_seed = _index_complete(rows, variant)
        if candidate_by_seed is not None:
            candidates.append(
                _paired_summary(
                    variant,
                    candidate_by_seed,
                    baseline_by_seed,
                    baseline_median,
                )
            )

    eligible = [row for row in candidates if row["all_guards_pass"]]
    best = max(
        eligible,
        key=lambda row: (
            row["median"]["f2"],
            row["median"]["recall"],
            -row["median"]["fde16"],
            -row["median"]["cpu_p95_ms"],
            row["variant"],
        ),
        default=None,
    )
    minimum_success = best is not None and best["f2_gain"] >= MIN_F2_GAIN
    selected_variant = best["variant"] if minimum_success else BASELINE_VARIANT
    selected_by_seed = (
        _index_complete(rows, selected_variant) or baseline_by_seed
    )
    selected_summary = (
        best
        if minimum_success
        else _paired_summary(
            BASELINE_VARIANT,
            baseline_by_seed,
            baseline_by_seed,
            baseline_median,
        )
    )
    seed_zero = selected_by_seed[0]
    return {
        "selected_variant": selected_variant,
        "minimum_success": minimum_success,
        "f2_gain": best["f2_gain"] if minimum_success else 0.0,
        "median": selected_summary["median"],
        "baseline_median": baseline_median,
        "paired": selected_summary["paired"],
        "candidates": candidates,
        "weights": seed_zero["weights"],
        "weights_sha256": seed_zero["weights_sha256"],
    }
