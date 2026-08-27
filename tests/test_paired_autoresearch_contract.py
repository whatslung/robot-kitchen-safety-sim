import importlib

import pytest


def _contract_module():
    return importlib.import_module("train.paired_autoresearch_contract")


def _row(variant, seed, f2, recall=0.90, fde16=0.20, latency=2.0, params=100):
    return {
        "status": "ok",
        "variant": variant,
        "seed": seed,
        "metrics": {
            "f2": f2,
            "recall": recall,
            "fde16": fde16,
            "cpu_p95_ms": latency,
            "parameters": params,
        },
        "weights": f"training/{variant}-s{seed}.pt",
        "weights_sha256": f"sha-{variant}-{seed}",
    }


def test_selection_compares_each_candidate_with_same_seed_baseline():
    contract = _contract_module()
    rows = [
        _row("transformer-lr1e3", 0, 0.70, recall=0.90),
        _row("transformer-lr1e3", 1, 0.70, recall=0.95),
        _row("transformer-lr1e3", 2, 0.70, recall=0.90),
        _row("transformer-lr6e4", 0, 0.73, recall=0.90),
        _row("transformer-lr6e4", 1, 0.73, recall=0.941),
        _row("transformer-lr6e4", 2, 0.73, recall=0.90),
    ]

    result = contract.select_paired_winner(rows)

    assert result["selected_variant"] == "transformer-lr6e4"
    assert result["minimum_success"]
    assert result["f2_gain"] == pytest.approx(0.03)
    assert [pair["seed"] for pair in result["paired"]] == [0, 1, 2]
    assert all(pair["guards"]["passed"] for pair in result["paired"])


def test_one_same_seed_guard_failure_rejects_candidate():
    contract = _contract_module()
    rows = [
        _row("transformer-lr1e3", seed, 0.70, recall=0.90)
        for seed in (0, 1, 2)
    ] + [
        _row(
            "transformer-lr6e4",
            seed,
            0.80,
            recall=0.889 if seed == 2 else 0.90,
        )
        for seed in (0, 1, 2)
    ]

    result = contract.select_paired_winner(rows)

    assert result["selected_variant"] == "transformer-lr1e3"
    assert not result["minimum_success"]
    failed = next(pair for pair in result["candidates"][0]["paired"] if pair["seed"] == 2)
    assert failed["guards"]["failures"] == ["recall"]


def test_selection_requires_three_baseline_seeds():
    contract = _contract_module()
    rows = [
        _row("transformer-lr1e3", 0, 0.70),
        _row("transformer-lr1e3", 1, 0.70),
    ]

    with pytest.raises(contract.PairedSelectionError, match="seed 0, 1, 2"):
        contract.select_paired_winner(rows)


def test_registered_variants_change_only_model_family_or_learning_rate():
    contract = _contract_module()

    assert set(contract.VARIANTS) == {
        "lstm-lr1e3",
        "lstm-lr6e4",
        "transformer-lr1e3",
        "transformer-lr6e4",
        "cvae-lr1e3",
        "cvae-lr6e4",
    }
    assert contract.VARIANTS["cvae-lr6e4"].learning_rate == pytest.approx(6e-4)
    assert contract.VARIANTS["lstm-lr1e3"].model == "lstm"


def test_selection_revalidates_every_input_record_when_validator_is_given():
    contract = _contract_module()
    rows = [
        _row("transformer-lr1e3", seed, 0.70)
        for seed in (0, 1, 2)
    ]
    seen = []

    contract.select_paired_winner(rows, record_validator=lambda row: seen.append(row))

    assert seen == rows
