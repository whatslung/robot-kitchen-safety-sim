import importlib


def _summary_module():
    return importlib.import_module("train.summarize_paired_autoresearch")


def _row(variant, seed, f2):
    return {
        "status": "ok",
        "variant": variant,
        "seed": seed,
        "training": {"steps": 70_000, "train_seconds": 10.0},
        "metrics": {
            "f2": f2,
            "recall": 0.90 + seed * 0.01,
            "precision": 0.60,
            "fde16": 0.20,
            "cpu_p95_ms": 0.30,
            "parameters": 100,
        },
        "weights": f"C:/repo/training/autoresearch-paired-v2/{variant}-s{seed}.pt",
    }


def test_summary_reports_three_seed_medians_and_guard_failures():
    summary_module = _summary_module()
    rows = [
        _row("transformer-lr1e3", seed, f2)
        for seed, f2 in enumerate((0.70, 0.72, 0.71))
    ] + [
        _row("cvae-lr1e3", seed, f2)
        for seed, f2 in enumerate((0.80, 0.82, 0.81))
    ]
    winner = {
        "selected_variant": "transformer-lr1e3",
        "minimum_success": False,
        "f2_gain": 0.0,
        "candidates": [
            {
                "variant": "cvae-lr1e3",
                "f2_gain": 0.10,
                "all_guards_pass": False,
                "paired": [
                    {"seed": 0, "guards": {"passed": False, "failures": ["recall"]}},
                    {"seed": 1, "guards": {"passed": True, "failures": []}},
                    {"seed": 2, "guards": {"passed": True, "failures": []}},
                ],
            }
        ],
    }

    summary = summary_module.summarize(rows, winner)

    groups = {row["variant"]: row for row in summary["variants"]}
    assert groups["transformer-lr1e3"]["median"]["f2"] == 0.71
    assert groups["cvae-lr1e3"]["guard_failures"] == {"recall": 1}
    assert summary["boundary"] == summary_module.BOUNDARY


def test_markdown_states_validation_only_selected_model_and_no_test_run():
    summary_module = _summary_module()
    summary = {
        "variants": [],
        "winner": {
            "selected_variant": "transformer-lr1e3",
            "minimum_success": False,
        },
        "boundary": summary_module.BOUNDARY,
    }

    markdown = summary_module.render_markdown(summary)

    assert "transformer-lr1e3" in markdown
    assert "locked test는 실행하지 않았다" in markdown
    assert "실제 급식실 안전 성능" in markdown
