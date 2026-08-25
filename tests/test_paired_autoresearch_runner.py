import importlib
from pathlib import Path

import pytest


def _runner_module():
    return importlib.import_module("train.run_paired_autoresearch")


def test_jobs_use_new_generation_directory_and_unique_names(tmp_path):
    runner = _runner_module()

    jobs = runner.build_jobs(
        variants=("lstm-lr1e3", "cvae-lr6e4"),
        seeds=(0, 2),
        output_dir=tmp_path,
    )

    assert [(job.variant, job.seed) for job in jobs] == [
        ("lstm-lr1e3", 0),
        ("lstm-lr1e3", 2),
        ("cvae-lr6e4", 0),
        ("cvae-lr6e4", 2),
    ]
    assert jobs[0].output_json == tmp_path / "lstm-lr1e3-s0.json"
    assert jobs[-1].weights_path == tmp_path / "cvae-lr6e4-s2.pt"


def test_formal_runner_rejects_non_registered_step_count():
    runner = _runner_module()

    with pytest.raises(SystemExit):
        runner.parse_args(["--steps", "10"])


def test_smoke_runner_allows_short_step_count_and_separate_output(tmp_path):
    runner = _runner_module()
    output = tmp_path / "smoke"

    args = runner.parse_args(
        [
            "--smoke",
            "--steps",
            "2",
            "--output-dir",
            str(output),
            "--variant",
            "transformer-lr1e3",
            "--seed",
            "0",
        ]
    )

    assert args.steps == 2
    assert args.output_dir == Path(output)
