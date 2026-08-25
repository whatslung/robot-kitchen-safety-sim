import importlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from train.paired_autoresearch_worker import state_dict_sha256


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


def test_smoke_runner_rejects_formal_output_directory():
    runner = _runner_module()

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--smoke",
                "--steps",
                "2",
                "--output-dir",
                str(runner.DEFAULT_OUTPUT_DIR),
            ]
        )


def _formal_record(runner, job, evidence):
    state = {"weight": torch.tensor([1.0, 2.0])}
    torch.save(state, job.weights_path)
    spec = runner.VARIANTS[job.variant]
    return {
        "status": "ok",
        "variant": job.variant,
        "model": spec.model,
        "seed": job.seed,
        "training": {"steps": runner.TRAINING_STEPS},
        "hyperparameters": asdict(spec),
        "weights": str(job.weights_path),
        "weights_sha256": state_dict_sha256(state),
        "contract_sha256": evidence.contract_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "environment": {"deterministic_algorithms": True},
    }


def test_formal_resume_rejects_wrong_step_count(tmp_path):
    runner = _runner_module()
    job = runner.build_jobs(("transformer-lr1e3",), (0,), tmp_path)[0]
    evidence = runner.ContractEvidence("contract", "manifest")
    record = _formal_record(runner, job, evidence)
    record["training"]["steps"] = 2

    with pytest.raises(runner.RunContractError, match="steps"):
        runner.validate_formal_record(record, job, evidence)


def test_formal_resume_rejects_wrong_contract_hash(tmp_path):
    runner = _runner_module()
    job = runner.build_jobs(("transformer-lr1e3",), (0,), tmp_path)[0]
    evidence = runner.ContractEvidence("contract", "manifest")
    record = _formal_record(runner, job, evidence)
    record["contract_sha256"] = "old-contract"

    with pytest.raises(runner.RunContractError, match="contract"):
        runner.validate_formal_record(record, job, evidence)


def test_formal_resume_rejects_changed_weights(tmp_path):
    runner = _runner_module()
    job = runner.build_jobs(("transformer-lr1e3",), (0,), tmp_path)[0]
    evidence = runner.ContractEvidence("contract", "manifest")
    record = _formal_record(runner, job, evidence)
    torch.save({"weight": torch.tensor([9.0])}, job.weights_path)

    with pytest.raises(runner.RunContractError, match="weights"):
        runner.validate_formal_record(record, job, evidence)


def test_formal_resume_rejects_duplicate_variant_seed_records(tmp_path):
    runner = _runner_module()
    job = runner.build_jobs(("transformer-lr1e3",), (0,), tmp_path)[0]
    evidence = runner.ContractEvidence("contract", "manifest")
    record = _formal_record(runner, job, evidence)

    with pytest.raises(runner.RunContractError, match="중복"):
        runner.validated_completed_keys(
            [record, dict(record)],
            {(job.variant, job.seed): job},
            evidence,
        )
