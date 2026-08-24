from pathlib import Path
import subprocess
import sys

import pytest

from train.eval_autoresearch_locked_test import LockedTestError, validate_gate


def test_gate_requires_exact_confirmation(tmp_path):
    with pytest.raises(LockedTestError, match="확인 문자열"):
        validate_gate(
            {
                "minimum_success": True,
                "selected_commit": "HEAD",
                "weights": str(tmp_path / "model.pt"),
                "weights_sha256": "x",
            },
            "no",
            tmp_path / "final.json",
            head_commit="HEAD",
        )


def test_gate_requires_validation_minimum_success(tmp_path):
    with pytest.raises(LockedTestError, match="최소 성공"):
        validate_gate(
            {
                "minimum_success": False,
                "selected_commit": "HEAD",
                "weights": str(tmp_path / "model.pt"),
                "weights_sha256": "x",
            },
            "RUN_LOCKED_TEST_ONCE",
            tmp_path / "final.json",
            head_commit="HEAD",
        )


def test_gate_requires_frozen_candidate_commit(tmp_path):
    with pytest.raises(LockedTestError, match="커밋"):
        validate_gate(
            {
                "minimum_success": True,
                "selected_commit": "old",
                "weights": str(tmp_path / "model.pt"),
                "weights_sha256": "x",
            },
            "RUN_LOCKED_TEST_ONCE",
            tmp_path / "final.json",
            head_commit="new",
        )


def test_gate_refuses_existing_result(tmp_path):
    output = tmp_path / "final.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(LockedTestError, match="이미 존재"):
        validate_gate(
            {
                "minimum_success": True,
                "selected_commit": "HEAD",
                "weights": str(tmp_path / "model.pt"),
                "weights_sha256": "x",
            },
            "RUN_LOCKED_TEST_ONCE",
            output,
            head_commit="HEAD",
        )


def test_gate_rejects_weight_hash_mismatch(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"actual")
    with pytest.raises(LockedTestError, match="weights SHA-256"):
        validate_gate(
            {
                "minimum_success": True,
                "selected_commit": "HEAD",
                "weights": str(weights),
                "weights_sha256": "wrong",
            },
            "RUN_LOCKED_TEST_ONCE",
            tmp_path / "final.json",
            head_commit="HEAD",
        )


def test_locked_test_cli_resolves_project_packages():
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "train" / "eval_autoresearch_locked_test.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
