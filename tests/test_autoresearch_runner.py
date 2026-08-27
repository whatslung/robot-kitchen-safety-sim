import json
from pathlib import Path

import pytest

from train.run_autoresearch_experiment import append_jsonl, classify_child
from train.lock_autoresearch_contract import (
    ContractLockError,
    build_lock,
    verify_lock,
)


def test_append_jsonl_preserves_existing_records(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"trial_id": "a", "status": "ok"})
    append_jsonl(path, {"trial_id": "b", "status": "failed"})
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["trial_id"] for row in rows] == ["a", "b"]


def test_timeout_is_a_failed_record():
    row = classify_child(
        "trial-1",
        "candidate",
        seed=0,
        returncode=None,
        timed_out=True,
        child_result=None,
        stderr="",
    )
    assert row["status"] == "failed"
    assert row["failure"] == "timeout"


def test_contract_lock_detects_changed_fixed_file(tmp_path):
    fixed = tmp_path / "fixed.py"
    fixed.write_text("A", encoding="utf-8")
    lock = build_lock(tmp_path, files=[Path("fixed.py")])
    fixed.write_text("B", encoding="utf-8")
    with pytest.raises(ContractLockError, match="fixed.py"):
        verify_lock(tmp_path, lock)
