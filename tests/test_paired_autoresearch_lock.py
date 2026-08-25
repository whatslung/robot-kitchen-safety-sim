import importlib
from pathlib import Path

import pytest


def test_paired_lock_detects_changed_fixed_file(tmp_path):
    lock_module = importlib.import_module("train.lock_paired_autoresearch_contract")
    fixed = tmp_path / "fixed.py"
    fixed.write_text("A", encoding="utf-8")
    lock = lock_module.build_paired_lock(tmp_path, files=(Path("fixed.py"),))

    fixed.write_text("B", encoding="utf-8")

    with pytest.raises(lock_module.ContractLockError, match="fixed.py"):
        lock_module.verify_lock(tmp_path, lock)
