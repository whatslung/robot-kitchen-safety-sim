import json

from train.run_autoresearch_experiment import append_jsonl, classify_child


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
