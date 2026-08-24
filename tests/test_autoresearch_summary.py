from train.summarize_autoresearch import summarize


def test_summary_counts_failures_and_omits_test_metrics():
    rows = [
        {"trial_id": "a", "status": "ok", "verdict": "keep"},
        {"trial_id": "b", "status": "failed", "failure": "timeout"},
    ]
    output = summarize(rows, winner={"trial_id": "a", "commit": "abc"})
    assert output["counts"] == {
        "total": 2,
        "ok": 1,
        "failed": 1,
        "keep": 1,
    }
    assert "test" not in output
