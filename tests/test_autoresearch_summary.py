from train.summarize_autoresearch import render_markdown, summarize


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
    assert not output["strict_filesystem_isolation"]
    assert "test JSON을 파싱하거나 평가하지는 않았지만" in render_markdown(output)
