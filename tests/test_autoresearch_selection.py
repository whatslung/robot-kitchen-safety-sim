import pytest

from train.select_autoresearch_winner import (
    select_winner,
    top_candidates,
)


def _fast(trial, f2):
    return {
        "trial_id": trial,
        "status": "ok",
        "verdict": "keep",
        "rerun": False,
        "metrics": {"f2": f2},
    }


def _rerun(trial, seed, f2, recall=0.8, fde16=0.2, latency=2.0, passed=True):
    return {
        "trial_id": f"{trial}-rerun-s{seed}",
        "base_trial_id": trial,
        "status": "ok",
        "rerun": True,
        "seed": seed,
        "guards": {"passed": passed},
        "metrics": {
            "f2": f2,
            "recall": recall,
            "fde16": fde16,
            "cpu_p95_ms": latency,
        },
        "weights": f"training/{trial}-s{seed}.pt",
        "weights_sha256": f"sha-{trial}-{seed}",
    }


def test_top_candidates_join_keep_commits_and_sort_f2():
    rows = [_fast("trial-a", 0.71), _fast("trial-b", 0.75)]
    top = top_candidates(rows, {"trial-a": "aaa", "trial-b": "bbb"}, limit=2)
    assert [row["trial_id"] for row in top] == ["trial-b", "trial-a"]
    assert top[0]["commit"] == "bbb"


def test_winner_uses_three_seed_median_and_all_seed_guards():
    rows = [
        _rerun("baseline-transformer", 0, 0.70),
        _rerun("baseline-transformer", 1, 0.70),
        _rerun("baseline-transformer", 2, 0.70),
        _rerun("trial-a", 0, 0.70),
        _rerun("trial-a", 1, 0.80),
        _rerun("trial-a", 2, 0.90),
        _rerun("trial-b", 0, 0.79),
        _rerun("trial-b", 1, 0.79),
        _rerun("trial-b", 2, 0.79),
    ]
    winner = select_winner(
        rows,
        {
            "baseline-transformer": "base",
            "trial-a": "aaa",
            "trial-b": "bbb",
        },
    )
    assert winner["trial_id"] == "trial-a"
    assert winner["median"]["f2"] == 0.80
    assert winner["f2_gain"] == pytest.approx(0.10)
    assert winner["minimum_success"]
    assert winner["source_commit"] == "aaa"
    assert winner["weights"].endswith("trial-a-s0.pt")


def test_failed_or_guard_failing_seed_falls_back_to_baseline():
    rows = [
        _rerun("baseline-transformer", 0, 0.7),
        _rerun("baseline-transformer", 1, 0.7),
        _rerun("baseline-transformer", 2, 0.7),
        _rerun("trial-a", 0, 0.9),
        _rerun("trial-a", 1, 0.9, passed=False),
        _rerun("trial-a", 2, 0.9),
    ]
    winner = select_winner(
        rows,
        {"baseline-transformer": "base", "trial-a": "aaa"},
    )
    assert winner["trial_id"] == "baseline-transformer"
    assert not winner["minimum_success"]
    assert winner["source_commit"] == "base"


def test_baseline_reruns_define_reference_even_if_point_guard_varies():
    rows = [
        _rerun("baseline-transformer", 0, 0.70, passed=False),
        _rerun("baseline-transformer", 1, 0.72, passed=False),
        _rerun("baseline-transformer", 2, 0.68, passed=False),
    ]
    winner = select_winner(rows, {"baseline-transformer": "base"})
    assert winner["trial_id"] == "baseline-transformer"
    assert winner["median"]["f2"] == 0.70
    assert not winner["minimum_success"]
