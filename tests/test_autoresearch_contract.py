import pytest
from types import SimpleNamespace

from trajectory.sim_traj import Window
from trajectory.types import Track, TrackScene
from train.autoresearch_contract import (
    Metrics,
    TestSplitLockedError,
    count_trainable_parameters,
    development_windows,
    evaluate_windows,
    evaluate_guards,
    f2_score,
    measure_cpu_p95,
    rank_key,
)


def _m(**overrides):
    base = dict(
        precision=0.60,
        recall=0.70,
        fde16=0.30,
        cpu_p95_ms=2.0,
        parameters=100_000,
        ade16=0.20,
        tp=70,
        fp=47,
        fn=30,
    )
    base.update(overrides)
    return Metrics(**base)


def test_f2_weights_recall_four_times():
    assert abs(f2_score(0.60, 0.70) - (5 * 0.60 * 0.70 / (4 * 0.60 + 0.70))) < 1e-12
    assert f2_score(0.0, 0.0) == 0.0


def test_guards_accept_exact_boundaries():
    baseline = _m()
    candidate = _m(
        recall=0.69,
        fde16=0.306,
        cpu_p95_ms=2.4,
        parameters=120_000,
    )
    report = evaluate_guards(candidate, baseline)
    assert report.passed
    assert report.failures == ()


def test_each_guard_reports_its_failure():
    baseline = _m()
    candidate = _m(
        recall=0.689,
        fde16=0.307,
        cpu_p95_ms=2.401,
        parameters=120_001,
    )
    report = evaluate_guards(candidate, baseline)
    assert set(report.failures) == {
        "recall",
        "fde16",
        "cpu_p95_ms",
        "parameters",
    }


def test_rank_prefers_f2_then_recall_then_lower_fde_and_latency():
    a = _m(precision=0.70, recall=0.70, fde16=0.25, cpu_p95_ms=2.0)
    b = _m(precision=0.70, recall=0.70, fde16=0.30, cpu_p95_ms=1.0)
    assert rank_key(a) > rank_key(b)


@pytest.mark.parametrize("split", ["test", "locked_test", "all"])
def test_development_loader_rejects_non_development_splits(split):
    with pytest.raises(TestSplitLockedError, match="train/val만"):
        development_windows(split)


class FixedPredictor:
    def predict_batch(self, hists):
        # The second mode enters stopR. Its probability mass is above tau.
        return [
            [
                {"path": [(4.0, 0.0)] * 12, "w": 0.85, "sigma": [0.0] * 12},
                {"path": [(3.0, 0.0)] * 12, "w": 0.15, "sigma": [0.0] * 12},
                {"path": [(5.0, 0.0)] * 12, "w": 0.0, "sigma": [0.0] * 12},
            ]
            for _ in hists
        ]


def _window():
    hist = [(i * 0.4, 5.0, 0.0) for i in range(8)]
    gt = [((i + 8) * 0.4, 3.0, 0.0) for i in range(12)]
    return Window(
        "scene-1",
        31,
        "extra_0",
        TrackScene(2.8, 4.8, [Track(0, hist)]),
        gt,
        None,
        True,
        (0.0, 0.0),
    )


def test_evaluation_uses_probability_mass_for_runtime_alert():
    out = evaluate_windows(FixedPredictor(), [_window()], bootstrap_samples=20)
    assert (out.metrics.tp, out.metrics.fp, out.metrics.fn) == (1, 0, 0)
    assert out.metrics.recall == 1.0 and out.metrics.precision == 1.0
    assert abs(out.metrics.fde16 - 1.0) < 1e-9


def test_cpu_p95_excludes_warmups_and_uses_95th_percentile():
    ticks = iter(range(0, 10_000_000, 1_000_000))
    predictor = SimpleNamespace(predict_batch=lambda _: None)
    p95 = measure_cpu_p95(
        predictor,
        [(0.0, 0.0)] * 8,
        warmups=1,
        repeats=3,
        clock_ns=lambda: next(ticks),
    )
    assert p95 == 1.0


def test_parameter_count_only_includes_trainable_values():
    import torch.nn as nn

    net = nn.Linear(2, 3)
    net.bias.requires_grad_(False)
    assert count_trainable_parameters(net) == 6
