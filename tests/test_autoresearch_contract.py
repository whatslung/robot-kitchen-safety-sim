from train.autoresearch_contract import Metrics, evaluate_guards, f2_score, rank_key


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
