"""P0-1 — scene 단위 bootstrap 신뢰구간 순수 로직 (설계 §4·§6)."""
from __future__ import annotations

import math

import pytest

from trajectory.bootstrap import scene_bootstrap_ci


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def test_point_equals_statistic_of_items():
    items = [1.0, 2.0, 3.0, 4.0]
    point, lo, hi = scene_bootstrap_ci(items, _mean, B=500, seed=0)
    assert point == pytest.approx(2.5)


def test_ci_contains_point():
    items = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    point, lo, hi = scene_bootstrap_ci(items, _mean, B=1000, seed=0)
    assert lo <= point <= hi


def test_deterministic_with_seed():
    items = [0.5, 1.5, 2.5, 3.5, 9.0]
    a = scene_bootstrap_ci(items, _mean, B=1000, seed=0)
    b = scene_bootstrap_ci(items, _mean, B=1000, seed=0)
    assert a == b


def test_single_item_zero_width():
    point, lo, hi = scene_bootstrap_ci([7.0], _mean, B=1000, seed=0)
    assert point == pytest.approx(7.0)
    assert lo == pytest.approx(7.0) and hi == pytest.approx(7.0)


def test_empty_returns_nan():
    point, lo, hi = scene_bootstrap_ci([], _mean, B=100, seed=0)
    assert math.isnan(point) and math.isnan(lo) and math.isnan(hi)


def test_lo_le_hi():
    items = [float(i) for i in range(20)]
    _p, lo, hi = scene_bootstrap_ci(items, _mean, B=1000, seed=0)
    assert lo <= hi


def test_works_with_compound_scene_values():
    # recall/precision 처럼 scene별 [TP,FP,FN] 를 합산해 통계 내는 경우.
    scenes = [[2, 0, 1], [1, 1, 0], [3, 0, 0], [0, 0, 2]]

    def recall(rows):
        tp = sum(r[0] for r in rows); fn = sum(r[2] for r in rows)
        return tp / (tp + fn) if (tp + fn) else float("nan")

    point, lo, hi = scene_bootstrap_ci(scenes, recall, B=1000, seed=0)
    assert point == pytest.approx(6 / 9)      # tp=6, fn=3
    assert 0.0 <= lo <= point <= hi <= 1.0
