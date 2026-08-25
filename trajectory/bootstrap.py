"""P0-1 — scene 단위 bootstrap 신뢰구간 (설계 §4).

윈도우는 stride-1 이라 강한 상관 → 윈도우 단위 CI는 불확실성을 과소추정한다.
그래서 **scene(=seed·레이아웃) 단위**로 복원추출해 통계의 95% CI를 낸다.
"""
from __future__ import annotations

import numpy as np


def scene_bootstrap_ci(items, statistic, B=2000, alpha=0.05, seed=0):
    """scene별 기여값 리스트를 scene 단위로 B회 복원추출해 통계의 (point, lo, hi) 반환.

    items      = [scene별 값 또는 카운트, …]  (ADE=scene평균 float, safety=scene별 [TP,FP,FN])
    statistic  = items 부분집합 → float  (예: 평균, recall)
    반환        = (point, lo, hi)  — point=statistic(items 전체), lo/hi=alpha/2·1-alpha/2 백분위.
                  items 비면 (nan, nan, nan).
    """
    n = len(items)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(items))
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    idx = np.arange(n)
    for b in range(B):
        pick = rng.choice(idx, size=n, replace=True)
        stats[b] = statistic([items[i] for i in pick])
    # 퇴화 리샘플(예: 양성 scene 0개 → recall NaN)은 무시하고 CI를 낸다.
    lo = float(np.nanpercentile(stats, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)
