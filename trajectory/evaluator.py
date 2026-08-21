"""예측 정확도 지표: ADE(평균 위치오차), FDE(최종시점 오차)."""
from __future__ import annotations

import math


def _dist(px: float, pz: float, gx: float, gz: float) -> float:
    return math.hypot(px - gx, pz - gz)


def ade(pred_steps, gt_points) -> float:
    """Average Displacement Error — 스텝별 위치오차의 평균(미터)."""
    n = min(len(pred_steps), len(gt_points))
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n):
        _, px, pz, _ = pred_steps[i]
        _, gx, gz = gt_points[i]
        total += _dist(px, pz, gx, gz)
    return total / n


def fde(pred_steps, gt_points) -> float:
    """Final Displacement Error — 마지막 공통 시점의 위치오차(미터)."""
    n = min(len(pred_steps), len(gt_points))
    if n == 0:
        return float("nan")
    _, px, pz, _ = pred_steps[n - 1]
    _, gx, gz = gt_points[n - 1]
    return _dist(px, pz, gx, gz)


# ── 안전 지표: 정지반경 진입 예측 recall/precision ─────────────────────────────
# ADE/FDE(위치오차)와 별개로, "정지반경 밖의 사람이 지평선 안에 반경 안으로 들어올지"를
# 미리 맞혔나를 잰다. 선제 안전층의 목표 직결 지표(놓치면 충돌 → recall 우선).

def min_dist_to(points, ref) -> float:
    """경로의 각 점과 기준점 ref=(x, z) 사이 최소 거리(미터). points=[(x, z), …]."""
    rx, rz = ref
    return min(math.hypot(x - rx, z - rz) for (x, z) in points)


def enters_radius(points, ref, radius: float) -> bool:
    """경로가 기준점 반경 안으로 한 번이라도 들어오면 True (최소거리 < radius)."""
    return min_dist_to(points, ref) < radius


def recall_precision(tp: int, fp: int, fn: int):
    """(recall, precision). 분모 0이면 해당 값은 NaN."""
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    pre = tp / (tp + fp) if (tp + fp) else float("nan")
    return rec, pre


def entry_confusion(cur_dist: float, gt_entry: bool, pred_entry: bool, radius: float):
    """정지반경 진입 예측의 혼동행렬 셀. 현재 반경 **밖**(cur>=radius)인 사람만 대상.
    반환: None(대상 아님, 이미 반경 안) | 'TP' | 'FP' | 'FN' | 'TN'."""
    if cur_dist < radius:
        return None                 # 이미 반경 안 → 반응형 몫, 진입 예측 대상 아님
    if pred_entry and gt_entry:
        return "TP"
    if pred_entry:
        return "FP"
    if gt_entry:
        return "FN"
    return "TN"
