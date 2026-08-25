"""P0-2 detector-track E2E 평가의 순수 함수 단위 테스트 (설계 §8).

파이프라인(YOLO+ByteTrack, 데이터·가중치 의존)은 pytest 비대상 —
여기서는 트랙 매칭·실패모드 분류·가상 로봇 위험·집계의 순수 로직만 검증한다.
"""
from __future__ import annotations

import math

import pytest

from trajectory.dettrack import (
    assign_per_frame,
    classify_failures,
    match_track,
    modes_from_prediction,
    virtual_robot_risk,
    aggregate,
)
from trajectory.types import Mode
from trajectory import risk


# ── assign_per_frame: GT 프레임마다 가장 가까운 검출 트랙(임계 이내) ──────────
def test_assign_per_frame_full_cover():
    gt = [(0, 0.10, 0.10), (1, 0.12, 0.10), (2, 0.14, 0.10)]
    det = {5: [(0, 0.10, 0.10), (1, 0.12, 0.10), (2, 0.14, 0.10)]}
    assert assign_per_frame(gt, det, max_dist=0.05) == [5, 5, 5]


def test_assign_per_frame_miss_when_no_detection_that_frame():
    gt = [(0, 0.10, 0.10), (1, 0.12, 0.10), (2, 0.14, 0.10)]
    det = {5: [(0, 0.10, 0.10), (2, 0.14, 0.10)]}   # 프레임 1 검출 없음
    assert assign_per_frame(gt, det, max_dist=0.05) == [5, None, 5]


def test_assign_per_frame_far_detection_is_miss():
    gt = [(0, 0.10, 0.10)]
    det = {5: [(0, 0.90, 0.90)]}                     # 임계 밖
    assert assign_per_frame(gt, det, max_dist=0.05) == [None]


def test_assign_per_frame_picks_nearest_among_candidates():
    gt = [(0, 0.50, 0.50)]
    det = {1: [(0, 0.54, 0.50)], 2: [(0, 0.51, 0.50)]}   # 2가 더 가깝다
    assert assign_per_frame(gt, det, max_dist=0.10) == [2]


def test_assign_per_frame_id_switch_across_frames():
    gt = [(0, 0.10, 0.10), (1, 0.20, 0.10), (2, 0.30, 0.10), (3, 0.40, 0.10)]
    det = {1: [(0, 0.10, 0.10), (1, 0.20, 0.10)],
           2: [(2, 0.30, 0.10), (3, 0.40, 0.10)]}
    assert assign_per_frame(gt, det, max_dist=0.05) == [1, 1, 2, 2]


# ── classify_failures: per-frame 할당열 → miss·fragment·id_switch ────────────
def test_classify_clean():
    assert classify_failures([9, 9, 9]) == {"miss": 0, "fragments": 1, "id_switches": 0}


def test_classify_miss_only():
    assert classify_failures([9, 9, None, 9]) == {"miss": 1, "fragments": 1, "id_switches": 0}


def test_classify_fragmentation_and_switch():
    assert classify_failures([1, 1, 2, 2]) == {"miss": 0, "fragments": 2, "id_switches": 1}


def test_classify_bounce_two_switches():
    assert classify_failures([1, 2, 1]) == {"miss": 0, "fragments": 2, "id_switches": 2}


def test_classify_switch_across_gap():
    # 갭(None)을 사이에 둔 id 변경도 같은 GT의 id 스위치로 센다(갭 압축).
    assert classify_failures([1, None, 2]) == {"miss": 1, "fragments": 2, "id_switches": 1}


def test_classify_all_missed():
    assert classify_failures([None, None]) == {"miss": 2, "fragments": 0, "id_switches": 0}


# ── match_track: 지배 검출 id(다수결, 동률은 작은 id) ─────────────────────────
def test_match_track_majority():
    assert match_track([1, 1, 2]) == 1


def test_match_track_tie_prefers_smaller_id():
    assert match_track([2, 2, 7, 7]) == 2


def test_match_track_none_when_all_missed():
    assert match_track([None, None]) is None


# ── virtual_robot_risk: 예측 모드 → risk.track_risk 어댑터 ────────────────────
def _mode(path_xz, w=1.0):
    steps = [(risk.STEP_DT * (i + 1), x, z, 0.0) for i, (x, z) in enumerate(path_xz)]
    return Mode(prob=w, steps=steps)


def test_modes_from_prediction_shape():
    m = _mode([(0.5, 0.0), (0.6, 0.0)], w=0.7)
    out = modes_from_prediction([m])
    assert out == [{"path": [(0.5, 0.0), (0.6, 0.0)], "w": 0.7, "sigma": [0.0, 0.0]}]


def test_virtual_robot_risk_enters_stop():
    # 첫 점이 정지반경(1.0) 안(로봇 원점) → 진입시각 = STEP_DT.
    m = _mode([(0.5, 0.0)])
    r = virtual_robot_risk([m], robot=(0.0, 0.0), stopR=1.0, slowR=2.0,
                           horizon=1.6, ksig=1.0, tau=0.1)
    assert r["tEntryStop"] == pytest.approx(risk.STEP_DT)
    assert r["riskMass"] == pytest.approx(1.0)


def test_virtual_robot_risk_matches_direct_track_risk():
    m = _mode([(1.5, 0.0), (0.8, 0.0)], w=1.0)
    got = virtual_robot_risk([m], robot=(0.0, 0.0), stopR=1.0, slowR=2.0,
                             horizon=1.6, ksig=1.0, tau=0.1)
    direct = risk.track_risk(modes_from_prediction([m]), robot=(0.0, 0.0),
                             stopR=1.0, slowR=2.0, horizon=1.6, ksig=1.0, tau=0.1)
    assert got == direct


def test_virtual_robot_risk_no_entry_when_far():
    m = _mode([(5.0, 0.0), (5.0, 0.0)])
    r = virtual_robot_risk([m], robot=(0.0, 0.0), stopR=1.0, slowR=2.0,
                           horizon=1.6, ksig=1.0, tau=0.1)
    assert r["tEntryStop"] is None and r["tEntrySlow"] is None


# ── aggregate: 그룹별 평균(전체·움직인 것만) ─────────────────────────────────
def test_aggregate_groups_and_means():
    records = [
        {"group": ("gt", "cv"), "ade": 2.0, "fde": 4.0, "moved": True},
        {"group": ("gt", "cv"), "ade": 4.0, "fde": 8.0, "moved": False},
        {"group": ("det", "cv"), "ade": 10.0, "fde": 20.0, "moved": True},
    ]
    out = aggregate(records)
    assert out[("gt", "cv")]["n"] == 2
    assert out[("gt", "cv")]["ade"] == pytest.approx(3.0)
    assert out[("gt", "cv")]["fde"] == pytest.approx(6.0)
    assert out[("gt", "cv")]["n_moved"] == 1
    assert out[("gt", "cv")]["ade_moved"] == pytest.approx(2.0)
    assert out[("det", "cv")]["n"] == 1
    assert out[("det", "cv")]["ade"] == pytest.approx(10.0)


def test_aggregate_moved_nan_when_none_moved():
    records = [{"group": ("gt", "kf"), "ade": 1.0, "fde": 2.0, "moved": False}]
    out = aggregate(records)
    assert out[("gt", "kf")]["n_moved"] == 0
    assert math.isnan(out[("gt", "kf")]["ade_moved"])
