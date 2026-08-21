"""ADE/FDE·안전 recall 지표 테스트 (손계산과 일치해야 함)."""
import math

from trajectory.evaluator import (ade, fde, min_dist_to, enters_radius,
                                   recall_precision, entry_confusion)


def test_ade_fde_match_hand_computation():
    # 예측 스텝 (t, x, z, sigma)
    pred = [(4.0, 1.0, 0.0, 0.0), (5.0, 2.0, 0.0, 0.0)]
    # 정답 (t, x, z): 첫 점은 정확, 마지막 점은 (3,4) 만큼 어긋남 → 거리 5
    gt = [(4.0, 1.0, 0.0), (5.0, 5.0, 4.0)]

    # 각 스텝 오차: 0.0, 5.0 → ADE = 2.5, FDE = 5.0
    assert math.isclose(ade(pred, gt), 2.5, abs_tol=1e-9)
    assert math.isclose(fde(pred, gt), 5.0, abs_tol=1e-9)


def test_min_dist_to():
    # 로봇 (0,0)으로 접근하다 (3,0)→(1,0). 최소거리 = 1.0
    path = [(3.0, 0.0), (2.0, 0.0), (1.0, 0.0)]
    assert math.isclose(min_dist_to(path, (0.0, 0.0)), 1.0, abs_tol=1e-9)
    # 3-4-5 직각삼각형: (3,4)까지 = 5
    assert math.isclose(min_dist_to([(3.0, 4.0)], (0.0, 0.0)), 5.0, abs_tol=1e-9)


def test_enters_radius():
    path = [(3.0, 0.0), (2.0, 0.0), (1.0, 0.0)]        # 최소거리 1.0
    assert enters_radius(path, (0.0, 0.0), 1.5) is True     # 1.0 < 1.5 → 진입
    assert enters_radius(path, (0.0, 0.0), 1.0) is False    # 1.0 < 1.0 아님(경계 배타)
    assert enters_radius(path, (0.0, 0.0), 0.5) is False


def test_recall_precision():
    # TP=6, FP=2, FN=4 → recall 6/10=0.6, precision 6/8=0.75
    rec, pre = recall_precision(6, 2, 4)
    assert math.isclose(rec, 0.6, abs_tol=1e-9)
    assert math.isclose(pre, 0.75, abs_tol=1e-9)
    # 분모 0 → NaN
    r2, p2 = recall_precision(0, 0, 0)
    assert math.isnan(r2) and math.isnan(p2)


def test_entry_confusion():
    R = 3.1
    # 이미 반경 안(cur < R) → 대상 아님
    assert entry_confusion(2.0, True, True, R) is None
    # 반경 밖(cur >= R)에서 4가지 조합
    assert entry_confusion(5.0, True, True, R) == "TP"     # 진입 예측·실제 진입
    assert entry_confusion(5.0, False, True, R) == "FP"    # 진입 예측·실제 안 함(헛정지)
    assert entry_confusion(5.0, True, False, R) == "FN"    # 예측 못 함·실제 진입(놓침=충돌)
    assert entry_confusion(5.0, False, False, R) == "TN"   # 둘 다 아님
    # 경계 cur == R 은 '밖'으로 취급(대상) — enters_radius와 같은 배타 규약
    assert entry_confusion(R, False, False, R) == "TN"
