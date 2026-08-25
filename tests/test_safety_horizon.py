"""안전 eval 지평선 절단 테스트 (감사 P0-C).

라이브 로봇 제어는 1.6s(4스텝) 지평선을 쓰고, 오프라인 평가는 4.8s(12스텝)를 쓴다.
두 수치가 섞이면 라이브 성능을 과대평가하게 된다. `eval_traj_safety.evaluate`는
`horizon_steps`로 예측·GT 경로를 같은 지평선으로 잘라 이 구분을 만들어낸다.

여기서는 데이터·모델 없이 순수 절단 로직과 그 진입판정 효과만 검증한다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.eval_traj_safety import _truncate, HORIZONS, STEP_DT, PRED
from trajectory.evaluator import enters_radius


def test_truncate_none_keeps_full():
    path = [(i, 0.0) for i in range(12)]
    assert _truncate(path, None) == path


def test_truncate_keeps_prefix():
    path = [(i, 0.0) for i in range(12)]
    assert _truncate(path, 4) == path[:4]
    assert len(_truncate(path, 4)) == 4


def test_horizons_cover_live_and_offline():
    labels = dict((steps, label) for label, steps in HORIZONS)
    assert 4 in labels, "라이브 1.6s(4스텝) 지평선이 있어야 한다"
    assert PRED in labels, "오프라인 4.8s(전체 스텝) 지평선이 있어야 한다"
    # 스텝×0.4s = 초 라는 계약을 고정
    assert abs(4 * STEP_DT - 1.6) < 1e-9
    assert abs(PRED * STEP_DT - 4.8) < 1e-9


def test_short_horizon_misses_late_entry():
    """반경 진입이 늦게(스텝 8) 일어나면 4.8s에선 잡히고 1.6s에선 안 잡힌다."""
    robot = (0.0, 0.0)
    R = 1.0
    # 스텝 0~4는 반경 밖(먼 곳), 스텝 8에서 반경 안으로 진입
    path = [(5.0, 0.0)] * 5 + [(3.0, 0.0), (2.0, 0.0), (1.5, 0.0), (0.5, 0.0)] + [(0.4, 0.0)] * 3
    assert enters_radius(_truncate(path, PRED), robot, R) is True     # 4.8s: 진입 봄
    assert enters_radius(_truncate(path, 4), robot, R) is False       # 1.6s: 아직 못 봄
