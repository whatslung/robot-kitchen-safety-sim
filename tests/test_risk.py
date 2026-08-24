"""다인원 위험 계산·중재 테스트 (감사 P0-5, 스펙 §4-1).

sim.html의 운영점 진입 로직(entry())을 파이썬으로 이식한 `trajectory/risk.py`를 검증한다.
좌표·반경은 씬 AU, 시각은 스텝 i → t=0.4·(i+1)초. 로봇은 원점(0,0) 기준으로 둔다.
"""
import math

from trajectory.risk import mode_entry, track_risk, arbitrate, STEP_DT


def _mode(path, w=1.0, sigma=None):
    return {"path": path, "w": w, "sigma": sigma if sigma is not None else [0.0] * len(path)}


ROBOT = (0.0, 0.0)


# ── mode_entry ─────────────────────────────────────────────────────────────
def test_step_dt_contract():
    assert abs(STEP_DT - 0.4) < 1e-9


def test_mode_entry_hits_radius_at_first_inside_step():
    # x: 5,4,3,2 (t=0.4,0.8,1.2,1.6). radius=3.1 → 첫 진입은 i=2(x=3, t=1.2).
    m = _mode([(5, 0), (4, 0), (3, 0), (2, 0)])
    t, mass = mode_entry([m], ROBOT, radius=3.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert abs(t - 1.2) < 1e-9
    assert abs(mass - 1.0) < 1e-9


def test_mode_entry_no_entry_returns_none():
    # 계속 반경 밖(4.0) → 진입 없음.
    m = _mode([(4, 0), (4, 0), (4, 0), (4, 0)])
    t, mass = mode_entry([m], ROBOT, radius=3.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert t is None
    assert mass == 0.0


def test_mode_entry_respects_horizon_cutoff():
    # 진입은 i=3(t=1.6)에서만 일어나는데 horizon=1.2면 못 본다.
    m = _mode([(5, 0), (5, 0), (5, 0), (2, 0)])
    t, _ = mode_entry([m], ROBOT, radius=3.1, horizon=1.2, ksig=1.0, tau=0.1)
    assert t is None
    t2, _ = mode_entry([m], ROBOT, radius=3.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert abs(t2 - 1.6) < 1e-9


def test_mode_entry_sigma_inflation_pulls_entry_earlier():
    # x=4 (반경 3.1 밖)이지만 σ=1.5, ksig=1 → 유효거리 4-1.5=2.5 < 3.1 → 진입으로 본다(보수적).
    m = _mode([(4, 0)], sigma=[1.5])
    t, _ = mode_entry([m], ROBOT, radius=3.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert abs(t - 0.4) < 1e-9
    # ksig=0 이면 팽창 없음 → 진입 아님.
    t0, _ = mode_entry([m], ROBOT, radius=3.1, horizon=1.6, ksig=0.0, tau=0.1)
    assert t0 is None


def test_mode_entry_mass_below_tau_returns_none():
    # 진입 모드 가중치 합이 tau 미만이면 경보 아님(tmin은 있어도 None 반환).
    m = _mode([(2, 0)], w=0.05)
    t, mass = mode_entry([m], ROBOT, radius=3.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert t is None
    assert abs(mass - 0.05) < 1e-9


def test_mode_entry_earliest_across_modes():
    early = _mode([(5, 0), (2, 0)], w=0.5)          # 진입 i=1, t=0.8
    late = _mode([(5, 0), (5, 0), (2, 0)], w=0.5)   # 진입 i=2, t=1.2
    t, mass = mode_entry([late, early], ROBOT, radius=3.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert abs(t - 0.8) < 1e-9
    assert abs(mass - 1.0) < 1e-9   # 두 모드 다 진입 → 합 1.0


# ── track_risk ─────────────────────────────────────────────────────────────
def test_track_risk_stop_and_slow():
    # x: 6,5,4,3 → stopR=3.1 진입은 i=3(x=3,t=1.6), slowR=5.1 진입은 i=1(x=5,t=0.8).
    m = _mode([(6, 0), (5, 0), (4, 0), (3, 0)])
    r = track_risk([m], ROBOT, stopR=3.1, slowR=5.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert abs(r["tEntryStop"] - 1.6) < 1e-9
    assert abs(r["tEntrySlow"] - 0.8) < 1e-9
    assert abs(r["riskMass"] - 1.0) < 1e-9
    assert abs(r["dMin"] - 3.0) < 1e-9    # 지평선 내 최소거리 = x=3


def test_track_risk_no_entry():
    m = _mode([(9, 0), (9, 0), (9, 0)])
    r = track_risk([m], ROBOT, stopR=3.1, slowR=5.1, horizon=1.6, ksig=1.0, tau=0.1)
    assert r["tEntryStop"] is None
    assert r["tEntrySlow"] is None
    assert r["riskMass"] == 0.0


# ── arbitrate — 최근접 ≠ 최고위험 ────────────────────────────────────────────
def _risk(id, tStop=None, tSlow=None, mass=0.0, dMin=99.0):
    return {"id": id, "tEntryStop": tStop, "tEntrySlow": tSlow, "riskMass": mass, "dMin": dMin}


def test_arbitrate_nearest_is_not_worst():
    # A: 가장 가깝지만(dMin 작음) 진입 없음 → 후보 아님.
    # B: 멀지만(dMin 큼) 정지반경에 진입 → worst.
    A = _risk("gt:0", tStop=None, tSlow=None, mass=0.0, dMin=3.2)   # 반경 밖 근접, 진입 안 함
    B = _risk("gt:1", tStop=0.8, tSlow=0.4, mass=0.6, dMin=2.5)
    worst = arbitrate([A, B])
    assert worst is not None
    assert worst["id"] == "gt:1"


def test_arbitrate_none_when_no_entry():
    A = _risk("gt:0")
    B = _risk("gt:1")
    assert arbitrate([A, B]) is None


def test_arbitrate_stop_entry_beats_slow_only():
    slow_only = _risk("gt:0", tStop=None, tSlow=0.4, mass=0.3, dMin=4.0)
    stop_late = _risk("gt:1", tStop=1.5, tSlow=1.0, mass=0.2, dMin=3.0)
    worst = arbitrate([slow_only, stop_late])
    assert worst["id"] == "gt:1"          # 정지진입이 감속전용보다 우선


def test_arbitrate_earliest_stop_entry_wins():
    a = _risk("gt:0", tStop=1.2, tSlow=0.8, mass=0.9, dMin=2.0)
    b = _risk("gt:1", tStop=0.4, tSlow=0.2, mass=0.2, dMin=3.0)
    assert arbitrate([a, b])["id"] == "gt:1"   # 더 이른 정지진입


def test_arbitrate_deterministic_tie_break_by_id():
    a = _risk("gt:5", tStop=0.8, tSlow=0.4, mass=0.5, dMin=2.0)
    b = _risk("gt:2", tStop=0.8, tSlow=0.4, mass=0.5, dMin=2.0)
    # 모든 값 동일 → id 오름차순으로 결정적.
    assert arbitrate([a, b])["id"] == "gt:2"
    assert arbitrate([b, a])["id"] == "gt:2"


def test_arbitrate_tie_break_is_numeric_not_lexicographic():
    # 사전식이면 'gt:10'이 'gt:2'보다 앞서는 함정 → 숫자 비교로 gt:2가 이겨야 한다.
    a = _risk("gt:10", tStop=0.8, tSlow=0.4, mass=0.5, dMin=2.0)
    b = _risk("gt:2", tStop=0.8, tSlow=0.4, mass=0.5, dMin=2.0)
    assert arbitrate([a, b])["id"] == "gt:2"
    assert arbitrate([b, a])["id"] == "gt:2"
