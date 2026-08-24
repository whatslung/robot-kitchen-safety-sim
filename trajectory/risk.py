"""다인원 미래 위험 계산·중재 (감사 P0-5, 스펙 §4-1).

sim.html의 멀티모달 운영점 진입 로직(entry(), sim.html ~5901)을 파이썬으로 이식한다.
라이브(시뮬)와 오프라인 평가가 같은 위험 코드를 공유하도록 순수 함수로 둔다.

단위: 좌표·반경은 씬 AU(모델 학습 단위와 동일), 시각은 스텝 i → t = STEP_DT·(i+1) 초.
모드 형식: {"path": [(x,z), …], "w": 가중치, "sigma": [σ, …]}  (LearnedPredictor.predict_modes 출력).
"""
from __future__ import annotations

import math
import re

# 궤적 캡처 간격(2.5Hz). 스텝 i의 예측 시각 = STEP_DT·(i+1).
STEP_DT = 0.4


def _id_key(idv):
    """중재 동률 시 결정적·직관적 정렬 키. 'gt:2'/'gt:10' 같은 id를 문자열이 아니라
    끝의 숫자로 비교해 gt:10이 gt:2보다 앞서는 사전식 함정을 피한다. (숫자 없으면 뒤로.)"""
    m = re.search(r"(\d+)\s*$", str(idv))
    return (int(m.group(1)) if m else math.inf, str(idv))


def _sig(mode, i):
    s = mode.get("sigma")
    return s[i] if (s and i < len(s)) else 0.0


def mode_entry(modes, robot, radius, horizon, ksig, tau):
    """정지/감속 반경 진입 판정 — sim.html entry()의 이식.

    각 모드에서 σ팽창 유효거리 `d = hypot(p - robot) - ksig·σ[i]`가 radius 미만이 되는
    **첫 점**(단, t ≤ horizon)을 찾고, 진입한 모드의 가중치 합 mass를 누적한다.

    반환: (t_entry, mass).
      - mass = 반경에 진입한 모드들의 가중치 합(경보 여부와 무관하게 보고용).
      - t_entry = mass ≥ tau 일 때 가장 이른 진입시각(초), 아니면 None(경보 아님).
    """
    rx, rz = robot
    mass = 0.0
    tmin = None
    for m in modes:
        w = m["w"]
        for i, (x, z) in enumerate(m["path"]):
            t = STEP_DT * (i + 1)
            if t > horizon + 1e-9:
                break
            d = math.hypot(x - rx, z - rz) - ksig * _sig(m, i)
            if d < radius:
                mass += w
                tmin = t if tmin is None else min(tmin, t)
                break
    if tmin is not None and mass >= tau:
        return tmin, mass
    return None, mass


def _dmin(modes, robot, horizon, ksig):
    """지평선 내 σ팽창 최소거리(보수적). 진입이 없어도 '얼마나 가까웠나'를 준다."""
    rx, rz = robot
    best = math.inf
    for m in modes:
        for i, (x, z) in enumerate(m["path"]):
            t = STEP_DT * (i + 1)
            if t > horizon + 1e-9:
                break
            d = math.hypot(x - rx, z - rz) - ksig * _sig(m, i)
            if d < best:
                best = d
    return best


def track_risk(modes, robot, stopR, slowR, horizon, ksig, tau):
    """한 사람(트랙)의 위험 요약.

    반환: {tEntryStop, tEntrySlow, riskMass, dMin}.
      riskMass = 정지반경에 진입한 모드 가중치 합. tEntry* = 경보 시 최이른 진입시각(초) or None.
    """
    t_stop, mass_stop = mode_entry(modes, robot, stopR, horizon, ksig, tau)
    t_slow, _ = mode_entry(modes, robot, slowR, horizon, ksig, tau)
    return {
        "tEntryStop": t_stop,
        "tEntrySlow": t_slow,
        "riskMass": mass_stop,
        "dMin": _dmin(modes, robot, horizon, ksig),
    }


def arbitrate(risks):
    """전 대상의 위험 중 로봇이 대응할 **가장 위험한 하나**를 고른다(스펙 §4-1).

    우선순위: ① 정지진입 여부 → ② 가장 이른 진입시각 → ③ 큰 riskMass → ④ 작은 dMin,
    동률은 id 오름차순으로 **결정적**. 정지·감속 어느 반경에도 진입하지 않는 대상은 후보에서 제외.

    risks = [{"id", "tEntryStop", "tEntrySlow", "riskMass", "dMin"}, …]  → worst dict | None.
    """
    candidates = [r for r in risks
                  if r["tEntryStop"] is not None or r["tEntrySlow"] is not None]
    if not candidates:
        return None

    def key(r):
        stop_flag = 0 if r["tEntryStop"] is not None else 1        # 0 = 정지진입(우선)
        if r["tEntryStop"] is not None:
            t = r["tEntryStop"]
        elif r["tEntrySlow"] is not None:
            t = r["tEntrySlow"]
        else:
            t = math.inf
        return (stop_flag, t, -r["riskMass"], r["dMin"], _id_key(r["id"]))

    return min(candidates, key=key)
