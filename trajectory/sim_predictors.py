"""sim 전용 예측기 — 스테이션 휴리스틱. 이슈 #2 3단계.

관측 마지막 지점에서 관측 속력으로 목표(gx,gz)를 향해 등속 직진하고, 목표에 닿으면 멈춘다.
목표를 모르는(통과 구간) 윈도우는 등속으로 폴백. 목표는 sim이 기록한 현재 목표를 쓴다 —
실제 배포엔 목표 추정기가 따로 필요하다(설계 스펙 §스테이션 휴리스틱 참조).
"""
from __future__ import annotations

import math

from trajectory.types import Mode, TrackScene
from trajectory.predictors import ConstantVelocityPredictor


class StationHeuristicPredictor:
    """목표를 아는 베이스라인. predict_steps(track, now, horizon, goal) -> [(t,x,z,sigma)…]."""

    def __init__(self, n_steps: int = 12):
        self.n_steps = n_steps
        self._cv = ConstantVelocityPredictor(n_steps=n_steps)

    def predict_steps(self, track, now: float, horizon: float, goal):
        if goal is None:                                   # 목표 없음 → 등속 폴백
            sc = TrackScene(now=now, horizon=horizon, agents=[track], map=None)
            return self._cv.predict(sc).per_agent[track.id][0].steps

        hist = track.history
        _, x0, z0 = hist[-1]
        gx, gz = goal
        speed = 0.0
        if len(hist) >= 2:
            t0, ax, az = hist[0]
            t1, bx, bz = hist[-1]
            dt = t1 - t0
            if dt > 1e-9:
                speed = math.hypot(bx - ax, bz - az) / dt
        dx, dz = gx - x0, gz - z0
        dist = math.hypot(dx, dz)
        ux, uz = (dx / dist, dz / dist) if dist > 1e-9 else (0.0, 0.0)
        steps = []
        for i in range(1, self.n_steps + 1):
            tau = horizon * i / self.n_steps
            travel = min(speed * tau, dist)                # 목표를 지나치지 않는다
            steps.append((now + tau, x0 + ux * travel, z0 + uz * travel, 0.0))
        return steps
