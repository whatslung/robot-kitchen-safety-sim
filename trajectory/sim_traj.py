"""sim 궤적(dataset/trajectories/*.json) → 예측 윈도우 로더. 이슈 #2 3단계.

2단계에서 모은 scene 연속 궤적을 관측 8스텝 / 예측 12스텝(Trajectron++ 관례)으로
슬라이딩 윈도우 잘라, 예측기가 먹는 TrackScene + GT로 만든다. scene 단위 train/val 분할.
설계: docs/chanwoo/specs/2026-08-19-baseline-ade-fde-design.md
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from trajectory.types import Track, TrackScene

OBS, PRED = 8, 12          # 관측 3.2s / 예측 4.8s (2.5Hz)
# '움직인' 윈도우 = 예측 구간 경로 길이(스텝 합)가 이 이상. 경로 길이라 왕복도 잡되,
# 임계 1.0m로 정지+지터(4.8s에 ~0.2m)를 걸러 걷기만 남긴다 — 이 값이 낮으면(0.2m)
# 지터만으로 거의 전부 '움직인'으로 잡혀 부분집합이 무의미해진다(리뷰 지적).
MOVE_EPS = 1.0

ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = ROOT / "dataset" / "trajectories"


@dataclass
class Window:
    """예측 윈도우 하나. scene=관측(8) 입력, gt=예측(12) 정답, goal=관측 마지막 목표."""
    scene_id: str
    seed: int
    node_id: str
    scene: TrackScene                       # now·horizon·agents=[Track(0, 8×(t,x,z))]
    gt: list                                # 12×(t, x, z)
    goal: Optional[tuple]                   # (gx, gz) at last obs step, 통과 구간이면 None
    moved: bool                             # 예측 구간 총 이동 > MOVE_EPS


def is_val(seed: int) -> bool:
    """scene 단위 분할 — seed%5==0 이 val(20%). 4단계도 같은 val을 쓴다(공정 비교)."""
    return seed % 5 == 0


def load_windows(split: str = "val", traj_dir=None) -> list:
    """split in {"val","train","all"}. 폐기 노드는 건너뛴다."""
    d = Path(traj_dir) if traj_dir else TRAJ_DIR
    wins = []
    for f in sorted(glob.glob(str(d / "*.json"))):
        with open(f, encoding="utf-8") as fh:
            sc = json.load(fh)
        seed = int(sc["seed"])
        val = is_val(seed)
        if split == "val" and not val:
            continue
        if split == "train" and val:
            continue
        for node in sc["nodes"]:
            if node.get("discarded"):
                continue
            fr = node["frames"]
            for s in range(0, len(fr) - (OBS + PRED) + 1):
                obs = fr[s:s + OBS]
                fut = fr[s + OBS:s + OBS + PRED]
                hist = [(o["t"], o["x"], o["z"]) for o in obs]
                gt = [(o["t"], o["x"], o["z"]) for o in fut]
                now = hist[-1][0]
                horizon = gt[-1][0] - now
                last = obs[-1]
                goal = (last["gx"], last["gz"]) if last.get("gx") is not None else None
                # moved = 예측 구간 실제 이동 거리(경로 길이). 끝점 변위가 아니라 스텝 합 —
                # 나갔다 되돌아오는(왕복) 윈도우도 '움직인' 것으로 잡는다.
                path = [(hist[-1][1], hist[-1][2])] + [(g[1], g[2]) for g in gt]
                dist = sum(((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5
                           for i in range(1, len(path)))
                moved = dist > MOVE_EPS
                scene = TrackScene(now=now, horizon=horizon, agents=[Track(0, hist)], map=None)
                wins.append(Window(sc["scene_id"], seed, node["id"], scene, gt, goal, moved))
    return wins
