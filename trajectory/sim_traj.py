"""sim 궤적(dataset/trajectories/*.json) → 예측 윈도우 로더. 이슈 #2 3단계.

2단계에서 모은 scene 연속 궤적을 관측 8스텝 / 예측 12스텝(Trajectron++ 관례)으로
슬라이딩 윈도우 잘라, 예측기가 먹는 TrackScene + GT로 만든다. scene 단위 train/val 분할.
설계: docs/chanwoo/specs/2026-08-19-baseline-ade-fde-design.md
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from trajectory.types import Track, TrackScene

MANIFEST_NAME = "split_manifest.json"

OBS, PRED = 8, 12          # 관측 3.2s / 예측 4.8s (2.5Hz)
# '움직인' 윈도우 = 예측 구간 경로 길이(스텝 합)가 이 이상. 경로 길이라 왕복도 잡되,
# 임계 1.0m로 정지+지터(4.8s에 ~0.2m)를 걸러 걷기만 남긴다 — 이 값이 낮으면(0.2m)
# 지터만으로 거의 전부 '움직인'으로 잡혀 부분집합이 무의미해진다(리뷰 지적).
MOVE_EPS = 1.0

ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = ROOT / "dataset" / "trajectories"

# 로봇 베이스 기본값 (x, z). 초기 궤적 캡처엔 scene에 robot 필드가 없다 —
# 그 시절에도 로봇은 이 자리(LAYOUT.robot.base)에 있었으므로 없을 때 이 값을 쓴다.
ROBOT_BASE = (-1.1, 0.815)


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
    robot: tuple = ROBOT_BASE               # scene 로봇 위치 (x, z) — 안전 진입 평가용


def is_val(seed: int) -> bool:
    """[deprecated] 예전 seed%5==0 val 분할. 새 코드는 split_manifest.json 기반
    load_windows 를 쓴다(P0-1). throwaway 스파이크 호환을 위해 남겨둔다."""
    return seed % 5 == 0


def load_manifest(traj_dir=None, manifest_path=None):
    """split_manifest.json 로드(없으면 None). {'train','val','test': [filename…]}."""
    d = Path(traj_dir) if traj_dir else TRAJ_DIR
    p = Path(manifest_path) if manifest_path else d / MANIFEST_NAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_windows(split: str = "val", traj_dir=None, manifest_path=None) -> list:
    """split in {"train","val","test","all"}. 폐기 노드는 건너뛴다.

    train/val/test 는 split_manifest.json 멤버십으로 필터(P0-1, seed 단위 분할).
    manifest 가 없으면 명확히 오류(silent 폴백 금지) — `python train/make_traj_split.py` 실행.
    'all' 은 manifest 무관하게 모든 sim scene(manifest 파일·비-scene 파일 제외).
    """
    d = Path(traj_dir) if traj_dir else TRAJ_DIR
    members = None
    if split in ("train", "val", "test"):
        man = load_manifest(d, manifest_path)
        if man is None:
            source = Path(manifest_path) if manifest_path else d / MANIFEST_NAME
            raise FileNotFoundError(
                f"manifest 없음: {source}. 먼저 split manifest를 생성하세요.")
        members = set(man[split])
    elif split != "all":
        raise ValueError(f"알 수 없는 split: {split!r} (train/val/test/all)")
    wins = []
    for f in sorted(glob.glob(str(d / "*.json"))):
        base = os.path.basename(f)
        if base == MANIFEST_NAME:
            continue
        if members is not None and base not in members:
            continue
        with open(f, encoding="utf-8") as fh:
            sc = json.load(fh)
        if "nodes" not in sc or "seed" not in sc:      # sim scene 아님(예: real_test_sample) → 제외
            continue
        seed = int(sc["seed"])
        rb = sc.get("robot") or {}
        robot = (float(rb.get("x", ROBOT_BASE[0])), float(rb.get("z", ROBOT_BASE[1])))
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
                # 예측 지평선 = 마지막 GT 시각 - now. 캡처가 0.4s 균일 간격이면 ≈ PRED*0.4=4.8s.
                # 등속·칼만은 이 horizon으로 위치를 스케일하므로(predictors.py), 캡처 주기가
                # 달라지면 베이스라인 예측이 달라진다 — 안전 eval 재현성은 균일 0.4s 가정에 의존.
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
                wins.append(Window(sc["scene_id"], seed, node["id"], scene, gt, goal, moved, robot))
    return wins
