"""detector-track E2E 평가의 순수 함수 (감사 P0-2, 설계 §4-1).

GT-트랙과 검출-트랙(YOLO+ByteTrack)을 같은 클립에서 비교하기 위한 좌표·라벨 무관
순수 로직만 담는다. 데이터·가중치에 의존하는 파이프라인은 train/eval_traj_dettrack.py.

좌표는 정규화(0~1, cx/cy). 위험 계산은 trajectory.risk 를 그대로 재사용해
라이브·오프라인이 같은 위험 코드를 공유한다(설계 §1).
"""
from __future__ import annotations

import math
from collections import Counter

from trajectory import risk
from trajectory.types import Mode


def assign_per_frame(gt_pts, det_tracks, max_dist):
    """GT 트랙의 각 프레임에, 그 프레임에 존재하고 중심거리가 max_dist 이내인
    **가장 가까운** 검출 트랙 id 를 배정한다. 없으면 None(그 프레임 미검출).

    gt_pts       = [(frame, x, y), …]
    det_tracks   = {det_id: [(frame, x, y), …], …}
    반환          = [det_id | None, …]  (gt_pts 와 같은 길이·순서)
    """
    # 프레임 → [(det_id, x, y)] 색인(프레임당 후보 조회를 O(1)로).
    by_frame: dict[int, list[tuple]] = {}
    for did, pts in det_tracks.items():
        for (f, x, y) in pts:
            by_frame.setdefault(f, []).append((did, x, y))

    out = []
    for (f, gx, gy) in gt_pts:
        best_id, best_d = None, max_dist
        for (did, dx, dy) in by_frame.get(f, ()):
            d = math.hypot(dx - gx, dy - gy)
            # 동률 시 작은 id 우선(결정성) — d < best_d 는 첫 최소를 유지하므로
            # 같은 거리의 더 큰 id 로 바뀌지 않는다. 후보를 id 순으로 돌지 않아도
            # 되게 <= 대신 < 를 쓰고, 동거리 타이는 아래에서 정렬로 보장.
            if d < best_d or (d == best_d and best_id is not None and did < best_id):
                best_d, best_id = d, did
        out.append(best_id)
    return out


def classify_failures(assigned):
    """per-frame 할당열(assign_per_frame 출력) → 검출 실패 지표.

    miss        = 미검출 프레임 수(None).
    fragments   = 이 GT 를 덮은 서로 다른 검출 id 수(정상=1, 여러 개면 트랙 쪼개짐).
    id_switches = 검출 id 가 바뀐 횟수 — None(갭)을 압축한 뒤 인접 비교(같은 GT가
                  다른 id 로 넘어간 횟수). 갭을 사이에 둔 변경도 1회로 센다.
    """
    non_none = [a for a in assigned if a is not None]
    fragments = len(set(non_none))
    id_switches = sum(1 for a, b in zip(non_none, non_none[1:]) if a != b)
    return {"miss": assigned.count(None), "fragments": fragments, "id_switches": id_switches}


def match_track(assigned):
    """per-frame 할당열에서 이 GT 를 대표하는 **지배 검출 id**(최다 프레임).
    동률은 작은 id(결정성). 전부 미검출이면 None."""
    non_none = [a for a in assigned if a is not None]
    if not non_none:
        return None
    counts = Counter(non_none)
    top = max(counts.values())
    return min(did for did, c in counts.items() if c == top)


def modes_from_prediction(pred_modes):
    """예측기 출력(list[Mode], Mode.steps=[(t,x,z,sigma)…]) → risk 가 기대하는
    모드 형식 [{"path":[(x,z)…], "w":prob, "sigma":[σ…]}]."""
    out = []
    for m in pred_modes:
        path = [(x, z) for (_t, x, z, _s) in m.steps]
        sigma = [s for (_t, _x, _z, s) in m.steps]
        out.append({"path": path, "w": m.prob, "sigma": sigma})
    return out


def virtual_robot_risk(pred_modes, robot, stopR, slowR, horizon, ksig=1.0, tau=0.1):
    """실사 클립엔 로봇이 없으므로 가상 로봇(robot, 반경)을 얹어 위험진입을 계산한다
    (설계 §1·§5). 예측기 출력을 risk 모드로 어댑트해 trajectory.risk.track_risk 재사용."""
    return risk.track_risk(modes_from_prediction(pred_modes), robot,
                           stopR, slowR, horizon, ksig, tau)


def aggregate(records):
    """평가 레코드를 그룹별 평균으로 집계.

    records = [{"group": hashable, "ade": float, "fde": float, "moved": bool}, …]
    반환      = {group: {"n", "ade", "fde", "n_moved", "ade_moved", "fde_moved"}}
      *_moved 는 예측 구간이 실제로 움직인 윈도우만(정지 다수인 실사에서 진짜 난이도).
      움직인 윈도우가 없으면 NaN.
    """
    groups: dict = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    out = {}
    for g, rs in groups.items():
        moved = [r for r in rs if r["moved"]]
        out[g] = {
            "n": len(rs),
            "ade": _mean([r["ade"] for r in rs]),
            "fde": _mean([r["fde"] for r in rs]),
            "n_moved": len(moved),
            "ade_moved": _mean([r["ade"] for r in moved]),
            "fde_moved": _mean([r["fde"] for r in moved]),
        }
    return out
