"""[SPIKE — 버리는 실험] 안전 결정 평가. 이슈 #2 — 목표 직결(충돌 방지·불필요 정지 방지).

ADE/FDE(위치오차) 대신, **"정지반경 진입을 미리 맞혔나"의 recall/precision**을 잰다.
현재 정지반경 **밖**에 있는 사람만 대상(진입 예측 = 예측의 값어치, 반응형이 못 하는 것):
  GT+  = 실제로 예측 지평선(4.8s) 안에 정지반경 진입(미래 최소거리 < R)
  Pred+= 예측이 진입이라 판정(예측 최소거리 < R). 학습형은 (a)최빈 모드 (b)전 모드 합집합.
  recall = 실제 진입 중 미리 잡은 비율(놓치면 충돌) · precision = 경보 중 진짜 비율(낮으면 헛정지)
robot 위치는 scene 메타(없으면 기본). R = 정지반경(SAFE.NOM_STOP=3.1) + 참고로 2.0.
실행: uv run python train/spike_safety.py
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.types import Track, TrackScene
from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor
from trajectory.learned_predictor import LearnedPredictor
from trajectory.sim_traj import is_val, OBS, PRED, TRAJ_DIR

ROBOT_DEFAULT = (-1.1, 0.815)
RADII = [3.1, 2.0]


def windows_val():
    """val 윈도우: (hist[(t,x,z)], gt_xz[(x,z)], robot(x,z))."""
    out = []
    for f in sorted(glob.glob(str(TRAJ_DIR / "*.json"))):
        sc = json.load(open(f, encoding="utf-8"))
        if not is_val(sc["seed"]):
            continue
        rb = sc.get("robot") or {}
        robot = (rb.get("x", ROBOT_DEFAULT[0]), rb.get("z", ROBOT_DEFAULT[1]))
        for n in sc["nodes"]:
            if n.get("discarded"):
                continue
            fr = n["frames"]
            for s in range(0, len(fr) - (OBS + PRED) + 1):
                obs = [(fr[i]["t"], fr[i]["x"], fr[i]["z"]) for i in range(s, s + OBS)]
                gt = [(fr[i]["x"], fr[i]["z"]) for i in range(s + OBS, s + OBS + PRED)]
                out.append((obs, gt, robot))
    return out


def _mind(pts, robot):
    return min(math.hypot(x - robot[0], z - robot[1]) for (x, z) in pts)


def evaluate(R):
    wins = windows_val()
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    lp = LearnedPredictor(weights_path=str(ROOT / "training/traj_predictor/model.pt"), device="cpu")
    hists = [[(o[1], o[2]) for o in w[0]] for w in wins]
    modes_all = lp.predict_batch(hists)

    # 카운터: 각 예측기 TP/FP/FN
    keys = ["등속", "칼만", "학습형(최빈)", "학습형(전모드)"]
    C = {k: [0, 0, 0] for k in keys}   # TP, FP, FN
    n_anti = 0
    for (obs, gt, robot), modes in zip(wins, modes_all):
        cur = math.hypot(obs[-1][1] - robot[0], obs[-1][2] - robot[1])
        if cur < R:
            continue                    # 이미 안쪽 → 반응형 몫, 진입 예측 대상 아님
        n_anti += 1
        gt_pos = _mind(gt, robot) < R   # 실제 진입?
        sc = TrackScene(now=obs[-1][0], horizon=PRED * 0.4, agents=[Track(0, obs)], map=None)
        cvp = [(x, z) for (_, x, z, _) in cv.predict(sc).per_agent[0][0].steps]
        kfp = [(x, z) for (_, x, z, _) in kf.predict(sc).per_agent[0][0].steps]
        ml = modes[0]["path"]
        preds = {
            "등속": _mind(cvp, robot) < R,
            "칼만": _mind(kfp, robot) < R,
            "학습형(최빈)": _mind(ml, robot) < R,
            "학습형(전모드)": any(_mind(m["path"], robot) < R for m in modes),  # 보수적 합집합
        }
        for k in keys:
            p = preds[k]
            if p and gt_pos: C[k][0] += 1
            elif p and not gt_pos: C[k][1] += 1
            elif (not p) and gt_pos: C[k][2] += 1
    return n_anti, C, sum(1 for (o, g, r) in wins if _mind(g, r) < R and math.hypot(o[-1][1]-r[0], o[-1][2]-r[1]) >= R)


def main():
    print("=== SPIKE: 안전 결정 평가 (진입 예측 recall/precision) ===")
    for R in RADII:
        n_anti, C, n_pos = evaluate(R)
        print(f"\n[R={R}m 정지반경] 대상(진입전 밖) 윈도우 {n_anti} · 실제 진입 {n_pos}")
        print(f"{'예측기':<16}{'recall':>8}{'precision':>11}   (TP/FP/FN)")
        for k, (tp, fp, fn) in C.items():
            rec = tp / (tp + fn) if tp + fn else float('nan')
            pre = tp / (tp + fp) if tp + fp else float('nan')
            print(f"{k:<16}{rec:>8.3f}{pre:>11.3f}   ({tp}/{fp}/{fn})")
        print("  참고: 반응형(예측 없음)은 이 윈도우들에서 recall=0 (지금 밖 → 진입을 못 봄).")


if __name__ == "__main__":
    main()
