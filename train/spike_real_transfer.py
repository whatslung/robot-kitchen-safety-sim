"""[SPIKE — 버리는 실험] 우리 sim-학습 LSTM을 실사에 zero-shot(스케일 다리). 이슈 #2 sim2real.

우리 LSTM은 미터 스케일 학습(스텝 ~0.22), 실사는 정규화-px(스텝 ~13px) → 스케일 ~50-80배 차이.
에이전트 중심 정규화는 평행이동·회전만 없애고 스케일은 안 없앤다. 그래서 **관측 스텝 크기를
학습 스케일에 맞추는 다리**를 걸어 우리 LSTM(최빈 모드)을 실사에 돌리고 등속과 비교한다.
정지 윈도우는 스케일이 발산하므로 '제자리 유지'로 처리. 실행: uv run python train/spike_real_transfer.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "train"))
from spike_real_baseline import windows, OBS, PRED, IMG   # 실사 로더 재사용
from trajectory.learned_predictor import LearnedPredictor

SIM_STEP = 0.22        # 우리 학습 데이터의 대략적 스텝 크기(m, =AU) — 다리의 목표 스케일
MIN_OBS_STEP = 2.0     # 관측 평균 스텝(px)이 이 미만이면 정지로 보고 제자리 유지


def ade_fde(pred_px, gt_px):
    d = [((pred_px[i][0]-gt_px[i][0])**2 + (pred_px[i][1]-gt_px[i][1])**2) ** 0.5 for i in range(len(gt_px))]
    return float(np.mean(d)), float(d[-1])


def main():
    wins = windows()
    lp = LearnedPredictor(weights_path=str(ROOT / "training/traj_predictor/model.pt"), device="cpu")
    # 배치 예측을 위해 스케일된 관측을 모아 둔다
    metas = []      # (last_px, scale, gt_px, moved)
    scaled_hists = []
    for w in wins:
        obs = np.array([[x*IMG, y*IMG] for (_, x, y) in w[:OBS]])
        gt = [(x*IMG, y*IMG) for (_, x, y) in w[OBS:]]
        last = obs[-1]
        steps = np.linalg.norm(np.diff(obs, axis=0), axis=1)
        obs_scale = float(np.mean(steps))
        moved = float(np.sum(np.linalg.norm(np.diff(np.array([[x*IMG, y*IMG] for (_, x, y) in w[OBS-1:]]), axis=0), axis=1))) > 10.0
        if obs_scale < MIN_OBS_STEP:
            metas.append((last, None, gt, moved))            # 정지 → 제자리
            scaled_hists.append([(0.0, 0.0)] * OBS)          # placeholder
        else:
            s = SIM_STEP / obs_scale
            sc = [(last[0] + (p[0]-last[0])*s, last[1] + (p[1]-last[1])*s) for p in obs]
            metas.append((last, s, gt, moved))
            scaled_hists.append(sc)
    modes_all = lp.predict_batch(scaled_hists)

    rec_lstm, rec_cv = [], []
    for (last, s, gt, moved), modes in zip(metas, modes_all):
        if s is None:
            pred = [(last[0], last[1])] * PRED                # 정지 유지
        else:
            ml = modes[0]["path"]                             # 최빈 모드(스케일 공간)
            pred = [(last[0] + (p[0]-last[0])/s, last[1] + (p[1]-last[1])/s) for p in ml]  # 역스케일→px
        rec_lstm.append((*ade_fde(pred, gt), moved))
    # 등속은 baseline 스파이크와 동일 정의로 별도 계산(간단히 마지막 관측 속도 외삽)
    for w in wins:
        obs = np.array([[x*IMG, y*IMG] for (_, x, y) in w[:OBS]])
        gt = [(x*IMG, y*IMG) for (_, x, y) in w[OBS:]]
        v = (obs[-1] - obs[0]) / (OBS - 1)
        pred = [(obs[-1][0] + v[0]*(i+1), obs[-1][1] + v[1]*(i+1)) for i in range(PRED)]
        moved = float(np.sum(np.linalg.norm(np.diff(np.array([[x*IMG, y*IMG] for (_, x, y) in w[OBS-1:]]), axis=0), axis=1))) > 10.0
        rec_cv.append((*ade_fde(pred, gt), moved))

    def agg(rec, mv):
        r = [x for x in rec if (x[2] or not mv)]
        return np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), len(r)

    print(f"실사 윈도우 {len(wins)}")
    print(f"{'':<18}{'ADE(px)':>9}{'FDE(px)':>9}")
    for lab, rec in (("등속", rec_cv), ("우리 LSTM(다리)", rec_lstm)):
        a, f, _ = agg(rec, False)
        am, fm, n = agg(rec, True)
        print(f"{lab:<18}{a:>9.2f}{f:>9.2f}   | 움직인 것만 {am:.2f}/{fm:.2f} (n={n})")
    print("\n해석: 우리 LSTM(다리)이 등속보다 낮으면 sim 모션 패턴이 실사에 전이됨. 비슷/높으면 전이 약함.")


if __name__ == "__main__":
    main()
