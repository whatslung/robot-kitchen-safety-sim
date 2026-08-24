"""이슈 #2 — 궤적 예측의 **안전 지표**(정지반경 진입 예측 recall/precision)를 val에서 잰다.

ADE/FDE(위치오차)와 별개로, 목표에 직결된 질문을 잰다:
  "지금 정지반경 **밖**에 있는 사람이 예측 지평선(4.8s) 안에 반경 **안**으로 들어올지 미리 맞혔나?"
현재 반경 안의 사람은 이미 반응형이 멈추므로 대상에서 제외한다(예측의 값어치 = 반응형이 못 하는 것).

  GT+   = 실제 미래 최소거리 < R (진짜 진입)
  Pred+ = 예측 최소거리 < R (진입 경보). 학습형은 (a)최빈 모드 (b)전 모드 합집합(보수적).
  recall = 실제 진입 중 미리 잡은 비율(놓치면 충돌) · precision = 경보 중 진짜 비율(낮으면 헛정지)

선제 안전층이라 **recall 우선**. ADE/FDE와 동일한 val 스플릿(seed%5==0)을 써 공정 비교한다.
(정식화 전 `train/spike_safety.py` 스파이크를 승격한 것 — 로직 동일, evaluator 순수함수 + 테스트로 검증.)

실행:  uv run python train/eval_traj_safety.py
비교:  train/eval_traj_baselines.py (ADE/FDE) · docs/chanwoo/prediction-eval.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                          # trajectory 패키지 import

from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor
from trajectory.learned_predictor import LearnedPredictor
from trajectory.evaluator import enters_radius, recall_precision, entry_confusion
from trajectory.sim_traj import load_windows, PRED

# 정지반경 3.1m = SAFE.NOM_STOP(시뮬 안전링). 2.0m는 더 촘촘한 참고 반경.
RADII = [3.1, 2.0]
PREDICTORS = ["등속(const-vel)", "칼만(Kalman)", "학습형 LSTM(최빈)", "학습형 LSTM(전모드)"]

# 평가 지평선. 캡처는 균일 0.4s(2.5Hz)라 스텝수×0.4 = 초.
#   라이브(1.6s=4스텝) — sim.html PRED.horizon 과 동일한 '실제 로봇 제어' 조건.
#   오프라인(4.8s=12스텝) — 학습·오프라인 평가 지평선(PRED). 발표 대표 수치는 라이브 조건을 쓴다.
STEP_DT = 0.4
HORIZONS = [("라이브 제어 1.6s(4스텝)", 4), ("오프라인 평가 4.8s(12스텝)", PRED)]


def _truncate(path, horizon_steps):
    """예측/GT 경로를 앞에서 horizon_steps개만 남긴다(라이브 지평선 모사). None이면 그대로."""
    return path[:horizon_steps] if horizon_steps else path


def _resolve_weights() -> str:
    """학습형 가중치: 로컬 있으면 그대로, 없으면 허깅페이스에서 받는다(검출기와 동일 방식)."""
    w = Path(os.environ.get("PREDICT_MODEL", str(ROOT / "training" / "traj_predictor" / "model.pt")))
    if w.exists():
        return str(w)
    repo = os.environ.get("PREDICT_MODEL_REPO", "chanubc/human-move-lstm")
    file = os.environ.get("PREDICT_MODEL_FILE", "model.pt")
    print(f"[eval] 예측기 가중치 로컬 없음 → 허브에서 받는다: {repo}/{file}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=file)


def _cur_dist(win) -> float:
    """관측 마지막 위치와 로봇 사이 현재 거리."""
    last = win.scene.agents[0].history[-1]                 # (t, x, z)
    return ((last[1] - win.robot[0]) ** 2 + (last[2] - win.robot[1]) ** 2) ** 0.5


def evaluate(R: float, split: str = "val", traj_dir=None, horizon_steps=None):
    """반경 R에서 예측기별 TP/FP/FN 을 센다. 대상 = 현재 반경 밖 윈도우.

    horizon_steps: 예측/GT 경로를 앞에서 이만큼 스텝만 보고 판정한다(라이브 1.6s=4스텝).
                   None이면 전체 지평선(4.8s=12스텝). GT·예측 양쪽을 같은 지평선으로 잘라
                   '지평선을 짧게 보면 진입을 덜 미리 본다'는 라이브 조건을 그대로 반영한다.
    """
    wins = load_windows(split, traj_dir=traj_dir)
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    lp = LearnedPredictor(weights_path=_resolve_weights(), device="cpu")
    modes_all = lp.predict_batch([[(o[1], o[2]) for o in w.scene.agents[0].history] for w in wins])

    C = {k: [0, 0, 0] for k in PREDICTORS}             # [TP, FP, FN]
    n_anti = n_pos = 0
    for win, modes in zip(wins, modes_all):
        cur = _cur_dist(win)
        if cur < R:
            continue                                    # 이미 안쪽 → 반응형 몫, 대상 아님
        n_anti += 1
        gt_xz = _truncate([(x, z) for (_, x, z) in win.gt], horizon_steps)
        gt_entry = enters_radius(gt_xz, win.robot, R)
        if gt_entry:
            n_pos += 1
        cv_xz = _truncate([(x, z) for (_, x, z, _) in cv.predict(win.scene).per_agent[0][0].steps], horizon_steps)
        kf_xz = _truncate([(x, z) for (_, x, z, _) in kf.predict(win.scene).per_agent[0][0].steps], horizon_steps)
        pred_entry = {
            "등속(const-vel)": enters_radius(cv_xz, win.robot, R),
            "칼만(Kalman)": enters_radius(kf_xz, win.robot, R),
            "학습형 LSTM(최빈)": enters_radius(_truncate(modes[0]["path"], horizon_steps), win.robot, R),
            "학습형 LSTM(전모드)": any(enters_radius(_truncate(m["path"], horizon_steps), win.robot, R) for m in modes),
        }
        for k in PREDICTORS:
            cell = entry_confusion(cur, gt_entry, pred_entry[k], R)
            if cell == "TP":
                C[k][0] += 1
            elif cell == "FP":
                C[k][1] += 1
            elif cell == "FN":
                C[k][2] += 1                            # TN 은 recall/precision에 불필요 → 미집계
    return {"R": R, "n_anti": n_anti, "n_pos": n_pos, "C": C}


def _table(res) -> str:
    lines = ["| 예측기 | recall | precision | (TP/FP/FN) |", "|---|---|---|---|"]
    for k in PREDICTORS:
        tp, fp, fn = res["C"][k]
        rec, pre = recall_precision(tp, fp, fn)
        lines.append(f"| {k} | {rec:.3f} | {pre:.3f} | {tp}/{fp}/{fn} |")
    return "\n".join(lines)


def main():
    # (지평선 라벨, 스텝수) × 반경 = 표 하나. 라이브 1.6s와 오프라인 4.8s를 나란히 낸다.
    results = {label: [evaluate(R, "val", horizon_steps=steps) for R in RADII]
               for label, steps in HORIZONS}
    for label, _ in HORIZONS:
        for res in results[label]:
            print(f"\n[{label} · R={res['R']}m] 대상(진입전 밖) {res['n_anti']} · 실제 진입 {res['n_pos']}")
            print(_table(res))
    print("  참고: 반응형(예측 없음)은 이 윈도우에서 recall=0 (지금 밖이라 진입을 못 봄).")

    doc = ROOT / "docs" / "chanwoo" / "prediction-safety-eval.md"
    body = [
        "# 궤적 예측 안전 지표 — 정지반경 진입 recall/precision (이슈 #2)\n",
        "> 자동 생성: `train/eval_traj_safety.py` · 데이터 `dataset/trajectories/`",
        "> ADE/FDE(위치오차)는 [prediction-eval.md](prediction-eval.md), 여기는 **안전 결정** 지표.",
        "> val = seed%5==0 · 관측 8스텝(3.2s) → 예측 최대 12스텝(4.8s)\n",
        "\"지금 정지반경 **밖**에 있는 사람이 지평선 안에 반경 안으로 진입하는지\"를 미리 맞혔나.",
        "현재 반경 안의 사람은 이미 반응형이 멈추므로 대상 제외. 선제 안전층이라 **recall(놓치면 충돌) 우선**.\n",
        "> ⚠️ **지평선을 반드시 구분한다.** 아래는 두 지평선의 값을 나란히 낸다.",
        "> **라이브 제어 1.6s** = `sim.html`의 `PRED.horizon`, 즉 실제 로봇이 감속·정지에 쓰는 조건이다.",
        "> **오프라인 평가 4.8s** = 학습·오프라인 지평선(12스텝). 지평선이 길수록 진입을 더 일찍 보므로",
        "> recall이 높게 나온다 — **발표 대표 수치는 라이브 1.6s 값을 쓴다.** 4.8s 수치를 라이브 성능처럼",
        "> 제시하지 않는다.\n",
    ]
    for label, _ in HORIZONS:
        body.append(f"# {label}\n")
        for res in results[label]:
            body.append(f"## 정지반경 R = {res['R']} m\n")
            body.append(f"대상(진입 전 밖) 윈도우 **{res['n_anti']}** · 실제 진입 **{res['n_pos']}**\n")
            body.append(_table(res) + "\n")
    body += [
        "## 읽는 법\n",
        "- **recall**: 실제 진입 중 미리 잡은 비율(놓치면 충돌). **precision**: 경보 중 진짜 비율(낮으면 헛정지).",
        "- **학습형(최빈)**: 가중치 최상위 단일 모드. **학습형(전모드)**: K개 모드 합집합(보수적, recall↑·헛정지↑).",
        "- 반응형(예측 없음)은 이 윈도우에서 recall=0 — 지금 밖이라 진입을 못 본다. 예측기의 값어치가 여기서 드러난다.",
        "- 운영점(경보 임계 τ)으로 recall↔precision 균형 조절은 `train/spike_oppoint.py` 참고.",
        "- **지평선 비교**: 같은 예측기라도 1.6s에서 recall이 4.8s보다 낮은 게 정상이다(짧게 보므로).",
        "  라이브 로봇은 1.6s로 판정하므로, 안전 성능 주장은 1.6s 값 기준이어야 한다.",
    ]
    doc.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"\n표를 {doc} 에 기록했다.")


if __name__ == "__main__":
    main()
