"""이슈 #2 3단계 — 베이스라인 3종 ADE/FDE를 val 궤적 윈도우에서 잰다.

등속 · 칼만 · 스테이션 휴리스틱(목표 앎)을 2단계 궤적(dataset/trajectories/*.json)의
val scene(seed%5==0) 윈도우(관측8/예측12)에 적용해 ADE/FDE를 집계하고,
표를 출력 + docs/chanwoo/prediction-eval.md 에 쓴다. 4단계 학습형의 비교 기준선.

실행:  uv run python train/eval_traj_baselines.py
설계:  docs/chanwoo/specs/2026-08-19-baseline-ade-fde-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                          # trajectory 패키지 import

from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor
from trajectory.sim_predictors import StationHeuristicPredictor
from trajectory.evaluator import ade, fde
from trajectory.sim_traj import load_windows, OBS, PRED


def _mean(xs):
    xs = [x for x in xs if x == x]                     # NaN 제거
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate(split="val", traj_dir=None):
    wins = load_windows(split, traj_dir=traj_dir)
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    sh = StationHeuristicPredictor(n_steps=PRED)

    rows = {"등속(const-vel)": [], "칼만(Kalman)": [], "스테이션(goal)": []}
    for w in wins:
        cv_steps = cv.predict(w.scene).per_agent[0][0].steps
        kf_steps = kf.predict(w.scene).per_agent[0][0].steps
        sh_steps = sh.predict_steps(w.scene.agents[0], w.scene.now, w.scene.horizon, w.goal)
        for name, steps in (("등속(const-vel)", cv_steps),
                            ("칼만(Kalman)", kf_steps),
                            ("스테이션(goal)", sh_steps)):
            rows[name].append((ade(steps, w.gt), fde(steps, w.gt), w.moved))

    n_all = len(wins)
    n_moved = sum(1 for w in wins if w.moved)
    seeds = sorted({w.seed for w in wins})

    def agg(recs, moved_only):
        sel = [(a, f) for (a, f, m) in recs if (m or not moved_only)]
        return _mean([a for a, _ in sel]), _mean([f for _, f in sel]), len(sel)

    return {"split": split, "n_all": n_all, "n_moved": n_moved, "seeds": seeds,
            "rows": {k: {"all": agg(v, False), "moved": agg(v, True)} for k, v in rows.items()}}


def _table(res, moved_only):
    key = "moved" if moved_only else "all"
    head = f"| 예측기 | ADE(m) | FDE(m) | 윈도우 수 |\n|---|---|---|---|"
    lines = [head]
    for name, d in res["rows"].items():
        a, f, n = d[key]
        lines.append(f"| {name} | {a:.3f} | {f:.3f} | {n} |")
    return "\n".join(lines)


def main():
    res = evaluate("val")
    print(f"val scene 시드: {res['seeds']}  · 윈도우 전체 {res['n_all']} · 움직인 것 {res['n_moved']}")
    print("\n[전체 윈도우]\n" + _table(res, False))
    print("\n[움직인 윈도우만]\n" + _table(res, True))

    doc = ROOT / "docs" / "chanwoo" / "prediction-eval.md"
    doc.write_text(
        "# 궤적 예측 평가 — 베이스라인 ADE/FDE (이슈 #2 3단계)\n\n"
        f"> 자동 생성: `train/eval_traj_baselines.py` · 데이터 `dataset/trajectories/`\n"
        f"> 관측 {OBS}스텝(3.2s) / 예측 {PRED}스텝(4.8s) · val = seed%5==0\n\n"
        f"val scene 시드: {res['seeds']}\n\n"
        f"윈도우: 전체 **{res['n_all']}** · 움직인 것 **{res['n_moved']}**\n\n"
        "## 전체 윈도우\n\n" + _table(res, False) + "\n\n"
        "## 움직인 윈도우만 (정지 구간 제외 — 예측 난이도가 드러난다)\n\n" + _table(res, True) + "\n\n"
        "## 읽는 법\n\n"
        "- **ADE**: 12스텝 예측 위치오차 평균(m). **FDE**: 12스텝째(4.8s 뒤) 위치오차(m).\n"
        "- 스테이션(goal)은 기록된 현재 목표를 아는 베이스라인 — 실배포엔 목표 추정기가 따로 필요.\n"
        "- 4단계 학습형(LSTM/CVAE)은 이 표를 기준선으로 삼아, 특히 **움직인 윈도우**에서 이겨야 한다.\n",
        encoding="utf-8")
    print(f"\n표를 {doc} 에 기록했다.")


if __name__ == "__main__":
    main()
