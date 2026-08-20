"""[SPIKE — 버리는 실험] 안전 운영점 튜닝. 이슈 #2 — recall/precision 곡선.

spike_safety의 "전모드 합집합"은 너무 보수적(recall 0.76 · precision 0.44). 여기선
**진입하는 모드의 확률질량 합 >= τ** 로 경보를 내고 τ를 훑어 recall/precision 곡선을 그린다.
τ→0 이면 "아무 모드나 진입"(recall↑·precision↓), τ↑ 이면 "확신할 때만"(precision↑·recall↓).
선제 안전층이라 recall 우선 — recall을 최대한 지키며 헛정지(1-precision)를 줄이는 지점을 찾는다.
+ σ 팽창(진입 판정에 -k·σ) 옵션도 함께 본다. 실행: uv run python train/spike_oppoint.py
"""
from __future__ import annotations
import math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "train"))
from spike_safety import windows_val, _mind      # 동일 윈도우 로더 재사용
from trajectory.learned_predictor import LearnedPredictor, PRED

R = 3.1
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def mode_enters(path, robot, sigma, ksig):
    """모드가 진입? min(거리 - k·σ) < R. ksig=0이면 순수 거리."""
    best = 1e9
    for i, (x, z) in enumerate(path):
        d = math.hypot(x - robot[0], z - robot[1]) - ksig * (sigma[i] if sigma else 0.0)
        best = min(best, d)
    return best < R


def run(ksig):
    wins = windows_val()
    lp = LearnedPredictor(weights_path=str(ROOT / "training/traj_predictor/model.pt"), device="cpu")
    anti = [(o, g, rb) for (o, g, rb) in wins
            if math.hypot(o[-1][1] - rb[0], o[-1][2] - rb[1]) >= R]
    modes_all = lp.predict_batch([[(o[1], o[2]) for o in w[0]] for w in anti])
    rows = []
    for w, modes in zip(anti, modes_all):
        gt = _mind(w[1], w[2]) < R
        mass = sum(m["w"] for m in modes if mode_enters(m["path"], w[2], m.get("sigma"), ksig))
        rows.append((mass, gt))
    n_pos = sum(1 for _, g in rows if g)
    out = []
    for tau in TAUS:
        tp = sum(1 for mass, g in rows if mass >= max(tau, 1e-9) and g)
        fp = sum(1 for mass, g in rows if mass >= max(tau, 1e-9) and not g)
        fn = n_pos - tp
        rec = tp / (tp + fn) if tp + fn else float('nan')
        pre = tp / (tp + fp) if tp + fp else float('nan')
        out.append((tau, rec, pre, tp, fp, fn))
    return len(anti), n_pos, out


def main():
    print(f"=== SPIKE: 안전 운영점 튜닝 (R={R}m, 진입 모드 확률질량 임계 τ) ===")
    for ksig, lab in [(0.0, "거리만"), (1.0, "σ팽창 k=1(보수적)")]:
        n_anti, n_pos, out = run(ksig)
        print(f"\n[{lab}] 대상 {n_anti} · 실제 진입 {n_pos}")
        print(f"{'τ':>6}{'recall':>9}{'precision':>11}   (TP/FP/FN)")
        for tau, rec, pre, tp, fp, fn in out:
            print(f"{tau:>6.2f}{rec:>9.3f}{pre:>11.3f}   ({tp}/{fp}/{fn})")
    print("\n선제 안전층 권장: recall 우선 — 낮은 τ에서 recall 최대, precision은 헛정지 허용선까지.")


if __name__ == "__main__":
    main()
