"""이슈 #2 4단계 — 학습형 멀티모달 예측기 학습·평가.

2단계 궤적(train scene)으로 경량 LSTM+혼합 헤드를 학습하고, val(seed%5)에서 ADE/FDE를
베이스라인과 나란히 재어 docs/chanwoo/prediction-eval.md 를 통합 표로 갱신한다.

실행:  uv run python train/train_traj_predictor.py
설계:  docs/chanwoo/specs/2026-08-19-learned-predictor-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from trajectory.sim_traj import load_windows, OBS, PRED
from trajectory.learned_predictor import (build_net, mtp_loss, frame_of, to_frame,
                                          LearnedPredictor, K)
from eval_traj_baselines import evaluate as eval_baselines   # 베이스라인 표 재사용

EPOCHS, BATCH, HIDDEN, LR, SEED = 140, 512, 64, 1e-3, 0
# 검출 노이즈 증강(sim-to-real) — 관측에 σ=0.06m(=PRED.sigM) 가우시안을 정규화 전에 주입한
# 사본을 train에만 추가. 실제 나디르→YOLO→추적 좌표의 흔들림에 강해지게. 스파이크 실측:
# 깨끗-학습은 0.06 노이즈에서 +36% 악화, 노이즈 증강이 -16% 회복. val은 깨끗 유지(베이스라인 비교).
NOISE_STD, NOISE_COPIES = 0.06, 2
WEIGHTS = ROOT / "training" / "traj_predictor" / "model.pt"


def _xz(seq):
    return [(p[1], p[2]) for p in seq]


def build_xy(split):
    """정규화된 관측/미래 텐서 + 원본 윈도우(미터 eval용).
    train은 노이즈 증강 사본을 추가(깨끗 1 + 노이즈 NOISE_COPIES). val은 깨끗만."""
    wins = load_windows(split)
    aug = (split == "train") and NOISE_STD > 0
    rng = np.random.default_rng(SEED)
    X, Y = [], []
    for w in wins:
        hist = np.asarray(_xz(w.scene.agents[0].history), float)   # (OBS,2) 미터
        gt = _xz(w.gt)
        variants = [hist]
        if aug:
            for _ in range(NOISE_COPIES):
                variants.append(hist + rng.normal(0, NOISE_STD, hist.shape))  # 정규화 전 노이즈
        for h in variants:
            origin, ang = frame_of(h)
            X.append(to_frame(h, origin, ang))
            Y.append(to_frame(gt, origin, ang))    # 미래=참값(관측과 같은 프레임)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), wins


def _ade(path, gt):
    return float(np.mean([((px - gx) ** 2 + (pz - gz) ** 2) ** 0.5
                          for (px, pz), (gx, gz) in zip(path, gt)]))


def _fde(path, gt):
    (px, pz), (gx, gz) = path[-1], gt[-1]
    return float(((px - gx) ** 2 + (pz - gz) ** 2) ** 0.5)


def eval_learned(lp, wins):
    """최빈 모드 + minADE/FDE@K 를 all/moved 로 집계. 전체 윈도우를 한 번에 배치 추론."""
    rec = {"ml": [], "mink": []}          # (ade, fde, moved)
    all_modes = lp.predict_batch([_xz(w.scene.agents[0].history) for w in wins])
    for w, modes in zip(wins, all_modes):
        gt = _xz(w.gt)
        ml = modes[0]["path"]             # 가중치 최상위
        ades = [_ade(m["path"], gt) for m in modes]
        fdes = [_fde(m["path"], gt) for m in modes]
        rec["ml"].append((_ade(ml, gt), _fde(ml, gt), w.moved))
        rec["mink"].append((min(ades), min(fdes), w.moved))
    return rec


def _agg(recs, moved_only):
    sel = [(a, f) for (a, f, m) in recs if (m or not moved_only)]
    n = len(sel)
    return (sum(a for a, _ in sel) / n, sum(f for _, f in sel) / n, n)


def main():
    import torch
    torch.manual_seed(SEED)
    # 재현성 — CPU는 결정적. GPU는 cuDNN LSTM에 결정적 알고리즘이 없어 런 간 미세 변동이
    # 남는다(아래 플래그로 최대한 억제하되 완전 고정은 불가). 지표는 런 간 소폭만 흔들린다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    Xtr, Ytr, _ = build_xy("train")
    print(f"train 윈도우 {len(Xtr)}")
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(Ytr, device=dev)
    net = build_net(h=HIDDEN).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    n = len(xt)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            paths, logits, logsig = net(xt[idx])
            loss = mtp_loss(paths, logits, logsig, yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3d}  loss {tot / n:.4f}")

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), WEIGHTS)
    print(f"가중치 저장 → {WEIGHTS}")

    # 평가 (val)
    val_wins = load_windows("val")
    lp = LearnedPredictor(net=net, device=dev)
    rec = eval_learned(lp, val_wins)
    base = eval_baselines("val")

    def table(moved_only):
        key = "moved" if moved_only else "all"
        rows = [("등속(const-vel)", base["rows"]["등속(const-vel)"][key]),
                ("칼만(Kalman)", base["rows"]["칼만(Kalman)"][key]),
                ("스테이션(goal)", base["rows"]["스테이션(goal)"][key]),
                ("학습형 LSTM(최빈)", _agg(rec["ml"], moved_only)),
                (f"학습형 LSTM(minADE@{K})", _agg(rec["mink"], moved_only))]
        out = ["| 예측기 | ADE(m) | FDE(m) | 윈도우 수 |", "|---|---|---|---|"]
        for name, (a, f, nn) in rows:
            out.append(f"| {name} | {a:.3f} | {f:.3f} | {nn} |")
        return "\n".join(out)

    print("\n[전체 윈도우]\n" + table(False))
    print("\n[움직인 윈도우만]\n" + table(True))

    doc = ROOT / "docs" / "chanwoo" / "prediction-eval.md"
    doc.write_text(
        "# 궤적 예측 평가 — 베이스라인 vs 학습형 (이슈 #2 3·4단계)\n\n"
        f"> 자동 생성: `train/train_traj_predictor.py` · 데이터 `dataset/trajectories/`\n"
        f"> 관측 {OBS}스텝(3.2s) / 예측 {PRED}스텝(4.8s) · val = seed%5==0 · 학습형 K={K}\n\n"
        f"val scene 시드: {base['seeds']} · 윈도우 전체 {base['n_all']} · 움직인 것 {base['n_moved']}\n\n"
        "## 전체 윈도우\n\n" + table(False) + "\n\n"
        "## 움직인 윈도우만 (정지 구간 제외 — 예측 난이도가 드러난다)\n\n" + table(True) + "\n\n"
        "## 읽는 법\n\n"
        "- **ADE/FDE**: 예측 위치오차 평균/최종(m). 낮을수록 좋다.\n"
        "- **스테이션(goal)**: 기록된 현재 목표를 아는 베이스라인(실배포엔 목표 추정기 필요).\n"
        "- **학습형(최빈)**: 가중치 최상위 모드 — 단봉 베이스라인과 직접 비교하는 대표값.\n"
        f"- **학습형(minADE@{K})**: K개 모드 중 최선 — 멀티모달이 정답 갈래를 담고 있는지(상한).\n"
        "- 목표 미조건 학습형이 등속·칼만을 이기고 스테이션(목표 앎)에 근접하면 성공.\n",
        encoding="utf-8")
    print(f"\n표를 {doc} 에 기록했다.")


if __name__ == "__main__":
    main()
