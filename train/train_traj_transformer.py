"""검증된 구조 비교(옵션 B) — Transformer 백본 궤적 예측기 학습.

train_traj_predictor 의 데이터 파이프라인(build_xy: 같은 train split·노이즈 증강)과 mtp_loss·
학습 루프를 그대로 재사용하고 **백본만 Transformer**로 바꿔 학습한다(공정 A/B). 문서는 쓰지 않는다
— 비교표는 eval_traj_split.py 가 단독 생성. 결정적(SEED=0).

실행:  uv run --group serve python train/train_traj_transformer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from trajectory.learned_predictor import build_transformer_net, mtp_loss   # noqa: E402
from train_traj_predictor import build_xy, EPOCHS, BATCH, HIDDEN, LR, SEED  # noqa: E402

WEIGHTS = ROOT / "training" / "traj_predictor" / "model_transformer.pt"


def main():
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} · arch=transformer")

    Xtr, Ytr, _ = build_xy("train")
    print(f"train 윈도우 {len(Xtr)}")
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(Ytr, device=dev)
    net = build_transformer_net(h=HIDDEN).to(dev)
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
    print("비교표는 `uv run --group serve python train/eval_traj_split.py` 로 생성.")


if __name__ == "__main__":
    main()
