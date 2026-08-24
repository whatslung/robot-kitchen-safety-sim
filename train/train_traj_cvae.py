"""이산 잠재 CVAE 궤적 예측기 학습 (설계 §3).

train_traj_predictor.build_xy(같은 train split·노이즈 증강) 재사용 + net.elbo 로 학습.
β KL 어닐링(초반 β≈0 → 선형 증가)으로 posterior collapse 방지. 결정적(SEED=0).
문서 미작성 — 비교표는 eval_traj_split.py 가 생성.

실행:  uv run --group serve python train/train_traj_cvae.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from trajectory.learned_predictor import build_cvae_net                     # noqa: E402
from train_traj_predictor import build_xy, EPOCHS, BATCH, HIDDEN, LR, SEED  # noqa: E402

WEIGHTS = ROOT / "training" / "traj_predictor" / "model_cvae.pt"
BETA_MAX = 1.0
ANNEAL_FRAC = 0.5           # 앞 절반 에폭에 걸쳐 β 0→BETA_MAX 선형 증가


def main():
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} · arch=cvae(discrete latent)")

    Xtr, Ytr, _ = build_xy("train")
    print(f"train 윈도우 {len(Xtr)}")
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(Ytr, device=dev)
    net = build_cvae_net(h=HIDDEN).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    n = len(xt)
    for ep in range(EPOCHS):
        beta = BETA_MAX * min(1.0, ep / max(1, int(EPOCHS * ANNEAL_FRAC)))
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = rec = klt = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = net.elbo(xt[idx], yt[idx], beta=beta)
            out["loss"].backward()
            opt.step()
            m = len(idx)
            tot += out["loss"].item() * m; rec += out["recon"].item() * m; klt += out["kl"].item() * m
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3d}  β {beta:.2f}  loss {tot/n:.4f}  recon {rec/n:.4f}  kl {klt/n:.4f}")

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), WEIGHTS)
    print(f"가중치 저장 → {WEIGHTS}")
    print("비교표는 `uv run --group serve python train/eval_traj_split.py` 로 생성.")


if __name__ == "__main__":
    main()
