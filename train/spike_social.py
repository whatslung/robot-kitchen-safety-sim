"""[SPIKE — 버리는 실험] 이웃(사회적) 특징이 궤적 예측을 개선하나?

이슈 #2 Trajectron++ 확장 검증용. 프로덕션 trajectory/ 는 건드리지 않는다.
현 모델(입력 2특징: 자기 위치)과 이웃 추가(4특징: +가장 가까운 이웃 상대위치)를
같은 아키텍처·에폭·시드로 재학습해 val ADE/FDE를 비교한다. 개선폭으로 전면 사회적
Trajectron++를 지을지 결정한다. 결과만 얻고 이 파일은 남겨도 되지만 프로덕션 아님.

실행: uv run python train/spike_social.py
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.learned_predictor import frame_of, to_frame, mtp_loss, K
from trajectory.sim_traj import is_val, OBS, PRED, TRAJ_DIR

EPOCHS, BATCH, HIDDEN, LR, SEED = 140, 512, 64, 1e-3, 0


def build(split, social):
    X, Y = [], []
    for f in sorted(glob.glob(str(TRAJ_DIR / "*.json"))):
        sc = json.load(open(f, encoding="utf-8"))
        seed = sc["seed"]
        if split == "val" and not is_val(seed):
            continue
        if split == "train" and is_val(seed):
            continue
        nodes = [n for n in sc["nodes"] if not n.get("discarded")]
        pos = {n["id"]: [(fr["x"], fr["z"]) for fr in n["frames"]] for n in nodes}
        for n in nodes:
            fr = pos[n["id"]]
            T = len(fr)
            for s in range(0, T - (OBS + PRED) + 1):
                obs = fr[s:s + OBS]
                fut = fr[s + OBS:s + OBS + PRED]
                origin, ang = frame_of(obs)
                obs_n = to_frame(obs, origin, ang)
                y = to_frame(fut, origin, ang)
                if social:
                    feats = np.zeros((OBS, 4), np.float32)
                    feats[:, :2] = obs_n
                    for k in range(OBS):
                        gi = s + k
                        ox, oz = fr[gi]
                        best, bd = None, 1e9
                        for m in nodes:
                            if m["id"] == n["id"]:
                                continue
                            mx, mz = pos[m["id"]][gi]
                            d = (mx - ox) ** 2 + (mz - oz) ** 2
                            if d < bd:
                                bd, best = d, (mx, mz)
                        if best is not None:
                            feats[k, 2:] = to_frame([best], origin, ang)[0]
                    X.append(feats)
                else:
                    X.append(obs_n.astype(np.float32))
                Y.append(y.astype(np.float32))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def make_net(in_dim, h=HIDDEN, k=K, pred=PRED):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.pred = k, pred
            self.enc = nn.LSTM(in_dim, h, batch_first=True)
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                      nn.Linear(h, k * pred * 2 + k + k * pred))

        def forward(self, x):
            _, (hn, _) = self.enc(x)
            o = self.head(hn[-1])
            b = o.shape[0]
            return (o[:, :k * pred * 2].reshape(b, k, pred, 2),
                    o[:, k * pred * 2:k * pred * 2 + k],
                    o[:, k * pred * 2 + k:].reshape(b, k, pred))
    return Net()


def train_eval(social, tag):
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = build("train", social)
    Xva, Yva = build("val", social)
    in_dim = Xtr.shape[2]
    net = make_net(in_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    n = len(xt)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = mtp_loss(*net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    # eval
    net.eval()
    with torch.no_grad():
        paths, logits, _ = net(torch.tensor(Xva, device=dev))
        paths = paths.cpu().numpy()
        ml = logits.argmax(1).cpu().numpy()
    gt = Yva
    ade_ml = fde_ml = ade_min = fde_min = 0.0
    for i in range(len(gt)):
        d = np.linalg.norm(paths[i] - gt[i][None], axis=2)   # (K,PRED)
        ades = d.mean(1)
        fdes = d[:, -1]
        ade_ml += ades[ml[i]]
        fde_ml += fdes[ml[i]]
        ade_min += ades.min()
        fde_min += fdes.min()
    m = len(gt)
    print(f"[{tag}] in_dim={in_dim} train={len(Xtr)} val={m}  "
          f"ML ADE/FDE {ade_ml/m:.3f}/{fde_ml/m:.3f}  "
          f"min@{K} ADE/FDE {ade_min/m:.3f}/{fde_min/m:.3f}")
    return ade_ml / m, fde_ml / m, ade_min / m, fde_min / m


def main():
    print("=== SPIKE: 이웃(사회적) 특징 효과 ===")
    base = train_eval(False, "no-neighbor")
    soc = train_eval(True, "social")
    d_ade = (base[0] - soc[0]) / base[0] * 100
    d_fde = (base[1] - soc[1]) / base[1] * 100
    print(f"\n최빈 ADE {d_ade:+.1f}% · FDE {d_fde:+.1f}%  (양수 = 이웃이 개선)")
    print("판정:", "이웃이 유의미하게 개선 → 전면 사회적 Trajectron++ 값어치 있음"
          if (d_ade > 3 or d_fde > 3) else
          "개선 미미 → 이 데이터는 직무사이클 지배, 사회적 풀링 이득 작음")


if __name__ == "__main__":
    main()
