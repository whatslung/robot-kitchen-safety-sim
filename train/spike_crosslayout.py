"""[SPIKE — 버리는 실험] 교차 레이아웃 일반화. 이슈 #2 다양성 검증.

island(+island_h58)만 학습 → 못 본 legacy에서 평가(A). all(legacy 포함) 학습 → legacy 평가(B).
A≈B면 에이전트 중심 정규화가 레이아웃을 안 봐도 일반화. A≫B면 레이아웃 다양성이 필요.
실행: uv run python train/spike_crosslayout.py
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.learned_predictor import frame_of, to_frame, build_net, mtp_loss
from trajectory.sim_traj import OBS, PRED, TRAJ_DIR

EPOCHS, BATCH, LR, SEED = 140, 512, 1e-3, 0


def load(layouts):
    """layouts: 포함할 layout 집합. → X(N,8,2), Y(N,12,2)."""
    X, Y = [], []
    for f in sorted(glob.glob(str(TRAJ_DIR / "*.json"))):
        sc = json.load(open(f, encoding="utf-8"))
        if sc.get("layout", "island") not in layouts:
            continue
        for n in sc["nodes"]:
            if n.get("discarded"):
                continue
            fr = [(p["x"], p["z"]) for p in n["frames"]]
            for s in range(0, len(fr) - (OBS + PRED) + 1):
                o, a = frame_of(fr[s:s + OBS])
                X.append(to_frame(fr[s:s + OBS], o, a).astype(np.float32))
                Y.append(to_frame(fr[s + OBS:s + OBS + PRED], o, a).astype(np.float32))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def train(Xtr, Ytr):
    import torch
    torch.manual_seed(SEED); torch.backends.cudnn.deterministic = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=LR)
    xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    n = len(xt)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]; opt.zero_grad()
            mtp_loss(*net(xt[idx]), yt[idx]).backward(); opt.step()
    return net, dev


def evalp(net, dev, Xva, Yva):
    import torch
    net.eval()
    with torch.no_grad():
        paths, logits, _ = net(torch.tensor(Xva, device=dev))
        paths = paths.cpu().numpy(); ml = logits.argmax(1).cpu().numpy()
    a_ml = f_ml = a_min = f_min = 0.0
    for i in range(len(Yva)):
        d = np.linalg.norm(paths[i] - Yva[i][None], axis=2)
        a_ml += d.mean(1)[ml[i]]; f_ml += d[:, -1][ml[i]]
        a_min += d.mean(1).min(); f_min += d[:, -1].min()
    m = len(Yva)
    return a_ml/m, f_ml/m, a_min/m, f_min/m


def main():
    isl = {"island"}
    Xisl, Yisl = load(isl)
    Xleg, Yleg = load({"legacy"})
    Xall, Yall = load({"island", "legacy"})
    print(f"윈도우 — island {len(Xisl)} · legacy {len(Xleg)} · all {len(Xall)}")

    print("\n[A] island만 학습 → legacy 평가(못 본 레이아웃)")
    netA, dev = train(Xisl, Yisl)
    a = evalp(netA, dev, Xleg, Yleg)
    print(f"    ML ADE/FDE {a[0]:.3f}/{a[1]:.3f} · min@3 {a[2]:.3f}/{a[3]:.3f}")

    print("[B] all 학습 → legacy 평가(레이아웃 봄, 상한)")
    netB, _ = train(Xall, Yall)
    b = evalp(netB, dev, Xleg, Yleg)
    print(f"    ML ADE/FDE {b[0]:.3f}/{b[1]:.3f} · min@3 {b[2]:.3f}/{b[3]:.3f}")

    gap = (a[0] - b[0]) / b[0] * 100
    print(f"\n일반화 격차(ML ADE): A가 B보다 {gap:+.1f}% 나쁨")
    print("판정:", "격차 작음 → 에이전트 중심 정규화가 레이아웃 안 봐도 일반화(다양성 이득 작음)"
          if gap < 8 else "격차 큼 → 레이아웃 다양성이 실제로 필요")


if __name__ == "__main__":
    main()
