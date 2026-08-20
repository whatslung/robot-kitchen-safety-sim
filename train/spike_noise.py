"""[SPIKE — 버리는 실험] 검출 노이즈 갭. 이슈 #2 sim-to-real 검증.

실제로는 나디르 이미지→YOLO→추적으로 좌표가 나와 노이즈가 낀다. 우리 학습 데이터는
깨끗한 GT라, 노이즈 낀 실제 트랙에서 성능이 떨어질 수 있다. 렌더/YOLO 없이 GT 관측에
가우시안 노이즈를 주입해:
  (1) 깨끗-학습 모델이 노이즈 관측에서 얼마나 나빠지나(노이즈 갭)
  (2) 노이즈-학습(증강)이 회복시키나
관측(obs 8점)에만 노이즈, 미래(GT 12점)는 참값. 노이즈는 정규화 전에 준다(heading 추정도 흔들리게).
실행: uv run python train/spike_noise.py
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.learned_predictor import frame_of, to_frame, build_net, mtp_loss
from trajectory.sim_traj import is_val, OBS, PRED, TRAJ_DIR

EPOCHS, BATCH, LR, SEED = 140, 512, 1e-3, 0
SIGM = 0.06   # 시뮬 자체 검출 노이즈 가정(m) = PRED.sigM


def load(split, noise_std, noise_seed=1234):
    rng = np.random.default_rng(noise_seed)
    X, Y = [], []
    for f in sorted(glob.glob(str(TRAJ_DIR / "*.json"))):
        sc = json.load(open(f, encoding="utf-8"))
        val = is_val(sc["seed"])
        if split == "val" and not val:
            continue
        if split == "train" and val:
            continue
        for n in sc["nodes"]:
            if n.get("discarded"):
                continue
            fr = [(p["x"], p["z"]) for p in n["frames"]]
            for s in range(0, len(fr) - (OBS + PRED) + 1):
                obs = np.asarray(fr[s:s + OBS], float)
                fut = np.asarray(fr[s + OBS:s + OBS + PRED], float)
                if noise_std > 0:
                    obs = obs + rng.normal(0, noise_std, obs.shape)   # 관측에만 노이즈
                o, a = frame_of(obs)
                X.append(to_frame(obs, o, a).astype(np.float32))
                Y.append(to_frame(fut, o, a).astype(np.float32))       # 미래=참값(노이즈 프레임 기준)
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
    a = f = 0.0
    for i in range(len(Yva)):
        d = np.linalg.norm(paths[i] - Yva[i][None], axis=2)
        a += d.mean(1)[ml[i]]; f += d[:, -1][ml[i]]
    m = len(Yva)
    return a / m, f / m


def main():
    print("=== SPIKE: 검출 노이즈 갭 (sim-to-real) ===")
    # 평가용 노이즈 val (train-noise와 다른 seed로 = 노이즈 자체를 외우지 못하게)
    Xv0 = load("val", 0.0)
    Xv6 = load("val", SIGM, noise_seed=777)
    Xv12 = load("val", SIGM * 2, noise_seed=777)

    print("\n[M1] 깨끗-학습")
    m1, dev = train(*load("train", 0.0))
    c = evalp(m1, dev, *Xv0)
    n6 = evalp(m1, dev, *Xv6)
    n12 = evalp(m1, dev, *Xv12)
    print(f"    깨끗 val      ADE/FDE {c[0]:.3f}/{c[1]:.3f}")
    print(f"    노이즈 0.06   ADE/FDE {n6[0]:.3f}/{n6[1]:.3f}  ({(n6[0]-c[0])/c[0]*100:+.0f}% ADE)")
    print(f"    노이즈 0.12   ADE/FDE {n12[0]:.3f}/{n12[1]:.3f}  ({(n12[0]-c[0])/c[0]*100:+.0f}% ADE)")

    print("[M2] 노이즈0.06-학습(증강)")
    m2, _ = train(*load("train", SIGM, noise_seed=42))
    r6 = evalp(m2, dev, *Xv6)
    print(f"    노이즈 0.06   ADE/FDE {r6[0]:.3f}/{r6[1]:.3f}  (M1@0.06 대비 {(r6[0]-n6[0])/n6[0]*100:+.0f}% ADE)")

    print("\n판정:")
    gap = (n6[0] - c[0]) / c[0] * 100
    rec = (n6[0] - r6[0]) / n6[0] * 100
    print(f"  노이즈 갭: 깨끗-학습 모델이 0.06 노이즈에서 {gap:+.0f}% 악화")
    print(f"  증강 회복: 노이즈-학습이 그걸 {rec:.0f}% 되돌림"
          + ("" if rec > 5 else " (미미)"))


if __name__ == "__main__":
    main()
