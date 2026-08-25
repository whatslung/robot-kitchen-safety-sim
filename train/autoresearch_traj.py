"""autoresearch-style 궤적 예측기 개선 실험 루프 (karpathy/autoresearch 방법 참고).

autoresearch(github.com/karpathy/autoresearch)의 핵심 방법론을 우리 궤적 예측에 적용:
  · 고정 설정(같은 데이터·같은 val 지표)에서 후보 학습 레시피를 하나씩 시험
  · 단일 val 지표(minADE_moved, 낮을수록↑)로 baseline 대비 개선 여부 판정
  · 모든 실험을 results 로그(tsv)에 기록, 개선분만 채택
  · '더 단순한데 같거나 나으면 채택'(autoresearch simplicity criterion)

autoresearch의 GPT/nanochat 코드는 우리 과제(좌표 회귀)와 무관해 재사용하지 않는다 —
가져오는 건 **실험 방법론 + 학습 레시피 아이디어**(AdamW·weight decay·LR warmup/warmdown 스케줄).
백본은 P0-1 승자 Transformer 고정, head·정규화·split은 동일(공정).

실행:  uv run --group serve python train/autoresearch_traj.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from trajectory.learned_predictor import build_transformer_net, mtp_loss, LearnedPredictor  # noqa: E402
from trajectory.evaluator import ade                                                        # noqa: E402
from trajectory.sim_traj import load_windows                                                # noqa: E402
from train_traj_predictor import build_xy, EPOCHS, BATCH, HIDDEN, LR, SEED                  # noqa: E402

STEP_DT = 0.4
LOG = ROOT / "docs" / "chanwoo" / "results" / "autoresearch-log.tsv"

# 후보 레시피 — baseline 먼저(현행 Transformer 학습과 동일). 이후 autoresearch train.py 레시피 요소.
RECIPES = [
    {"name": "baseline",       "optim": "adam",  "wd": 0.0,  "sched": False, "layers": 2, "heads": 4},
    {"name": "adamw_wd",       "optim": "adamw", "wd": 0.01, "sched": False, "layers": 2, "heads": 4},
    {"name": "lr_sched",       "optim": "adam",  "wd": 0.0,  "sched": True,  "layers": 2, "heads": 4},
    {"name": "adamw_sched",    "optim": "adamw", "wd": 0.01, "sched": True,  "layers": 2, "heads": 4},
    {"name": "adamw_sched_big","optim": "adamw", "wd": 0.01, "sched": True,  "layers": 3, "heads": 8},
]

WARMUP_FRAC, WARMDOWN_FRAC = 0.1, 0.5      # autoresearch: 후반 절반 LR warmdown


def _lr_mult(progress):
    if progress < WARMUP_FRAC:
        return progress / WARMUP_FRAC
    if progress > 1 - WARMDOWN_FRAC:
        return max(0.0, (1 - progress) / WARMDOWN_FRAC)
    return 1.0


def train_recipe(r, xt, yt, dev):
    import torch
    torch.manual_seed(SEED)                # 결정적(레시피 간 공정)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = build_transformer_net(h=HIDDEN, layers=r["layers"], heads=r["heads"]).to(dev)
    if r["optim"] == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=r["wd"])
    else:
        opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(xt)
    for ep in range(EPOCHS):
        if r["sched"]:
            for g in opt.param_groups:
                g["lr"] = LR * _lr_mult(ep / EPOCHS)
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = mtp_loss(*net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return net


def val_metrics(net, val_wins, dev):
    """val minADE_moved(최빈)·minADE@3(moved) — 낮을수록 좋다."""
    lp = LearnedPredictor(net=net, device=dev)
    hists = [[(o[1], o[2]) for o in w.scene.agents[0].history] for w in val_wins]
    modes_all = lp.predict_batch(hists)
    top, mink = [], []
    for w, modes in zip(val_wins, modes_all):
        if not w.moved:
            continue
        steps0 = [(STEP_DT * (i + 1), x, z, 0.0) for i, (x, z) in enumerate(modes[0]["path"])]
        top.append(ade(steps0, w.gt))
        mink.append(min(ade([(STEP_DT * (i + 1), x, z, 0.0) for i, (x, z) in enumerate(m["path"])], w.gt)
                        for m in modes))
    return float(np.mean(top)), float(np.mean(mink))


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} · autoresearch-style 궤적 예측기 개선 (백본=Transformer 고정)")

    Xtr, Ytr, _ = build_xy("train")
    xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(Ytr, device=dev)
    val_wins = load_windows("val")
    print(f"train {len(xt)} · val {len(val_wins)} 윈도우\n")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    rows = ["recipe\toptim\twd\tsched\tlayers\theads\tval_minADE_moved\tval_minADE@3_moved\tkept"]
    baseline_top = None
    best = None
    for r in RECIPES:
        net = train_recipe(r, xt, yt, dev)
        top, mink = val_metrics(net, val_wins, dev)
        if baseline_top is None:
            baseline_top = top
        kept = "baseline" if r["name"] == "baseline" else ("keep" if top < baseline_top - 1e-4 else "discard")
        rows.append(f"{r['name']}\t{r['optim']}\t{r['wd']}\t{int(r['sched'])}\t{r['layers']}\t{r['heads']}"
                    f"\t{top:.4f}\t{mink:.4f}\t{kept}")
        print(f"  {r['name']:16s} val minADE_moved={top:.4f} (baseline={baseline_top:.4f}) minADE@3={mink:.4f} → {kept}")
        if best is None or top < best[1]:
            best = (r, top, mink, net)

    LOG.write_text("\n".join(rows) + "\n", encoding="utf-8")
    b_r, b_top, b_mink, b_net = best
    gain = 100 * (baseline_top - b_top) / baseline_top
    print(f"\n최고: {b_r['name']} · val minADE_moved={b_top:.4f} (baseline 대비 {gain:+.1f}%) · minADE@3={b_mink:.4f}")
    print(f"로그 → {LOG}")
    if b_r["name"] != "baseline" and b_top < baseline_top - 1e-4:
        out = ROOT / "training" / "traj_predictor" / "model_transformer_tuned.pt"
        torch.save(b_net.state_dict(), out)
        print(f"개선 채택 → {out} (레시피 {b_r})")
    else:
        print("baseline이 최고 — 개선 없음(정직 보고). 채택 없음.")


if __name__ == "__main__":
    main()
