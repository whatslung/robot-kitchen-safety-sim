"""[SPIKE — 버리는 실험] 실사 오버헤드 궤적의 예측 난이도. 이슈 #2 sim-to-real.

dataset/overhead-person-v3(Roboflow, 24 클립) 라벨 → IoU 추적 → (cx,cy) 정규화 궤적.
등속·칼만(단위·스케일 무관)을 실사 윈도우(obs8/pred12)에 적용해 ADE/FDE(×640px)를 잰다.
목적: 실사 예측이 어려운 문제인가? 대부분 정지면 등속이 이미 강해 학습형이 낄 자리가 적다.
실행: uv run python train/spike_real_baseline.py
"""
from __future__ import annotations
import os, re, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.types import Track, TrackScene
from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor
from trajectory.evaluator import ade, fde

BASE = ROOT / "dataset" / "overhead-person-v3"
CLIP_RE = re.compile(r'(.+?)[_-](\d{6,8})_jpg\.rf\.')
OBS, PRED, IMG = 8, 12, 640
MOVE_PX = 10.0   # 예측 구간 경로가 이 이상(px) 움직이면 '움직인' 윈도우


def load_clips():
    clips = defaultdict(dict)
    for split in ["train", "valid", "test"]:
        for lb in glob.glob(str(BASE / split / "labels" / "*.txt")):
            m = CLIP_RE.search(os.path.basename(lb))
            if not m:
                continue
            clip, frame = m.group(1), int(m.group(2))
            boxes = []
            for ln in Path(lb).read_text().splitlines():
                p = ln.split()
                if len(p) >= 5:
                    boxes.append(tuple(map(float, p[1:5])))
            clips[clip][frame] = boxes
    return clips


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix, iy = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1))
    inter = ix*iy
    ua = a[2]*a[3] + b[2]*b[3] - inter
    return inter/ua if ua > 0 else 0


def track(fd):
    tracks = defaultdict(list); active = {}; nid = 0
    for fr in sorted(fd):
        boxes = fd[fr]; assigned = set()
        for tid, last in list(active.items()):
            best, bi = 0.3, -1
            for i, bx in enumerate(boxes):
                if i in assigned:
                    continue
                v = iou(last, bx)
                if v > best:
                    best, bi = v, i
            if bi >= 0:
                assigned.add(bi); active[tid] = boxes[bi]
                tracks[tid].append((fr, boxes[bi][0], boxes[bi][1]))
        for i, bx in enumerate(boxes):
            if i in assigned:
                continue
            nid += 1; active[nid] = bx; tracks[nid].append((fr, bx[0], bx[1]))
    return tracks


def windows():
    out = []
    for clip, fd in load_clips().items():
        for tid, pts in track(fd).items():
            pts = sorted(pts); seg = [pts[0]]
            for i in range(1, len(pts)):
                if pts[i][0]-pts[i-1][0] == 1:
                    seg.append(pts[i])
                else:
                    seg = _emit(seg, out); seg = [pts[i]]
            _emit(seg, out)
    return out


def _emit(seg, out):
    if len(seg) >= OBS + PRED:
        for s in range(0, len(seg) - (OBS + PRED) + 1):
            w = seg[s:s + OBS + PRED]
            out.append(w)
    return []


def main():
    wins = windows()
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    rec = {"등속": [], "칼만": []}
    nmoved = 0
    for w in wins:
        obs = [(f, x*IMG, y*IMG) for (f, x, y) in w[:OBS]]        # 프레임=t, px
        gt = [(f, x*IMG, y*IMG) for (f, x, y) in w[OBS:]]
        gtp = [(t, x, y) for (t, x, y) in gt]
        path = np.array([[x, y] for _, x, y in w])
        moved = np.sum(np.linalg.norm(np.diff(path[OBS-1:], axis=0), axis=1)) * IMG > MOVE_PX
        if moved:
            nmoved += 1
        sc = TrackScene(now=obs[-1][0], horizon=PRED, agents=[Track(0, obs)], map=None)
        for name, pr in (("등속", cv), ("칼만", kf)):
            st = pr.predict(sc).per_agent[0][0].steps
            rec[name].append((ade(st, gtp), fde(st, gtp), moved))
    n = len(wins)
    print(f"실사 윈도우 {n} · 움직인 것 {nmoved} ({100*nmoved/n:.0f}%)")
    print(f"{'예측기':<8}{'ADE(px)':>9}{'FDE(px)':>9}   |  움직인 것만 ADE/FDE")
    for name, r in rec.items():
        a = np.mean([x[0] for x in r]); f = np.mean([x[1] for x in r])
        mv = [x for x in r if x[2]]
        am = np.mean([x[0] for x in mv]) if mv else float('nan')
        fm = np.mean([x[1] for x in mv]) if mv else float('nan')
        print(f"{name:<8}{a:>9.2f}{f:>9.2f}   |  {am:.2f}/{fm:.2f}")
    print("\n해석: 전체가 작으면 대부분 정지라 쉬운 문제. '움직인 것만'이 실제 난이도.")


if __name__ == "__main__":
    main()
