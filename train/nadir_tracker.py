"""나디르 person 추적 — 안전용 recall 우선 배포 파이프라인.

설계 근거(실험 5-6): ByteTrack을 그대로 쓰면 자체 신뢰도 게이팅이 저신뢰 검출을 버려
base recall이 깎였다(0.982→0.943). 안전에선 놓침이 오탐보다 훨씬 위험하므로,
  ① 검출은 저신뢰(conf~0.15)로 전부 살리고
  ② 추적기는 'id 부여 + 잠깐 놓친 프레임 coast'에만 쓴다.
즉 추적기가 검출을 '거르는' 게 아니라 검출 위에 '보태는' 역할.

구현: SORT형 칼만 필터(등속) + IoU 헝가리안 매칭.
  - 매 프레임 출력 = (그 프레임의 모든 검출, id 부여) + (확정 트랙이 잠깐 놓친 자리의 칼만 예측 coast).
  - 확정(min_hits) 트랙만 coast → 스퍼리어스 1회 검출이 유령으로 번지는 것 차단(precision 보호).
  - coast는 max_age(K) 프레임까지만 → recall↔precision 트레이드를 K로 제한.

사용:
  from nadir_tracker import NadirTracker
  trk = NadirTracker(iou_thresh=0.3, min_hits=2, max_age=5)
  for frame in stream:
      dets = detector(frame)                 # [[x1,y1,x2,y2,score], ...]
      outs = trk.update(dets)                # [{box,id,source,score}, ...]  source: 'det'|'coast'
CLI(검증):  python nadir_tracker.py <dataset_dir_under_dataset/> [--fuse]
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _to_z(b):                       # bbox[x1,y1,x2,y2] -> 관측 z[cx,cy,s,r]
    w, h = b[2]-b[0], b[3]-b[1]
    return np.array([b[0]+w/2, b[1]+h/2, w*h, (w/h if h > 0 else 1.0)], dtype=float)


def _to_box(x):                     # 상태 x[cx,cy,s,r,...] -> bbox
    s, r = max(x[2], 1.0), max(x[3], 1e-3)
    w = np.sqrt(s*r); h = s/w if w > 0 else 0
    return [x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2]


class _KF:
    """등속 칼만 필터. 상태 [cx,cy,s,r,cx',cy',s'] (r은 상수 가정)."""
    def __init__(self, box):
        self.x = np.zeros(7); self.x[:4] = _to_z(box)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)  # 속도 불확실성 크게
        self.F = np.eye(7)
        for i in range(3): self.F[i, i+4] = 1.0
        self.H = np.zeros((4, 7));
        for i in range(4): self.H[i, i] = 1.0
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 1e-4]).astype(float)
        self.R = np.diag([1, 1, 10, 10]).astype(float)

    def predict(self):
        self.x = self.F @ self.x
        if self.x[2] <= 0: self.x[2] = 1.0
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _to_box(self.x)

    def update(self, box):
        z = _to_z(box); y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class _Track:
    def __init__(self, box, tid):
        self.kf = _KF(box); self.box = list(box[:4])
        self.id = tid                       # id는 소속 NadirTracker가 발급(카메라별 독립)
        self.hits = 1; self.time_since_update = 0

    def predict(self):
        self.box = self.kf.predict(); return self.box

    def update(self, box):
        self.kf.update(box); self.box = _to_box(self.kf.x)
        self.hits += 1; self.time_since_update = 0


class NadirTracker:
    def __init__(self, iou_thresh=0.3, min_hits=2, max_age=5):
        self.iou_thresh = iou_thresh; self.min_hits = min_hits; self.max_age = max_age
        self.tracks = []
        self._next = 1                      # 이 추적기 전용 id 시퀀스(다른 추적기와 독립)

    def update(self, dets):
        """dets: [[x1,y1,x2,y2,score], ...] → outputs: [{box,id,source,score}]."""
        for t in self.tracks: t.predict()
        # IoU 헝가리안 매칭
        matches, un_d, un_t = self._match(dets)
        det_to_track = {}
        for di, ti in matches:
            self.tracks[ti].update(dets[di][:4]); det_to_track[di] = self.tracks[ti]
        # 미매칭 검출 → 새 트랙
        for di in un_d:
            t = _Track(dets[di][:4], self._next); self._next += 1; self.tracks.append(t); det_to_track[di] = t
        # 미매칭 트랙 → 놓침 카운트
        for ti in un_t: self.tracks[ti].time_since_update += 1

        outs = []
        # ① 이 프레임의 모든 검출을 id 붙여 그대로 출력 (recall 우선)
        for di, d in enumerate(dets):
            t = det_to_track[di]
            outs.append({"box": list(d[:4]), "id": t.id, "source": "det",
                         "score": (d[4] if len(d) > 4 else 1.0)})
        # ② 확정 트랙이 잠깐(≤max_age) 놓친 자리 → 칼만 예측 coast
        matched_tids = {det_to_track[di].id for di in det_to_track}
        for t in self.tracks:
            if t.id in matched_tids: continue
            if t.hits >= self.min_hits and 1 <= t.time_since_update <= self.max_age:
                outs.append({"box": list(t.box), "id": t.id, "source": "coast", "score": None})
        # 오래 놓친 트랙 제거
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return outs

    def _match(self, dets):
        T, D = len(self.tracks), len(dets)
        if T == 0: return [], list(range(D)), []
        if D == 0: return [], [], list(range(T))
        iou = np.zeros((D, T))
        for di in range(D):
            for ti in range(T):
                iou[di, ti] = _iou(dets[di][:4], self.tracks[ti].box)
        di_idx, ti_idx = linear_sum_assignment(-iou)
        matches, un_d, un_t = [], [], []
        matched_d = set(); matched_t = set()
        for di, ti in zip(di_idx, ti_idx):
            if iou[di, ti] >= self.iou_thresh:
                matches.append((di, ti)); matched_d.add(di); matched_t.add(ti)
        un_d = [di for di in range(D) if di not in matched_d]
        un_t = [ti for ti in range(T) if ti not in matched_t]
        return matches, un_d, un_t


# ────────────────────────── 검증 CLI ──────────────────────────
def _eval(dataset, fuse=False):
    import json, sys
    from pathlib import Path
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / ("dataset/" + dataset)
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, K = 960, 720, 0.15, 5

    metas = sorted((FT / "meta").glob("*.json"),
                   key=lambda p: (int(''.join(ch for ch in p.stem if ch.isdigit()) or 0)))
    # 프레임 순서 결정: combo는 (frame,cam), 단일캠 클립은 frame
    def det_of(stem):
        res = m.predict(str(FT/"images"/(stem+".png")), conf=CONF, verbose=False, device=0)[0]
        return [[*[float(v) for v in b.xyxy[0]], float(b.conf[0])]
                for b in res.boxes if int(b.cls[0]) == 0]

    # 카메라별로 독립 추적기 운용 (배포도 카메라마다 한 대씩)
    metas_by_cam = defaultdict(list)
    for mp in metas:
        meta = json.loads(mp.read_text()); metas_by_cam[meta.get("cam", 0)].append((meta, mp.stem))
    for cam in metas_by_cam:
        metas_by_cam[cam].sort(key=lambda x: x[0]["frame"])

    # 각 (cam, frame)의 추적기 출력 저장
    outs_cf = {}   # (cam, frame) -> outputs
    raw_cf = {}    # (cam, frame) -> raw det boxes
    for cam, seq in metas_by_cam.items():
        trk = NadirTracker(iou_thresh=0.3, min_hits=2, max_age=K)
        for meta, stem in seq:
            dets = det_of(stem)
            outs = trk.update(dets)
            outs_cf[(cam, meta["frame"])] = outs
            raw_cf[(cam, meta["frame"])] = dets

    # GT: (cam, frame) -> {pid: box}
    gt_cf = {}
    frames_by = defaultdict(set)   # frame -> pids present (any cam)
    for mp in metas:
        meta = json.loads(mp.read_text()); cam, t = meta.get("cam", 0), meta["frame"]
        g = {}
        for p in meta["persons"]:
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            g[p["id"]] = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
            frames_by[t].add(p["id"])
        gt_cf[(cam, t)] = g

    def recall_prec(get_boxes, per_cam=True):
        hit = tot = ot = tp = 0
        if per_cam:
            for (cam, t), g in gt_cf.items():
                boxes = get_boxes(cam, t)
                for pid, gb in g.items():
                    tot += 1
                    if any(_iou(gb, b) > 0.3 for b in boxes): hit += 1
                for b in boxes:
                    ot += 1
                    if any(_iou(gb, b) > 0.3 for gb in g.values()): tp += 1
        return hit/tot if tot else 0, (tp/ot if ot else 0), tot

    cams = sorted(metas_by_cam)
    raw_r, raw_p, tot = recall_prec(lambda c, t: [d[:4] for d in raw_cf.get((c, t), [])])
    trk_r, trk_p, _ = recall_prec(lambda c, t: [o["box"] for o in outs_cf.get((c, t), [])])

    print("NADIR_TRACKER_EVAL", flush=True)
    print(f"데이터셋 {dataset} · 카메라 {len(cams)}대 · 사람관측 {tot}개 · K={K}", flush=True)
    print(f"RAW(검출만)              recall {raw_r:.3f}  precision {raw_p:.3f}", flush=True)
    print(f"NadirTracker(검출+coast) recall {trk_r:.3f}  precision {trk_p:.3f}", flush=True)

    if fuse and len(cams) > 1:
        # 공간융합 + 시간축(추적기 출력을 카메라 간 OR): frame별 pid를 어느 카메라 출력이라도 맞추면 성공
        fh = ftot = 0
        for t, pids in frames_by.items():
            for pid in pids:
                ftot += 1
                gb_any = None
                for c in cams:
                    gb_any = gt_cf.get((c, t), {}).get(pid)
                    if gb_any is None: continue
                    if any(_iou(gb_any, o["box"]) > 0.3 for o in outs_cf.get((c, t), [])):
                        fh += 1; break
        print(f"+공간융합(추적기 출력 OR)  recall {fh/ftot:.3f}   (세 축 결합, 사람 {ftot}관측)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "temporal-clip"
    _eval(ds, fuse=("--fuse" in sys.argv))
