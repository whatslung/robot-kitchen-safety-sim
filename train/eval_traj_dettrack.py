"""P0-2 — detector-track E2E 예측 평가 (설계 docs/chanwoo/specs/2026-08-24-detection-track-eval-design.md).

overhead-person-v3(실사 오버헤드 사람 클립) 위에서
  (A) GT-트랙: GT 라벨 → IoU 추적
  (B) 검출-트랙: 이미지 → YOLO(best.pt) → ByteTrack
두 입력으로 같은 윈도우(obs8/pred12)를 CV·칼만·LSTM 예측기에 돌려 ADE/FDE와
가상 로봇 위험진입을 비교한다. 검출 노이즈(미검출·fragmentation·ID switch)가
예측·안전 결정에 주는 타격을 정량화한다.

가정(설계 §5, 결과 JSON에 기록): 프레임 가로 = FRAME_W_M(6.0m), 프레임간격 0.4s,
로봇 = 클립별 GT 통행 중심, 절대 위험진입은 가정 기반 → GT vs 검출 '차이'가 핵심 신호.

실행: uv run --group serve python train/eval_traj_dettrack.py --split test
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trajectory.types import Track, TrackScene           # noqa: E402
from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor  # noqa: E402
from trajectory.evaluator import ade, fde                # noqa: E402
from trajectory import risk                              # noqa: E402
from trajectory.dettrack import (                        # noqa: E402
    assign_per_frame, classify_failures, aggregate,
)

OBS, PRED, DT = 8, 12, risk.STEP_DT
FRAME_W_M = 6.0            # 가정: 프레임 가로 6.0m(셀 가로 ~6.1m)
MOVE_M = 0.15             # 예측 구간이 이 이상(m) 움직이면 '움직인' 윈도우(진짜 난이도)
MATCH_DIST = 0.08        # GT↔검출 프레임별 중심거리 임계(정규화, 프레임의 8%)
STOP_R, SLOW_R, HORIZON, KSIG, TAU = 1.2, 2.0, 1.6, 1.0, 0.1   # 가상 로봇(m)
CLIP_RE = re.compile(r"(.+?)[_-](\d{6,8})_jpg\.rf\.")


# ── GT: 라벨 → 클립·프레임 → IoU 추적 ────────────────────────────────────────
def clip_frame(name):
    m = CLIP_RE.search(name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load_gt(split_dir):
    """{clip: {frame: [(cx,cy,w,h)]}} + {clip: {frame: image_path}}."""
    boxes, images = defaultdict(dict), defaultdict(dict)
    for lb in glob.glob(str(split_dir / "labels" / "*.txt")):
        base = os.path.basename(lb)
        clip, frame = clip_frame(base)
        if clip is None:
            continue
        bxs = []
        for ln in Path(lb).read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                bxs.append(tuple(map(float, p[1:5])))
        boxes[clip][frame] = bxs
        img = split_dir / "images" / (base[:-4] + ".jpg")
        if img.exists():
            images[clip][frame] = img
    return boxes, images


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix, iy = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1))
    inter = ix*iy
    ua = a[2]*a[3] + b[2]*b[3] - inter
    return inter/ua if ua > 0 else 0.0


def iou_track(frame_boxes):
    """{frame: [box]} → {tid: [(frame, cx, cy)]}  (GT 라벨 IoU 추적, spike 방식)."""
    tracks = defaultdict(list); active = {}; nid = 0
    for fr in sorted(frame_boxes):
        boxes = frame_boxes[fr]; assigned = set()
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


# ── 검출: 이미지 → YOLO → ByteTrack (클립마다 새 트래커) ─────────────────────
def build_det_tracks(ds, frame_images):
    """{frame: image_path} → {det_id: [(frame, cx, cy)]}. 검출 불가면 None."""
    if ds is None or not str(getattr(ds, "MODE", "off")).startswith("yolo"):
        return None
    from PIL import Image
    tracker = ds._new_tracker()
    if tracker is None:
        return None
    det = defaultdict(list)
    for fr in sorted(frame_images):
        img = Image.open(frame_images[fr]).convert("RGB")
        w, h = img.size
        detections = ds._detections_from(ds.run_detect(img), w, h)
        try:
            tracked = tracker.update(detections)
        except Exception as e:                      # noqa: BLE001
            print(f"  [det] tracker.update 실패(frame {fr}): {e}")
            continue
        if tracked.tracker_id is None:
            continue
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            if tid < 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            det[tid].append((fr, (x1+x2)/2/w, (y1+y2)/2/h))
    return dict(det)


# ── 윈도우화: GT 트랙의 연속프레임 구간 → obs8+pred12 ────────────────────────
def windows_of(track_pts):
    pts = sorted(track_pts)
    segs, seg = [], [pts[0]] if pts else []
    for i in range(1, len(pts)):
        if pts[i][0] - pts[i-1][0] == 1:
            seg.append(pts[i])
        else:
            segs.append(seg); seg = [pts[i]]
    if seg:
        segs.append(seg)
    out = []
    for s in segs:
        for i in range(0, len(s) - (OBS+PRED) + 1):
            out.append(s[i:i+OBS+PRED])
    return out


def to_m(x, y, hm):
    return (x * FRAME_W_M, y * hm)


def hold_last(seq):
    """[(x,y)|None]*OBS → 결측을 앞/뒤 최근값으로 채움. 전부 None이면 None."""
    known = [p for p in seq if p is not None]
    if not known:
        return None
    out, last = [], None
    for p in seq:
        last = p if p is not None else last
        out.append(last)
    first = known[0]                                  # 선행 결측은 첫 관측으로 채움
    return [p if p is not None else first for p in out]


def top_steps(modes):
    """risk 모드 리스트(가중치 내림차순) → 최빈 모드의 pred_steps=[(t,x,z,sigma)]."""
    m = modes[0]
    return [(DT*(i+1), x, z, (m["sigma"][i] if i < len(m["sigma"]) else 0.0))
            for i, (x, z) in enumerate(m["path"])]


def predict_all(obs_m, predictors):
    """obs_m=[(t,x,z)]*OBS → {name: (pred_steps, risk_modes)}. risk_modes = [{path,w,sigma}]."""
    out = {}
    scene = TrackScene(now=obs_m[-1][0], horizon=PRED, agents=[Track(0, obs_m)], map=None)
    for name, pr in predictors.items():
        if name == "LSTM":
            hist_xz = [(x, z) for (_t, x, z) in obs_m]
            modes = pr.predict_modes(hist_xz)
            out[name] = (top_steps(modes), modes)
        else:
            modes_obj = pr.predict(scene).per_agent[0]
            steps = modes_obj[0].steps
            risk_modes = [{"path": [(x, z) for (_t, x, z, _s) in mo.steps],
                           "w": mo.prob,
                           "sigma": [s for (_t, _x, _z, s) in mo.steps]} for mo in modes_obj]
            out[name] = (steps, risk_modes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--out", default=str(ROOT / "docs" / "chanwoo" / "results" / "detection-track-eval.json"))
    ap.add_argument("--dataset", default=str(ROOT / "dataset" / "overhead-person-v3"))
    args = ap.parse_args()

    split_dir = Path(args.dataset) / args.split
    boxes, images = load_gt(split_dir)
    print(f"[데이터] split={args.split} · 클립 {len(boxes)}개")

    # 검출기(YOLO+ByteTrack) 로드 — 없거나 실패하면 GT-트랙만 산출(설계 §7, silent 금지).
    ds = None
    try:
        from backend import detect_server as ds  # noqa: PLC0415
        print(f"[검출기] mode={ds.MODE}")
    except Exception as e:                        # noqa: BLE001
        print(f"[검출기] 로드 실패 → 검출-트랙 생략, GT-트랙만 산출: {e}")

    predictors = {"CV": ConstantVelocityPredictor(n_steps=PRED),
                  "Kalman": KalmanPredictor(n_steps=PRED)}
    try:
        from trajectory.learned_predictor import LearnedPredictor  # noqa: PLC0415
        w = Path(os.environ.get("PREDICT_MODEL", str(ROOT/"training"/"traj_predictor"/"model.pt")))
        if w.exists():
            predictors["LSTM"] = LearnedPredictor(weights_path=str(w), device="cpu")
            print(f"[예측기] LSTM 로드: {w}")
        else:
            print(f"[예측기] LSTM 가중치 없음({w}) → CV·칼만만")
    except Exception as e:                        # noqa: BLE001
        print(f"[예측기] LSTM 로드 실패 → CV·칼만만: {e}")

    records = []            # ADE/FDE 표: {group:(source,pred[,'clean'|'degraded']), ade,fde,moved}
    risk_rows = []          # 위험진입 비교: {pred, has_fail, gt_entry, det_entry, dt_gt, dt_det}
    fail_totals = {"miss": 0, "fragments": 0, "id_switches": 0, "gt_tracks": 0}
    n_win = n_win_det = 0

    for clip in sorted(boxes):
        gt_tracks = iou_track(boxes[clip])
        # 이미지 크기(높이 스케일용) — 첫 프레임에서.
        hm = FRAME_W_M
        if images[clip]:
            from PIL import Image
            iw, ih = Image.open(next(iter(images[clip].values()))).size
            hm = FRAME_W_M * ih / iw
        det_tracks = build_det_tracks(ds, images[clip]) if images[clip] else None
        # 검출 위치 색인: (det_id, frame) -> (x,y)
        det_pos_at = {}
        if det_tracks:
            for did, pts in det_tracks.items():
                for (f, x, y) in pts:
                    det_pos_at[(did, f)] = (x, y)
        robot = _clip_robot(gt_tracks, hm)

        for _tid, gpts in gt_tracks.items():
            fail_totals["gt_tracks"] += 1
            # 검출 실패모드(트랙 전체) 집계
            assigned_all = assign_per_frame(gpts, det_tracks or {}, MATCH_DIST)
            f = classify_failures(assigned_all)
            for k in ("miss", "fragments", "id_switches"):
                fail_totals[k] += f[k] if not (k == "fragments" and det_tracks is None) else 0

            for w20 in windows_of(gpts):
                n_win += 1
                gwin = [(fr, *to_m(x, y, hm)) for (fr, x, y) in w20]
                obs_gt = [(fr*DT, x, z) for (fr, x, z) in gwin[:OBS]]
                gt_future = [(fr*DT, x, z) for (fr, x, z) in gwin[OBS:]]
                moved = _moved(gwin[OBS-1:])

                gt_preds = predict_all(obs_gt, predictors)
                for name, (steps, rmodes) in gt_preds.items():
                    records.append({"group": ("gt", name),
                                    "ade": ade(steps, gt_future), "fde": fde(steps, gt_future),
                                    "moved": moved})

                # ── 검출-트랙 입력 (있을 때만) ──
                if not det_tracks:
                    continue
                frames = [fr for (fr, _x, _y) in w20]
                assigned = assign_per_frame([(fr, x, y) for (fr, x, y) in w20], det_tracks, MATCH_DIST)
                obs_pos_norm = [det_pos_at.get((assigned[i], frames[i])) if assigned[i] is not None else None
                                for i in range(OBS)]
                filled = hold_last(obs_pos_norm)
                if filled is None:                      # obs 전 구간 미검출 → 검출 입력 불가
                    continue
                n_win_det += 1
                obs_det = [(frames[i]*DT, *to_m(filled[i][0], filled[i][1], hm)) for i in range(OBS)]
                fwin = classify_failures(assigned)
                degraded = (fwin["miss"] > 0 or fwin["id_switches"] > 0 or fwin["fragments"] > 1)
                tag = "degraded" if degraded else "clean"

                det_preds = predict_all(obs_det, predictors)
                for name, (steps, rmodes) in det_preds.items():
                    records.append({"group": ("det", name),
                                    "ade": ade(steps, gt_future), "fde": fde(steps, gt_future),
                                    "moved": moved})
                    records.append({"group": ("det", name, tag),
                                    "ade": ade(steps, gt_future), "fde": fde(steps, gt_future),
                                    "moved": moved})
                    # 공정 비교(같은 윈도우): GT-예측 vs 검출-예측을 검출이 성립한 윈도우에서만.
                    gsteps = gt_preds[name][0]
                    records.append({"group": ("gt", name, "paired"),
                                    "ade": ade(gsteps, gt_future), "fde": fde(gsteps, gt_future),
                                    "moved": moved})
                    records.append({"group": ("det", name, "paired"),
                                    "ade": ade(steps, gt_future), "fde": fde(steps, gt_future),
                                    "moved": moved})
                    # 위험진입: 같은 가상 로봇으로 GT-예측 vs 검출-예측
                    rg = risk.track_risk(gt_preds[name][1], robot, STOP_R, SLOW_R, HORIZON, KSIG, TAU)
                    rd = risk.track_risk(rmodes, robot, STOP_R, SLOW_R, HORIZON, KSIG, TAU)
                    risk_rows.append({"pred": name, "has_fail": degraded,
                                      "gt_entry": rg["tEntryStop"] is not None,
                                      "det_entry": rd["tEntryStop"] is not None,
                                      "dt_gt": rg["tEntryStop"], "dt_det": rd["tEntryStop"]})

    agg = aggregate(records)
    risk_summary = _risk_summary(risk_rows)
    _print_report(args, len(boxes), n_win, n_win_det, fail_totals, agg, risk_summary, list(predictors))
    _write_json(args, boxes, n_win, n_win_det, fail_totals, agg, risk_summary, list(predictors))


def _clip_robot(gt_tracks, hm):
    xs, zs = [], []
    for pts in gt_tracks.values():
        for (_f, x, y) in pts:
            mx, mz = to_m(x, y, hm)
            xs.append(mx); zs.append(mz)
    return (float(np.mean(xs)), float(np.mean(zs))) if xs else (FRAME_W_M/2, hm/2)


def _moved(future_m):
    path = np.array([[x, z] for (_f, x, z) in future_m])
    return bool(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) > MOVE_M) if len(path) > 1 else False


def _risk_summary(rows):
    out = {}
    preds = sorted({r["pred"] for r in rows})
    for p in preds:
        rs = [r for r in rows if r["pred"] == p]
        both = [r for r in rs if r["gt_entry"] and r["det_entry"] and r["dt_gt"] is not None and r["dt_det"] is not None]
        out[p] = {
            "n": len(rs),
            "gt_entry": sum(r["gt_entry"] for r in rs),
            "det_entry": sum(r["det_entry"] for r in rs),
            "missed_by_det": sum(r["gt_entry"] and not r["det_entry"] for r in rs),   # GT는 위험, 검출은 놓침
            "false_by_det": sum(r["det_entry"] and not r["gt_entry"] for r in rs),    # 검출만 위험(헛)
            "mean_abs_dt": (float(np.mean([abs(r["dt_gt"]-r["dt_det"]) for r in both])) if both else float("nan")),
        }
    return out


def _fmt(v):
    return "  n/a" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:6.3f}"


def _print_report(args, n_clips, n_win, n_win_det, fails, agg, risk_summary, preds):
    print("\n" + "="*72)
    print(f"P0-2 detector-track E2E — split={args.split} · 클립 {n_clips} · 윈도우 GT {n_win}/검출 {n_win_det}")
    print(f"GT 트랙 {fails['gt_tracks']} · 검출 실패 합계: 미검출 {fails['miss']} · "
          f"fragmentation {fails['fragments']} · ID switch {fails['id_switches']}")
    print("-"*72)
    print("공정 비교 — 검출이 성립한 같은 윈도우에서 GT-예측 vs 검출-예측(ADE/FDE, m):")
    print(f"{'예측기':<8}{'입력':<12}{'ADE(m)':>9}{'FDE(m)':>9}{'ADE움직임':>11}{'FDE움직임':>11}{'n':>6}")
    for name in preds:
        for src, label in (("gt", "GT-예측"), ("det", "검출-예측")):
            a = agg.get((src, name, "paired"))
            if a:
                print(f"{name:<8}{label:<12}{_fmt(a['ade'])}{_fmt(a['fde'])}"
                      f"{_fmt(a['ade_moved']):>11}{_fmt(a['fde_moved']):>11}{a['n']:>6}")
    print("-"*72)
    print("참고 — GT-트랙 예측 상한(전체 GT 윈도우, 검출 무관):")
    for name in preds:
        a = agg.get(("gt", name))
        if a:
            print(f"  {name:<8} ADE {_fmt(a['ade'])} · FDE {_fmt(a['fde'])} · n {a['n']}")
    print("-"*72)
    print("검출-트랙 실패모드별(ADE/FDE, m):")
    for name in preds:
        c, d = agg.get(("det", name, "clean")), agg.get(("det", name, "degraded"))
        cc = f"clean {_fmt(c['ade']) if c else '  n/a'}/{_fmt(c['fde']) if c else 'n/a'}(n{c['n'] if c else 0})"
        dd = f"degraded {_fmt(d['ade']) if d else '  n/a'}/{_fmt(d['fde']) if d else 'n/a'}(n{d['n'] if d else 0})"
        print(f"  {name:<8} {cc}   {dd}")
    print("-"*72)
    print("가상 로봇 위험진입(정지반경) — GT-예측 vs 검출-예측:")
    for name in preds:
        s = risk_summary.get(name)
        if s:
            print(f"  {name:<8} GT진입 {s['gt_entry']} · 검출진입 {s['det_entry']} · "
                  f"검출이 놓침 {s['missed_by_det']} · 검출 헛경보 {s['false_by_det']} · "
                  f"|Δ진입시각| {_fmt(s['mean_abs_dt'])}s")
    print("="*72)
    print("※ 가정: 프레임 가로 6.0m·0.4s/frame·로봇=클립 GT 통행중심. 절대값은 가정 기반, "
          "핵심은 GT vs 검출 '차이'. 데이터=overhead-person(일반 오버헤드, 주방 특정 아님).")


def _write_json(args, boxes, n_win, n_win_det, fails, agg, risk_summary, preds):
    out = {
        "spec": "docs/chanwoo/specs/2026-08-24-detection-track-eval-design.md",
        "split": args.split, "dataset": args.dataset,
        "assumptions": {"frame_width_m": FRAME_W_M, "dt_s": DT, "robot": "clip GT centroid",
                        "stopR_m": STOP_R, "slowR_m": SLOW_R, "horizon_s": HORIZON,
                        "match_dist_norm": MATCH_DIST, "note": "absolute risk-entry is assumption-based; signal = GT vs det diff; data is generic overhead-person, not kitchen"},
        "clips": len(boxes), "windows_gt": n_win, "windows_det": n_win_det,
        "failures": fails, "predictors": preds,
        "ade_fde": {f"{g[0]}|{'|'.join(g[1:])}" if isinstance(g, tuple) else str(g): v
                    for g, v in agg.items()},
        "risk_entry": risk_summary,
        "reproduce": f"uv run --group serve python train/eval_traj_dettrack.py --split {args.split}",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[결과] {p}")


if __name__ == "__main__":
    main()
