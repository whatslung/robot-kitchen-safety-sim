"""시간축 추적 평가 — 연속 클립에서 '추적 없이 프레임별' vs '추적으로 놓친 프레임 메우기' recall.

두 가지를 측정:
 1) RAW: 매 프레임 검출만. (지금까지의 단일 프레임 recall)
 2) TRACKED(실제 추적기): 그리디 IoU 연관 + 등속 예측으로 최근 K프레임 안에 보이던 트랙을
    잠깐 놓친 프레임에 '이어붙임(coast)'. 배포 가능한 인과적(과거만) 방식.
 3) BRIDGE(오프라인 상한): 앞뒤로 확인된 트랙만 보간. 시간축이 이론상 메울 수 있는 최대.

precision도 함께 — coast가 유령 박스를 만들어 오탐(허위 경보)을 늘리지 않는지 확인(안전에 중요).
사람 id(person_N)로 GT를 매칭하되, 추적기 자체는 id를 모르고 IoU로만 연관한다(현실적)."""
from pathlib import Path


def main():
    import json, sys
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / ("dataset/" + (sys.argv[1] if len(sys.argv) > 1 else "temporal-clip"))
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, K = 960, 720, 0.15, 5   # K = 이어붙이기 최대 프레임 수

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0

    # 프레임 순서대로: GT 사람박스(id별) + 검출박스
    frames = []
    for mp in sorted((FT / "meta").glob("*.json"), key=lambda p: int(p.stem.split("_")[1])):
        meta = json.loads(mp.read_text())
        res = m.predict(str(FT/"images"/(mp.stem+".png")), conf=CONF, verbose=False, device=0)[0]
        dets = [[float(v) for v in b.xyxy[0]] for b in res.boxes if int(b.cls[0]) == 0]
        gts = {}
        for p in meta["persons"]:
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            gts[p["id"]] = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
        frames.append({"f": meta["frame"], "gts": gts, "dets": dets})

    # ── 1) RAW recall + 검출기 자체 precision ─────────────────────
    raw_hit = raw_tot = 0
    det_total = det_tp = 0
    for fr in frames:
        for gid, gb in fr["gts"].items():
            raw_tot += 1
            if any(iou(gb, d) > 0.3 for d in fr["dets"]):
                raw_hit += 1
        for d in fr["dets"]:
            det_total += 1
            if any(iou(gb, d) > 0.3 for gb in fr["gts"].values()): det_tp += 1

    # ── 2) TRACKED(실제 그리디 IoU 추적 + 인과 coast) ─────────────
    # 트랙: {box, vel(dx,dy), last_seen, prev_box}
    tracks = []
    per_frame_outputs = []   # 프레임별 추적기 출력 박스(실검출+coast)
    for fr in frames:
        f = fr["f"]; dets = [d[:] for d in fr["dets"]]
        # 예측 위치로 매칭
        for t in tracks:
            gap = f - t["last_seen"]
            t["pred"] = [t["box"][i] + t["vel"][i % 2] * gap for i in range(4)]
        used = [False]*len(dets)
        # 그리디: (트랙,검출) IoU 큰 순
        pairs = []
        for ti, t in enumerate(tracks):
            for di, d in enumerate(dets):
                v = iou(t["pred"], d)
                if v > 0.2: pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        matched_t = set(); matched_d = set()
        for v, ti, di in pairs:
            if ti in matched_t or di in matched_d: continue
            matched_t.add(ti); matched_d.add(di)
            t = tracks[ti]; d = dets[di]
            cx0 = (t["box"][0]+t["box"][2])/2; cy0 = (t["box"][1]+t["box"][3])/2
            cx1 = (d[0]+d[2])/2; cy1 = (d[1]+d[3])/2
            gap = max(1, f - t["last_seen"])
            t["vel"] = [(cx1-cx0)/gap, (cy1-cy0)/gap]
            t["box"] = d; t["last_seen"] = f; t["hits"] = t.get("hits", 1) + 1
        # 미매칭 검출 → 새 트랙(미확정)
        for di, d in enumerate(dets):
            if di in matched_d: continue
            tracks.append({"box": d, "vel": [0, 0], "last_seen": f, "hits": 1})
        # 이 프레임 출력 = 실검출 + '확정 트랙'의 coast 박스
        #   확정(hits>=2)만 coast → 스퍼리어스 1회 검출이 유령으로 번지는 것 차단(precision 보호).
        #   화면을 벗어난 예측 박스는 버림(사람이 나갔음).
        out = list(dets)
        for ti, t in enumerate(tracks):
            if ti in matched_t: continue
            gap = f - t["last_seen"]
            if t.get("hits", 1) >= 2 and 1 <= gap <= K:
                pb = [t["box"][i] + t["vel"][i % 2] * gap for i in range(4)]
                ccx, ccy = (pb[0]+pb[2])/2, (pb[1]+pb[3])/2
                if 0 <= ccx <= W and 0 <= ccy <= H:      # 프레임 안일 때만 coast
                    out.append(pb)
        per_frame_outputs.append(out)

    trk_hit = trk_tot = 0; trk_out_total = trk_out_tp = 0
    for fr, out in zip(frames, per_frame_outputs):
        for gid, gb in fr["gts"].items():
            trk_tot += 1
            if any(iou(gb, o) > 0.3 for o in out): trk_hit += 1
        # precision: 출력 박스 중 GT와 매칭되는 비율
        for o in out:
            trk_out_total += 1
            if any(iou(gb, o) > 0.3 for gb in fr["gts"].values()): trk_out_tp += 1

    # ── 3) BRIDGE(오프라인 상한) — id로 앞뒤 확인된 프레임 사이를 메움 ──
    # 사람별 '검출된 프레임' 집합을 만들고, 그 사이(gap<=K) 프레임은 메울 수 있다고 본다.
    seen_frames = defaultdict(list)   # id -> 검출된 프레임 목록
    all_frames = defaultdict(list)    # id -> 등장한 프레임 목록
    for fr in frames:
        for gid, gb in fr["gts"].items():
            all_frames[gid].append(fr["f"])
            if any(iou(gb, d) > 0.3 for d in fr["dets"]): seen_frames[gid].append(fr["f"])
    br_hit = br_tot = 0
    for gid, fl in all_frames.items():
        sset = set(seen_frames[gid]); slist = sorted(sset)
        for f in fl:
            br_tot += 1
            if f in sset: br_hit += 1; continue
            # 앞뒤로 K 이내에 검출된 프레임이 모두 있으면 보간 가능
            before = [x for x in slist if x < f and f - x <= K]
            after = [x for x in slist if x > f and x - f <= K]
            if before and after: br_hit += 1

    # ── 사람별 검출률 분포: 잠깐 놓침(간헐) vs 지속 놓침(항상 작아 안 잡힘) 구분 ──
    #   각 사람의 (검출프레임/등장프레임). 100%=완전검출, 0%=한번도 못잡음(추적 불가).
    rates = []
    for gid, fl in all_frames.items():
        rates.append(len(seen_frames[gid]) / len(fl))
    rates.sort()
    never = sum(1 for r in rates if r == 0)          # 한 번도 검출 안 됨 → 추적 시작 불가
    partial = sum(1 for r in rates if 0 < r < 1)      # 간헐 검출 → 추적으로 메울 수 있음
    always = sum(1 for r in rates if r == 1)          # 항상 검출

    print("TEMPORAL_RESULT", flush=True)
    print(f"데이터셋 {FT.name} · 프레임 {len(frames)}장 · 사람관측 {raw_tot}개 · 고유인물 {len(all_frames)}명 · K={K}", flush=True)
    print(f"1) RAW      (프레임별 검출만)        recall {raw_hit/raw_tot:.3f}  precision {det_tp/det_total:.3f}", flush=True)
    print(f"2) TRACKED  (실제 추적+인과 coast)   recall {trk_hit/trk_tot:.3f}  precision {trk_out_tp/trk_out_total:.3f}", flush=True)
    print(f"3) BRIDGE   (오프라인 상한, 앞뒤보간) recall {br_hit/br_tot:.3f}", flush=True)
    print(f"인물별 검출률: 항상검출 {always}명 · 간헐(추적가능) {partial}명 · 한번도못잡음(추적불가) {never}명", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
