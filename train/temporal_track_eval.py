"""제대로 된 추적기(ByteTrack)로 시간축 recall 재측정.
   단순 그리디 IoU 대신 ultralytics 내장 ByteTrack(칼만+2단계 매칭+트랙버퍼)을 써서
   id가 붙은 검출을 얻고, 최근 K프레임 안에 있던 트랙이 이번 프레임에 없으면
   등속 예측으로 이어붙인다(배포 가능한 인과 방식). BRIDGE(GT-id 상한)와 비교.

   비교 대상:
   - RAW        : 검출만 (ByteTrack의 실검출, id 무시)
   - BYTETRACK  : ByteTrack id + 인과 coast (실제 배포 시 얻는 값, 연관오류 포함)
   - BRIDGE     : GT id로 앞뒤 확인 사이 보간 (시간축 상한)
"""
from pathlib import Path


def main():
    import json, sys
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / ("dataset/" + (sys.argv[1] if len(sys.argv) > 1 else "temporal-clip"))
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, K = 960, 720, 0.15, 5

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0

    metas = sorted((FT / "meta").glob("*.json"), key=lambda p: int(p.stem.split("_")[1]))
    frames = []
    for mp in metas:
        meta = json.loads(mp.read_text())
        # persist=True 로 프레임 간 트랙 상태 유지, 프레임 순서대로 호출
        res = m.track(str(FT/"images"/(mp.stem+".png")), persist=True, tracker="bytetrack.yaml",
                      conf=CONF, verbose=False, device=0)[0]
        dets = []
        for b in res.boxes:
            if int(b.cls[0]) != 0: continue
            box = [float(v) for v in b.xyxy[0]]
            tid = int(b.id[0]) if b.id is not None else -1
            dets.append((box, tid))
        gts = {}
        for p in meta["persons"]:
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            gts[p["id"]] = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
        frames.append({"f": meta["frame"], "gts": gts, "dets": dets})

    # ── RAW ──
    raw_hit = raw_tot = det_total = det_tp = 0
    for fr in frames:
        for gid, gb in fr["gts"].items():
            raw_tot += 1
            if any(iou(gb, d[0]) > 0.3 for d in fr["dets"]): raw_hit += 1
        for d in fr["dets"]:
            det_total += 1
            if any(iou(gb, d[0]) > 0.3 for gb in fr["gts"].values()): det_tp += 1

    # ── BYTETRACK + 인과 coast ──
    # 트랙 id별 마지막 상태(box, vel, last_frame). 이번 프레임에 없고 최근 K 안이면 등속 예측 출력.
    last = {}   # tid -> {box, vel, lf}
    per_out = []
    for fr in frames:
        f = fr["f"]; cur_ids = set()
        out = []
        for box, tid in fr["dets"]:
            out.append(box)
            if tid >= 0:
                cur_ids.add(tid)
                if tid in last:
                    lb = last[tid]["box"]; gap = max(1, f - last[tid]["lf"])
                    vel = [((box[0]+box[2]) - (lb[0]+lb[2]))/2/gap, ((box[1]+box[3]) - (lb[1]+lb[3]))/2/gap]
                else:
                    vel = [0, 0]
                last[tid] = {"box": box, "vel": vel, "lf": f}
        # coast: 최근 K 안에 있었으나 이번에 없는 트랙 → 예측 박스
        for tid, st in last.items():
            if tid in cur_ids: continue
            gap = f - st["lf"]
            if 1 <= gap <= K:
                pb = [st["box"][i] + st["vel"][i % 2] * gap for i in range(4)]
                ccx, ccy = (pb[0]+pb[2])/2, (pb[1]+pb[3])/2
                if 0 <= ccx <= W and 0 <= ccy <= H: out.append(pb)
        per_out.append(out)

    bt_hit = bt_tot = bt_ot = bt_tp = 0
    for fr, out in zip(frames, per_out):
        for gid, gb in fr["gts"].items():
            bt_tot += 1
            if any(iou(gb, o) > 0.3 for o in out): bt_hit += 1
        for o in out:
            bt_ot += 1
            if any(iou(gb, o) > 0.3 for gb in fr["gts"].values()): bt_tp += 1

    # ── BRIDGE (GT-id 상한) ──
    seen = defaultdict(set); allf = defaultdict(list)
    for fr in frames:
        for gid, gb in fr["gts"].items():
            allf[gid].append(fr["f"])
            if any(iou(gb, d[0]) > 0.3 for d in fr["dets"]): seen[gid].add(fr["f"])
    br_hit = br_tot = 0
    for gid, fl in allf.items():
        ss = sorted(seen[gid])
        for f in fl:
            br_tot += 1
            if f in seen[gid]: br_hit += 1; continue
            if any(x < f and f-x <= K for x in ss) and any(x > f and x-f <= K for x in ss): br_hit += 1

    # id 연속성: GT 사람당 ByteTrack이 붙인 서로 다른 track id 수(1이 이상적, 클수록 id 스위치 많음)
    print("BYTETRACK_RESULT", flush=True)
    print(f"데이터셋 {FT.name} · 프레임 {len(frames)}장 · 사람관측 {raw_tot}개 · K={K}", flush=True)
    print(f"RAW       recall {raw_hit/raw_tot:.3f}  precision {det_tp/det_total:.3f}", flush=True)
    print(f"BYTETRACK recall {bt_hit/bt_tot:.3f}  precision {bt_tp/bt_ot:.3f}   (배포 가능, 연관오류 포함)", flush=True)
    print(f"BRIDGE    recall {br_hit/br_tot:.3f}                     (시간축 상한)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
