"""세 축 결합 측정 — 존(좋은검출) × 공간융합 × 시간축추적.
   존 카메라 4대(겹침)가 같은 연속 클립을 촬영. 사람 id는 카메라·시간 모두에서 고정.
   메타: {frame(시간), cam, persons:[{id,cx,cy,w,h}]}.

   사다리(각 단계가 앞 단계 위에 쌓임):
   - 1) 단일          : 한 대·한 프레임 검출 (평균)
   - 2) +공간융합     : 프레임 t에서 '어느 카메라라도' 검출하면 성공
   - 3) +시간축(인과) : 위에 더해, id가 [t-K,t-1]에 '어느 카메라라도' 검출됐으면 이어붙임
   - 4) BRIDGE(상한)  : id 앞뒤 확인 사이 보간 (공간+시간 정보의 이론 상한)
   커버리지 = 그 프레임에 어느 카메라라도 라벨한 사람(그 순간 시야 안)."""
from pathlib import Path


def main():
    import json
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / "dataset/combo"
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, K = 960, 720, 0.15, 5

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0

    # (frame, cam) -> {pid: hit}, 그리고 (frame)별 등장 pid
    hit = defaultdict(dict)          # (t,c) -> {pid: bool}
    present = defaultdict(set)       # t -> {pid}  (그 프레임 어느 카메라든 라벨)
    single_hit = single_tot = 0
    for mp in sorted((FT / "meta").glob("*.json")):
        meta = json.loads(mp.read_text()); t, c = meta["frame"], meta["cam"]
        res = m.predict(str(FT/"images"/(mp.stem+".png")), conf=CONF, verbose=False, device=0)[0]
        dets = [[float(v) for v in b.xyxy[0]] for b in res.boxes if int(b.cls[0]) == 0]
        for p in meta["persons"]:
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            gb = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
            hh = any(iou(gb, d) > 0.3 for d in dets)
            hit[(t, c)][p["id"]] = hh
            present[t].add(p["id"])
            single_tot += 1; single_hit += 1 if hh else 0

    frames_t = sorted(present)
    cams = sorted({c for (t, c) in hit})

    # 공간융합: 프레임 t에서 pid를 '어느 카메라라도' 검출?
    def spatial_det(t, pid):
        return any(hit.get((t, c), {}).get(pid, False) for c in cams)

    # 2) +공간융합 recall
    sp_hit = sp_tot = 0
    for t in frames_t:
        for pid in present[t]:
            sp_tot += 1
            if spatial_det(t, pid): sp_hit += 1

    # 3) +시간축(인과): 공간융합으로 잡거나, 최근 K 프레임에 공간융합으로 잡혔으면 성공
    detected_frames = defaultdict(set)   # pid -> {t: 공간융합 검출된 프레임}
    for t in frames_t:
        for pid in present[t]:
            if spatial_det(t, pid): detected_frames[pid].add(t)
    st_hit = st_tot = 0
    for t in frames_t:
        for pid in present[t]:
            st_tot += 1
            if t in detected_frames[pid]: st_hit += 1; continue
            if any(t-k in detected_frames[pid] for k in range(1, K+1)): st_hit += 1

    # 4) BRIDGE 상한: 그 순간(공간융합) 검출됐거나, 앞이든 뒤든 K 이내에 검출됐으면 메움.
    #    (양쪽 필요조건을 빼야 진짜 상한 — 인과 coast는 뒤가 없어도 메우므로 그걸 포함해야 함)
    br_hit = br_tot = 0
    for pid in present_pids(present):
        df = sorted(detected_frames[pid])
        appear = [t for t in frames_t if pid in present[t]]
        for t in appear:
            br_tot += 1
            if t in detected_frames[pid]: br_hit += 1; continue
            if any(abs(x - t) <= K for x in df): br_hit += 1

    print("COMBO_RESULT", flush=True)
    print(f"프레임(시간) {len(frames_t)} · 카메라 {len(cams)}대 · 단일관측 {single_tot}개", flush=True)
    print(f"1) 단일 카메라·단일 프레임        recall {single_hit/single_tot:.3f}", flush=True)
    print(f"2) +공간융합(4대 중 하나라도)     recall {sp_hit/sp_tot:.3f}", flush=True)
    print(f"3) +시간축 추적(인과 coast K={K})  recall {st_hit/st_tot:.3f}", flush=True)
    print(f"4) BRIDGE(공간+시간 상한)         recall {br_hit/br_tot:.3f}", flush=True)
    print("DONE", flush=True)


def present_pids(present):
    s = set()
    for t in present: s |= present[t]
    return s


if __name__ == "__main__":
    main()
