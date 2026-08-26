"""9대(3×3, 겹침 큼) 융합 평가 — 카메라 수 늘리면 유효 recall이 상한에 가는지.
   같은 씬 9대 촬영 + 사람 id. 카메라 부분집합(1/4/9)별 융합 recall 비교."""
from pathlib import Path


def main():
    import json
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / "dataset/fusion9-test"
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF = 960, 720, 0.15

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0

    # scene -> cam -> {person_id: detected(bool)}
    data = defaultdict(lambda: defaultdict(dict))
    cam_total = cam_det = 0
    for mp in sorted((FT / "meta").glob("*.json")):
        meta = json.loads(mp.read_text()); S, C = meta["scene"], meta["cam"]
        res = m.predict(str(FT / "images" / (mp.stem + ".png")), conf=CONF, verbose=False, device=0)[0]
        dets = [[float(v) for v in b.xyxy[0]] for b in res.boxes if int(b.cls[0]) == 0]
        for p in meta["persons"]:
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            gb = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
            hit = any(iou(gb, d) > 0.3 for d in dets)
            data[S][C][p["id"]] = data[S][C].get(p["id"], False) or hit
            cam_total += 1; cam_det += 1 if hit else 0

    def fused(cam_subset):
        cov = det = 0
        for S in data:
            seen = defaultdict(bool); found = defaultdict(bool)
            for C in cam_subset:
                for pid, hit in data[S].get(C, {}).items():
                    seen[pid] = True
                    if hit: found[pid] = True
            cov += len(seen); det += sum(1 for pid in seen if found[pid])
        return det / cov if cov else 0, cov

    ALL9 = list(range(9))
    CORNERS4 = [0, 2, 6, 8]           # 3×3의 네 모서리 (성긴 겹침)
    CENTER_CROSS = [1, 3, 4, 5, 7]    # 참고
    per_cam = cam_det / cam_total
    f9, c9 = fused(ALL9)
    f4, c4 = fused(CORNERS4)
    print("FUSION9_RESULT", flush=True)
    print(f"단일 카메라 (라벨 {cam_total}): recall {per_cam:.3f}", flush=True)
    print(f"4대 모서리 융합 (사람 {c4}): recall {f4:.3f}", flush=True)
    print(f"9대 융합       (사람 {c9}): recall {f9:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
