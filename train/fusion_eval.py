"""4캠 융합 평가 — 단일 카메라 recall vs 4대 융합 recall.
   같은 씬을 4대가 찍고, 각 사람에 id(person_N)를 붙였다. '한 대라도 잡으면 성공'으로 융합 계산."""
from pathlib import Path


def main():
    import json
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / "dataset/fusion-test"
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    CONF = 0.15

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0

    metas = sorted((FT / "meta").glob("*.json"))
    # scene → { person_id → detected_in_any_cam }, and coverage
    scene_person_cov = defaultdict(set)       # (scene) -> set of person_ids seen by >=1 cam
    scene_person_det = defaultdict(set)       # (scene) -> set of person_ids detected by >=1 cam
    cam_total = cam_det = 0                    # 단일카메라(라벨 단위) 집계
    W, H = 960, 720

    for mp in metas:
        meta = json.loads(mp.read_text())
        S, C, persons = meta["scene"], meta["cam"], meta["persons"]
        img = FT / "images" / (mp.stem + ".png")
        res = m.predict(str(img), conf=CONF, verbose=False, device=0)[0]
        dets = [[float(v) for v in b.xyxy[0]] for b in res.boxes if int(b.cls[0]) == 0]
        for p in persons:
            # GT box → 픽셀 (cy는 bottom-origin: cy=1이 위 → pixel_y=(1-cy)*H)
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            gb = [(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H]
            hit = any(iou(gb, d) > 0.3 for d in dets)
            cam_total += 1; cam_det += 1 if hit else 0
            scene_person_cov[S].add(p["id"])
            if hit:
                scene_person_det[S].add(p["id"])

    # 융합 recall: 씬별 (한 대라도 잡은 사람) / (한 대라도 본 사람)
    fused_cov = fused_det = 0
    for S in scene_person_cov:
        fused_cov += len(scene_person_cov[S])
        fused_det += len(scene_person_det[S])

    per_cam = cam_det / cam_total if cam_total else 0
    fused = fused_det / fused_cov if fused_cov else 0
    print("FUSION_RESULT", flush=True)
    print(f"단일 카메라 recall (라벨 단위 {cam_total}개): {per_cam:.3f}", flush=True)
    print(f"4대 융합 recall   (고유 사람 {fused_cov}명):   {fused:.3f}", flush=True)
    print(f"→ 융합으로 놓친 사람 {fused_cov-fused_det}명 / 단일로 놓친 라벨 {cam_total-cam_det}개", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
