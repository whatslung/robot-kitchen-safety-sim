"""9대 융합의 '놓침 상관성' 분석 — 여러 대가 같은 사람을 봐도 같이 놓치는가?
   독립이라면 k대가 보면 융합 놓침 = (단일놓침)^k. 실제와 비교해 상관 정도를 본다."""
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
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0

    # (scene, person_id) -> list of per-camera hit(bool)
    views = defaultdict(list)
    single_hit = single_tot = 0
    for mp in sorted((FT / "meta").glob("*.json")):
        meta = json.loads(mp.read_text()); S = meta["scene"]
        res = m.predict(str(FT/"images"/(mp.stem+".png")), conf=CONF, verbose=False, device=0)[0]
        dets = [[float(v) for v in b.xyxy[0]] for b in res.boxes if int(b.cls[0]) == 0]
        for p in meta["persons"]:
            cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
            gb = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
            hit = any(iou(gb, d) > 0.3 for d in dets)
            views[(S, p["id"])].append(hit)
            single_hit += hit; single_tot += 1

    p_single = single_hit/single_tot            # 단일 검출률
    q = 1 - p_single                             # 단일 놓침률
    # k대가 보는 사람만 모아, 실제 융합놓침 vs 독립가정 놓침 비교
    from collections import Counter
    by_k = defaultdict(lambda: [0, 0])           # k -> [총사람수, 융합놓침수(전원놓침)]
    for _, hits in views.items():
        k = len(hits)
        by_k[k][0] += 1
        if not any(hits):                        # 전원 놓침
            by_k[k][1] += 1
    print("CORR_RESULT", flush=True)
    print(f"단일 검출률 p={p_single:.3f}, 단일 놓침률 q={q:.3f}", flush=True)
    print(f"{'k대노출':>6} {'사람수':>5} {'실제융합놓침률':>10} {'독립가정 q^k':>12}", flush=True)
    for k in sorted(by_k):
        tot, miss = by_k[k]
        emp = miss/tot if tot else 0
        print(f"{k:>6} {tot:>5} {emp:>12.3f} {q**k:>12.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
