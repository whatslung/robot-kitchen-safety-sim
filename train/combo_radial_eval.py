"""반경대별 커버리지 평가 — 예측/회피 목적 검증.
   로봇(원점) 기준 반경대(정지링 3.1 · 감속링 3.9 · 접근링 3.9~5 · 먼거리)에서
   월드 융합 추적이 사람을 얼마나 잡는지. 4대(combo2/3)가 못 덮던 접근링(3.9~5m)을
   6대(combo6)가 덮어 조기 예측이 가능한지 본다.

   추가: '예측 리드타임' — 감속링(3.9m) 진입 전에 그 사람이 몇 프레임 연속 추적됐는지
   (속도 추정에 필요한 이력). 6대가 접근링을 덮으면 진입 전 이력이 길어진다.
"""
from pathlib import Path


def main():
    import json, sys, numpy as np
    from collections import defaultdict
    from ultralytics import YOLO
    from scipy.optimize import linear_sum_assignment
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    ds = sys.argv[1] if len(sys.argv) > 1 else "combo6"
    FT = R / ("dataset/" + ds)
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, K, MERGE, RM = 960, 720, 0.15, 5, 0.4, 0.6
    STOP, SLOW, APPROACH = 3.10, 3.90, 5.0

    metas = [json.loads(p.read_text()) for p in sorted((FT / "meta").glob("*.json"))]
    corr = defaultdict(lambda: ([], []))
    for mt in metas:
        for p in mt["persons"]:
            if p["id"] in mt["world"]:
                corr[mt["cam"]][0].append([p["cx"], p["cy"], 1.0]); corr[mt["cam"]][1].append(mt["world"][p["id"]])
    aff = {c: np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0] for c, (A, B) in corr.items() if len(A) >= 3}
    def tw(c, cxn, cyn): v = np.array([cxn, cyn, 1.0]) @ aff[c]; return (float(v[0]), float(v[1]))

    by_cf = defaultdict(list)
    for mt in metas:
        c, f = mt["cam"], mt["frame"]
        if c not in aff: continue
        base = "t" + str(f).zfill(3) + "_c" + str(c)
        res = m.predict(str(FT/"images"/(base+".png")), conf=CONF, verbose=False, device=0)[0]
        by_cf[(c, f)] = [tw(c, (x1+x2)/2/W, (y1+y2)/2/H) for b in res.boxes if int(b.cls[0]) == 0
                         for (x1, y1, x2, y2) in [[float(v) for v in b.xyxy[0]]]]
    frames = sorted({mt["frame"] for mt in metas}); cams = sorted(aff)
    gt_world = defaultdict(dict)
    for mt in metas:
        for p in mt["persons"]:
            if p["id"] in mt["world"]: gt_world[mt["frame"]][p["id"]] = tuple(mt["world"][p["id"]])

    def dist(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    def rad(p): return (p[0]**2 + p[1]**2)**0.5

    # 월드 추적기
    tracks = []; outs = {}
    for f in frames:
        raw = [pp for c in cams for pp in by_cf.get((c, f), [])]
        mg = []
        for p in raw:
            h = next((i for i, q in enumerate(mg) if dist(p, q) < MERGE), None)
            if h is None: mg.append(list(p))
            else: mg[h] = [(mg[h][0]+p[0])/2, (mg[h][1]+p[1])/2]
        for t in tracks: g = f-t["last"]; t["pred"] = (t["pos"][0]+t["vel"][0]*g, t["pos"][1]+t["vel"][1]*g)
        mt_ = set(); md = set()
        if tracks and mg:
            C = np.array([[dist(d, t["pred"]) for t in tracks] for d in mg])
            for di, ti in zip(*linear_sum_assignment(C)):
                if C[di, ti] <= 0.8:
                    t = tracks[ti]; d = mg[di]; g = max(1, f-t["last"]); t["vel"] = ((d[0]-t["pos"][0])/g, (d[1]-t["pos"][1])/g); t["pos"] = (d[0], d[1]); t["last"] = f; t["hits"] += 1; mt_.add(ti); md.add(di)
        for di, d in enumerate(mg):
            if di not in md: tracks.append({"pos": (d[0], d[1]), "vel": (0, 0), "last": f, "hits": 1})
        out = list(mg)
        for ti, t in enumerate(tracks):
            if ti in mt_: continue
            g = f-t["last"]
            if t["hits"] >= 2 and 1 <= g <= K: out.append(t["pred"])
        outs[f] = out; tracks = [t for t in tracks if f-t["last"] <= K]

    # 반경대별 recall (커버리지 OR vs 월드융합)
    bands = [("정지<3.1", 0, STOP), ("감속3.1-3.9", STOP, SLOW), ("접근3.9-5.0", SLOW, APPROACH), ("먼거리>5", APPROACH, 99)]
    print("RADIAL_RESULT", flush=True)
    print(f"데이터셋 {ds} · 카메라 {len(cams)}대 · 프레임 {len(frames)}", flush=True)
    print(f"{'반경대(m)':<14}{'사람':>6}{'커버OR':>9}{'월드융합':>9}", flush=True)
    for name, lo, hi in bands:
        tot = ohit = whit = 0
        for f in frames:
            allp = [pp for c in cams for pp in by_cf.get((c, f), [])]
            for pid, gp in gt_world[f].items():
                r = rad(gp)
                if not (lo <= r < hi): continue
                tot += 1
                if any(dist(gp, d) <= RM for d in allp): ohit += 1
                if any(dist(gp, o) <= RM for o in outs[f]): whit += 1
        if tot:
            print(f"{name:<14}{tot:>6}{ohit/tot:>9.3f}{whit/tot:>9.3f}", flush=True)
        else:
            print(f"{name:<14}{tot:>6}{'-':>9}{'-':>9}", flush=True)

    # 예측 리드타임: 감속링(3.9) 진입 시점 직전 연속 추적 프레임 수
    #   각 사람 id의 반경 시계열에서 3.9 밑으로 처음 들어오는 프레임 t*를 찾고,
    #   [t*-L, t*-1]에서 그 사람이 검출(커버OR)된 프레임 비율 = 진입 전 이력 충실도.
    idr = defaultdict(dict)  # pid -> {frame: radius}
    iddet = defaultdict(dict)  # pid -> {frame: detected(bool)}
    for f in frames:
        allp = [pp for c in cams for pp in by_cf.get((c, f), [])]
        for pid, gp in gt_world[f].items():
            idr[pid][f] = rad(gp)
            iddet[pid][f] = any(dist(gp, d) <= RM for d in allp)
    L = 8  # 진입 전 살펴볼 프레임 수(≈1s @ ~8fps 상당)
    crossings = 0; hist_frac = []
    for pid, rr in idr.items():
        fs = sorted(rr)
        for i in range(1, len(fs)):
            if rr[fs[i-1]] >= SLOW and rr[fs[i]] < SLOW:   # 감속링 진입
                crossings += 1
                prev = [fs[j] for j in range(max(0, i-L), i)]
                if prev: hist_frac.append(sum(iddet[pid][pf] for pf in prev) / len(prev))
    if hist_frac:
        import statistics
        print(f"감속링 진입 {crossings}회 · 진입 전 {L}프레임 추적충실도 median {statistics.median(hist_frac):.3f} mean {statistics.mean(hist_frac):.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
