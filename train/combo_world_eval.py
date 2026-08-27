"""월드좌표 융합 추적 — 카메라 간 트랙 연결이 recall을 0.900→? 로 얼마나 올리는지.

combo2 데이터(메타에 사람 월드좌표 + 카메라 intrinsics 저장, 4대 동시 순간 캡처).
방법:
 1) 카메라별 아핀(이미지 cx,cy → 월드 X,Z)을 GT 대응으로 1회 피팅(=카메라 캘리브레이션).
 2) 각 카메라 검출 박스중심을 월드로 매핑 → 프레임별 전 카메라 검출을 월드에 pool.
 3) 근접(0.4m) 검출 병합(같은 사람이 두 카메라에 보인 중복 제거).
 4) 월드공간 단일 추적기(거리게이트 그리디 + 등속 + coast)로 id 부여·잔여 프레임 coast.
    → 카메라 간 트랙이 하나로 이어짐(핸드오프 구간 놓침을 coast가 메움).
 5) recall = GT 사람(월드 스냅샷, 어느 카메라라도 라벨) 이 추적기 출력 R(0.6m) 안에 있으면 성공.

비교: 단일 0.848 · 커버리지 OR(연결 없음) 0.900 · GT-id 완벽연결 이상치 0.950 사이 어디에 앉는지.
"""
from pathlib import Path


def main():
    import json, sys, numpy as np
    from collections import defaultdict
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    FT = R / ("dataset/" + (sys.argv[1] if len(sys.argv) > 1 else "combo2"))
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, K, MERGE, RMATCH = 960, 720, 0.15, 5, 0.4, 0.6

    metas = [json.loads(p.read_text()) for p in sorted((FT / "meta").glob("*.json"))]

    # 1) 카메라별 아핀 (cx,cy,1)->(X,Z), GT 대응 최소자승
    corr = defaultdict(lambda: ([], []))
    for mt in metas:
        for p in mt["persons"]:
            if p["id"] in mt["world"]:
                corr[mt["cam"]][0].append([p["cx"], p["cy"], 1.0])
                corr[mt["cam"]][1].append(mt["world"][p["id"]])
    aff = {}
    for c, (A, B) in corr.items():
        sol, *_ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None); aff[c] = sol  # 3x2
    def to_world(c, cxn, cyn):
        v = np.array([cxn, cyn, 1.0]) @ aff[c]; return float(v[0]), float(v[1])

    # 2) 검출 → 월드. (cam,frame)별 검출 월드점 + GT
    by_cf = {}
    stems = {(mt["cam"], mt["frame"]): None for mt in metas}
    meta_of = {(mt["cam"], mt["frame"]): mt for mt in metas}
    for (c, f), _ in stems.items():
        mt = meta_of[(c, f)]
        base = "t" + str(f).zfill(3) + "_c" + str(c)
        res = m.predict(str(FT/"images"/(base+".png")), conf=CONF, verbose=False, device=0)[0]
        pts = []
        for b in res.boxes:
            if int(b.cls[0]) != 0: continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cxn = (x1+x2)/2/W; cyn = (y1+y2)/2/H           # 아핀은 GT(cx,cy)로 피팅 — 뒤집지 않음(실측 확인)
            pts.append(to_world(c, cxn, cyn))
        by_cf[(c, f)] = pts

    frames = sorted({f for (_, f) in by_cf})
    cams = sorted({c for (c, _) in by_cf})

    # GT 월드: frame별 {pid: (X,Z)} — 어느 카메라라도 라벨된 사람만(그 순간 시야 안)
    gt_world = defaultdict(dict)
    for mt in metas:
        for p in mt["persons"]:
            if p["id"] in mt["world"]:
                gt_world[mt["frame"]][p["id"]] = tuple(mt["world"][p["id"]])

    def dist(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    # 3)+4) 월드공간 추적기 (거리게이트 그리디 + 등속 + coast)
    from scipy.optimize import linear_sum_assignment
    tracks = []   # {pos,vel,last,hits,id}
    nid = [1]
    outs_by_f = {}
    for f in frames:
        # pool + 근접 병합
        raw = []
        for c in cams: raw += by_cf.get((c, f), [])
        merged = []
        for p in raw:
            hit = next((mi for mi, mp in enumerate(merged) if dist(p, mp) < MERGE), None)
            if hit is None: merged.append(list(p))
            else: merged[hit] = [(merged[hit][0]+p[0])/2, (merged[hit][1]+p[1])/2]
        # 예측
        for t in tracks:
            g = f - t["last"]; t["pred"] = (t["pos"][0]+t["vel"][0]*g, t["pos"][1]+t["vel"][1]*g)
        # 거리게이트 헝가리안
        matched_t = set(); matched_d = set()
        if tracks and merged:
            C = np.zeros((len(merged), len(tracks)))
            for di, d in enumerate(merged):
                for ti, t in enumerate(tracks): C[di, ti] = dist(d, t["pred"])
            di_i, ti_i = linear_sum_assignment(C)
            for di, ti in zip(di_i, ti_i):
                if C[di, ti] <= 0.8:
                    t = tracks[ti]; d = merged[di]; g = max(1, f - t["last"])
                    t["vel"] = ((d[0]-t["pos"][0])/g, (d[1]-t["pos"][1])/g)
                    t["pos"] = (d[0], d[1]); t["last"] = f; t["hits"] += 1
                    matched_t.add(ti); matched_d.add(di)
        for di, d in enumerate(merged):
            if di not in matched_d:
                tracks.append({"pos": (d[0], d[1]), "vel": (0, 0), "last": f, "hits": 1, "id": nid[0]}); nid[0] += 1
        # 출력 = 검출(병합) + 확정트랙 coast
        out = list(merged)
        for ti, t in enumerate(tracks):
            if ti in matched_t: continue
            g = f - t["last"]
            if t["hits"] >= 2 and 1 <= g <= K: out.append(t["pred"])
        outs_by_f[f] = out
        tracks = [t for t in tracks if f - t["last"] <= K]

    # 5) recall (단계별)
    def cover(get):  # get(f)->list of world points
        hit = tot = 0
        for f in frames:
            outs = get(f)
            for pid, gp in gt_world[f].items():
                tot += 1
                if any(dist(gp, o) <= RMATCH for o in outs): hit += 1
        return hit/tot if tot else 0, tot

    # 단일카메라 커버리지 없이(검출 그대로, 카메라별) — 참고: 커버리지 OR(연결 없음)
    or_cov = lambda f: [pp for c in cams for pp in by_cf.get((c, f), [])]
    r_or, tot = cover(or_cov)
    r_world, _ = cover(lambda f: outs_by_f[f])

    print("COMBO_WORLD_RESULT", flush=True)
    print(f"데이터셋 {FT.name} · 프레임 {len(frames)} · 카메라 {len(cams)}대 · 사람관측 {tot}개 · 매칭 {RMATCH}m", flush=True)
    for c in cams:
        A, B = corr[c]; sol = aff[c]; pred = np.array(A) @ sol; err = np.linalg.norm(pred - np.array(B), axis=1)
        print(f"  cam{c} 아핀잔차 median {np.median(err):.3f}m", flush=True)
    print(f"공간 커버리지 OR (연결 없음)   recall {r_or:.3f}", flush=True)
    print(f"월드 융합 추적 (카메라간 연결)  recall {r_world:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
