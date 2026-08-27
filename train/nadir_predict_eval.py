"""FULL PIPELINE — 나디르 추적 → 궤적 예측 → 안전, 4대 vs 6대 비교.

나디르 6대 긴 클립(combolong, dt=0.4s, 월드좌표+intrinsics)에서:
 1) 카메라별 아핀(이미지→바닥) GT 피팅 → 검출을 월드로 매핑.
 2) 두 구성:
    - 6cam: 6대 전 검출.
    - 4cam: 검출을 '4대 레이아웃 커버리지(여유 0.3m)'로 필터 → 동일 궤적을 4대가 관측.
 3) GT 사람별 관측 시계열(프레임마다 최근접 검출 within RM, 없으면 결측→hold-last).
 4) obs8/pred12 윈도우(0.4s) → CV·칼만·LSTM·Transformer 예측.
 5) ADE/FDE + 위험진입(로봇 원점, 정지링 3.10m·감속 3.90m) recall/precision.
 4대는 커버리지 구멍으로 관측 결측↑ → 예측·안전 저하. 그 차이가 카메라 수의 값.

가정: scene AU == m (궤적 학습셋 mPerAU=1). 로봇=원점(조리 셀). 예측기는 sim AU로 학습됨 → 변환 불필요.
LSTM/Transformer 가중치는 HF(chanubc/human-move-lstm·transformer)에서 자동.
"""
from pathlib import Path


def main():
    import json, numpy as np, math
    from collections import defaultdict
    import sys
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    sys.path.insert(0, str(R))
    from ultralytics import YOLO
    import torch
    from huggingface_hub import hf_hub_download
    from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor
    from trajectory.types import TrackScene, Track
    from trajectory.learned_predictor import LearnedPredictor, build_net, build_transformer_net, OBS, PRED
    from trajectory import risk as riskmod, evaluator as ev

    FT = R / "dataset/combolong"
    m = YOLO(str(R / "training/sweep_r3389/weights/best.pt"))
    W, H, CONF, RM, DT = 960, 720, 0.15, 0.6, 0.4
    STOP, SLOW, KSIG, TAU = 3.10, 3.90, 1.0, 0.1
    HOR = PRED * DT
    ROBOT = (0.0, 0.0)

    metas = [json.loads(p.read_text()) for p in sorted((FT / "meta").glob("*.json"))]

    # 카메라별 아핀
    corr = defaultdict(lambda: ([], []))
    for mt in metas:
        for p in mt["persons"]:
            if p["id"] in mt["world"]:
                corr[mt["cam"]][0].append([p["cx"], p["cy"], 1.0]); corr[mt["cam"]][1].append(mt["world"][p["id"]])
    aff = {c: np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0] for c, (A, B) in corr.items() if len(A) >= 3}
    def tw(c, cxn, cyn): v = np.array([cxn, cyn, 1.0]) @ aff[c]; return (float(v[0]), float(v[1]))

    # (cam,frame) 검출 월드점
    det_cf = {}
    for mt in metas:
        c, f = mt["cam"], mt["frame"]
        if c not in aff: continue
        res = m.predict(str(FT/"images"/("t"+str(f).zfill(3)+"_c"+str(c)+".png")), conf=CONF, verbose=False, device=0)[0]
        det_cf[(c, f)] = [tw(c, (x1+x2)/2/W, (y1+y2)/2/H) for b in res.boxes if int(b.cls[0]) == 0
                          for (x1, y1, x2, y2) in [[float(v) for v in b.xyxy[0]]]]
    frames = sorted({mt["frame"] for mt in metas}); cams = sorted(aff)

    # GT 사람 월드 궤적
    gt = defaultdict(dict)   # pid -> {frame: (x,z)}
    for mt in metas:
        for pid, xz in mt["world"].items(): gt[pid][mt["frame"]] = (float(xz[0]), float(xz[1]))

    # 4대 레이아웃 커버리지(여유 0.3m) — 검출 필터
    L4 = [(-1.5,-1.0),(2.3,-1.0),(-1.5,2.6),(2.3,2.6)]
    def in4(x, z, mgn=0.3): return any(abs(x-cx) <= 2.9-mgn and abs(z-cz) <= 2.3-mgn for cx, cz in L4)

    def dets_for(cfg, f):
        d = [pp for c in cams for pp in det_cf.get((c, f), [])]
        if cfg == "4cam": d = [p for p in d if in4(*p)]
        return d

    def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    # 예측기
    def load(repo, netfn):
        w = hf_hub_download(repo_id=repo, filename="model.pt"); net = netfn()
        net.load_state_dict(torch.load(w, map_location="cpu")); net.eval()
        return LearnedPredictor(net=net, device="cpu")
    lstm = load("chanubc/human-move-lstm", build_net)
    trans = load("chanubc/human-move-transformer", build_transformer_net)
    cv = ConstantVelocityPredictor(n_steps=PRED); kf = KalmanPredictor(n_steps=PRED)

    def predict_modes(name, obs_tf):   # obs_tf=[(t,x,z)]*OBS → modes[{path,w,sigma}]
        if name == "CV" or name == "Kalman":
            sc = TrackScene(now=obs_tf[-1][0], horizon=HOR, agents=[Track(0, obs_tf)], map=None)
            pr = (cv if name == "CV" else kf).predict(sc)
            mode0 = pr.per_agent[0][0]
            path = [(s[1], s[2]) for s in mode0.steps]; sig = [s[3] for s in mode0.steps]
            return [{"path": path, "w": 1.0, "sigma": sig}]
        lp = lstm if name == "LSTM" else trans
        return lp.predict_modes([(x, z) for (_t, x, z) in obs_tf])

    # 윈도우: 사람별 연속 프레임 obs8+pred12
    def observed(cfg, pid, f):
        gp = gt[pid].get(f)
        if gp is None: return None
        ds = dets_for(cfg, f)
        best = min((d for d in ds), key=lambda d: dist(d, gp), default=None)
        return best if (best is not None and dist(best, gp) <= RM) else None

    def hold_last(seq):   # [(x,z)|None]*OBS → 결측을 앞/뒤 최근값으로
        out = list(seq); last = None
        for i in range(len(out)):
            if out[i] is not None: last = out[i]
            elif last is not None: out[i] = last
        nxt = None
        for i in range(len(out)-1, -1, -1):
            if out[i] is not None: nxt = out[i]
            elif nxt is not None: out[i] = nxt
        return out if all(o is not None for o in out) else None

    names = ["CV", "Kalman", "LSTM", "Transformer"]
    res = {cfg: {n: {"ade": [], "fde": [], "tp": 0, "fp": 0, "fn": 0, "obs_miss": []} for n in names} for cfg in ["4cam", "6cam"]}
    Wlen = OBS + PRED
    for cfg in ["4cam", "6cam"]:
        for pid, fr in gt.items():
            fs = sorted(fr)
            # 연속 구간
            runs = []; cur = [fs[0]]
            for a, b in zip(fs, fs[1:]):
                if b == a+1: cur.append(b)
                else: runs.append(cur); cur = [b]
            runs.append(cur)
            for run in runs:
                for i in range(0, len(run)-Wlen+1):
                    win = run[i:i+Wlen]
                    obs_f, fut_f = win[:OBS], win[OBS:]
                    obs_raw = [observed(cfg, pid, f) for f in obs_f]
                    miss = sum(1 for o in obs_raw if o is None)
                    filled = hold_last(obs_raw)
                    if filled is None: continue
                    obs_tf = [(obs_f[k]*DT, filled[k][0], filled[k][1]) for k in range(OBS)]
                    gt_fut = [(fut_f[k]*DT, gt[pid][fut_f[k]][0], gt[pid][fut_f[k]][1]) for k in range(PRED)]
                    gt_fut_xz = [(x, z) for (_t, x, z) in gt_fut]
                    gt_enters = ev.enters_radius(gt_fut_xz, ROBOT, STOP)
                    for n in names:
                        modes = predict_modes(n, obs_tf)
                        pred_steps = [(DT*(j+1), modes[0]["path"][j][0], modes[0]["path"][j][1], modes[0]["sigma"][j] if j < len(modes[0]["sigma"]) else 0.0) for j in range(len(modes[0]["path"]))]
                        res[cfg][n]["ade"].append(ev.ade(pred_steps, gt_fut))
                        res[cfg][n]["fde"].append(ev.fde(pred_steps, gt_fut))
                        res[cfg][n]["obs_miss"].append(miss)
                        rr = riskmod.track_risk(modes, ROBOT, STOP, SLOW, HOR, KSIG, TAU)
                        pred_enter = rr["tEntryStop"] is not None
                        if gt_enters and pred_enter: res[cfg][n]["tp"] += 1
                        elif gt_enters and not pred_enter: res[cfg][n]["fn"] += 1
                        elif (not gt_enters) and pred_enter: res[cfg][n]["fp"] += 1

    def mean(a): return sum(a)/len(a) if a else float("nan")
    print("NADIR_PREDICT_RESULT", flush=True)
    nwin = len(res["6cam"]["CV"]["ade"])
    print(f"윈도우 {nwin} · dt={DT}s · 로봇 원점 · 정지링 {STOP}m", flush=True)
    print(f"관측결측(obs8 중): 4cam {mean(res['4cam']['CV']['obs_miss']):.2f} vs 6cam {mean(res['6cam']['CV']['obs_miss']):.2f} 프레임/윈도우", flush=True)
    print(f"{'구성':>5} {'예측기':>12} {'ADE(m)':>8} {'FDE(m)':>8} {'진입recall':>10} {'진입prec':>9}", flush=True)
    for cfg in ["4cam", "6cam"]:
        for n in names:
            d = res[cfg][n]; tp, fp, fn = d["tp"], d["fp"], d["fn"]
            rec = tp/(tp+fn) if tp+fn else float("nan"); prec = tp/(tp+fp) if tp+fp else float("nan")
            print(f"{cfg:>5} {n:>12} {mean(d['ade']):>8.3f} {mean(d['fde']):>8.3f} {rec:>10.3f} {prec:>9.3f}  (TP{tp} FN{fn} FP{fp})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
