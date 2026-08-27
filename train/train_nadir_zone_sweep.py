"""real:sim 비율 스윕 — zone·real recall 최적점 탐색. zsim 237 고정, real 양만 조절."""
from pathlib import Path


def main():
    import json, torch
    from ultralytics import YOLO
    R = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
    ND = R / "dataset/nadir-zone"; NP = R / "dataset/nadir-person"; P = str(R / "training")
    real_all = [l for l in (NP / "real_train.txt").read_text().split() if l.strip()]
    zsim = [l for l in (ND / "zsim_train.txt").read_text().split() if l.strip()]

    def pm(res):
        idx = [int(c) for c in res.box.ap_class_index]
        if 0 not in idx: return None
        i = idx.index(0)
        return {k: round(float(v[i]), 3) for k, v in
                [("R", res.box.r), ("P", res.box.p), ("map50", res.box.ap50)]}

    results = []
    for rn in [3389, 700, 237]:
        real = real_all[:rn]
        mix = real + zsim
        pct = round(100 * len(zsim) / len(mix))
        mtxt = ND / f"sweep_train_r{rn}.txt"
        mtxt.write_text("\n".join(mix) + "\n", encoding="utf-8")
        yml = ND / f"sweep_r{rn}.yaml"
        yml.write_text(f"train: {mtxt.as_posix()}\nval: {(ND/'zsim_val.txt').as_posix()}\n"
                       f"test: {(ND/'zsim_test.txt').as_posix()}\nnc: 1\nnames: ['person']\n", encoding="utf-8")
        name = f"sweep_r{rn}"
        print(f"\n### real {rn} + zsim {len(zsim)} (sim {pct}%) ###", flush=True)
        m = YOLO("yolo11s.pt")
        m.train(data=str(yml), epochs=100, imgsz=640, batch=-1, device=0,
                patience=20, seed=42, deterministic=True, project=P, name=name,
                exist_ok=True, plots=False, verbose=False, workers=4)
        zone = pm(m.val(data=str(ND / "zone_simonly.yaml"), split="test", imgsz=640,
                        project=P, name=name + "_ez", exist_ok=True, workers=0))
        realm = pm(m.val(data=str(ND / "real_test.yaml"), split="test", imgsz=640,
                         project=P, name=name + "_er", exist_ok=True, workers=0))
        results.append({"real": rn, "sim_pct": pct, "zone": zone, "real": realm})
        print(f"RESULT real{rn} sim{pct}% zone={json.dumps(zone)} real={json.dumps(realm)}", flush=True)

    (R / "training" / "zone_ratio_sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 비율 스윕 요약 ==", flush=True)
    for r in results:
        print(f"real {r['real']:>4} (sim {r['sim_pct']:>2}%): zone R={r['zone']['R']} | real R={r['real']['R']}", flush=True)


if __name__ == "__main__":
    main()
