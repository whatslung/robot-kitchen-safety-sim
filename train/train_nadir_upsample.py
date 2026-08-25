"""나디르 person-only 재학습 — sim 업샘플링 검증.

가설: 1차(real 3389 + sim 140 = sim 4%)에서 sim recall 0.67로 낮았던 건 sim 과소대표 때문.
데이터를 새로 안 만들고 **sim 경로를 N배 반복**해 비율만 4%→~25%로 올려 재학습하면
sim recall이 오르는지 본다(양성이면 '비율 문제' 확정 → 다음은 sim 대량 생성).

val/test는 절대 업샘플하지 않는다(평가 왜곡 방지). 평가는 동일한 sim_test/real_test.

    uv run --group serve python train/train_nadir_upsample.py [--repeat 8]
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = next(
    (c for c in [ROOT, *ROOT.parents] if (c / "dataset" / "nadir-person").is_dir()), ROOT)
ND = DATA_ROOT / "dataset" / "nadir-person"
PROJECT = str(DATA_ROOT / "training")


def person_metrics(res):
    idx = [int(c) for c in res.box.ap_class_index]
    if 0 not in idx:
        return None
    i = idx.index(0)
    return {k: round(float(v[i]), 4) for k, v in
            [("precision", res.box.p), ("recall", res.box.r),
             ("map50", res.box.ap50), ("map50_95", res.box.ap)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=8, help="sim 반복 배수 (8 → sim≈25%)")
    args = ap.parse_args()

    real = [l for l in (ND / "real_train.txt").read_text().split("\n") if l.strip()]
    sim = [l for l in (ND / "sim_train.txt").read_text().split("\n") if l.strip()]
    up = real + sim * args.repeat                            # sim 을 N배 반복
    pct = round(100 * len(sim) * args.repeat / len(up), 1)
    up_txt = ND / f"mix_train_up{args.repeat}.txt"
    up_txt.write_text("\n".join(up) + "\n", encoding="utf-8")

    up_yaml = ND / f"nadir_mix_up{args.repeat}.yaml"
    up_yaml.write_text(
        f"# 자동 생성 — train_nadir_upsample.py (sim ×{args.repeat} ≈ {pct}%)\n"
        f"train: {up_txt.as_posix()}\n"
        f"val: {(ND / 'mix_val.txt').as_posix()}\n"
        f"test: {(ND / 'real_test.txt').as_posix()}\n"
        f"nc: 1\nnames: ['person']\n", encoding="utf-8")

    name = f"yolo11s_nadir_up{args.repeat}"
    print(f"[upsample] real {len(real)} + sim {len(sim)}×{args.repeat} = {len(up)} "
          f"(sim {pct}%)", flush=True)

    on_gpu = torch.cuda.is_available()
    m = YOLO("yolo11s.pt")
    m.train(
        data=str(up_yaml), epochs=100, imgsz=640,
        batch=-1 if on_gpu else 8, device=0 if on_gpu else "cpu",
        patience=20, seed=42, deterministic=True,
        project=PROJECT, name=name, exist_ok=True, plots=True, verbose=True,
    )
    sim_m = person_metrics(m.val(data=str(ND / "sim_test.yaml"), split="test",
                                 project=PROJECT, name=name + "_eval_sim", exist_ok=True))
    real_m = person_metrics(m.val(data=str(ND / "real_test.yaml"), split="test",
                                  project=PROJECT, name=name + "_eval_real", exist_ok=True))
    out = {"model": f"yolo11s nadir mix, sim×{args.repeat} ({pct}%)",
           "person_sim_test": sim_m, "person_real_test": real_m,
           "baseline_ref": {"sim": 0.665, "real": 0.852, "note": "sim 4% (repeat=1)"}}
    p = DATA_ROOT / "training" / (name + "_summary.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 업샘플 결과 ==")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
