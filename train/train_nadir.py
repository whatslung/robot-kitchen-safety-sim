"""나디르(top-down) person-only 검출기 — sim(v2)+real 혼합 파인튜닝 (헤지 경로).

핸드오프(나디르 YOLO 병렬 헤지)의 목표: 사선 CCTV 주력이 안 될 경우를 대비해,
나디르 도메인에서 **사람만** 검출하는 YOLO를 같은 eval로 확보한다.
데이터는 train/prep_nadir.py가 만든 dataset/nadir-person/ (person-only nc:1,
도메인별 val/test 분리, 누수 방지). sim은 GT 오라벨 수정본(v2)을 쓴다.

레시피: COCO 사전학습 yolo11s → 나디르 혼합 파인튜닝. epochs=100 imgsz=640
batch=-1 seed=42 결정적. (온디바이스면 yolo11n + ONNX/INT8.)

    uv run --group serve python train/train_nadir.py

끝나면 training/yolo11s_nadir/weights/best.pt (+ best.onnx),
training/yolo11s_nadir_summary.json 에 **도메인별**(sim/real) person 지표가 남는다.
"""
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = next(
    (c for c in [ROOT, *ROOT.parents] if (c / "dataset" / "nadir-person").is_dir()), ROOT)
ND = DATA_ROOT / "dataset" / "nadir-person"
TRAIN_YAML = str(ND / "nadir_mix.yaml")
SIM_YAML = str(ND / "sim_test.yaml")
REAL_YAML = str(ND / "real_test.yaml")
PROJECT = str(DATA_ROOT / "training")
NAME = "yolo11s_nadir"


def person_metrics(res):
    """DetMetrics에서 person(class 0)만. nc:1이면 인덱스 0."""
    idx = [int(c) for c in res.box.ap_class_index]
    if 0 not in idx:
        return None
    i = idx.index(0)
    return {
        "precision": round(float(res.box.p[i]), 4),
        "recall": round(float(res.box.r[i]), 4),
        "map50": round(float(res.box.ap50[i]), 4),
        "map50_95": round(float(res.box.ap[i]), 4),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640, help="입력 해상도 (960=sim 네이티브, 작은 사람 검출↑)")
    ap.add_argument("--name", default=None, help="run 이름 (기본 yolo11s_nadir[_imgsz])")
    args = ap.parse_args()
    name = args.name or (NAME if args.imgsz == 640 else f"{NAME}_{args.imgsz}")

    on_gpu = torch.cuda.is_available()
    print("CUDA:", on_gpu,
          torch.cuda.get_device_name(0) if on_gpu else "CPU only", "· imgsz", args.imgsz, flush=True)

    m = YOLO("yolo11s.pt")                                   # COCO 사전학습
    m.train(
        data=TRAIN_YAML, epochs=100, imgsz=args.imgsz,
        batch=-1 if on_gpu else 8,
        device=0 if on_gpu else "cpu",
        patience=20, seed=42, deterministic=True,
        project=PROJECT, name=name, exist_ok=True,
        plots=True, verbose=True,
    )

    # 도메인별 평가 — 사선 경로와 나란히 비교 가능하게 sim/real 분리. 평가도 학습 해상도로.
    sim = person_metrics(m.val(data=SIM_YAML, split="test", imgsz=args.imgsz, project=PROJECT,
                               name=name + "_eval_sim", exist_ok=True))
    real = person_metrics(m.val(data=REAL_YAML, split="test", imgsz=args.imgsz, project=PROJECT,
                                name=name + "_eval_real", exist_ok=True))
    out = {
        "model": f"yolo11s finetuned (nadir sim-v3 + real mix, person-only, imgsz{args.imgsz})",
        "train_data": TRAIN_YAML,
        "person_sim_test": sim,
        "person_real_test": real,
        "goal": "나디르 도메인(sim+real) person recall >= 0.9 (안전이면 recall 우선)",
    }
    p = DATA_ROOT / "training" / (name + "_summary.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 도메인별 person 지표 ==")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("summary →", p)


if __name__ == "__main__":
    main()
