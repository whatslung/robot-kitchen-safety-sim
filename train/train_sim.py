"""직교 나디르(ds_top) 시뮬 데이터로 YOLO11s를 파인튜닝한다 — top-down 검출 make-or-break.

참고: overhead-person-yolo11/scripts/train_phase1.py (같은 저자, 실사 overhead 학습).
그 실사 모델의 시뮬 top-down 성능은 recall 0.17 / precision 0.30이었다(핸드오프 §4-2).
이 파인튜닝이 그 수치를 넘어 설비 오탐을 잡느냐가 나디르 top-down 채택의 유일한 생존 조건.

    uv run python train/train_sim.py

끝나면 training/yolo11s_orthotop/weights/best.pt (+ best.onnx),
그리고 training/summary.json 에 person 지표가 남는다.
"""
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "sim-person" / "data.yaml")
PROJECT = str(ROOT / "training")
NAME = "yolo11s_orthotop"


def person_metrics(res):
    """DetMetrics에서 person(class 0)만 뽑는다. 없으면 None."""
    idx = list(int(c) for c in res.box.ap_class_index)
    if 0 not in idx:
        return None
    i = idx.index(0)
    return {
        "precision": float(res.box.p[i]),
        "recall": float(res.box.r[i]),
        "map50": float(res.box.ap50[i]),
        "map50_95": float(res.box.ap[i]),
    }


def main():
    on_gpu = torch.cuda.is_available()
    print("CUDA:", on_gpu,
          torch.cuda.get_device_name(0) if on_gpu else "CPU only", flush=True)

    m = YOLO("yolo11s.pt")
    m.train(
        data=DATA, epochs=100, imgsz=640,
        batch=-1 if on_gpu else 8,      # GPU면 VRAM 맞춰 자동배치
        device=0 if on_gpu else "cpu",
        patience=20, seed=42,
        project=PROJECT, name=NAME, exist_ok=True,
        plots=True, verbose=True,
    )

    res = m.val()
    person = person_metrics(res)
    out = {
        "model": "yolo11s finetuned (ds_top orthotop)",
        "data": DATA,
        "person": person,                       # ← make-or-break 핵심 지표
        "all_classes": {
            "map50": float(res.box.map50),
            "map50_95": float(res.box.map),
            "precision": float(res.box.mp),
            "recall": float(res.box.mr),
        },
        "best_weights": str(m.trainer.best),
    }

    # 후속 파이프라인용 ONNX (opset 12 — sim.html ONNX 경로 · detect_server 호환)
    try:
        onnx = m.export(format="onnx", opset=12, imgsz=640, simplify=True,
                        dynamic=False, nms=False)
        out["onnx"] = str(onnx)
    except Exception as e:
        out["onnx_error"] = str(e)

    Path(PROJECT).mkdir(parents=True, exist_ok=True)
    (Path(PROJECT) / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("make-or-break 판정 (기준선: 실사 overhead 모델 recall 0.17 / prec 0.30)")
    if person:
        print(f"  person recall    : {person['recall']:.3f}")
        print(f"  person precision : {person['precision']:.3f}")
        print(f"  person mAP50     : {person['map50']:.3f}")
    else:
        print("  ⚠ val에 person 예측/라벨이 없음 — 데이터 확인 필요")
    print("=" * 60)
    print("SUMMARY:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
