"""파인튜닝한 sim 모델을 '실사' 데이터셋에서 평가한다 — sim-to-real 갭 정량화.

우리 모델은 직교 나디르 합성으로 학습됐고, 실사(예: Roboflow overhead-person v3,
실사 top-down)는 원근·실제 질감·평상복이라 도메인 갭이 있다. 그 갭을 person 지표로 잰다.
6클래스 모델이지만 person=0으로 인덱스가 맞으므로 classes=[0]로 person만 평가한다.

    uv run python train/eval_real.py <real_data.yaml> [--split test] [--weights <best.pt>]

real_data.yaml 은 Roboflow yolov11 export가 만든 것(train/valid/test + nc:1 person).
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_W = ROOT / "training" / "yolo11s_orthotop" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser(description="sim 모델의 실사 person 검출 평가 (sim-to-real)")
    ap.add_argument("data", help="실사 데이터셋 data.yaml 경로")
    ap.add_argument("--split", default="test", help="평가 split (기본 test)")
    ap.add_argument("--weights", default=str(DEFAULT_W), help="가중치 (기본 best.pt)")
    args = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else "cpu"
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only", flush=True)
    print("weights:", args.weights, flush=True)

    m = YOLO(args.weights)
    r = m.val(data=args.data, split=args.split, classes=[0], device=dev, workers=0,
              project=str(ROOT / "training"), name="real_eval", exist_ok=True,
              plots=True, verbose=True)

    out = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "person_precision": float(r.box.mp),
        "person_recall": float(r.box.mr),
        "person_map50": float(r.box.map50),
        "person_map50_95": float(r.box.map),
    }
    print("\n" + "=" * 60)
    print("sim-to-real (실사 person 검출)")
    print(f"  recall    : {out['person_recall']:.3f}")
    print(f"  precision : {out['person_precision']:.3f}")
    print(f"  mAP50     : {out['person_map50']:.3f}")
    print("=" * 60)
    (ROOT / "training" / "real_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("REAL:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
