"""파인튜닝 전 stock YOLO11s(COCO)의 person 성능을 우리 val셋에서 잰다.

make-or-break 비교의 '이전(before)' 기준선. 핸드오프 실측(§4-1)은 stock이
시뮬 top-down에서 검출 0이었다 — 이 스크립트로 우리 직교 데이터에서 재확인한다.
COCO person=0, 우리 데이터 person=0로 인덱스가 일치하므로 classes=[0]로 person만 평가.

    uv run python train/eval_stock.py
"""
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "sim-person" / "data.yaml")


def main():
    dev = 0 if torch.cuda.is_available() else "cpu"
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only", flush=True)

    m = YOLO("yolo11s.pt")   # stock COCO 가중치 (자동 다운로드)
    r = m.val(data=DATA, classes=[0], device=dev, workers=0,
              project=str(ROOT / "training"), name="before_stock", exist_ok=True,
              plots=False, verbose=False)

    out = {
        "model": "yolo11s.pt (stock COCO, 파인튜닝 전)",
        "split": "val",
        "person_precision": float(r.box.mp),
        "person_recall": float(r.box.mr),
        "person_map50": float(r.box.map50),
        "person_map50_95": float(r.box.map),
    }
    print("BEFORE(stock):", json.dumps(out, ensure_ascii=False), flush=True)
    (ROOT / "training" / "before_stock.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "training" / "before_stock.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
