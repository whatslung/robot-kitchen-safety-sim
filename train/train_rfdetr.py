"""RF-DETR(Roboflow, Apache-2.0)를 sim-person(COCO)으로 파인튜닝 → 실사 test 평가.

YOLO와의 sim-to-real 비교용: sim만 학습한 RF-DETR이 실사 test에서 YOLO(sim-only, mAP50 0.048)
대비 전이가 나은지 본다. 별도 격리 env로 실행(우리 .venv 안 건드림):

    C:/Users/chanwoo/rfdetr-env/Scripts/python.exe train/train_rfdetr.py

데이터: dataset/rfdetr/{train,valid,test}/ (train/yolo_to_coco.py 산출, person 단일).
"""
import argparse
import json
from pathlib import Path

from rfdetr import RFDETRNano

ROOT = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default="dataset/rfdetr")   # COCO 데이터셋 폴더
_ap.add_argument("--name", default="rfdetr_nano")      # training/ 하위 출력명
_args, _ = _ap.parse_known_args()
DATA = str((ROOT / _args.data) if not Path(_args.data).is_absolute() else Path(_args.data))
OUT = str(ROOT / "training" / _args.name)


def main():
    model = RFDETRNano()
    model.train(
        dataset_dir=DATA,
        epochs=50,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=OUT,
        num_workers=0,          # Windows 데이터로더 안정
        early_stopping=True,
    )
    # 실사 test 평가 (COCO 지표)
    metrics = model.evaluate(split="test", dataset_dir=DATA)
    print("RFDETR_TEST_METRICS:", json.dumps({k: float(v) for k, v in metrics.items()}, ensure_ascii=False))
    Path(OUT).mkdir(parents=True, exist_ok=True)
    (Path(OUT) / "test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
