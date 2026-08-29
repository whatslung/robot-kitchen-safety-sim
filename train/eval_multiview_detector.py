"""Measure an existing YOLO person's precision/recall on the 4+1 sim views.

The live detector uses a confidence threshold before ByteTrack, so this evaluator reports
fixed-threshold metrics instead of hiding the deployment trade-off behind mAP alone.

    uv run --group serve python train/eval_multiview_detector.py \
      --dataset dataset/mv5-baseline-20260826
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset" / "mv5-baseline-20260826"
DEFAULT_MODEL = ROOT / "training" / "island_yolo11s" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "training" / "mv5_detector_baseline.json"
DEFAULT_HUB_REPO = "chanubc/robot-kitchen-nadir-yolo11s"


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def score_image(gt_boxes, pred_boxes, confidence=0.25, iou_threshold=0.5):
    selected = sorted(
        (pred for pred in pred_boxes if pred[4] >= confidence),
        key=lambda pred: pred[4],
        reverse=True,
    )
    unmatched = set(range(len(gt_boxes)))
    tp = 0
    for pred in selected:
        candidates = [(box_iou(pred, gt_boxes[index]), index) for index in unmatched]
        best_iou, best_index = max(candidates, default=(0.0, None))
        if best_index is not None and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            tp += 1
    return {"tp": tp, "fp": len(selected) - tp, "fn": len(unmatched)}


def _empty_counts():
    return {"images": 0, "ground_truth": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}


def _finalize(counts):
    precision_den = counts["tp"] + counts["fp"]
    recall_den = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_den if precision_den else 0.0
    recall = counts["tp"] / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def score_dataset(samples, confidence=0.25, iou_threshold=0.5):
    overall = _empty_counts()
    cameras = {}
    for sample in samples:
        camera = sample["camera"]
        camera_counts = cameras.setdefault(camera, _empty_counts())
        result = score_image(sample["gt"], sample["pred"], confidence, iou_threshold)
        predictions = sum(pred[4] >= confidence for pred in sample["pred"])
        for counts in (overall, camera_counts):
            counts["images"] += 1
            counts["ground_truth"] += len(sample["gt"])
            counts["predictions"] += predictions
            for key in ("tp", "fp", "fn"):
                counts[key] += result[key]
    return {
        "overall": _finalize(overall),
        "cameras": {camera: _finalize(counts) for camera, counts in sorted(cameras.items())},
    }


def best_f1(samples, thresholds=None, iou_threshold=0.5):
    thresholds = thresholds or [index / 100 for index in range(1, 100)]
    candidates = []
    for confidence in thresholds:
        metrics = score_dataset(samples, confidence, iou_threshold)["overall"]
        candidates.append({"confidence": confidence, "metrics": metrics})
    return max(
        candidates,
        key=lambda row: (
            row["metrics"]["f1"],
            row["metrics"]["recall"],
            row["metrics"]["precision"],
        ),
    )


def _read_person_labels(path):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5 or int(fields[0]) != 0:
            continue
        cx, cy, width, height = map(float, fields[1:5])
        boxes.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
    return boxes


def _resolve_model(requested):
    # 명시한 Ultralytics 이름(yolo11s.pt)이나 로컬 경로는 그대로 존중한다.
    # 기본 배포 가중치가 없을 때만 프로젝트의 Hugging Face 체크포인트로 폴백한다.
    if requested:
        return Path(requested)
    candidate = Path(os.environ.get("DETECT_MODEL", DEFAULT_MODEL))
    if candidate.exists():
        return candidate
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=os.environ.get("DETECT_MODEL_REPO", DEFAULT_HUB_REPO),
            filename=os.environ.get("DETECT_MODEL_FILE", "best.pt"),
        )
    )


def _camera_from(path):
    return path.stem.rsplit("_", 1)[0]


def _collect_samples(model, images, args, person_classes):
    results = model.predict(
        source=[str(path) for path in images],
        classes=person_classes,
        conf=args.min_confidence,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )
    samples = []
    for result in results:
        image_path = Path(result.path)
        height, width = result.orig_shape
        predictions = []
        for xyxy, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            predictions.append(
                [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height, confidence]
            )
        samples.append(
            {
                "name": image_path.name,
                "camera": _camera_from(image_path),
                "gt": _read_person_labels(args.dataset / "labels" / f"{image_path.stem}.txt"),
                "pred": predictions,
            }
        )
    return samples


def _worst_images(samples, confidence, iou_threshold, limit=20):
    rows = []
    for sample in samples:
        scored = score_image(sample["gt"], sample["pred"], confidence, iou_threshold)
        if scored["fn"] or scored["fp"]:
            rows.append({"name": sample["name"], "camera": sample["camera"], **scored})
    return sorted(rows, key=lambda row: (row["fn"], row["fp"]), reverse=True)[:limit]


def _inference_settings(args):
    return {
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "min_confidence": args.min_confidence,
        "nms_iou": args.nms_iou,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", help="Local .pt path; defaults to deploy weights/Hugging Face")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    images = sorted((args.dataset / "images").glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No PNG images found under {args.dataset / 'images'}")

    from ultralytics import YOLO

    model_path = _resolve_model(args.model)
    model = YOLO(str(model_path))
    person_classes = [index for index, name in model.names.items() if str(name).lower() == "person"]
    if not person_classes:
        raise SystemExit(f"Model has no person class: {model.names}")

    samples = _collect_samples(model, images, args, person_classes)
    fixed = {}
    for threshold in (0.05, 0.10, 0.25, 0.35):
        fixed[f"{threshold:.2f}"] = score_dataset(samples, threshold, args.match_iou)
    best = best_f1(samples, iou_threshold=args.match_iou)
    report = {
        "model": str(model_path),
        "dataset": str(args.dataset.resolve()),
        "images": len(samples),
        **_inference_settings(args),
        "person_class_ids": person_classes,
        "match_iou": args.match_iou,
        "fixed_thresholds": fixed,
        "best_f1": best,
        "worst_at_0.25": _worst_images(samples, 0.25, args.match_iou),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
