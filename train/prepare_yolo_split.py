#!/usr/bin/env python3
"""ds_top(직교 나디르) 데이터셋을 YOLO 학습용으로 분할한다.

시뮬 generateDataset이 만든 폴더(images/ + labels/)를 받아
train/val로 나눠 train.txt·val.txt·data.yaml을 쓴다. 파일을 옮기지 않고
이미지 경로 목록만 만들므로(Ultralytics가 .txt 목록을 그대로 읽음) 안전·빠르다.

카메라 1대라 샘플=이미지가 1:1 → 멀티뷰 누수 걱정이 없어 무작위 분할로 충분하다.
(여러 카메라를 섞어 뽑았다면 같은 장면의 다른 뷰가 train/val에 갈리지 않게
 base 인덱스 단위로 나눠야 하지만, orthotop 단독이면 해당 없음.)

사용:
    python train/prepare_yolo_split.py "D:/path/to/dataset" --val-ratio 0.2

그러면 <dataset>/data.yaml 이 생기고, 학습은:
    yolo detect train model=yolo11n.pt data="D:/path/to/dataset/data.yaml" \
        epochs=80 imgsz=640 batch=16
"""
import argparse
import random
from pathlib import Path

# 데이터셋 클래스 순서 — sim.html GT_CLASSES와 반드시 일치시킨다.
# (MODEL_HANDOFF.md 2026-08-13: fire·smoke를 앞에 둬 계획서 person·fire·smoke=0·1·2 보장)
CLASSES = ["person", "fire", "smoke", "robot", "kettle", "equipment"]
IMG_EXT = {".png", ".jpg", ".jpeg"}


def main():
    ap = argparse.ArgumentParser(description="ds_top YOLO train/val 분할 + data.yaml 생성")
    ap.add_argument("dataset", help="images/ 와 labels/ 를 담은 데이터셋 폴더")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="검증셋 비율 (기본 0.2)")
    ap.add_argument("--seed", type=int, default=0, help="셔플 시드 (재현용)")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    img_dir, lbl_dir = root / "images", root / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"✗ images/ 폴더가 없습니다: {img_dir}")
    if not lbl_dir.is_dir():
        raise SystemExit(f"✗ labels/ 폴더가 없습니다: {lbl_dir}")

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"✗ images/ 안에 이미지가 없습니다: {img_dir}")

    # 라벨 짝 확인 — 라벨 파일이 없으면 그 이미지는 학습에서 배경 취급되지만,
    # 여기서는 명시적으로 짚어준다(생성 누락 조기 발견).
    paired, missing, empty = [], 0, 0
    for im in imgs:
        lbl = lbl_dir / (im.stem + ".txt")
        if not lbl.exists():
            missing += 1
            continue
        if lbl.stat().st_size == 0:
            empty += 1
        paired.append(im)

    rng = random.Random(args.seed)
    rng.shuffle(paired)
    n_val = max(1, round(len(paired) * args.val_ratio))
    val, train = paired[:n_val], paired[n_val:]

    (root / "train.txt").write_text(
        "\n".join(str(p) for p in train) + "\n", encoding="utf-8")
    (root / "val.txt").write_text(
        "\n".join(str(p) for p in val) + "\n", encoding="utf-8")

    yaml = (
        f"# 자동 생성 — train/prepare_yolo_split.py\n"
        f"path: {root.as_posix()}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (root / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"✅ 분할 완료 — train {len(train)} · val {len(val)} (총 {len(paired)})")
    if empty:
        print(f"   ℹ 빈 라벨(사람 0명 배경) {empty}장 — 정상, 학습에 그대로 쓰임")
    if missing:
        print(f"   ⚠ 라벨 없는 이미지 {missing}장 — 제외됨 (생성 누락 여부 확인)")
    print(f"   → data.yaml: {root / 'data.yaml'}")
    print(f"\n다음 학습 명령:")
    print(f'   yolo detect train model=yolo11n.pt data="{(root / "data.yaml").as_posix()}" '
          f"epochs=80 imgsz=640 batch=16")


if __name__ == "__main__":
    main()
