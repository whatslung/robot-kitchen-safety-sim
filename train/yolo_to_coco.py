"""YOLO 라벨 → COCO(person-only) 변환 — RF-DETR 학습/평가용 데이터셋 구성.

RF-DETR은 COCO 포맷(각 split 폴더에 이미지 + _annotations.coco.json)을 요구한다.
각 split은 image-list(.txt) 또는 images 디렉터리로 지정한다. person(class 0)만 담는다.

기본(인자 없음) = sim-only:
    uv run python train/yolo_to_coco.py
3-way용 예:
    uv run python train/yolo_to_coco.py --out dataset/rfdetr_real \
        --train dataset/3way/real_train.txt --valid dataset/3way/real_val.txt --test dataset/3way/real_test.txt
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CAT = [{"id": 1, "name": "person", "supercategory": "none"}]
EXT = {".jpg", ".jpeg", ".png"}


def img_list(spec):
    p = Path(spec)
    if p.suffix == ".txt":
        return [Path(l.strip()) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(q for q in p.iterdir() if q.suffix.lower() in EXT)


def label_path(img):
    return Path(str(img.parent).replace("/images", "/labels").replace("\\images", "\\labels")) / (img.stem + ".txt")


def build(outdir, split, images):
    d = outdir / split
    d.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": CAT}
    ann_id = 1
    for img_id, img in enumerate(images, 1):
        if not img.exists():
            continue
        W, H = Image.open(img).size
        shutil.copy2(img, d / img.name)
        coco["images"].append({"id": img_id, "file_name": img.name, "width": W, "height": H})
        lbl = label_path(img)
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0] != "0":
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x, y, w, h = (cx - bw/2)*W, (cy - bh/2)*H, bw*W, bh*H
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [round(x,2), round(y,2), round(w,2), round(h,2)], "area": round(w*h,2), "iscrowd": 0})
            ann_id += 1
    (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"{split:6} images {len(coco['images'])}  person-boxes {len(coco['annotations'])}  → {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/rfdetr")
    ap.add_argument("--train", default="sim-person/train.txt")
    ap.add_argument("--valid", default="sim-person/val.txt")
    ap.add_argument("--test",  default="dataset/overhead-person-v3/test/images")
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    for split, spec in [("train", a.train), ("valid", a.valid), ("test", a.test)]:
        s = spec if Path(spec).is_absolute() else str(ROOT / spec)
        build(out, split, img_list(s))
    print(f"\n✅ RF-DETR 데이터셋: {out}")


if __name__ == "__main__":
    main()
