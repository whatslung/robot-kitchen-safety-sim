"""지식 증류(pseudo-label) — RF-DETR(sim-only) 교사로 실사 이미지에 YOLO 라벨 생성.

목적: 실사 GT 라벨을 한 장도 안 쓰고, 파운데이션 교사(RF-DETR, 실사 전이 mAP50 0.411)가
     실사 이미지에 단 pseudo-label로 경량 YOLO 학생을 학습 → 실사 test에서 얼마나 회복되나.
     (교사 = sim만 학습 → 실사엔 라벨이 원래 없다는 시나리오)

출력: dataset/distill/{train,val}/{images,labels}  (YOLO 포맷, person 단일) + data.yaml
      test는 실사 GT(overhead-person-v3/test)로 평가.

    C:/Users/chanwoo/rfdetr-env/Scripts/python.exe train/distill_pseudo.py
"""
import shutil
from pathlib import Path

from PIL import Image
from rfdetr import RFDETRNano

ROOT = Path(__file__).resolve().parent.parent
TEACHER = str(ROOT / "training" / "rfdetr_nano" / "checkpoint_best_ema.pth")  # sim-only 교사
OUT = ROOT / "dataset" / "distill"
CONF = 0.5   # pseudo-label 신뢰도 하한
SPLITS = {"train": ROOT / "dataset/3way/real_train.txt",   # 실사 이미지(라벨은 무시)
          "val":   ROOT / "dataset/3way/real_val.txt"}


def imgs(txt):
    return [Path(l.strip()) for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    model = RFDETRNano(pretrain_weights=TEACHER)
    lists = {}
    for split, txt in SPLITS.items():
        idir = OUT / split / "images"; ldir = OUT / split / "labels"
        idir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)
        paths, nbox = [], 0
        for im in imgs(txt):
            if not im.exists():
                continue
            W, H = Image.open(im).size
            det = model.predict(str(im), threshold=CONF)
            lines = []
            for (x1, y1, x2, y2) in det.xyxy:
                cx, cy, w, h = ((x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H)
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            shutil.copy2(im, idir / im.name)
            (ldir / (im.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            paths.append((idir / im.name).as_posix()); nbox += len(lines)
        (OUT / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        lists[split] = (len(paths), nbox)
        print(f"{split}: images {len(paths)}  pseudo-boxes {nbox}")

    test_dir = (ROOT / "dataset/overhead-person-v3/test/images").as_posix()
    (OUT / "data.yaml").write_text(
        f"train: {(OUT/'train.txt').as_posix()}\n"
        f"val: {(OUT/'val.txt').as_posix()}\n"
        f"test: {test_dir}\n"
        f"nc: 1\nnames: ['person']\n", encoding="utf-8")
    print(f"\n✅ 증류 데이터셋: {OUT}  (교사 CONF={CONF})")


if __name__ == "__main__":
    main()
