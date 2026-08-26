"""zone(4구역) sim + real 혼합 person-only split — 빠른 테스트용.
   sim-nadir-zone(6클래스 라벨) → person(class0)만 필터 → 연속블록 split → real 혼합."""
import shutil
from pathlib import Path

ROOT = Path(r"C:/Users/chanwoo/workspace/robot-kitchen-safety-sim")
ZSRC = ROOT / "dataset" / "sim-nadir-zone"
REAL = ROOT / "dataset" / "overhead-person-v3"
OUT = ROOT / "dataset" / "nadir-zone"
ZOUT = OUT / "sim"
EXT = {".png", ".jpg", ".jpeg"}
R_TRAIN, R_VAL = 0.70, 0.15


def imgs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)


def wlist(path, items):
    path.write_text("\n".join(p.as_posix() for p in items) + "\n", encoding="utf-8")


def wyaml(path, tr, va, te):
    path.write_text(f"train: {tr.as_posix()}\nval: {va.as_posix()}\ntest: {te.as_posix()}\n"
                    f"nc: 1\nnames: ['person']\n", encoding="utf-8")


def main():
    (ZOUT / "images").mkdir(parents=True, exist_ok=True)
    (ZOUT / "labels").mkdir(parents=True, exist_ok=True)
    # person-only 필터 + 이미지 복사
    kept, nbox = [], 0
    for im in imgs(ZSRC / "images"):
        lb = ZSRC / "labels" / (im.stem + ".txt")
        plines = [l for l in (lb.read_text().splitlines() if lb.exists() else [])
                  if l.split() and l.split()[0] == "0"]
        shutil.copy2(im, ZOUT / "images" / im.name)
        (ZOUT / "labels" / (im.stem + ".txt")).write_text(
            ("\n".join(plines) + "\n") if plines else "", encoding="utf-8")
        nbox += len(plines); kept.append(ZOUT / "images" / im.name)
    kept.sort()
    n = len(kept); a, b = int(n * R_TRAIN), int(n * (R_TRAIN + R_VAL))
    ztr, zva, zte = kept[:a], kept[a:b], kept[b:]
    # real (전량 pool, 클립 무시하고 기존 3way real_train/val/test 재사용)
    rtr = [Path(l) for l in (ROOT / "dataset/nadir-person/real_train.txt").read_text().split() if l.strip()]
    rva = [Path(l) for l in (ROOT / "dataset/nadir-person/real_val.txt").read_text().split() if l.strip()]
    rte = [Path(l) for l in (ROOT / "dataset/nadir-person/real_test.txt").read_text().split() if l.strip()]
    wlist(OUT / "zsim_train.txt", ztr); wlist(OUT / "zsim_val.txt", zva); wlist(OUT / "zsim_test.txt", zte)
    wlist(OUT / "mix_train.txt", rtr + ztr); wlist(OUT / "mix_val.txt", rva + zva)
    wyaml(OUT / "zone_mix.yaml", OUT / "mix_train.txt", OUT / "mix_val.txt", OUT / "real_test.yaml")
    wyaml(OUT / "zone_test.yaml", OUT / "mix_train.txt", OUT / "zsim_val.txt", OUT / "zsim_test.txt")
    wyaml(OUT / "real_test.yaml", OUT / "mix_train.txt", OUT / "real_val.txt", ROOT / "dataset/nadir-person/real_test.txt")
    # real_val.txt 도 필요
    wlist(OUT / "real_val.txt", rva)
    print(f"[zone] sim {n}장 (person박스 {nbox}) → train {len(ztr)}/val {len(zva)}/test {len(zte)}")
    print(f"[mix ] train {len(rtr)+len(ztr)} (real {len(rtr)} + zsim {len(ztr)})")
    print(f"yaml → zone_mix.yaml(학습) · zone_test.yaml · real_test.yaml")


if __name__ == "__main__":
    main()
