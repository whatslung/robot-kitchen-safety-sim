"""나디르(top-down) person-only 검출용 데이터셋 구성 — sim+real 혼합, 누수 방지.

목적(헤지): 사선 CCTV가 주력이지만 안 될 경우를 대비해, 나디르 도메인에서 **사람만**
검출하는 YOLO를 사선 경로와 같은 eval로 확보한다. 안전 로직(trajectory/risk.py)은
사람 위치만 비전으로 필요하고 로봇은 컨트롤러가, 설비는 고정 맵이 위치를 알므로
검출기는 person 하나로 충분하다 → 클래스 공간을 nc:1(person)로 통일.

두 소스:
  - sim  : dataset/sim-person-island (orthotop_*.png, 6클래스). person=class 0만 남긴
           라벨 사본을 dataset/nadir-person/sim 에 만든다(원본 6클래스 셋 불변).
  - real : dataset/overhead-person-v3 (실사 오버헤드, 이미 nc:1 person). 원본 그대로 참조.

누수 방지(P0-1 교훈):
  - real: Roboflow 기본 train/valid/test는 프레임을 랜덤 분할해 같은 클립(예: cam_1_2min…)이
          여러 split에 걸친다 → 전량 pool 후 **클립 prefix 단위**로 재분할.
  - sim : 파일명에 세션/시드 마커가 없는 연속 프레임 → 정렬 후 **연속 블록**으로 분할해
          split 경계 밖 인접 프레임 누수를 없앤다.

평가는 도메인별로 분리한다(sim_test.yaml, real_test.yaml). 사선 경로와 같은 방식.

    uv run python train/prep_nadir.py
"""
import re
import shutil
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 데이터셋은 gitignore라 워크트리엔 없을 수 있다 → dataset/ 를 가진 상위를 찾아 그곳을 데이터 루트로.
DATA_ROOT = next(
    (c for c in [ROOT, *ROOT.parents] if (c / "dataset" / "overhead-person-v3").is_dir()),
    ROOT,
)

# sim 소스: 최신 재생성본 우선. v3(다양성: 옷·바닥 색 + 인원↑) → v2(오라벨 제거) → 원본.
_V3 = DATA_ROOT / "dataset" / "sim-person-island-v3"
_V2 = DATA_ROOT / "dataset" / "sim-person-island-v2"
SIM_SRC = (_V3 if (_V3 / "images").is_dir()
           else _V2 if (_V2 / "images").is_dir()
           else DATA_ROOT / "dataset" / "sim-person-island")
REAL = DATA_ROOT / "dataset" / "overhead-person-v3"
OUT = DATA_ROOT / "dataset" / "nadir-person"       # 산출 루트
SIM_OUT = OUT / "sim"                               # person-only 라벨 사본 + 이미지

CLASSES = ["person"]                               # nc:1
EXT = {".jpg", ".jpeg", ".png"}
SEED = 42
# split 비율 (train/val/test). 나머지가 test.
R_TRAIN, R_VAL = 0.70, 0.15


def imgs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)


def write_list(path, items):
    path.write_text("\n".join(p.as_posix() for p in items) + "\n", encoding="utf-8")


def write_yaml(path, train_txt, val_txt, test_txt):
    path.write_text(
        f"# 자동 생성 — train/prep_nadir.py (나디르 person-only)\n"
        f"train: {train_txt.as_posix()}\n"
        f"val: {val_txt.as_posix()}\n"
        f"test: {test_txt.as_posix()}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )


def build_sim_person_only():
    """sim 이미지+라벨을 person(class 0)만 남겨 SIM_OUT/{images,labels}로 복사."""
    src_img, src_lbl = SIM_SRC / "images", SIM_SRC / "labels"
    out_img, out_lbl = SIM_OUT / "images", SIM_OUT / "labels"
    for d in (out_img, out_lbl):
        d.mkdir(parents=True, exist_ok=True)

    kept_imgs, n_person_box, n_dropped_box = [], 0, 0
    for im in imgs(src_img):
        lbl = src_lbl / (im.stem + ".txt")
        person_lines = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "0":                 # person
                    person_lines.append(line)
                else:
                    n_dropped_box += 1
        # 이미지 복사(원본 불변) + 필터 라벨 기록(빈 라벨=배경, 정상)
        shutil.copy2(im, out_img / im.name)
        (out_lbl / (im.stem + ".txt")).write_text(
            ("\n".join(person_lines) + "\n") if person_lines else "", encoding="utf-8")
        n_person_box += len(person_lines)
        kept_imgs.append(out_img / im.name)
    print(f"[sim] {len(kept_imgs)}장 · person 박스 {n_person_box} 유지 · "
          f"비-person 박스 {n_dropped_box} 제거")
    return kept_imgs


def split_contiguous(items):
    """연속 블록 분할(인접 프레임 누수 방지) — 정렬된 순서 그대로 앞/중/뒤."""
    items = sorted(items)
    n = len(items)
    a, b = int(n * R_TRAIN), int(n * (R_TRAIN + R_VAL))
    return items[:a], items[a:b], items[b:]


def clip_id(name):
    """실사 파일명에서 클립 식별자 추출: 프레임번호(6자리+) 이후는 버린다."""
    return re.sub(r"_?\d{6,}.*$", "", name)


def split_by_clip(items):
    """클립 prefix 단위 분할 — 같은 클립이 여러 split에 걸치지 않게."""
    groups = {}
    for p in items:
        groups.setdefault(clip_id(p.name), []).append(p)
    clips = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(clips)
    n = len(clips)
    a, b = int(n * R_TRAIN), int(n * (R_TRAIN + R_VAL))
    sel = lambda cs: [p for c in cs for p in sorted(groups[c])]
    return sel(clips[:a]), sel(clips[a:b]), sel(clips[b:]), clips


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- sim: person-only 사본 + 연속 블록 분할 ---
    sim_all = build_sim_person_only()
    sim_tr, sim_va, sim_te = split_contiguous(sim_all)

    # --- real: 전량 pool 후 클립 단위 분할 ---
    real_pool = []
    for sub in ("train", "valid", "test"):
        d = REAL / sub / "images"
        if d.is_dir():
            real_pool += imgs(d)
    real_tr, real_va, real_te, clips = split_by_clip(real_pool)

    # --- 리스트 파일 ---
    write_list(OUT / "sim_train.txt", sim_tr)
    write_list(OUT / "sim_val.txt", sim_va)
    write_list(OUT / "sim_test.txt", sim_te)
    write_list(OUT / "real_train.txt", real_tr)
    write_list(OUT / "real_val.txt", real_va)
    write_list(OUT / "real_test.txt", real_te)
    write_list(OUT / "mix_train.txt", sim_tr + real_tr)
    write_list(OUT / "mix_val.txt", sim_va + real_va)

    # --- yaml: 학습용(혼합) + 도메인별 평가용 ---
    write_yaml(OUT / "nadir_mix.yaml", OUT / "mix_train.txt",
               OUT / "mix_val.txt", OUT / "real_test.txt")
    write_yaml(OUT / "sim_test.yaml", OUT / "mix_train.txt",
               OUT / "sim_val.txt", OUT / "sim_test.txt")
    write_yaml(OUT / "real_test.yaml", OUT / "mix_train.txt",
               OUT / "real_val.txt", OUT / "real_test.txt")

    print(f"[real] pool {len(real_pool)} · 클립 {len(clips)}개 → "
          f"train {len(real_tr)} · val {len(real_va)} · test {len(real_te)}")
    print(f"[sim ] train {len(sim_tr)} · val {len(sim_va)} · test {len(sim_te)}")
    print(f"[mix ] train {len(sim_tr) + len(real_tr)} · val {len(sim_va) + len(real_va)}")
    print(f"yaml → {OUT / 'nadir_mix.yaml'} (학습)")
    print(f"yaml → {OUT / 'sim_test.yaml'} · {OUT / 'real_test.yaml'} (도메인별 평가)")


if __name__ == "__main__":
    main()
