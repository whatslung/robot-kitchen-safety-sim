"""3-way 비교(sim-only / real-only / real+sim)용 데이터 구성.

목적: "합성 데이터가 실사 검출에 도움이 되는가"를 **제한된 실사(limited-real) 체제**에서 측정.
실사 전량(4,120장)이면 real-only가 이미 천장(참고 recall 0.98)이라 sim 기여가 안 보인다.
실사를 500장으로 제한하면 = 실사 주방 라벨이 부족한 현실을 모사 → sim의 보탬이 드러난다.

클래스 공간은 6클래스로 통일(실사 라벨 class 0 = person = 우리 sim의 person). 실사엔
person만 존재하고 나머지 5클래스는 0개 — 학습/평가에 무해. 덕분에 sim-only(6클래스) 모델을
그대로 재사용해 같은 실사 test에서 비교할 수 있다.

평가는 세 조건 모두 **동일한 실사 test 137장**에서 person(class 0)만.

    uv run python train/prep_3way.py
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "dataset" / "overhead-person-v3"
SIM_IMG = ROOT / "sim-person" / "images"
OUT = ROOT / "dataset" / "3way"
CLASSES = ["person", "fire", "smoke", "robot", "kettle", "equipment"]
N_TRAIN, N_VAL, SEED = 500, 150, 42
EXT = {".jpg", ".jpeg", ".png"}


def imgs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)


def write_list(path, items):
    path.write_text("\n".join(p.as_posix() for p in items) + "\n", encoding="utf-8")


def write_yaml(path, train_txt, val_txt, test_txt):
    path.write_text(
        f"# 자동 생성 — train/prep_3way.py (limited-real 3-way 비교)\n"
        f"train: {train_txt.as_posix()}\n"
        f"val: {val_txt.as_posix()}\n"
        f"test: {test_txt.as_posix()}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = imgs(REAL / "train" / "images")
    rng = random.Random(SEED)
    rng.shuffle(pool)
    real_train, real_val = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_VAL]
    real_test = imgs(REAL / "test" / "images")
    sim = imgs(SIM_IMG)

    write_list(OUT / "real_train.txt", real_train)
    write_list(OUT / "real_val.txt", real_val)
    write_list(OUT / "real_test.txt", real_test)
    write_list(OUT / "real_plus_sim_train.txt", real_train + sim)

    write_yaml(OUT / "real_only.yaml", OUT / "real_train.txt",
               OUT / "real_val.txt", OUT / "real_test.txt")
    write_yaml(OUT / "real_sim.yaml", OUT / "real_plus_sim_train.txt",
               OUT / "real_val.txt", OUT / "real_test.txt")

    print(f"real_train {len(real_train)} · real_val {len(real_val)} · "
          f"real_test {len(real_test)} · sim {len(sim)}")
    print(f"real+sim train = {len(real_train) + len(sim)}")
    print(f"yaml → {OUT / 'real_only.yaml'}")
    print(f"yaml → {OUT / 'real_sim.yaml'}")


if __name__ == "__main__":
    main()
