"""데이터셋 프레임 중복도 측정 (2026-08-24)

왜 필요한가
-----------
화재 모델 피드백에 "프레임 중복이 너무 많으면 문제가 될 수도"가 있었는데,
지금까지 **아무도 그 숫자를 몰랐다.** 모르는 채로는 두 가지를 구분할 수 없다.

  · 합성 데이터가 사실상 같은 그림을 반복해서 학습이 안 되는 것인가
  · 데이터는 다양한데 하이퍼파라미터/split이 문제인가

이 스크립트는 그 숫자를 낸다. 결과가 낮으면 시뮬 데이터는 용의선상에서 빠지고,
높으면 어느 카메라가 범인인지까지 같이 나온다.

무엇을 재는가
-------------
dHash(64비트) — 이미지를 9x8 그레이로 줄여 가로 이웃 픽셀의 밝기 대소를 비트로 만든다.
밝기·대비·노이즈가 조금 달라도 **구도가 같으면 같은 해시**가 나온다. 시뮬 데이터의
중복은 "완전히 같은 파일"이 아니라 "사람만 조금 움직인 같은 그림"이라 픽셀 비교로는
안 잡히고 이 방식이 맞다.

해밍거리 <= THRESH 면 사실상 같은 프레임으로 본다(기본 5 = 64비트 중 5비트 차이).

카메라별로 따로 집계하는 이유 — 이 데이터셋은 한 샘플을 카메라 20대가 동시에 찍는다.
중복이 있다면 샘플 사이가 아니라 **한 카메라 안**에서 생긴다(고정 나디르 뷰처럼
화면 대부분이 매번 같은 카메라). 전체 평균만 내면 그게 묻힌다.

쓰는 법
-------
    uv run python tools/dataset_dup.py <데이터셋폴더>
    uv run python tools/dataset_dup.py <데이터셋폴더> --thresh 8 --show 5

<데이터셋폴더>는 images/ 를 가진 곳이다(시뮬 '데이터셋 생성'이 만든 구조).
images/ 가 없으면 폴더 아래 이미지를 전부 훑는다.

파일명 규약: `{카메라id}_{샘플번호}.png` — 시뮬이 이렇게 저장한다.
"""

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow가 필요하다:  uv pip install pillow")

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def dhash(path: Path) -> int:
    """9x8 그레이 축소 후 가로 이웃 대소 비교 → 64비트 정수."""
    img = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            bits = (bits << 1) | (1 if px[row + x] > px[row + x + 1] else 0)
    return bits


def collect(root: Path) -> list[Path]:
    base = root / "images" if (root / "images").is_dir() else root
    files = sorted(p for p in base.rglob("*") if p.suffix.lower() in EXTS)
    # GT 부산물은 학습 이미지가 아니다 — 섞이면 중복률이 거짓으로 올라간다
    return [p for p in files if not any(t in p.name for t in ("_mask", "_inst", "_depth"))]


def camera_of(path: Path) -> str:
    """`cvN_0007.png` → `cvN`. 규약에 안 맞으면 통째로 한 그룹."""
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else "(전체)"


def main() -> int:
    ap = argparse.ArgumentParser(description="데이터셋 프레임 중복도 측정")
    ap.add_argument("dataset", type=Path, help="images/ 를 가진 데이터셋 폴더")
    ap.add_argument("--thresh", type=int, default=5,
                    help="해밍거리 임계 (기본 5). 올릴수록 '비슷하면 중복'으로 본다")
    ap.add_argument("--show", type=int, default=3, help="카메라별로 예시 쌍 몇 개를 보일지")
    args = ap.parse_args()

    files = collect(args.dataset)
    if not files:
        return print(f"이미지가 없다: {args.dataset}") or 1
    print(f"이미지 {len(files)}장 — 해시 계산 중…", flush=True)

    by_cam: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for i, p in enumerate(files, 1):
        by_cam[camera_of(p)].append((p, dhash(p)))
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    print(f"\n{'카메라':<14}{'장수':>6}{'중복쌍':>8}{'중복프레임':>11}{'비율':>8}")
    print("-" * 47)

    total_dup, total_n = 0, 0
    worst: list[tuple[float, str, list[tuple[str, str, int]]]] = []

    for cam, items in sorted(by_cam.items()):
        pairs = [(a, b, (ha ^ hb).bit_count())
                 for (a, ha), (b, hb) in combinations(items, 2)]
        near = [(a, b, d) for a, b, d in pairs if d <= args.thresh]
        # '중복 프레임'은 쌍의 개수가 아니라 **적어도 하나와 겹치는 파일 수**다.
        # 쌍으로 세면 20장이 전부 같을 때 190쌍이 나와 규모가 과장된다.
        dup_files = {p for a, b, _ in near for p in (a, b)}
        rate = len(dup_files) / len(items) * 100 if items else 0.0
        total_dup += len(dup_files)
        total_n += len(items)
        print(f"{cam:<14}{len(items):>6}{len(near):>8}{len(dup_files):>11}{rate:>7.1f}%")
        worst.append((rate, cam, [(a.name, b.name, d) for a, b, d in near[:args.show]]))

    overall = total_dup / total_n * 100 if total_n else 0.0
    print("-" * 47)
    print(f"{'전체':<14}{total_n:>6}{'':>8}{total_dup:>11}{overall:>7.1f}%")

    print(f"\n판정 (임계 해밍거리 {args.thresh})")
    if overall < 5:
        print("  ✅ 중복 낮음 — 학습이 안 되는 원인이 데이터 중복은 아니다.")
    elif overall < 20:
        print("  🟡 중복 보통 — 아래 상위 카메라만 손보면 된다.")
    else:
        print("  🔴 중복 높음 — randomizeScene의 변동 폭이 부족하다. 아래 카메라부터.")

    worst.sort(reverse=True)
    for rate, cam, examples in worst[:3]:
        if rate <= 0:
            continue
        print(f"\n  [{cam}] {rate:.1f}% — 예시")
        for a, b, d in examples:
            print(f"    거리 {d:>2}  {a}  ==  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
