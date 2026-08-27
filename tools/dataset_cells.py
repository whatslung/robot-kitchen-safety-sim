"""12칸 균등 생성 결과 점검 — conditions.csv 한 장만 읽는다.

    python3 tools/dataset_cells.py <데이터셋폴더 | conditions.csv>

시뮬의 `cell`은 **요청값**이고, 같은 줄의 fire_progress·smoke_particles·steam_lens가
**실제로 찍힌 값**이다. fireSeekTo가 느린 PC에서 목표에 못 미친 전례가 있어
(HANDOFF「조건부 미해결」) 둘이 어긋날 수 있다 — 이 스크립트가 그걸 잡는다.

표준 라이브러리만 쓴다. pandas 설치 없이 어디서든 돌아가야 하는 점검 도구다.
"""
import csv
import sys
import unicodedata
from pathlib import Path


def pad(s, width, right=False):
    """한글은 터미널에서 두 칸을 먹는다 — 그걸 세서 맞춘다(f-string 정렬은 1칸으로 센다)."""
    s = str(s)
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    fill = " " * max(0, width - w)
    return fill + s if right else s + fill

FULL_S = 90.0  # FIRE_FOG.full — fire_progress 1.0 = 90초

# sim.html의 CELL_FIRE / CELL_FOG와 같은 값이어야 한다. 바꾸면 양쪽 다 바꾼다.
CELL_FIRE = [
    ("none", "평상시", False, None),
    ("early", "화재초기", True, (3.0, 30.0)),
    ("mid", "화재중기", True, (30.0, 60.0)),
    ("late", "화재후기", True, (60.0, 90.0)),
]
# 두 번째 축은 **렌즈 김서림**이다(2026-08-27). 솥 김 축은 화재 칸에서 죽어 있었다 —
# 김을 껐다 켜도 화재 중에는 화면이 0.07%밖에 안 바뀐다(연기에 묻힌다).
CELL_FOG = [
    ("clear", "김서림없음", (0.00, 0.00)),
    ("mid", "김서림중간", (0.20, 0.50)),
    ("heavy", "김서림심함", (0.50, 0.90)),
]
FIRE_BY_KEY = {k: (label, fire, stage) for k, label, fire, stage in CELL_FIRE}
FOG_BY_KEY = {k: (label, rng) for k, label, rng in CELL_FOG}

# 경계 여유. fireSeekTo가 프레임을 돌리는 동안 FIRE_FOG.t가 조금 더 간다(실측 +0.7s).
STAGE_TOL = 1.5


def find_csv(arg):
    p = Path(arg)
    if p.is_dir():
        p = p / "conditions.csv"
    if not p.exists():
        sys.exit(f"conditions.csv를 못 찾았다: {p}")
    return p


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("빈 CSV다.")
    if "cell" not in rows[0]:
        sys.exit(
            "`cell` 열이 없다 — 12칸 균등 생성으로 뽑은 데이터가 아니다.\n"
            "데이터 탭에서 「12칸 균등 생성」을 켜고 다시 뽑아야 한다."
        )
    return rows


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return default


def summarize(rows):
    cams = sorted({r["camera"] for r in rows})
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)

    unlabeled = len(by_cell.get("", []))
    print(f"이미지 {len(rows)}장 · 카메라 {len(cams)}대 · 칸 {len([k for k in by_cell if k])}개")
    if unlabeled:
        print(f"⚠️ cell이 빈 행 {unlabeled}장 — 12칸 모드가 아닌 회차가 섞였다")
    print()

    print(pad("칸", 20) + pad("샘플", 6, True) + pad("이미지", 8, True) +
          pad("찍힌 초", 16, True) + pad("연기입자(중앙)", 16, True) + pad("충전율", 8, True) +
          pad("김서림", 14, True) + pad("이탈", 6, True))
    print("-" * 94)

    problems = []
    smoke_med = {}          # 단계 축이 살아 있는지 보려고 칸별 중앙값을 모은다
    for fk, flabel, want_fire, stage in CELL_FIRE:
        for sk, slabel, frange in CELL_FOG:
            key = f"{fk}|{sk}"
            rs = by_cell.get(key, [])
            n_img = len(rs)
            n_sample = n_img // max(1, len(cams))
            if not rs:
                print(pad(flabel + "·" + slabel, 20) + pad(0, 6, True) + pad(0, 8, True) +
                      pad("—", 16, True) + pad("—", 16, True) + pad("—", 8, True) +
                      pad("—", 14, True) + pad("—", 6, True))
                problems.append(f"{key} 칸이 비었다")
                continue

            secs = [fnum(r, "fire_progress") * FULL_S for r in rs]
            smoke = sorted(int(fnum(r, "smoke_particles")) for r in rs)
            ks = [fnum(r, "steam_lens") for r in rs]     # 렌즈 김서림 실측값
            fires = [r["fire"] in ("1", "True", "true") for r in rs]

            bad = 0
            for r, sec, k, fire in zip(rs, secs, ks, fires):
                if fire != want_fire:
                    bad += 1
                elif want_fire and not (stage[0] - STAGE_TOL <= sec <= stage[1] + STAGE_TOL):
                    bad += 1
                elif not (frange[0] - 0.02 <= k <= frange[1] + 0.02):
                    bad += 1

            # 충전율 — fireSeekTo가 이 단계에 채우려던 개수(smoke_target) 대비 실제.
            # 느린 PC에서 후반 칸이 목표의 20~30%에 그치는 일이 있다. 그러면 라벨은
            # "화재후기"인데 그림은 초기 수준이 된다 — 칸이 이름만 후기가 된다.
            fills = [fnum(r, "smoke_particles") / fnum(r, "smoke_target", 0)
                     for r in rs if fnum(r, "smoke_target", 0) > 0]
            fill_med = sorted(fills)[len(fills) // 2] if fills else None
            fill_txt = "—" if fill_med is None else f"{fill_med * 100:.0f}%"

            sec_txt = "화재없음" if not want_fire else f"{min(secs):.1f}~{max(secs):.1f}"
            print(pad(flabel + "·" + slabel, 20) + pad(n_sample, 6, True) + pad(n_img, 8, True) +
                  pad(sec_txt, 16, True) + pad(smoke[len(smoke) // 2], 16, True) +
                  pad(fill_txt, 8, True) + pad(f"{min(ks):.2f}~{max(ks):.2f}", 14, True) +
                  pad(bad, 6, True))
            if bad:
                problems.append(f"{key}: 요청 범위를 벗어난 이미지 {bad}/{n_img}장")
            if fill_med is not None and fill_med < 0.8:
                problems.append(f"{key}: 연기가 목표의 {fill_med * 100:.0f}%까지만 찼다 "
                                f"— 라벨은 {flabel}인데 그림은 그만큼 옅다")
            smoke_med[key] = smoke[len(smoke) // 2]

    # 단계 축이 실제로 갈렸는가 — 이름만 다르고 그림이 같으면 칸을 나눈 의미가 없다.
    def stage_med(fk):
        v = [smoke_med[f"{fk}|{sk}"] for sk, *_ in CELL_FOG if f"{fk}|{sk}" in smoke_med]
        return sum(v) / len(v) if v else 0
    early, late = stage_med("early"), stage_med("late")
    if early > 0 and late / early < 1.5:
        problems.append(f"화재 단계 축이 무너졌다 — 초기 평균 {early:.0f}개 vs 후기 평균 {late:.0f}개 "
                        f"({late / early:.2f}배). 연기량으로는 단계가 구분되지 않는다")

    print()
    counts = [len(by_cell.get(f"{fk}|{sk}", [])) for fk, *_ in CELL_FIRE for sk, *_ in CELL_FOG]
    spread = max(counts) - min(counts)
    per_sample = max(1, len(cams))
    print(f"칸별 이미지 수 {min(counts)}~{max(counts)}장 (차이 {spread}장 = 샘플 {spread // per_sample}개)")
    if spread > per_sample:
        problems.append(f"칸 사이 차이가 샘플 {spread // per_sample}개 — 라운드로빈이면 최대 1개여야 한다")

    # 카메라 × 화재단계 교차표 — 카메라별로 쪼개서 볼 만큼 두꺼운지 확인용
    print()
    print("카메라별 이미지 수 (화재 단계 기준)")
    print(pad("카메라", 12) + "".join(pad(lbl, 11, True) for _, lbl, *_ in CELL_FIRE) + pad("합계", 8, True))
    for cam in cams:
        cells = []
        for fk, *_ in CELL_FIRE:
            cells.append(sum(1 for r in rows if r["camera"] == cam and r["cell"].startswith(fk + "|")))
        print(pad(cam, 12) + "".join(pad(c, 11, True) for c in cells) + pad(sum(cells), 8, True))

    print()
    if problems:
        print("⚠️ 확인할 것")
        for p in problems:
            print("  · " + p)
    else:
        print("✅ 12칸 전부 요청 범위 안이고 장수도 고르다.")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = find_csv(sys.argv[1])
    print(f"# {path}\n")
    sys.exit(summarize(load(path)))
