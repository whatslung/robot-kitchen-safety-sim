"""문서 지표 이름 자동 검증 (감사 P0-A).

검출 sim-to-real 수치는 `recall 0.270 / precision 0.072 / mAP50 0.048`이다.
과거 문서들이 mAP50 값 `0.048`을 'recall'로 잘못 불렀다(지표 뒤바뀜). 이 값이
발표·모델카드로 새어 나가면 성능을 20배 이상 낮게 오해하게 만든다. 아래 두 성질을
회귀 테스트로 고정한다.

1. 권위 문서(detection-eval.md)에 세 수치가 `recall|precision|mAP50` 순서로 함께 있다.
2. 어떤 문서도 `0.048`을 recall로 부르지 않는다(표 셀 제외 — 표는 헤더가 지표를 명시).
"""
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs" / "chanwoo"

# sim-only(합성만 학습) → 실사 test 검출 성능. 출처: detection-eval.md §2.
REAL_RECALL = "0.270"
REAL_PRECISION = "0.072"
REAL_MAP50 = "0.048"


def _md_files():
    # handoff/ 는 특정 시점 기록이라 옛 오기를 '인용'해 설명한다(감사 문서 자체가 그 예).
    # 인용까지 막으면 결함을 문서화할 수 없으므로 산 문서(성적표·평가·카드)만 검사한다.
    return sorted(p for p in DOCS.rglob("*.md") if "handoff" not in p.parts)


def test_docs_dir_exists():
    assert DOCS.is_dir(), f"문서 경로 없음: {DOCS}"
    assert _md_files(), "검사할 .md 문서가 없다"


def test_canonical_transfer_numbers_present():
    """detection-eval.md에 세 수치가 recall|precision|mAP50 순서로 한 줄에 함께 있어야 한다."""
    text = (DOCS / "detection-eval.md").read_text(encoding="utf-8")
    ok = any(
        (REAL_RECALL in line and REAL_PRECISION in line and REAL_MAP50 in line
         and line.index(REAL_RECALL) < line.index(REAL_PRECISION) < line.index(REAL_MAP50))
        for line in text.splitlines()
    )
    assert ok, (
        "detection-eval.md에 sim-only→실사 행(recall 0.270 / precision 0.072 / mAP50 0.048)이 "
        "그 순서로 보이지 않는다 — 권위 수치가 훼손됐을 수 있다."
    )


@pytest.mark.parametrize("md", _md_files(), ids=lambda p: p.name)
def test_no_doc_mislabels_map50_as_recall(md):
    """어떤 산문 줄도 mAP50 값(0.048)을 recall로 부르면 안 된다.

    표 셀(줄이 '|'로 시작)은 헤더가 지표를 명시하므로 제외한다. 산문에서 'recall'과
    '0.048'이 같은 줄에 있으면서 'mAP'/올바른 recall값(0.270)이 함께 있지 않으면,
    0.048을 recall로 오기한 것으로 본다.
    """
    offenders = []
    for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith("|"):            # 표 셀 — 헤더가 지표명을 준다
            continue
        low = s.lower()
        if "recall" in low and REAL_MAP50 in s and "map" not in low and REAL_RECALL not in s:
            offenders.append(f"{md.name}:{i}: {s}")
    assert not offenders, (
        "mAP50 값 0.048을 recall로 오기한 줄:\n" + "\n".join(offenders)
        + f"\n→ 올바른 표기: recall {REAL_RECALL} / precision {REAL_PRECISION} / mAP50 {REAL_MAP50}"
    )
