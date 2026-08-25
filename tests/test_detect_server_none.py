"""DETECT_MODEL=none 예측 전용 부팅 테스트 (감사 P0-B).

README·HANDOFF는 `DETECT_MODEL=none`이면 검출을 끄고 GT 좌표만 쓰는 '예측 전용'
실행이라고 안내한다. 그러나 과거 구현은 `Path("none").exists()`가 False라
허깅페이스 허브에서 기본 모델을 **다운로드**했다(오프라인이면 부팅 실패).

이 테스트는 `none`(및 off/빈값)이 명시적 검출 비활성 모드로 처리되어,
모델 로드도 허브 다운로드도 하지 않고 서버가 뜨는지 고정한다.

serve 그룹(fastapi 등)이 없으면 detect_server가 import 단계에서 SystemExit 하므로
그 경우 이 테스트는 skip 된다:  uv run --group serve --with pytest python -m pytest tests/
"""
import importlib

import pytest

pytest.importorskip("fastapi", reason="serve 그룹 필요 (uv sync --group serve)")


def _load(monkeypatch, model_val):
    """DETECT_MODEL을 지정하고 detect_server를 새로 로드한다.

    HF_HUB_OFFLINE=1 을 켜 두어, 만약 코드가 실수로 허브를 받으려 하면 예외가 나고
    모델 로드 try/except가 이를 삼켜 MODE가 'off'가 아닌 'dummy'로 떨어진다 →
    "받으려는 시도"가 있으면 아래 MODE=='off' 단언이 깨져 회귀를 잡는다.
    """
    monkeypatch.setenv("DETECT_MODEL", model_val)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    mod = importlib.import_module("backend.detect_server")
    return importlib.reload(mod)


@pytest.mark.parametrize("val,off", [
    ("none", True), ("NONE", True), (" none ", True), ("off", True), ("", True),
    ("training/island_yolo11s/weights/best.pt", False), ("yolo11s.pt", False),
])
def test_is_detect_off(monkeypatch, val, off):
    ds = _load(monkeypatch, "none")   # 모듈 자체는 안전하게 로드해 두고
    assert ds._is_detect_off(val) is off


def test_none_boots_without_model_or_download(monkeypatch):
    ds = _load(monkeypatch, "none")
    assert ds.MODE == "off", f"none인데 MODE={ds.MODE!r} — 검출 비활성으로 뜨지 않았다"
    assert ds.MODEL is None, "none인데 모델이 로드됐다(허브 다운로드 가능성)"


def test_off_detect_returns_no_boxes(monkeypatch):
    ds = _load(monkeypatch, "none")
    assert ds.run_detect(None) == [], "검출 비활성 모드는 빈 박스를 돌려줘야 한다"


def test_health_reports_off_mode(monkeypatch):
    ds = _load(monkeypatch, "none")
    assert ds.health()["mode"] == "off"
