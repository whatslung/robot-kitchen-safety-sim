"""/predict 배치 계약 + 하위호환 테스트 (감사 P0-5, 스펙 §3·§4-2).

detect_server의 순수 헬퍼 `_predict_response(body, predictor)`를 가짜 예측기로 직접 호출한다
(HTTP·모델 다운로드 없이). serve 그룹(fastapi)이 없으면 import 단계에서 skip.
"""
import importlib

import pytest

pytest.importorskip("fastapi", reason="serve 그룹 필요 (uv sync --group serve)")


@pytest.fixture
def ds(monkeypatch):
    monkeypatch.setenv("DETECT_MODEL", "none")   # 검출 비활성 — 모델 다운로드 없이 로드
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    mod = importlib.import_module("backend.detect_server")
    return importlib.reload(mod)


class FakePredictor:
    """호출된 hists 개수만큼 모드를 돌려주는 가짜 예측기. 경로는 인자로 주입."""
    def __init__(self, mode_maker):
        self.mode_maker = mode_maker
        self.batch_calls = 0

    def predict_batch(self, hists):
        self.batch_calls += 1
        return [self.mode_maker(i) for i, _ in enumerate(hists)]

    def predict_modes(self, hist):
        return self.mode_maker(0)


def _straight_in(_i):
    # 원점으로 곧장 들어오는 단일 모드(정지반경 진입).
    return [{"path": [(3.0, 0.0), (2.0, 0.0), (1.0, 0.0)], "w": 1.0, "sigma": [0.0, 0.0, 0.0]}]


def _far(_i):
    # 계속 멀리 있는 단일 모드(진입 없음).
    return [{"path": [(9.0, 0.0), (9.0, 0.0), (9.0, 0.0)], "w": 1.0, "sigma": [0.0, 0.0, 0.0]}]


def test_batch_returns_per_track_modes_and_risk(ds):
    body = {
        "tracks": [{"id": "gt:0", "hist": [[6, 0]] * 8}, {"id": "gt:1", "hist": [[6, 0]] * 8}],
        "robot": {"x": 0, "z": 0}, "stopR": 3.1, "slowR": 5.1,
        "horizon": 1.6, "safeKsig": 1.0, "safeTau": 0.1,
    }
    p = FakePredictor(_straight_in)
    resp = ds._predict_response(body, p)
    assert p.batch_calls == 1                       # forward 한 번(배치)
    assert len(resp["tracks"]) == 2
    for t in resp["tracks"]:
        assert "modes" in t and "risk" in t
        assert t["risk"]["tEntryStop"] is not None  # 진입 감지
    assert resp["worst"] is not None
    assert resp["worst"]["id"] == "gt:0"            # 동률 → id 오름차순


def test_batch_worst_is_none_when_no_entry(ds):
    body = {
        "tracks": [{"id": 0, "hist": [[9, 0]] * 8}],
        "robot": {"x": 0, "z": 0}, "stopR": 3.1, "slowR": 5.1,
        "horizon": 1.6, "safeKsig": 1.0, "safeTau": 0.1,
    }
    resp = ds._predict_response(body, FakePredictor(_far))
    assert resp["worst"] is None
    assert resp["tracks"][0]["risk"]["tEntryStop"] is None


def test_backward_compatible_single_hist(ds):
    body = {"hist": [[6, 0]] * 8}
    resp = ds._predict_response(body, FakePredictor(_straight_in))
    assert "modes" in resp                          # 옛 응답 형태 유지
    assert "tracks" not in resp
    assert resp["modes"][0]["path"][0] == [3.0, 0.0]


def test_batch_single_equivalence(ds):
    """같은 이력은 배치 응답의 modes와 단일 응답의 modes가 동일해야 한다."""
    p = FakePredictor(_straight_in)
    single = ds._predict_response({"hist": [[6, 0]] * 8}, p)["modes"]
    batch = ds._predict_response({
        "tracks": [{"id": 0, "hist": [[6, 0]] * 8}],
        "robot": {"x": 0, "z": 0}, "stopR": 3.1, "slowR": 5.1,
    }, p)["tracks"][0]["modes"]
    assert single == batch
