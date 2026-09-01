"""PREDICT_NET 아키텍처 선택 회귀 테스트 (PR #40 후속).

`_predictor_net_config` 가 별칭·대소문자·공백을 정규화해 (arch, repo, local) 을 돌려주는지,
로컬 파일명이 학습·평가 스크립트와 일치하는지, `/health` 가 그 arch 를 보고하는지 고정한다.
향후 아키텍처 추가/이름 변경 시 회귀를 잡는다.

serve 그룹(fastapi 등)이 없으면 detect_server가 import 단계에서 SystemExit 하므로 skip 된다:
  uv run --group serve --with pytest python -m pytest tests/
"""
import importlib

import pytest

pytest.importorskip("fastapi", reason="serve 그룹 필요 (uv sync --group serve)")


def _load(monkeypatch, **env):
    """detect_server 를 검출 비활성(none)·오프라인으로 안전하게 재로드한다.

    DETECT_MODEL=none·HF_HUB_OFFLINE=1 로, 모듈 import 시 모델 로드나 허브 다운로드가
    일어나지 않게 한다(_predictor_net_config·health 는 가중치 없이 순수하게 동작).
    """
    monkeypatch.setenv("DETECT_MODEL", "none")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = importlib.import_module("backend.detect_server")
    return importlib.reload(mod)


@pytest.mark.parametrize("val,arch", [
    ("transformer", "transformer"), ("tf", "transformer"), ("xf", "transformer"),
    ("Transformer", "transformer"), ("  TF  ", "transformer"),
    ("lstm", "lstm"), ("LSTM", "lstm"), ("", "lstm"), ("garbage", "lstm"), (None, "lstm"),
])
def test_net_arch_normalized(monkeypatch, val, arch):
    ds = _load(monkeypatch)
    assert ds._predictor_net_config(val)[0] == arch


@pytest.mark.parametrize("val,repo,local", [
    ("transformer", "chanubc/human-move-transformer", "model_transformer.pt"),
    ("lstm", "chanubc/human-move-lstm", "model.pt"),
])
def test_net_repo_and_local(monkeypatch, val, repo, local):
    ds = _load(monkeypatch)
    _, r, l = ds._predictor_net_config(val)
    assert (r, l) == (repo, local)


def test_transformer_local_matches_training_script(monkeypatch):
    """Transformer 로컬 기본 파일명이 train/eval 스크립트(model_transformer.pt)와 일치해야
    로컬 재학습본이 조용히 무시되고 허브 버전을 받는 사고가 안 난다(PR #40 리뷰 지적)."""
    ds = _load(monkeypatch)
    assert ds._predictor_net_config("transformer")[2] == "model_transformer.pt"


@pytest.mark.parametrize("val,arch", [("transformer", "transformer"), ("lstm", "lstm"), (None, "lstm")])
def test_health_reports_predict_net(monkeypatch, val, arch):
    env = {} if val is None else {"PREDICT_NET": val}
    ds = _load(monkeypatch, **env)
    assert ds.health()["predict_net"] == arch
