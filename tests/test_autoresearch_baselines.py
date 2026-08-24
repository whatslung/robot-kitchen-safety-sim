import pytest

from train.run_autoresearch_baselines import BaselineError, validate_baselines


def test_baselines_require_all_three_models():
    with pytest.raises(BaselineError, match="cvae"):
        validate_baselines({"lstm": {}, "transformer": {}})


def test_transformer_is_guard_reference():
    rows = {
        name: {"metrics": {"f2": index}}
        for index, name in enumerate(("lstm", "transformer", "cvae"))
    }
    assert validate_baselines(rows)["guard_reference"] == "transformer"
