import pytest
import torch

from trajectory.autoresearch_candidate import CandidateConfig, build_candidate
from trajectory.learned_predictor import K, OBS, PRED


def test_candidate_output_contract():
    net = build_candidate(
        CandidateConfig(
            hidden=32,
            layers=1,
            heads=4,
            ff_ratio=2,
            dropout=0.0,
            norm_first=True,
            pooling="last",
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=64,
        )
    )
    paths, logits, logsig = net(torch.zeros(5, OBS, 2))
    assert paths.shape == (5, K, PRED, 2)
    assert logits.shape == (5, K)
    assert logsig.shape == (5, K, PRED)


def test_hidden_must_be_divisible_by_heads():
    with pytest.raises(ValueError, match="hidden.*heads"):
        build_candidate(CandidateConfig(hidden=30, heads=4))


def test_pooling_is_explicit():
    with pytest.raises(ValueError, match="pooling"):
        build_candidate(CandidateConfig(pooling="max"))
