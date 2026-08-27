import torch

from train.autoresearch_worker import build_model, model_hyperparameters


def test_worker_builds_all_fixed_baselines_and_candidate():
    for name in ("lstm", "transformer", "cvae", "candidate"):
        net = build_model(name)
        assert isinstance(net, torch.nn.Module)


def test_candidate_hyperparameters_come_from_candidate_file():
    hyperparameters = model_hyperparameters("candidate")
    assert hyperparameters.batch_size > 0
    assert hyperparameters.learning_rate > 0
