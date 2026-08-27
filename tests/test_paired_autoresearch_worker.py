import importlib

import pytest
import torch


def _worker_module():
    return importlib.import_module("train.paired_autoresearch_worker")


def test_worker_builds_each_registered_variant_with_its_learning_rate():
    worker = _worker_module()

    for variant in worker.VARIANTS:
        net, hyperparameters = worker.build_variant(variant)
        assert isinstance(net, torch.nn.Module)
        assert hyperparameters.learning_rate == pytest.approx(
            worker.VARIANTS[variant].learning_rate
        )


def test_deterministic_setup_enables_strict_torch_algorithms():
    worker = _worker_module()

    worker.set_determinism(7)

    assert torch.are_deterministic_algorithms_enabled()


def test_state_dict_hash_depends_on_tensor_values_not_archive_metadata():
    worker = _worker_module()
    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    second.load_state_dict(first.state_dict())

    assert worker.state_dict_sha256(first.state_dict()) == worker.state_dict_sha256(
        second.state_dict()
    )

    with torch.no_grad():
        second.weight[0, 0] += 1.0
    assert worker.state_dict_sha256(first.state_dict()) != worker.state_dict_sha256(
        second.state_dict()
    )
