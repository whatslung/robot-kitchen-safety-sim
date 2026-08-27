import importlib

import pytest
import torch


def _training_module():
    return importlib.import_module("train.paired_autoresearch_training")


def test_fixed_step_training_runs_exact_requested_updates():
    training = _training_module()
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.zeros(8, 2)
    y = torch.zeros(8, 2)
    progress_values = []

    result = training.train_for_steps(
        net,
        optimizer,
        lambda model, batch_x, batch_y, progress: (
            progress_values.append(progress)
            or ((model(batch_x) - batch_y) ** 2).mean()
        ),
        x,
        y,
        batch_size=4,
        seed=0,
        steps=3,
    )

    assert result.steps == 3
    assert progress_values == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_fixed_step_training_rejects_non_positive_step_count():
    training = _training_module()
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.zeros(2, 2)
    y = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="steps"):
        training.train_for_steps(
            net,
            optimizer,
            lambda model, batch_x, batch_y, progress: model(batch_x).sum(),
            x,
            y,
            batch_size=2,
            seed=0,
            steps=0,
        )


def test_fixed_step_training_aborts_on_non_finite_loss():
    training = _training_module()
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.zeros(2, 2)
    y = torch.zeros(2, 2)

    with pytest.raises(training.NonFiniteTrainingError, match="finite"):
        training.train_for_steps(
            net,
            optimizer,
            lambda model, batch_x, batch_y, progress: (
                model(batch_x).sum() * float("nan")
            ),
            x,
            y,
            batch_size=2,
            seed=0,
            steps=1,
        )
