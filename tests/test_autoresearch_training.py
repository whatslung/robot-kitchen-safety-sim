import pytest
import torch

from train.autoresearch_training import NonFiniteTrainingError, train_for_budget


def test_training_budget_starts_after_warmup():
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.zeros(8, 2)
    y = torch.zeros(8, 2)
    ticks = iter([0.0, 0.0, 0.4, 0.8, 1.2, 1.2])
    result = train_for_budget(
        net,
        optimizer,
        lambda model, batch_x, batch_y, progress: (
            (model(batch_x) - batch_y) ** 2
        ).mean(),
        x,
        y,
        batch_size=8,
        seed=0,
        budget_seconds=1.0,
        warmup_steps=2,
        clock=lambda: next(ticks),
    )
    assert result.warmup_steps == 2
    assert result.steps == 3


def test_non_finite_loss_aborts_trial():
    net = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.zeros(2, 2)
    y = torch.zeros(2, 2)
    with pytest.raises(NonFiniteTrainingError, match="finite"):
        train_for_budget(
            net,
            optimizer,
            lambda model, batch_x, batch_y, progress: model(batch_x).sum()
            * float("nan"),
            x,
            y,
            2,
            seed=0,
            budget_seconds=0.1,
            warmup_steps=0,
        )
