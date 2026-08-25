"""Exact-step training loop for paired trajectory experiments."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    train_seconds: float
    final_loss: float


class NonFiniteTrainingError(RuntimeError):
    pass


def train_for_steps(
    net,
    optimizer,
    loss_fn,
    x,
    y,
    batch_size,
    seed,
    steps,
    clock=time.perf_counter,
):
    if steps <= 0:
        raise ValueError("steps must be positive")

    generator = torch.Generator(device=x.device).manual_seed(seed)
    order = torch.randperm(len(x), generator=generator, device=x.device)
    cursor = 0
    final_loss = float("nan")
    start = clock()

    for step_index in range(steps):
        if cursor + batch_size > len(order):
            order = torch.randperm(
                len(x),
                generator=generator,
                device=x.device,
            )
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        optimizer.zero_grad()
        progress = (step_index + 1) / steps
        loss = loss_fn(net, x[indices], y[indices], progress)
        if not torch.isfinite(loss):
            raise NonFiniteTrainingError("loss is not finite")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    return TrainingResult(
        steps=steps,
        train_seconds=clock() - start,
        final_loss=final_loss,
    )
