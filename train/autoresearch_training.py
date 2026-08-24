"""Deterministic time-budgeted training loop for autoresearch trials."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    warmup_steps: int
    train_seconds: float
    final_loss: float


class NonFiniteTrainingError(RuntimeError):
    pass


def train_for_budget(
    net,
    optimizer,
    loss_fn,
    x,
    y,
    batch_size,
    seed,
    budget_seconds=300.0,
    warmup_steps=10,
    clock=time.perf_counter,
):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    order = torch.randperm(len(x), generator=generator, device=x.device)
    cursor = 0

    def step(progress):
        nonlocal order, cursor
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
        loss = loss_fn(net, x[indices], y[indices], progress)
        if not torch.isfinite(loss):
            raise NonFiniteTrainingError("loss is not finite")
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    final_loss = float("nan")
    for _ in range(warmup_steps):
        final_loss = step(0.0)
    start = clock()
    steps = 0
    while True:
        now = clock()
        if now - start >= budget_seconds:
            break
        progress = min(1.0, (now - start) / budget_seconds)
        final_loss = step(progress)
        steps += 1
    return TrainingResult(
        steps=steps,
        warmup_steps=warmup_steps,
        train_seconds=clock() - start,
        final_loss=final_loss,
    )
