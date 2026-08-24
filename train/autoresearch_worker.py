"""Train and evaluate one isolated trajectory autoresearch model."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from trajectory.autoresearch_candidate import CONFIG, build_candidate
from trajectory.learned_predictor import (
    LearnedPredictor,
    build_cvae_net,
    build_net,
    build_transformer_net,
    frame_of,
    mtp_loss,
    to_frame,
)
from trajectory.traj_v2 import sha256_file
from train.autoresearch_contract import (
    count_trainable_parameters,
    development_windows,
    evaluate_windows,
    measure_cpu_p95,
)
from train.autoresearch_training import train_for_budget


CANDIDATE_PATH = ROOT / "trajectory" / "autoresearch_candidate.py"


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    weight_decay: float
    batch_size: int


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    seed: int
    budget_seconds: float
    output_json: Path
    weights_path: Path


def build_model(name: str):
    if name == "lstm":
        return build_net(h=64)
    if name == "transformer":
        return build_transformer_net(h=64, layers=2, heads=4)
    if name == "cvae":
        return build_cvae_net(h=64, layers=2, heads=4)
    if name == "candidate":
        return build_candidate(CONFIG)
    raise ValueError(f"지원하지 않는 model: {name}")


def model_hyperparameters(name: str) -> Hyperparameters:
    if name == "candidate":
        return Hyperparameters(
            CONFIG.learning_rate,
            CONFIG.weight_decay,
            CONFIG.batch_size,
        )
    if name in {"lstm", "transformer", "cvae"}:
        return Hyperparameters(1e-3, 0.0, 512)
    raise ValueError(f"지원하지 않는 model: {name}")


def build_training_arrays(windows, seed):
    rng = np.random.default_rng(seed)
    x_values = []
    y_values = []
    for window in windows:
        history = np.asarray(
            [(x, z) for _, x, z in window.scene.agents[0].history],
            dtype=float,
        )
        future = [(x, z) for _, x, z in window.gt]
        variants = (
            history,
            history + rng.normal(0, 0.06, history.shape),
            history + rng.normal(0, 0.06, history.shape),
        )
        for observed in variants:
            origin, angle = frame_of(observed)
            x_values.append(to_frame(observed, origin, angle))
            y_values.append(to_frame(future, origin, angle))
    return np.asarray(x_values, np.float32), np.asarray(y_values, np.float32)


def mtp_loss_adapter(net, batch_x, batch_y, progress):
    del progress
    return mtp_loss(*net(batch_x), batch_y)


def cvae_loss_adapter(net, batch_x, batch_y, progress):
    beta = min(1.0, progress / 0.5)
    return net.elbo(batch_x, batch_y, beta=beta)["loss"]


def candidate_sha256() -> str:
    return sha256_file(CANDIDATE_PATH)


def _atomic_torch_save(state_dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(state_dict, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_worker(config: WorkerConfig) -> dict:
    _set_determinism(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "cpu"
    )
    hyperparameters = model_hyperparameters(config.model)
    train_windows = development_windows("train")
    x_values, y_values = build_training_arrays(train_windows, config.seed)
    x = torch.tensor(x_values, device=device)
    y = torch.tensor(y_values, device=device)
    net = build_model(config.model).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    loss_fn = cvae_loss_adapter if config.model == "cvae" else mtp_loss_adapter
    net.train()
    training_result = train_for_budget(
        net,
        optimizer,
        loss_fn,
        x,
        y,
        batch_size=hyperparameters.batch_size,
        seed=config.seed,
        budget_seconds=config.budget_seconds,
    )

    cpu_net = build_model(config.model)
    cpu_net.load_state_dict(net.state_dict())
    cpu_predictor = LearnedPredictor(net=cpu_net, device="cpu")
    val_windows = development_windows("val")
    latency_hist = [
        (px, pz)
        for _, px, pz in val_windows[0].scene.agents[0].history
    ]
    cpu_p95_ms = measure_cpu_p95(cpu_predictor, latency_hist)
    parameters = count_trainable_parameters(cpu_net)

    predictor = LearnedPredictor(net=net, device=device)
    evaluation = evaluate_windows(
        predictor,
        val_windows,
        cpu_p95_ms=cpu_p95_ms,
        parameters=parameters,
    )
    _atomic_torch_save(net.state_dict(), config.weights_path)
    metric_record = asdict(evaluation.metrics)
    metric_record["f2"] = evaluation.metrics.f2
    record = {
        "status": "ok",
        "model": config.model,
        "seed": config.seed,
        "training": asdict(training_result),
        "metrics": metric_record,
        "ci": evaluation.ci,
        "candidate_sha256": candidate_sha256(),
        "weights": str(config.weights_path),
        "weights_sha256": sha256_file(config.weights_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device_name,
            "cpu_threads": torch.get_num_threads(),
        },
    }
    _atomic_json_write(record, config.output_json)
    return record


def parse_args(argv=None) -> WorkerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("lstm", "transformer", "cvae", "candidate"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget-seconds", required=True, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--weights-path", required=True, type=Path)
    args = parser.parse_args(argv)
    return WorkerConfig(
        model=args.model,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        output_json=args.output_json,
        weights_path=args.weights_path,
    )


def main(argv=None) -> int:
    record = run_worker(parse_args(argv))
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
