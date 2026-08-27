"""Train and validate one fixed-step paired autoresearch variant."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from trajectory.learned_predictor import LearnedPredictor
from trajectory.traj_v2 import sha256_file
from train.autoresearch_contract import (
    count_trainable_parameters,
    development_windows,
    evaluate_windows,
    measure_cpu_p95,
)
from train.autoresearch_worker import (
    _atomic_json_write,
    _atomic_torch_save,
    _set_determinism,
    build_model,
    build_training_arrays,
    cvae_loss_adapter,
    mtp_loss_adapter,
)
from train.paired_autoresearch_contract import VARIANTS
from train.paired_autoresearch_training import train_for_steps


@dataclass(frozen=True)
class WorkerConfig:
    variant: str
    seed: int
    steps: int
    output_json: Path
    weights_path: Path


def build_variant(name: str):
    spec = VARIANTS[name]
    return build_model(spec.model), spec


def set_determinism(seed: int) -> None:
    _set_determinism(seed)
    torch.use_deterministic_algorithms(True)


def state_dict_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def environment_fingerprint() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "cpu"
    )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device_name,
    }


def run_worker(config: WorkerConfig) -> dict:
    set_determinism(config.seed)
    spec = VARIANTS[config.variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_windows = development_windows("train")
    x_values, y_values = build_training_arrays(train_windows, config.seed)
    x = torch.tensor(x_values, device=device)
    y = torch.tensor(y_values, device=device)
    net, _ = build_variant(config.variant)
    net = net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    loss_fn = cvae_loss_adapter if spec.model == "cvae" else mtp_loss_adapter
    net.train()
    training_result = train_for_steps(
        net,
        optimizer,
        loss_fn,
        x,
        y,
        batch_size=spec.batch_size,
        seed=config.seed,
        steps=config.steps,
    )

    cpu_net, _ = build_variant(config.variant)
    cpu_net.load_state_dict(net.state_dict())
    cpu_predictor = LearnedPredictor(net=cpu_net, device="cpu")
    val_windows = development_windows("val")
    latency_history = [
        (px, pz)
        for _, px, pz in val_windows[0].scene.agents[0].history
    ]
    cpu_p95_ms = measure_cpu_p95(cpu_predictor, latency_history)
    parameters = count_trainable_parameters(cpu_net)
    predictor = LearnedPredictor(net=net, device=device)
    evaluation = evaluate_windows(
        predictor,
        val_windows,
        cpu_p95_ms=cpu_p95_ms,
        parameters=parameters,
    )

    state_dict = net.state_dict()
    _atomic_torch_save(state_dict, config.weights_path)
    metrics = asdict(evaluation.metrics)
    metrics["f2"] = evaluation.metrics.f2
    record = {
        "status": "ok",
        "variant": config.variant,
        "model": spec.model,
        "seed": config.seed,
        "training": asdict(training_result),
        "hyperparameters": asdict(spec),
        "metrics": metrics,
        "ci": evaluation.ci,
        "weights": str(config.weights_path),
        "weights_sha256": state_dict_sha256(state_dict),
        "weights_file_sha256": sha256_file(config.weights_path),
        "environment": {
            **environment_fingerprint(),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
    }
    _atomic_json_write(record, config.output_json)
    return record


def parse_args(argv=None) -> WorkerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--weights-path", required=True, type=Path)
    args = parser.parse_args(argv)
    return WorkerConfig(
        variant=args.variant,
        seed=args.seed,
        steps=args.steps,
        output_json=args.output_json,
        weights_path=args.weights_path,
    )


def main(argv=None) -> int:
    record = run_worker(parse_args(argv))
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
