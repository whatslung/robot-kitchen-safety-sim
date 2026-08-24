"""Single editable model seam for trajectory autoresearch experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trajectory.learned_predictor import K, OBS, PRED


@dataclass(frozen=True)
class CandidateConfig:
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    ff_ratio: int = 4
    dropout: float = 0.0
    norm_first: bool = False
    pooling: str = "last"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512


CONFIG = CandidateConfig()


def build_candidate(config: CandidateConfig = CONFIG) -> nn.Module:
    if config.hidden % config.heads:
        raise ValueError("hidden은 heads로 나누어져야 함")
    if config.pooling not in {"last", "mean"}:
        raise ValueError(f"지원하지 않는 pooling: {config.pooling}")

    class CandidateTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = config.hidden
            self.inp = nn.Linear(2, hidden)
            self.pos = nn.Parameter(torch.zeros(1, OBS, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.heads,
                dim_feedforward=hidden * config.ff_ratio,
                dropout=config.dropout,
                batch_first=True,
                norm_first=config.norm_first,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, K * PRED * 2 + K + K * PRED),
            )

        def forward(self, x):
            encoded = self.enc(self.inp(x) + self.pos[:, : x.shape[1], :])
            if config.pooling == "last":
                pooled = encoded[:, -1, :]
            else:
                pooled = encoded.mean(dim=1)
            output = self.head(pooled)
            batch = output.shape[0]
            paths = output[:, : K * PRED * 2].reshape(batch, K, PRED, 2)
            logits = output[:, K * PRED * 2 : K * PRED * 2 + K]
            logsig = output[:, K * PRED * 2 + K :].reshape(batch, K, PRED)
            return paths, logits, logsig

    return CandidateTransformer()
