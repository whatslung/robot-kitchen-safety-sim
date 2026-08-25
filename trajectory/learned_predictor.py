"""학습형 멀티모달 예측기 — 경량 LSTM + 혼합 헤드(MTP). 이슈 #2 4단계.

관측 8스텝(미터) → K개 미래 모드(각 12스텝 경로 + 가중치 + 스텝별 σ). 에이전트 중심
정규화(마지막 관측 원점 + 진행방향 +x 회전) 후 학습·추론하고, 예측을 미터 원좌표로 역변환한다.
설계: docs/chanwoo/specs/2026-08-19-learned-predictor-design.md
"""
from __future__ import annotations

import math

import numpy as np

OBS, PRED, K = 8, 12, 3


# ── 에이전트 중심 정규화 ─────────────────────────────────────────────────────
def frame_of(hist_xz):
    """관측 (OBS,2) → (origin(2,), ang). last=원점, 관측 첫→끝 벡터가 heading."""
    xs = np.asarray(hist_xz, dtype=np.float64)
    origin = xs[-1].copy()
    v = xs[-1] - xs[0]
    ang = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 1e-3 else 0.0
    return origin, ang


def _rot(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], axis=1)


def to_frame(xz, origin, ang):
    """미터 좌표 (N,2) → 정규화 프레임(원점 상대 + -ang 회전)."""
    return _rot(np.asarray(xz, dtype=np.float64) - origin, -ang)


def from_frame(xz_norm, origin, ang):
    """정규화 프레임 (N,2) → 미터 원좌표(+ang 회전 후 원점 복원)."""
    return _rot(xz_norm, ang) + origin


# ── 모델 (torch는 함수 안에서 import — numpy만 쓰는 추론 경로와 분리) ──────────
def build_net(h=64, k=K, pred=PRED):
    import torch.nn as nn

    class TrajNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.pred = k, pred
            self.enc = nn.LSTM(2, h, batch_first=True)
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                      nn.Linear(h, k * pred * 2 + k + k * pred))

        def forward(self, x):                      # x (B,OBS,2)
            _, (hn, _) = self.enc(x)
            o = self.head(hn[-1])
            b = o.shape[0]
            paths = o[:, :k * pred * 2].reshape(b, k, pred, 2)
            logits = o[:, k * pred * 2:k * pred * 2 + k]
            logsig = o[:, k * pred * 2 + k:].reshape(b, k, pred)
            return paths, logits, logsig

    return TrajNet()


def build_transformer_net(h=64, k=K, pred=PRED, layers=2, heads=4):
    """검증된 구조 비교(옵션 B) — LSTM 백본을 self-attention Transformer 인코더로 교체.
    head·출력 계약은 build_net 과 **완전히 동일**(paths/logits/logsig) → LearnedPredictor·
    mtp_loss 를 그대로 재사용한다. 관측 8스텝에 위치 인코딩을 더해 인코딩하고 마지막
    관측 토큰으로 미래를 뽑는다(ego 정규화상 마지막 관측 = 원점, '현재'를 질의점으로)."""
    import torch
    import torch.nn as nn

    class TrajTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.pred = k, pred
            self.inp = nn.Linear(2, h)
            self.pos = nn.Parameter(torch.zeros(1, 64, h))     # 위치 인코딩(최대 64스텝, OBS=8 사용)
            enc_layer = nn.TransformerEncoderLayer(d_model=h, nhead=heads, dim_feedforward=h * 4,
                                                   batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                      nn.Linear(h, k * pred * 2 + k + k * pred))

        def forward(self, x):                                  # x (B,OBS,2)
            t = x.shape[1]
            z = self.inp(x) + self.pos[:, :t, :]
            z = self.enc(z)                                    # (B,OBS,h)
            o = self.head(z[:, -1, :])                         # 마지막 관측 토큰 = '현재'
            b = o.shape[0]
            paths = o[:, :k * pred * 2].reshape(b, k, pred, 2)
            logits = o[:, k * pred * 2:k * pred * 2 + k]
            logsig = o[:, k * pred * 2 + k:].reshape(b, k, pred)
            return paths, logits, logsig

    return TrajTransformer()


def mtp_loss(paths, logits, logsig, gt):
    """best-of-K 회귀(가우시안 NLL) + 모드 분류(CE). gt (B,PRED,2)."""
    import torch
    import torch.nn.functional as F
    diff = paths - gt.unsqueeze(1)                 # (B,K,PRED,2)
    d2 = (diff ** 2).sum(-1)                        # (B,K,PRED)
    kstar = d2.sqrt().mean(-1).argmin(1)            # (B,) ADE(평균 L2) 최소 모드 = winner
    sig = logsig.exp().clamp(min=1e-2)             # (B,K,PRED)
    gi = kstar.view(-1, 1, 1)
    d2s = d2.gather(1, gi.expand(-1, 1, d2.shape[-1])).squeeze(1)     # (B,PRED)
    ss = sig.gather(1, gi.expand(-1, 1, sig.shape[-1])).squeeze(1)    # (B,PRED)
    nll = 0.5 * d2s / (ss ** 2) + 2.0 * torch.log(ss)                 # 2D 등방 가우시안
    return nll.mean() + F.cross_entropy(logits, kstar)


# ── 추론 래퍼 ────────────────────────────────────────────────────────────────
class LearnedPredictor:
    """학습된 가중치로 K모드를 낸다. predict_modes(hist_xz) → 미터 좌표 모드 리스트."""

    def __init__(self, weights_path=None, net=None, device="cpu"):
        import torch
        self.torch = torch
        self.device = device
        self.net = net if net is not None else build_net()
        if weights_path is not None:
            self.net.load_state_dict(torch.load(weights_path, map_location=device))
        self.net.to(device).eval()

    def predict_modes(self, hist_xz):
        """hist_xz = [(x,z)]*OBS (미터). → [{path:[(x,z)]*PRED, w, sigma:[..]}] (미터, 가중치 내림차순)."""
        return self.predict_batch([hist_xz])[0]

    def predict_batch(self, hists):
        """hists = [hist_xz, …] → 각 입력의 모드 리스트. 한 번의 forward로 배치 처리."""
        torch = self.torch
        if not hists:
            return []
        frames = [frame_of(h) for h in hists]
        obs = np.stack([to_frame(h, o, a) for h, (o, a) in zip(hists, frames)]).astype(np.float32)
        with torch.no_grad():
            x = torch.tensor(obs, device=self.device)                    # (N,OBS,2)
            paths, logits, logsig = self.net(x)
            w = torch.softmax(logits, dim=1).cpu().numpy()               # (N,K)
            paths = paths.cpu().numpy()                                  # (N,K,PRED,2)
            sig = logsig.exp().clamp(0.01, 5.0).cpu().numpy()            # (N,K,PRED) 밀도 viz 보호(상·하한)
        out = []
        for i, (origin, ang) in enumerate(frames):
            modes = []
            for k in range(paths.shape[1]):
                path_m = from_frame(paths[i, k], origin, ang)            # (PRED,2) 미터
                modes.append({"path": [tuple(p) for p in path_m], "w": float(w[i, k]),
                              "sigma": [float(s) for s in sig[i, k]]})
            modes.sort(key=lambda m: -m["w"])
            out.append(modes)
        return out
