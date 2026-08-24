"""이산 잠재 CVAE 궤적 예측기 — 추론 계약(드롭인)·ELBO 학습·결정성 (설계 §5)."""
import numpy as np

from trajectory.learned_predictor import (build_cvae_net, LearnedPredictor,
                                          OBS, PRED, K)


def test_cvae_predict_modes_shape():
    lp = LearnedPredictor(net=build_cvae_net(), device="cpu")      # 미학습 net
    hist = [(0.2 * i, 0.0) for i in range(OBS)]
    modes = lp.predict_modes(hist)                                 # 추론 = 기존 head 계약과 동형
    assert len(modes) == K
    assert all(len(m["path"]) == PRED and len(m["sigma"]) == PRED for m in modes)
    assert abs(sum(m["w"] for m in modes) - 1.0) < 1e-5           # prior softmax 합=1
    assert all(modes[i]["w"] >= modes[i + 1]["w"] for i in range(K - 1))


def test_cvae_forward_shapes():
    import torch
    net = build_cvae_net(h=32)
    x = torch.zeros(4, OBS, 2)
    paths, logits, logsig = net(x)
    assert paths.shape == (4, K, PRED, 2)
    assert logits.shape == (4, K)
    assert logsig.shape == (4, K, PRED)


def test_cvae_forward_deterministic():
    import torch
    net = build_cvae_net(h=32).eval()
    x = torch.randn(3, OBS, 2)
    with torch.no_grad():
        a = net(x)[0]
        b = net(x)[0]
    assert torch.allclose(a, b)                                    # 추론 비샘플링 → 결정적


def _batch():
    B = 16
    X = np.zeros((B, OBS, 2), np.float32)
    Y = np.zeros((B, PRED, 2), np.float32)
    for b in range(B):
        for i in range(OBS):
            X[b, i] = (0.2 * (i - (OBS - 1)), 0.0)
        for j in range(PRED):
            Y[b, j] = (0.2 * (j + 1), 0.0)
    return X, Y


def test_cvae_elbo_finite_and_kl_nonneg():
    import torch
    net = build_cvae_net(h=32)
    X, Y = _batch()
    out = net.elbo(torch.tensor(X), torch.tensor(Y), beta=1.0)
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["recon"]) and torch.isfinite(out["kl"])
    assert out["kl"].item() >= -1e-6                               # 범주형 KL ≥ 0


def test_cvae_elbo_overfits_one_batch():
    import torch
    torch.manual_seed(0)
    net = build_cvae_net(h=32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    X, Y = _batch()
    xt, yt = torch.tensor(X), torch.tensor(Y)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = net.elbo(xt, yt, beta=1.0)["loss"]
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first - 1.0                               # 학습 경로 정상
