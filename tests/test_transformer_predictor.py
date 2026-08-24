"""Transformer 백본 궤적 예측기 — 출력 계약·과적합 sanity (LSTM과 동일 head 계약)."""
import numpy as np

from trajectory.learned_predictor import (build_transformer_net, mtp_loss,
                                          LearnedPredictor, OBS, PRED, K)


def test_transformer_predict_modes_shape():
    lp = LearnedPredictor(net=build_transformer_net(), device="cpu")   # 미학습 net
    hist = [(0.2 * i, 0.0) for i in range(OBS)]
    modes = lp.predict_modes(hist)
    assert len(modes) == K
    assert all(len(m["path"]) == PRED and len(m["sigma"]) == PRED for m in modes)
    assert abs(sum(m["w"] for m in modes) - 1.0) < 1e-5           # softmax 합=1
    assert all(modes[i]["w"] >= modes[i + 1]["w"] for i in range(K - 1))  # 내림차순


def test_transformer_forward_shapes():
    import torch
    net = build_transformer_net(h=32)
    x = torch.zeros(4, OBS, 2)
    paths, logits, logsig = net(x)
    assert paths.shape == (4, K, PRED, 2)
    assert logits.shape == (4, K)
    assert logsig.shape == (4, K, PRED)


def test_transformer_overfits_one_batch():
    import torch
    torch.manual_seed(0)
    net = build_transformer_net(h=32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    B = 16
    X = np.zeros((B, OBS, 2), np.float32)
    Y = np.zeros((B, PRED, 2), np.float32)
    for b in range(B):
        for i in range(OBS):
            X[b, i] = (0.2 * (i - (OBS - 1)), 0.0)
        for j in range(PRED):
            Y[b, j] = (0.2 * (j + 1), 0.0)
    xt, yt = torch.tensor(X), torch.tensor(Y)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = mtp_loss(*net(xt), yt)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first - 1.0                              # 학습 경로 정상
