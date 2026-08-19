"""학습형 예측기 — 정규화 역변환·출력 형상·과적합 sanity. 이슈 #2 4단계."""
import numpy as np

from trajectory.learned_predictor import (frame_of, to_frame, from_frame,
                                          build_net, mtp_loss, LearnedPredictor,
                                          OBS, PRED, K)


def test_frame_roundtrip():
    hist = [(1.0, 2.0), (1.3, 2.1), (1.7, 2.0), (2.2, 1.8),
            (2.6, 1.7), (3.1, 1.6), (3.5, 1.6), (3.9, 1.5)]
    origin, ang = frame_of(hist)
    n = to_frame(hist, origin, ang)
    back = from_frame(n, origin, ang)
    assert np.allclose(back, np.asarray(hist, float), atol=1e-9)
    assert np.allclose(n[-1], [0.0, 0.0], atol=1e-9)          # 마지막 관측 = 원점


def test_predict_modes_shape():
    lp = LearnedPredictor(net=build_net(), device="cpu")      # 미학습 net
    hist = [(0.2 * i, 0.0) for i in range(OBS)]
    modes = lp.predict_modes(hist)
    assert len(modes) == K
    assert all(len(m["path"]) == PRED and len(m["sigma"]) == PRED for m in modes)
    assert abs(sum(m["w"] for m in modes) - 1.0) < 1e-5       # softmax 가중치 합=1
    assert all(modes[i]["w"] >= modes[i + 1]["w"] for i in range(K - 1))  # 내림차순


def test_overfits_one_batch():
    import torch
    torch.manual_seed(0)
    net = build_net(h=32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    # 작은 합성 배치: 오른쪽으로 등속 → 미래도 직진
    B = 16
    X = np.zeros((B, OBS, 2), np.float32)
    Y = np.zeros((B, PRED, 2), np.float32)
    for b in range(B):
        for i in range(OBS):
            X[b, i] = (0.2 * (i - (OBS - 1)), 0.0)            # 원점 기준 과거
        for j in range(PRED):
            Y[b, j] = (0.2 * (j + 1), 0.0)                    # 앞으로 계속 직진
    xt, yt = torch.tensor(X), torch.tensor(Y)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = mtp_loss(*net(xt), yt)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first - 1.0                          # 확실히 내려간다(학습 경로 정상)
