"""이슈 #2 4→5단계 — 학습형 예측기를 ONNX로 export(브라우저 배포용).

결정적 forward(정규화 관측 (B,OBS,2) → paths (B,K,PRED,2)·logits (B,K)·logsig (B,K,PRED))만
내보낸다. 정규화·역변환·softmax·샘플링은 JS(5단계)에서 한다. onnxruntime이 있으면 파리티 확인.

실행:  uv run --with onnxscript --with onnxruntime python train/export_traj_onnx.py
       (torch 2.11 export는 onnxscript 필요 · onnxruntime은 파리티 확인용. 둘 다 배포 전용이라
        프로젝트 의존성엔 안 넣고 --with 로 일회성 사용.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trajectory.learned_predictor import build_net, OBS
from trajectory.sim_traj import load_windows
from trajectory.learned_predictor import frame_of, to_frame

WEIGHTS = ROOT / "training" / "traj_predictor" / "model.pt"
OUT = ROOT / "training" / "traj_predictor" / "model.onnx"


def main():
    import torch
    net = build_net()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    net.eval()

    dummy = torch.zeros(1, OBS, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net, dummy, str(OUT),
        input_names=["obs"], output_names=["paths", "logits", "logsig"],
        dynamic_axes={"obs": {0: "batch"}, "paths": {0: "batch"},
                      "logits": {0: "batch"}, "logsig": {0: "batch"}},
        opset_version=17,
    )
    # 단일 파일화 — torch 2.11 exporter는 가중치를 model.onnx.data 로 분리한다.
    # 브라우저(onnxruntime-web)는 단일 self-contained .onnx 가 다루기 쉬우므로 임베드해 다시 쓴다.
    import onnx
    m = onnx.load(str(OUT))                              # .data 를 함께 읽어들임
    onnx.save_model(m, str(OUT), save_as_external_data=False)
    ext = OUT.with_name(OUT.name + ".data")
    if ext.exists():
        ext.unlink()
    print(f"ONNX 저장(단일 파일) → {OUT} ({OUT.stat().st_size} bytes)")

    # 파리티 — onnxruntime 있으면 torch 출력과 비교
    try:
        import onnxruntime as ort
    except Exception as e:                              # noqa: BLE001
        print(f"onnxruntime 없음 → 파리티 생략(5단계에서 확인). ({e})")
        return
    w = load_windows("val")[0]
    hist = [(p[1], p[2]) for p in w.scene.agents[0].history]
    origin, ang = frame_of(hist)
    x = to_frame(hist, origin, ang).astype(np.float32)[None]
    with torch.no_grad():
        tp, tl, ts = (t.numpy() for t in net(torch.tensor(x)))
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    op, ol, os_ = sess.run(None, {"obs": x})
    err = max(np.abs(tp - op).max(), np.abs(tl - ol).max(), np.abs(ts - os_).max())
    print(f"torch vs onnxruntime 최대 오차: {err:.2e}  {'OK' if err < 1e-4 else '⚠불일치'}")


if __name__ == "__main__":
    main()
