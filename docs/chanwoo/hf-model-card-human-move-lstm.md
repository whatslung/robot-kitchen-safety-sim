---
license: mit
tags:
  - trajectory-prediction
  - human-motion
  - lstm
  - multimodal
  - robotics
  - safety
library_name: pytorch
pipeline_tag: other
---

# human-move-lstm — 사람 이동 궤적 멀티모달 예측기

급식 조리로봇 안전 셀 시뮬레이터([robot-kitchen-safety-sim](https://github.com/whatslung/robot-kitchen-safety-sim))에서
**사람의 다음 이동 경로를 예측**해 로봇이 선제적으로 감속·정지하도록 하는 경량 예측 모델입니다.

## 구조

- **인코더**: 경량 LSTM (입력 2차원 `(x, z)`, 은닉 64)
- **헤드**: MLP 혼합 헤드(MTP, Multiple-Trajectory Prediction) → `K=3`개 미래 모드
- 각 모드 = 미래 경로 `(PRED=12스텝 × (x,z))` + 모드 가중치 `w` + 스텝별 불확실성 `σ`
- **정규화**: 에이전트 중심(마지막 관측을 원점, 진행 방향을 +x로 회전) → 예측 후 원좌표로 역변환
- **학습 손실**: best-of-K 가우시안 NLL + 모드 분류 교차엔트로피

## 입출력 계약

- **입력**: 관측 `OBS=8`스텝, 각 `(x, z)` 좌표. 약 0.4초 간격 리샘플. 단위 = 미터(기본 스케일에서 1 scene-unit = 1 m).
- **출력**: `K=3`개 모드 리스트. 각 `{ path: [[x,z]×12], w, sigma: [×12] }`. 가중치 내림차순.

## 평가 지표

평가 조건: val 스플릿(`seed % 5 == 0`), 관측 8스텝(3.2s) → 예측 12스텝(4.8s), 합성 궤적.
베이스라인(등속·칼만)과 동일한 val 윈도우에서 측정. **낮을수록 좋은 값 ↓, 높을수록 좋은 값 ↑.**

### 위치 오차 — ADE / FDE (전체 val 윈도우 8,646)

| 예측기 | ADE(m) ↓ | FDE(m) ↓ |
|---|---|---|
| 등속 (const-vel) | 1.114 | 2.129 |
| 칼만 (Kalman) | 1.031 | 2.069 |
| **학습형 LSTM (최빈 모드)** | **0.748** | **1.420** |
| 학습형 LSTM (minADE@3) | 0.432 | 0.797 |
| _참고: 스테이션(목표 앎)_ | _0.694_ | _1.233_ |

- **ADE**: 12스텝 예측 위치오차 평균(m). **FDE**: 12스텝째(4.8s 뒤) 최종 위치오차(m).
- **최빈 모드**: 가중치 최상위 단일 모드 — 단봉 베이스라인과 직접 비교하는 대표값.
- **minADE@3**: K=3 모드 중 최선 — 멀티모달이 정답 갈래를 담는지(상한).
- 목표를 모르는 학습형이 등속·칼만을 크게 이기고, 목표를 아는 스테이션에 근접.

### 안전 recall — 정지반경 진입 예측 (R = 3.1 m)

"지금 정지반경 **밖**에 있는 사람이 4.8s 안에 반경 안으로 진입하는지"를 미리 맞혔나.
대상(진입 전 밖) val 윈도우 5,086 · 실제 진입 1,199. 선제 안전층이라 **recall(놓치면 충돌) 우선**.

| 예측기 | recall ↑ | precision ↑ |
|---|---|---|
| 등속 (const-vel) | 0.164 | 0.883 |
| 칼만 (Kalman) | 0.248 | 0.911 |
| **학습형 LSTM (최빈 모드)** | **0.433** | 0.707 |
| 학습형 LSTM (전 모드 합집합) | 0.756 | 0.442 |

- **recall**: 실제 진입 중 미리 잡은 비율(놓치면 충돌). **precision**: 경보 중 진짜 비율(낮으면 헛정지).
- 반응형(예측 없음)은 이 윈도우에서 recall = 0(지금 밖이라 진입을 못 봄) — 예측기의 값어치가 여기서 드러난다.
- 멀티모달 **전 모드 합집합**은 여러 갈래를 다 경계해 recall이 가장 높다(헛정지는 늘어남 → 운영점 τ로 조절).

> 재현: `train/eval_traj_baselines.py`·`train/train_traj_predictor.py`(ADE/FDE), `train/eval_traj_safety.py`(recall).
> 상세: `docs/chanwoo/prediction-eval.md`(ADE/FDE), `docs/chanwoo/prediction-safety-eval.md`(recall).

## 파일

| 파일 | 용도 |
|---|---|
| `model.pt` | PyTorch 가중치(`state_dict`). 백엔드 서빙용(권장). |
| `model.onnx` | ONNX export. 인브라우저/타 런타임 추론용(옵션). |

## 사용

```python
from huggingface_hub import hf_hub_download
from trajectory.learned_predictor import LearnedPredictor   # 리포의 trajectory 모듈

w = hf_hub_download("chanubc/human-move-lstm", "model.pt")
pred = LearnedPredictor(weights_path=w, device="cpu")
modes = pred.predict_modes([[0,0],[0.1,0],[0.2,0],[0.3,0],[0.4,0],[0.5,0],[0.6,0],[0.7,0]])
# → [{"path": [[x,z]…12], "w": .., "sigma": […12]}, … 3개]
```

시뮬레이터 백엔드(`backend/detect_server.py`)는 로컬 가중치가 없으면 이 저장소에서 자동으로 내려받습니다.

## 재현

학습·export 스크립트는 리포에 있습니다: `train/train_traj_predictor.py`, `train/export_traj_onnx.py`.
설계 문서: `docs/chanwoo/specs/2026-08-19-learned-predictor-design.md`.

## 라이선스

MIT. 시뮬레이션(합성) 궤적으로 학습된 연구/데모용 모델입니다.
