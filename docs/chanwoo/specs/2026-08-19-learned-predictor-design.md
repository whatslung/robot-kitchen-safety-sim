# 학습형 멀티모달 예측기 — 설계 (확정)

> 2026-08-19 브레인스토밍 확정. 이슈 #2의 **4단계**. 2단계 궤적으로 학습해 3단계 베이스라인
> (등속·칼만·스테이션)을 이기는 **경량 멀티모달 LSTM**을 만든다. 5단계(브라우저 배포)의
> `window.__customPredictor` / `PRED.mix`(밀도 구름 시각화)에 꽂힐 형식으로 출력한다.

## 결정 (브레인스토밍)

| 결정 | 값 |
|---|---|
| 구조 | **경량 LSTM 인코더 + 멀티모달 혼합 헤드**(MTP/MultiPath 식). CVAE 대신 — 학습 안정·ONNX 자명·기존 GMM 시각화 정합 |
| 멀티모달 | **K=3 모드**(기존 viz maxModes·modeCols 3개와 일치). 각 모드 = 12스텝 경로 + 가중치 + 스텝별 σ |
| 목표 조건화 | **안 함.** 모델이 스스로 목표 분포를 배워 봉우리로 낸다 → 베이스라인을 공정하게 이기는 이야기 |
| 정규화 | 에이전트 중심 — 마지막 관측을 원점으로 평행이동 + 진행방향을 +x로 회전(정지 시 회전 생략). 예측 후 원좌표로 역변환 |
| 손실 | best-of-K 회귀(GT에 가장 가까운 모드만 회귀) + 모드 분류(그 모드로 CE). posterior collapse 없음 |
| 분할 | 3단계와 같은 val=seed%5(공정 비교). train=나머지 32 scene |
| 지표 | 최빈 모드 ADE/FDE(단봉 베이스라인과 비교) + minADE/FDE@K(멀티모달 이점) |

## 데이터 (2단계 로더 재사용)

- `trajectory/sim_traj.load_windows("train"/"val")` — 관측 8 / 예측 12, 월드 미터.
- 정규화: 윈도우마다 last-obs 원점 평행이동 + heading 회전(관측 첫→끝 벡터, 속력<eps면 θ=0).
  변환 (θ, origin)을 들고 있다가 예측을 역변환해 미터 원좌표에서 ADE/FDE.

## 모델 (torch)

- 입력: 정규화 관측 8스텝의 (x,z) (원점 상대). LSTM(input=2, hidden=H, batch_first).
- 헤드: 마지막 은닉 → `K·12·2`(모드별 경로 offset) + `K`(모드 로짓) + `K·12`(모드별 스텝 등방 log σ).
- 출력 해석: 경로는 정규화 프레임 offset → 역변환. 가중치=softmax(로짓). σ=exp(logσ).
- 규모: H≈64, 파라미터 수만~십수만. CUDA 있으면 GPU, 없으면 CPU(작아서 무방). seed 고정.

## 손실 (MTP)

각 샘플: GT에 ADE 최소인 모드 k\*를 고른다.
- 회귀: k\* 경로에 Gaussian NLL(예측 σ 사용) 또는 L2. σ가 지평선 따라 커지도록 NLL 채택.
- 분류: 모드 로짓에 CE(정답 = k\*). 이걸로 가중치가 "어느 모드가 맞을 확률"을 배운다.
- 학습된 모드가 서로 다른 갈래(스테이션)로 벌어지며 봉우리가 생긴다.

## 산출물

- `trajectory/learned_predictor.py` — 모델 정의 + `LearnedPredictor(weights)`.
  `.predict_modes(hist_m) -> [{path:[(t,x,z)], w, sigma:[…]}, …]`(미터). eval용 최빈 경로 어댑터.
- `train/train_traj_predictor.py` — 학습 → `training/traj_predictor/model.pt` 저장, val 지표 출력 +
  `docs/chanwoo/prediction-eval.md`에 학습형 행 추가.
- `tests/test_learned_predictor.py` — 형상·역변환·과적합 sanity(작은 배치 1개를 외워 손실↓).
- (5단계 준비) ONNX export 스크립트 — 결정적 forward(관측→K모드 params)를 export.
  onnxruntime 있으면 파리티 확인, 없으면 export만 하고 5단계로 미룸.

## 배포 정합 (5단계 전제, 여기선 형식만 맞춤)

- 브라우저 `__customPredictor(obs)`는 씬 AU + mPerAU를 준다 → 미터 변환 후 추론 → AU로 역변환.
- 최빈 모드 = `{path, sigma}` (현행 단일 경로 계약). 전체 K모드 = `PRED.mix`(밀도 구름) 확장용.
- 실제 배선·ONNX 브라우저 로드는 **5단계**. 여기선 모델이 이 형식을 낼 수 있음을 보증.

## 검증 (성공 기준)

1. **형상·역변환** — predict_modes가 K개 (12,·) 경로 반환, 역변환 왕복 오차 ~0.
2. **과적합 sanity** — 작은 배치를 외우면 train 손실이 확실히 내려간다(학습 경로 정상).
3. **베이스라인 대비** — val 최빈 모드 FDE < 등속·칼만. minFDE@3는 그보다 더 낮다.
   목표 미조건인데도 스테이션(목표 앎, FDE 1.28)에 근접/도달하면 성공.
4. **멀티모달** — 여러 목표가 가능한 관측에서 모드들이 서로 다른 스테이션으로 벌어진다(정성 확인).
5. **재현성** — seed 고정 시 학습 곡선·지표 재현(런 단위).

## 범위 밖

- 5단계 실제 브라우저 배선·ONNX 로드·AU 단위 왕복·`PRED.mix` 확장 UI.
- 이웃 상호작용(멀티에이전트 사회적 풀링) — 지금은 단일 에이전트. 데이터에 공유 타임라인이
  있으니 후속 확장 여지(스펙 2단계 참조).
