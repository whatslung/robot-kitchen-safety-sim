# 이산 잠재 CVAE 궤적 예측 — Trajectron++식 (옵션 B 임팩트판)

> 작성 2026-08-24 · 담당 chanwoo · 상태: **설계 승인 — 구현 착수** (사용자: 머지 후 착수·Z=3)
> 선행: P0-1 split·CI·백본비교(PR #19, main 머지) · 설계 재사용.
> 배경: 백본 비교(Transformer)는 정확도·안전을 끌어올렸으나 **진입 recall이 여전히 부족**(~0.58).
> 진짜 임팩트는 멀티모달 학습(minADE@3 FDE 0.20 — 잠재력 있음)을 top-1·recall로 끌어오는 것.

## 0. 목표

> **best-of-K(MTP) 학습을 Trajectron++식 이산 잠재 CVAE(ELBO)로 바꿔, 같은 모드 예산(Z=3)·같은
> 추론 계약·같은 split/eval/CI에서 멀티모달 품질과 진입 recall을 끌어올린다.**

## 1. 왜 이산 잠재인가 (연속 CVAE의 위험을 피함)

- **결정적 추론**: z가 이산(Z=3)이라 모든 z를 **열거**(샘플링 X) → K모드. 우리 결정적 eval·bootstrap CI에 그대로.
- **결정적 학습**: Z가 작아 z를 **정확히 주변화**(recon=Σ_z q(z)·NLL, KL=범주형). Gumbel/샘플링 불필요.
- **posterior collapse 완화**: 범주형 KL이 가우시안 KL보다 안정적.
- **드롭인 추론 계약**: forward(x) → `(paths(B,Z,PRED,2), logits(B,Z), logsig(B,Z,PRED))` — 기존 head와
  동형(Z=K) → `LearnedPredictor`·`eval_traj_split` **무수정 재사용**. 학습만 다르다.

## 2. 아키텍처 (`trajectory/learned_predictor.py`)

`build_cvae_net(h=64, z=K, pred=PRED, layers=2, heads=4)` — nn.Module:
- **과거 인코더**: Transformer 인코더(백본 비교 승자 재사용) → 컨텍스트 c=(B,h)(마지막 관측 토큰).
- **사전분포** p(z|past): `Linear(h→Z)` → prior_logits (B,Z).
- **인식망**(학습 전용) q(z|past,future): 미래 12스텝을 작은 인코더(Linear/GRU)로 요약 f=(B,h) →
  `Linear([c;f]→Z)` → post_logits (B,Z).
- **디코더** decode(c, z): z 임베딩(Embedding(Z,h)) + c → MLP → path(PRED×2) + logsig(PRED). 전 z 벡터화로
  paths=(B,Z,PRED,2)·logsig=(B,Z,PRED).
- **forward(x)** = 추론: c → prior_logits, 전 z 디코드 → (paths, prior_logits, logsig). (인식망 미사용.)
- **elbo(x, gt, beta)** = 학습: c → prior_logits·post_logits(gt 사용)·전 z 디코드.
  - per-z NLL n_z (B,Z) = mean_pred(0.5·‖path_z−gt‖²/σ² + 2·logσ)  (기존 mtp NLL 형식).
  - q=softmax(post_logits). **recon = Σ_z q(z)·n_z** (B,). **KL = Σ_z q·(log q − log p)** (범주형, B,).
  - loss = recon.mean() + beta·KL.mean().

## 3. 학습 (`train/train_traj_cvae.py`, 신규)

- `train_traj_predictor.build_xy` 재사용(같은 train split·노이즈 증강) + 동일 루프·SEED=0(결정적).
- **β KL 어닐링**: 초반 β≈0(디코더 먼저 학습) → 선형 증가하여 목표 β(예: 1.0)까지. posterior collapse 방지.
- 저장 → `training/traj_predictor/model_cvae.pt`. 문서 미작성(eval_traj_split 단독 소유).

## 4. 평가 (`eval_traj_split.py` — 한 줄 확장)

- `learned_predictors()`에 model_cvae.pt 있으면 `LearnedPredictor(net=build_cvae_net(), weights=...)` 추가
  → "학습형 CVAE(최빈)"·"(minADE@3)"·안전 "(전모드)" 행이 val/test·CI·1.6s 표에 자동 등장.
- LSTM·Transformer·CVAE 3-way 비교. **같은 Z=3·같은 head 계약** → 차이는 **학습 목적함수(MTP vs ELBO)**.

## 5. 테스트 (TDD seam) — `tests/test_cvae_predictor.py`

- forward 출력 계약: predict_modes → K모드·각 PRED길이·softmax 합=1·내림차순(미학습 net).
- forward 형상: paths(B,Z,PRED,2)/logits(B,Z)/logsig(B,Z,PRED).
- **elbo overfit sanity**: 합성 배치에서 elbo가 확실히 하강(학습 경로 정상).
- **결정성**: 같은 입력 forward 2회 동일(추론 비샘플링).
- KL·recon 비음수/유한성 소규모 확인.

## 6. 완료 정의

- CVAE가 LSTM·Transformer와 **같은 Z=3·head·split·eval/CI**로 학습·평가되어 3-way 표에 등장.
- val(선택)·test(1회) ADE/FDE(4.8s·1.6s)·안전 recall/precision + scene-level 95% CI.
- 순수 net 테스트 통과, 전체 스위트 green.
- 정직 경계 명시: "Trajectron++ 실행"이 아니라 "**Trajectron++식 이산 잠재 CVAE 학습**"(graph/map/dynamics 없음).

## 7. 기대·후속

- 기대: 멀티모달 책임 분배(ELBO)로 top-1(최빈) 품질·진입 recall이 MTP 대비 개선될 여지. 검증은 수치로.
- 실패 시 정직 보고(ELBO가 MTP를 못 이기면 그대로 기록 — 음성 결과도 결과).
- 후속: 목표 조건화, 상호작용(graph), 실사 미세조정(P0-2 격차), 운영점 τ 스윕.
