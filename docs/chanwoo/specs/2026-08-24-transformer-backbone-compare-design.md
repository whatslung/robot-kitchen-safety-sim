# 검증된 구조 비교 — Transformer 백본 vs LSTM (궤적 예측)

> 작성 2026-08-24 · 담당 chanwoo · 상태: **설계 승인 — 구현 착수**
> 맥락: "검증된 오픈 궤적모델과 비교" 요청 → 이 분야엔 받아 쓰는 사전학습 모델이 없어(연구용
> repo·옛 torch·가중치 미제공), **검증된 구조를 우리 데이터로 학습해 공정 비교**(옵션 B)로 합니다.
> P0-1([2026-08-24-traj-split-ci-design.md](2026-08-24-traj-split-ci-design.md)) split·eval·CI 재사용.

## 0. 목표

> **우리 LSTM 백본을, 검증된 Transformer 백본으로 한 가지만 바꿔, 같은 데이터·같은 평가에서
> 공정하게 비교한다.**

Trajectron++를 그대로 돌리는 게 아니다(그 repo는 CVAE+dynamics+maps+graph 대형 시스템·torch 1.x·
공개 가중치 없음 → 이 환경에서 비현실적). Trajectron++의 **핵심 원리(멀티모달 K모드 + 불확실성 σ)**는
우리 헤드가 이미 담고 있으므로, **temporal 백본만 LSTM → Transformer**로 교체해 원인을 명확히 한다
(Giuliari 2020, Transformer for trajectory forecasting 계열).

## 1. 범위 (확정)

- **바꾸는 것: 인코더 1개뿐.** 헤드(K*pred*2 + K + K*pred), MTP 손실, ego 정규화(frame_of/to_frame),
  학습 루프, split, eval, CI 전부 동일 → "차이 = 백본"이 깨끗하다.
- **결정적**(샘플링 없음) → P0-1 결정적 eval·scene-level CI에 그대로 한 행 추가.
- **정직 경계**: "Trajectron++를 실행했다"가 아니라 "Trajectron++식 멀티모달+불확실성 출력에
  Transformer 백본"이다. 결과 문서에 명시.

## 2. 컴포넌트

### 2-1. `trajectory/learned_predictor.py` — `build_transformer_net(h, k, pred, layers=2, heads=4)`
- 입력 (B,OBS,2) → Linear(2→h) + 위치 인코딩 → `nn.TransformerEncoder`(layers층, heads헤드,
  batch_first) → 마지막 관측 토큰(또는 mean pool) → 기존과 **동일한 head** → (paths, logits, logsig).
- forward 출력 계약은 `build_net`과 동일(형상 (B,K,PRED,2)/(B,K)/(B,K,PRED)) → `LearnedPredictor`·
  `mtp_loss`·`predict_batch` 그대로 재사용.

### 2-2. `train/train_traj_transformer.py` (신규, LSTM 트레이너 무수정)
- `train_traj_predictor.build_xy` 재사용(같은 train split·노이즈 증강) + `mtp_loss` + 동일 루프
  (SEED=0 결정적) → `training/traj_predictor/model_transformer.pt` 저장. **docs는 쓰지 않는다**
  (문서는 eval_traj_split.py가 단독 소유 — 클로버 방지).

### 2-3. `train/eval_traj_split.py` — 비교 행 추가
- transformer 가중치가 있으면 두 번째 `LearnedPredictor(net=build_transformer_net(), weights=...)`로
  "학습형 Transformer(최빈)"·minADE 행을 val/test·CI 표에 추가. 없으면 우아하게 생략(로그).

## 3. 테스트 (TDD seam)

- **`tests/test_transformer_predictor.py`**: (a) 출력 형상이 head 계약과 일치(predict_modes → K모드·
  각 PRED길이·softmax 합=1·내림차순), (b) 작은 합성 배치 overfit(loss 확실히 하강) — test_learned_predictor
  의 transformer 판. 순수 net 레벨.
- 전체: `uv run --group serve --with pytest python -m pytest tests/ -q`.

## 4. 재현 · 산출

```
uv run --group serve python train/train_traj_predictor.py       # LSTM (기존)
uv run --group serve python train/train_traj_transformer.py     # Transformer (신규)
uv run --group serve python train/eval_traj_split.py            # val/test·CI 비교표
```
- 산출: prediction-eval.md·prediction-safety-eval.md에 Transformer 행 추가, results/traj-split-eval.json.

## 5. 완료 정의

- Transformer 백본이 LSTM과 **동일 head·손실·정규화·split·eval**로 학습·평가됨(공정).
- val/test ADE·FDE·안전 recall/precision + scene-level 95% CI 표에 LSTM vs Transformer 나란히.
- 순수 net 테스트 통과, 전체 스위트 green.
- 문서에 "백본 교체 비교(≠ Trajectron++ 실행)" 경계 명시.

## 6. 후속 (사용자 지적 — 진짜 임팩트는 CVAE)

- **CVAE(Trajectron++ 근사) 후속**: 잠재변수 샘플링 멀티모달 → minADE@N. 별도 스펙. 샘플링 eval·
  KL 튜닝(posterior collapse)·비결정 CI 배선이 필요해 이번 범위 밖. 이 Transformer A/B가 그 발판.
- 실사 미세조정(P0-2 격차)·목표 조건화도 후속.
