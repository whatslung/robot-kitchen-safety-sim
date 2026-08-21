# docs/chanwoo — 작업 문서 색인

조리로봇 안전: **검출 → 추적 → 궤적 예측 → 선제 안전** 파이프라인의 설계·평가·인계 문서.
파이프라인 실행법은 리포 루트 [README](../../README.md) "검출·예측 파이프라인" 참조.

## 시작점
- **[HANDOFF.md](HANDOFF.md)** — 롤링 핸드오프(현재 상태·다음 단계). **여기부터 읽는다.**

## 검출 (YOLO, 나디르 top-down)
- [detection-eval.md](detection-eval.md) — 검출 학습·평가(sim in-domain / 실사 / 3-way)

## 궤적 예측 (이슈 #2)
- [prediction-eval.md](prediction-eval.md) — 베이스라인 vs 학습형 ADE/FDE 비교표(자동 생성)
- [prediction-safety-eval.md](prediction-safety-eval.md) — 정지반경 진입 예측 recall/precision(안전 지표, 자동 생성)
- [hf-model-card-human-move-lstm.md](hf-model-card-human-move-lstm.md) — HF 공개 모델 [`chanubc/human-move-lstm`](https://huggingface.co/chanubc/human-move-lstm) 카드 원본
- [prediction-sim2real-notes.md](prediction-sim2real-notes.md) — Trajectron++ 조사·레이아웃 다양성·
  검출 노이즈 증강·안전 결정(recall/precision)·운영점 튜닝·**실사 zero-shot 전이** 결론

## 설계 스펙 (specs/) — 예측 5단계
- [1단계 스테이션 전이](specs/2026-08-20-station-transition-design.md) — 직무 사이클 확률 전이 동선
- [2단계 궤적 수집](specs/2026-08-19-trajectory-collection-design.md) — 고정 dt 하네스·자기기술 scene
- [3단계 베이스라인](specs/2026-08-19-baseline-ade-fde-design.md) — 등속·칼만·스테이션 휴리스틱
- [4단계 학습형](specs/2026-08-19-learned-predictor-design.md) — 경량 LSTM + 멀티모달 MTP 헤드
- [5단계 배포](specs/2026-08-20-predictor-deploy-design.md) — /predict + __customPredictor
- 예측 시각화: [예측 뷰](specs/2026-08-18-prediction-viz-design.md) · [바닥 밀도 구름](specs/2026-08-19-density-floor-design.md)

## 계획 (plans/) · 세션 핸드오프 (handoff/)
- `plans/` — 단계별 실행 계획
- `handoff/` — 세션별 인계 기록

## 참고 조사
- [tensorrt-model-connect-research.md](tensorrt-model-connect-research.md) — NVIDIA TRTMC 조사
  (ONNX 대체가 아님 — 결론 요약)
