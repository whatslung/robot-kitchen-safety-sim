# 발표 에셋 — before/after 결과 그래프

캡스톤 최종 발표용 결과 시각화. **모든 수치는 저장소 정본 리포트 실측**이며, 각 그래프에
출처 파일·도메인(SIM/REAL)을 명시했다. 잡스식 무대 톤(검정 배경·엠버/틸).

재생성: `python scripts/make_charts.py` (matplotlib + Malgun Gothic 필요)

## 그래프 목록 (`charts/`)

| 파일 | 내용 | 핵심 수치 | 도메인 | 출처 |
|---|---|---|---|---|
| `1_person_finetune.png` | 천장 사람 검출 · 기성 vs 파인튜닝 | R 0.175 → 0.871 · P 0.374 → 0.872 | 합성 val | detection-eval.md §1·§6 |
| `2_person_fusion_ladder.png` | 4분할·월드융합·추적 사다리 | recall 0.831 → 0.988 | 합성/존 | nadir-zone-fusion.md §5-7~§5-9 |
| `3_person_real_transfer.png` | 실사 전이(정직성) | R 0.270 → 0.844 (RF-DETR 0.917) | **실사** test 137 | detection-eval.md §3·§5-2 |
| `4_pred_entry_recall.png` | 위험 진입 예측 recall ⭐ | 57.1% → 92.9% (FN48→FN8) | 풀 파이프라인 4대 | nadir-zone-fusion.md §5-15 |
| `5_pred_ade.png` | 예측 오차 ADE@1.6s | CV 0.372 → Transformer 0.217 m | test | prediction-eval.md 1.6s |
| `6_fire_ablation.png` | 화재 ablation(불을 보는가) | 0.81 → 0.005 | 합성 | kitchen-fire-noise-poc |
| `7_fire_noise_robust.png` | 노이즈 붕괴 → 학습 회복 | 저조도 0.31→0.79 등 | 합성 | kitchen-fire-noise-poc TIMELINE |
| `8_fire_transfer_honesty.png` | 합성 in-domain vs 실사 전이 | 0.81 → 0.31 (한계) | 실사 화재 | kitchen-fire-noise-poc SUMMARY |

## ⚠️ 발표 전 반드시 확인할 것

1. **도메인 구분** — 0.871·0.988은 **합성(SIM)** 수치다. 실사 사람 검출은 `3_person_real_transfer`
   기준(sim-only 0.270 붕괴 → real+sim 0.844)이며 **실사 검출 회복은 남은 과제**다.
   발표에서 SIM 수치를 실사인 것처럼 말하지 않는다.

2. **화재 수치 정정** — 기존 덱의 화재 "현장 닮은 실데이터 0.85~0.90", "하드네거티브
   fpr 0.46→0.066(7배)"는 `kitchen-fire-noise-poc` 정본 리포트에 **없다**(0.85~0.90은 사람 검출
   수치로 보임). 화재 정본 서사 = **합성 in-domain은 되지만(0.81) 실사 전이는 약함(0.31) →
   그래서 시뮬(v3) 커리큘럼**. `6·7·8`번 그래프가 이 정본 서사를 따른다.

## 미완 — 박스 검출 이미지 (raw 프레임 필요)

"검출 박스 친 이미지 before/after"는 **원본(raw) 프레임**이 있어야 생성 가능. 현재 로컬에 없음:
- **실사**: Roboflow `overhead-person-szky0` 원본 필요 (`.env`의 `ROBOFLOW_API_KEY`) — 채우면
  기성 vs 실사모델(HF `chanubc/overhead-person-yolo11`) before/after 자동 생성.
- **합성**: `sim.html`(island 나디르 CAM11)에서 오버레이 없는 프레임 캡처 필요 — 기성 vs
  나디르모델(HF `chanubc/robot-kitchen-nadir-yolo11s`) before/after 생성.
- 모델·추론 파이프라인은 로컬에서 동작 확인됨(Colab 불필요). `assets/samples/*.jpg`는 이미
  박스가 구워진 데모 출력이라 입력으로 못 씀.
