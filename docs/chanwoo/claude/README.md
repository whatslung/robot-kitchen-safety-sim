# 발표 에셋 — Claude 팩

캡스톤 발표 개편용 결과 이미지. `docs/chanwoo/codex/`(병렬 팩)와 별도로 Claude가 정리했다.
모든 수치는 저장소 정본/팀 리포트 실측이며 각 항목에 도메인(SIM/REAL)·출처를 명시한다.
잡스식 무대 톤(검정 배경·엠버/틸). 재생성 스크립트는 `scripts/`.

## 화재 (`fire/`)

- `fire_before_after.png` — 합성-only vs 실사 학습 (동일 Indoor Fire Smoke grouped test)
  - recall **23.7% → 89.9%**, precision **84.2% → 97.9%**
  - 출처: `kitchen-fire-noise-poc` commit `b0c9d726` · `AFTER_meeting.md §5B` · YOLOv8s 60ep · conf 0.25 · frame-level · fire test 358장
  - 정본 서사: 합성만으론 실사에서 약함 → **현장 실사 화재 데이터로 학습해 0.899 달성**.
  - 생성: `scripts/fire_chart.py`
- 화재 검출 **박스 이미지**는 미확보 — 권위 체크포인트가 다른 Google Drive에 있어 Colab 재현 필요
  (codex `fire/FIRE_BOX_ASSET_STATUS.txt` 참조). 임의 생성 금지.

## 천장 사람 검출 (`person/`)

| 파일 | 내용 | 수치 | 도메인 |
|---|---|---|---|
| `person_finetune.png` | 기성 vs 파인튜닝 | R 0.175 → 0.871 · P 0.374 → 0.872 | 합성 val |
| `person_fusion_ladder.png` | 4분할·월드융합·추적 사다리 | R 0.831 → 0.988 | 합성/존 |
| `person_real_transfer.png` | 실사 전이(정직성) | R 0.270 → 0.844 (RF-DETR 0.917) | **실사** test 137 |
| `sim_person_ba.png` | 검출 박스 · 기성 vs 나디르 4분할 (인상용) | **0명 → 13명** | 합성 나디르(무화재) |
| `sim_person_verified.png` | 검출 박스 · **GT 대조 검증** (방 전체·잘림 0) | **0명 → 8/8 전원** (FP 0·누락 0) | 합성 나디르(무화재) |
| `sim_zone4_exploded.png` | 4구역 타일별 재검출 | 6·5·4·1명 | 합성 나디르 |
| `real_person_ba.png` | 검출 박스 · 기성 vs 실사모델 | **0명 → 13명** | 실사 overhead test 원본 |

- 박스별 `person 0.XX` 라벨 표기. 모델 클래스는 `person`(head 아님).
- 생성: `scripts/infer_boxes.py`(박스), `scripts/split4_detect.py`(4분할), `scripts/make_charts.py`(차트).
- ⚠️ **도메인 구분**: 0.871·0.988은 **합성(SIM)** 수치. 실사 사람 검출은 `person_real_transfer` 기준
  (sim-only 0.270 붕괴 → real+sim 0.844)이며 **실사 검출 회복은 남은 과제**. SIM을 실사처럼 말하지 않는다.
- **박스 이미지 두 버전**: `sim_person_ba`(13명·인상용)와 `sim_person_verified`(8/8·GT 검증). 단일
  정지 프레임은 밀집 시 1~2명을 놓칠 수 있고, 그 놓침을 프레임 간 추적(ByteTrack+칼만 coast)이
  메워 배포 recall 0.988이 된다 → "안 놓친다"의 정량 근거는 `person_fusion_ladder`.

## 움직임 예측 (`motion/`)

| 파일 | 내용 | 수치 | 출처 |
|---|---|---|---|
| `pred_entry_recall.png` | 위험 진입 예측 recall ⭐ | 57.1% → 92.9% (FN48→FN8) | nadir-zone-fusion.md §5-15 |
| `pred_ade.png` | 예측 오차 ADE@1.6s | CV 0.372 → Transformer 0.217 m | prediction-eval.md 1.6s test |

- "다음 위치 예측(미래 경로)" 시각화는 codex `future/lstm_three_future_prediction.png`에 있음(실 LSTM 추론).

## 참고 — codex 팩과의 관계

- `docs/chanwoo/codex/`는 병렬 팩(화재 0.899, 미래 예측 viz, 궤적 정확도 등). 두 팩은 별도 유지.
- 화재 정본 수치(0.899)는 codex와 동일 소스로 맞췄다.
