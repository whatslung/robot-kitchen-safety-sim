# 발표 최종 에셋 (통합본)

캡스톤 발표에 실제로 들어갈 에셋을 **주제별 한 폴더로 통합**한 것이다.
원본은 `docs/chanwoo/{claude,codex,assets}/`에 그대로 남아 있고, 여기는 복사본이다.

- 화재·사람·움직임처럼 **claude 팩과 codex 팩이 같은 주제를 각자 렌더**한 경우, 두 버전을
  `-claude` / `-codex` 접미사로 **함께** 담았다. 발표 때 톤에 맞는 쪽을 고르면 된다.
  - claude 팩 = 검정 배경·엠버/틸의 무대 톤(잡스식). codex 팩 = 담백한 증거 톤.
- 모든 수치는 저장소 정본·팀 리포트 실측이며, 각 항목에 **도메인(SIM=시뮬 / REAL=실사)**과 출처를 붙였다.

## 폴더 지도

| 폴더 | 주제 | 슬라이드 쓰임 |
|---|---|---|
| `00-overview-pipeline/` | 파이프라인·방법론 그림 14장 (교수님 발표용 최종본) | 개요·방법 슬라이드 |
| `01-fire/` | 화재 검출 before/after | 화재 성능 |
| `02-person/` | 천장(나디르) 사람 검출 | 사람 검출 성능 |
| `03-motion-prediction/` | 위험진입 예측 recall·궤적 오차 차트 | "왜 학습형 예측" 방어 |
| `04-future-prediction/` | 다음 위치(미래 경로) 예측 시각화 | 예측 데모 |
| `05-architecture/` | 예측기 인코더-디코더 구조도 | 슬라이드 09 |
| `06-history-handoff/` | 동선 히스토리 나디르 + Figure Labs 핸드오프 | 슬라이드 08 |
| `07-extra-evidence/` | 나디르 평가·학습 결과 등 보충 증거 (큐레이션 밖) | 백업 슬라이드/부록 |

---

## 00 · 파이프라인 개요 (`00-overview-pipeline/`)

이동경로 예측 안전 파이프라인 발표 그림. 실제 sim 스크린샷 + 데이터 근거 차트. 모두 `@2x` 고해상도.

| 파일 | 내용 | 방어 질문 |
|---|---|---|
| `fig1_pipeline@2x.png` | 5단계 개요 (입력→검출→추적→예측→안전) | "이 시스템이 뭘 하나" |
| `fig2_why_6_cameras@2x.png` | 안전링 커버리지 4대 vs 6대 (24 seed) | "카메라 왜 6대?" |
| `fig3_why_learned_prediction@2x.png` | 위험진입 recall — CV/Kalman vs LSTM/Transformer | "직선 예측이면 안 되나?" |
| `fig4_dataset_sample@2x.png` | 궤적 학습 데이터 샘플 (obs8→pred12) | "무슨 데이터로 학습했나" |
| `fig5_world_fusion@2x.png` | 멀티카메라 월드 융합 (아핀→바닥좌표→병합→트랙) | "여러 카메라를 어떻게 합치나" |
| `fig6_fusion_ladder@2x.png` | 검출 융합 사다리 (단일 0.85→공간 0.89→+시간축 0.95) | "정확도를 어떻게 올렸나 / 왜 시간축" |
| `fig7_ssm_safety@2x.png` | SSM 안전거리 예산 (정지/감속링 + 도달 5.7m) | "왜 예측이 필요한가" |
| `fig8_collision_vs_avoidance@2x.png` | 충돌 vs 회피 — **탑뷰** (실제 sim) | "예측이 실제로 뭘 바꾸나" |
| `fig8f_collision_vs_avoidance_front@2x.png` | 충돌 vs 회피 — **정면뷰** | fig8의 아이레벨 버전 |
| `fig9_recall_precision@2x.png` | recall·precision 트레이드 (CV→학습) | "recall 올린 대가는?" |
| `fig10_scene_graph@2x.png` | 씬→상호작용 그래프 (Trajectron++ Fig.1 스타일) | "사람·로봇 그래프 모델링" |
| `fig11_methodology@2x.png` | 컬러 넘버 5단계 메서드 파이프라인 | 논문식 개요 |
| `fig12_multimodal_prediction@2x.png` | 멀티모달 예측 도식 (예측경로가 정지링 진입→회피) | "왜 회피?" |
| `fig13_avoidance_sequence@2x.png` | 회피 3컷 시퀀스 (정상→진입→팔 후퇴) 실제 sim | "예측→어디로 회피" |
| `fig14_predictor_architecture@2x.png` | 예측기 인코더-디코더 + MTP 헤드 | 논문식 모델 구조도 |

- 재현용 소스(`figures.html`, `src_*.png`, `make_dataset_figure.py`)는 원본 `docs/chanwoo/assets/pipeline/`에 있다.
- 근거 출처: `docs/chanwoo/nadir-zone-fusion.md` §5-14(커버리지)·§5-15(예측기)·§5-9(월드 융합).

## 01 · 화재 (`01-fire/`)

동일 Indoor Fire Smoke grouped test · YOLOv8s 60ep · conf 0.25 · frame-level · fire test 358장.
합성-only → 실사 학습: **recall 23.7% → 89.9%**, **precision 84.2% → 97.9%**.
출처: `K-H-MOON/kitchen-fire-noise-poc` commit `b0c9d726` · `docs/AFTER_meeting.md §5B`.

| 파일 | 내용 |
|---|---|
| `fire-before-after-claude.png` | 무대 톤 차트 (claude 팩) |
| `fire-before-after-codex.png` | 담백 톤 차트 (codex 팩) |
| `fire-synthesis-comparison-codex.jpg` | C0 알파 합성 vs C3 발광 합성 (검출 아님, 합성 방식 비교) |
| `_FIRE_BOX_ASSET_STATUS.txt` | 화재 검출 **박스 이미지 미확보** 사유 + Colab 재현 경로. 임의 생성 금지. |

## 02 · 천장 사람 검출 (`02-person/`)

> ⚠️ **도메인 구분**: 0.871·0.988은 **합성(SIM)** 수치. 실사 검출은 `person-real-transfer`(sim-only 0.270 붕괴 → real+sim 0.844)가 정직한 기준이고, 실사 회복은 남은 과제다. SIM을 실사처럼 말하지 않는다.

**claude 팩** (검출 박스에 `person 0.XX` 라벨, 클래스는 head 아님):

| 파일 | 내용 | 수치 | 도메인 |
|---|---|---|---|
| `person-finetune-sim-claude.png` | 기성 vs 파인튜닝 | R 0.175→0.871 · P 0.374→0.872 | SIM val |
| `person-fusion-ladder-claude.png` | 4분할·월드융합·추적 사다리 | R 0.831→0.988 | SIM/존 |
| `person-real-transfer-claude.png` | 실사 전이(정직성) | R 0.270→0.844 (RF-DETR 0.917) | **REAL** test 137 |
| `person-boxes-sim-before-after-claude.png` | 박스 · 기성 vs 나디르 4분할 (인상용) | 0명→13명 | SIM 나디르 |
| `person-boxes-sim-GTverified-claude.png` | 박스 · **GT 대조 검증** | 0명→8/8 전원 (FP0·누락0) | SIM 나디르 |
| `person-zone4-exploded-claude.png` | 4구역 타일별 재검출 | 6·5·4·1명 | SIM 나디르 |
| `person-boxes-real-before-after-claude.png` | 박스 · 기성 vs 실사모델 | 0명→13명 | REAL overhead |

**codex 팩** (Roboflow overhead-person v3, 동일 iid test 427장):

| 파일 | 내용 | 수치 |
|---|---|---|
| `person-overhead-before-after-codex.png` | stock YOLO11s vs overhead 파인튜닝 | R 0.442→0.980 · P 0.627→0.969 |
| `person-overhead-boxes-montage-codex.png` | 실제 overhead test 추론 박스 몽타주 | — |
| `person-overhead-boxes-raw-codex.jpg` | 발표 재편집용 원본 샘플 | — |

- codex 0.98은 **동일 분포 iid** 결과 → 새 조리실 cross-site 성능으로 해석 금지.
- 박스 두 버전: `sim-before-after`(13명·인상용)와 `sim-GTverified`(8/8·검증). 단일 정지 프레임의 놓침을 프레임 간 추적(ByteTrack+칼만)이 메워 배포 recall 0.988 → 근거는 `person-fusion-ladder`.

## 03 · 움직임(위험진입) 예측 (`03-motion-prediction/`)

| 파일 | 내용 | 수치 | 출처 |
|---|---|---|---|
| `danger-entry-recall-claude.png` | 위험 진입 예측 recall ⭐ | 57.1%→92.9% (FN48→FN8) | nadir-zone-fusion.md §5-15 |
| `trajectory-ade-1p6s-claude.png` | 예측 오차 ADE@1.6s | CV 0.372→Transformer 0.217 m | prediction-eval.md |
| `danger-entry-before-after-codex.png` | 정지링 밖에서 1.6s 내 3.1m 진입 예측 (CV vs Transformer K=3) | — | prediction-safety-eval.md |
| `trajectory-accuracy-1p6s-codex.png` | CV·Kalman·LSTM·Transformer ADE/FDE@1.6s + 95% CI | test 18 scenes·5,895 windows | results/traj-split-eval.json |

## 04 · 다음 위치(미래 경로) 예측 (`04-future-prediction/`)

- `lstm-three-future-codex.png` / `.json` — 체크포인트 **실제 실행** LSTM K=3 추론. 관측 8스텝·세 예측 경로·실제 미래 GT·정지/감속 링. seed 11 학습 예시(held-out 아님).
- `sim-live-prediction-overlay-codex.png` — 시뮬레이터 실제 다중 미래 오버레이.
- `lstm_move_01~05.png` (LSTM) / `tf_move_01~03.png` (Transformer) — **나디르(탑뷰) 밀도 구름** 최신본(PR #40). 흰 곡선=과거 관측 동선, 색 곡선+구름=백엔드 `/predict` K=3 미래(teal=최빈). 시간지평 3.5s(12×0.4s). 안전링 감속 3.9m·정지 3.1m·팔 1.87m. 자세한 범례는 원본 `docs/chanwoo/assets/prediction/README.md`.

## 05 · 예측기 아키텍처 (`05-architecture/`)

`predictor-architecture.svg`(벡터·편집용) / `.png`(3040×1600·삽입용). 관측 8스텝·ego 정규화 → 인코더 교체형(LSTM⇄Transformer) → 64-d 문맥 → 멀티모달 헤드 → K=3 미래 + best-of-K. `trajectory/learned_predictor.py`에 1:1 대응.

## 06 · 동선 히스토리 핸드오프 (`06-history-handoff/`)

슬라이드 08("과거 히스토리 학습") 왼쪽 그림을 실제 나디르 렌더 위에 재현하는 Figure Labs 핸드오프.

- `figurelab_handoff.md` — Figure Labs에 넣는 단일 파일(프롬프트 + 궤적 JSON).
- `kitchen-nadir-wholescene.png` — 배경(1440×1080 직교 나디르).
- `kitchen-trajectories-nadir.json` — 조리원 3명 궤적(60s·2.5Hz) + 스테이션·로봇·안전링. 각 점에 `x,z`(m)·`u,v`(px).
- `overlay-preview.png` — 검증 예시(JSON `u,v`를 PNG에 직접 찍음).
- `figurelab_prompt.md` — 프롬프트만.

## 07 · 보충 증거 (`07-extra-evidence/`)

발표 큐레이션 밖이지만 백업·부록용으로 모아둔 것.

- `nadir/` — 나디르 검출·평가 그림 (final_montage, loss_curve, occ_check, sim_gt_vs_pred, smoke_zoom, zone4_test, zonefog_check).
- `misc/` — 학습/데모 차트 (island_* YOLO 학습결과, bytetrack_ids, density_floor_demo, live_full_chain, predviz_*, fig1_style_*) + `codex-contact-sheet.png`(codex 팩 전체 한눈보기).
