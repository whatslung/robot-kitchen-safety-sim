# 오블리크 멀티캠 융합·위치추정 조사 — 정직한 측정 (2026-08-27)

> **배경:** "6캠 융합 recall 0.938"이 실제로 전역 ID 융합·ByteTrack 시간축 recall·false positive를
> 측정한 게 아니라는 지적(Codex #28) → 이 셋을 제대로 재고, 위치추정 병목의 해법을 실험했다.
> **한 줄 결론:** 0.938은 "6캠 중 1대라도 IoU 검출"한 낙관값. 정직한 전역 BEV 융합은
> **recall ~0.75 · FP 35~51%**, 병목은 **오블리크 발-픽셀 위치추정 ~1m 오차**이며,
> 발목 keypoint·다중뷰 삼각측량 모두 실제 검출에선 이를 못 넘는다. **나디르 카메라가 정밀 위치추정의 답.**

## 0. 측정 조건 (재현)

- 캡처: `tools/headless_gen/capture_fusion_still.cjs` — **센서 OFF(#28 정렬)** 단일샷, 6캠, clean RGB.
  · 사람 GT image box(+global id) · GT 바닥좌표(peopleList) · 호모그래피 calib(바닥4점) · P(3×4) DLT용 3D점.
  · ⚠ Codex #28의 "GT 재질 준비 10초 reject" 가드가 headless 캡처를 막아, worktree sim.html에서
    임시로 2.5초 resolve로 로컬 패치(커밋 안 함). 원복: `setTimeout(resolve,2500)` → `reject 10000`.
- 검출: `training/simfixed_yolo11s_1280/best.pt`(우리 sim 검출기) · pose: `yolo11n-pose.pt`(COCO).
- 평가: `scratch_eval_fusion_still.py` · `scratch_eval_footcalib.py` · `scratch_eval_pose_fusion.py` · `scratch_eval_triangulate.py`.
- 표본: 24 독립 샘플 · person 60~66 · conf 0.10. **표본 작음 — 경향 지표.**

## 1. 전역 BEV 융합 recall + FP (0.938의 정직한 대체)

호모그래피로 발끝→바닥 → 근접 융합 → GT 바닥좌표 대조.

| 지표 | merge 0.9m | merge 1.6m |
|---|---|---|
| BEV 융합 recall(거리기반) | 0.767 | 0.783 |
| **융합 false-positive** | **59.3%** | **37.3%** |
| 'anycam' IoU(옛 0.938식) | 0.650 | 0.650 |

- **재현율보다 정밀도(FP)가 병목.** 클러스터가 60명인데 75~113개(사람당 ~2) → 같은 사람이 카메라마다 흩어져 안 합쳐짐.
- ByteTrack **시간축 id-switch는 미측정** — 연속 비디오 캡처가 sim 구조상 막힘(정지 프레임용 설계: 자율이동 텔레포트 / freeze는 GT 재질 마스크 오염 / 정적+비프리즈는 재질 타임아웃).

## 2. 원인 = 발-픽셀 위치추정 오차 (GT 완벽 박스로도)

발끝점 → 호모그래피 → 바닥, GT 바닥좌표와의 오차:

| 발끝점 | median | mean | p90 |
|---|---|---|---|
| **이상 발-접점**(GT 바닥점 투영, 도달 불가) | **0.00m** | 0.63 | 2.5 |
| 박스 하단(검출기) | 1.09m | 1.36 | 2.7 |
| 발목 pose(단일뷰) | 0.99m | 1.25 | 2.6 |

→ 기하(호모그래피)는 정확(이상점 median 0). **박스하단·발목이 ~1m인 건 그 픽셀이 실제 발-바닥 접점이 아니기 때문**(발목 관절은 바닥 위 ~10cm, grazing 각도서 증폭).

## 3. 해법 실험 — 발목 keypoint & 다중뷰 삼각측량

**발목 keypoint(pose):** COCO pose가 sim 사람 발목을 잘 잡음(성공률 97~100%). 융합 FP를
박스하단 51% → **발목 35%** → 이상 20%로 **부분 개선**. 단 위치추정 오차 0.99m로 박스하단(1.09m)과 큰 차 없음.

**다중뷰 삼각측량**(P(3×4) DLT 보정, ≥2캠 광선 교차):

| 발끝점 | 단일뷰 호모 | 삼각측량(나이브) | 삼각측량(강건 쌍-median) | ≥2캠 가능 |
|---|---|---|---|---|
| 이상 발-접점 | 0.00m | 0.00m | 0.00m | 98.5% |
| **발목 pose** | 0.99m | **1.26m** | **1.26m** | 67.7% |

→ **삼각측량은 실제 발목에서 오히려 악화.** 기하는 맞지만(이상점 median 0), 발목 검출 노이즈가
카메라마다 ~1m로 일관돼 광선이 어긋나고(skew), 강건화(쌍-median)도 무효.

## 4. 결론 · 방향

- **오블리크 바닥 위치추정 병목 = 발-접점 픽셀 정확도(~1m).** 융합방법·삼각측량·깊이가 아니라 검출 픽셀이 한계.
- **발목·삼각측량 모두 실제 검출에선 이 벽을 못 넘음.** 이상점(median 0)만 정확 = 검출로는 도달 불가.
- **진짜 해법:**
  1. **나디르(천장 수직, `orthotop`) 카메라로 정밀 위치추정** — 발점이 사람 바로 아래·깊이 모호성 0 → 바닥 정확(핸드오프 "나디르 GT exact"). **오블리크=검출·커버리지 / 나디르=위치추정** 역할 분담.
  2. 또는 발-접점 전용 검출(발바닥 세그멘테이션) — 관절이 아니라 접지점.
  3. 또는 안전 로직을 ~1m 오차·높은 FP 전제로 거칠게.
- **역할 재확인:** 실배포 사람 검출=실사(chef1 0.97). sim 오블리크 검출=커버리지. 정밀 위치추정=나디르. 예측/안전=GT 트랙(Transformer ADE 0.12m).

## 5. 스크립트 (gitignore 자산: dataset·weights)

`tools/headless_gen/capture_fusion_still.cjs`(+p3d) · `scratch_eval_{fusion_still,footcalib,pose_fusion,triangulate}.py`
· `train/pose_foot.py`(발목 추출) · `trajectory/{homography,multicam}.py`(순수 함수 재사용).
