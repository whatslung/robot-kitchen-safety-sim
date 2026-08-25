# detector-track E2E 예측 평가 — 설계 (감사 P0-2)

> 작성 2026-08-24 · 담당 chanwoo · 상태: **설계 승인 — 구현 착수**
> 근거: [handoff/2026-08-22-motion-quality-audit.md](../handoff/2026-08-22-motion-quality-audit.md) §P0-2
> 선결: 검출 모드 관측(P0-5 마무리, PR #17) — 라이브 검출 관측 체인이 성립.

## 0. 목표와 현재 한계

북극성 한 문장:

> **실제 검출 노이즈(미검출·박스 지터·ID 스위치)가 궤적 예측·안전 결정에 얼마나 타격을 주는지를 같은 클립에서 정량화한다.**

**현재**: 실사 예측 평가([spike_real_baseline.py](../../train/spike_real_baseline.py))는 **GT 라벨** 궤적 기반이라, 검출기가 사람을 놓치거나 박스가 떨리거나 트랙 ID가 바뀌는 영향이 예측에 전달되는 정도를 재지 않는다. P0-2는 **같은 클립**에서 GT-트랙 입력과 실제 검출-트랙 입력을 나란히 돌려 성능 차이를 표로 만든다.

## 1. 범위 결정 (확정)

- **예측기**: 등속(CV)·칼만(가중치 불필요, 항상 실행) + LSTM(`LearnedPredictor`, HF 가중치 자동 다운로드). CV·칼만이 보장 산출물, LSTM은 가중치가 받아지면 함께.
- **위험진입 = 가상 로봇 주입**: 실사 클립엔 로봇이 없다(제3자 person-only 데이터). 프로젝트가 "현실에 없는 걸 가상에서 학습·검증한다"는 sim-to-real 취지이므로, 실사 사람 트랙 위에 **가상 로봇 위치·반경을 얹어** 시뮬의 위험 모델([risk.py](../../trajectory/risk.py))을 실제 검출 트랙에 노출시킨다. §5의 가정을 반드시 명시한다.
- **해석은 상대값 우선**: 절대 위험진입 시각은 §5 가정(스케일·로봇 위치)에 의존한다. 핵심 신호는 **GT-트랙 vs 검출-트랙의 *차이*** — 같은 가상 로봇·같은 스케일을 두 경로에 똑같이 적용하므로 이 차이는 가정과 무관하게 공정하다. 절대값에는 "가정 기반" 꼬리표를 붙인다.

## 2. 데이터

- **`dataset/overhead-person-v3`** (Roboflow `riccardo-kxtut/overhead-person`, CC BY 4.0): 천장 시점 사람 클립을 프레임별 이미지 + GT 사람 박스(YOLO 라벨)로 담음. `nc:1, names:['person']`. train 4120 · valid 1 · test 137 라벨. 파일명(`(clip)_(frame)_jpg.rf.*`)으로 클립·프레임 복원(spike의 `CLIP_RE` 재사용).
- **⚠️ 완료 조건 한계(정직 표기)**: P0-2 완료 조건은 "실제 **주방/조리실 유사** 클립 1개"인데 이 데이터는 **일반 오버헤드 사람**이라 주방 특정이 아니다. 현재 워크트리에서 구할 수 있는 가장 가까운 실사 데이터로 진행하고, 결과 문서에 이 한계를 명시한다. 별도 주방 클립 확보 시 같은 파이프라인으로 재실행.

## 3. 아키텍처 (데이터 흐름)

```
overhead-person-v3 프레임(이미지+GT라벨)
   ├─(A) GT-트랙:  GT 라벨 → IoU 추적(spike track()) → per-track (cx,cy) 시계열
   └─(B) 검출-트랙: 이미지 → YOLO best.pt → ByteTrack → per-track (cx,cy) 시계열
          │
          ▼  GT↔검출 트랙 매칭(프레임별 중심거리, 다수결로 1:1)
   윈도우화(obs8/pred12, 연속프레임, MOVE_PX 필터 — spike 재사용)
          │
          ▼  각 윈도우: 예측기(CV·칼만·LSTM)로 pred12, 가상 로봇으로 위험진입
   지표 집계
          ▼
   { GT-트랙 vs 검출-트랙 ADE/FDE·Δ위험진입, 실패모드별 하락 } → 결과 JSON + 표
```

## 4. 컴포넌트

### 4-1. `trajectory/dettrack.py` (신규, 순수 함수 — pytest 대상)
JS 모놀리스와 달리 파이썬 순수 함수라 단위 테스트가 깔끔하다(이 저장소 방식).
- `match_tracks(gt_tracks, det_tracks, max_center_dist) -> dict[gt_id, det_id|None]`
  프레임별 중심거리로 겹치는 구간을 모아 다수결 1:1 매칭. 매칭 없으면 `None`.
- `classify_failures(gt_track, det_track_matched) -> dict`
  `{miss_frames, fragments, id_switches}` — 검출 실패(해당 프레임 검출 없음)·fragmentation(한 GT가 여러 det로 쪼개짐)·ID switch(같은 GT의 det id 변경) 카운트.
- `virtual_robot_risk(pred_modes, robot, stopR, slowR, horizon, ksig, tau) -> dict`
  [risk.track_risk](../../trajectory/risk.py) 얇은 래퍼 — 예측기 출력(`Mode.steps`)을 risk가 기대하는 `{path,[x,z]], w, sigma}` 모드 형식으로 어댑트해 위험진입 시각 산출.
- `aggregate(records) -> dict` — GT/검출 × 예측기 × (전체·움직인것·실패모드별) ADE/FDE·Δ위험진입 평균.

### 4-2. `train/eval_traj_dettrack.py` (신규, 파이프라인)
- GT-트랙: spike의 `load_clips`/`iou`/`track`/`windows` 로직 재사용(공통화 또는 import).
- 검출-트랙: ultralytics `YOLO(best.pt)`(로컬 없으면 HF `chanubc/robot-kitchen-nadir-yolo11s`) + supervision/trackers `ByteTrackTracker`로 프레임 순차 추론 → per-track 시계열. 검출 실패로 러너블하지 않으면(가중치·네트워크 없음) 명확히 `log`하고 GT-트랙만이라도 산출(silent 실패 금지).
- 예측기: `ConstantVelocityPredictor`·`KalmanPredictor`·`LearnedPredictor`.
- 가상 로봇: §5 가정으로 배치, 몇 위치 민감도 스윕.
- 출력: 결과 JSON(`docs/chanwoo/results/detection-track-eval.json` 등) + 재현 명령.

## 5. 명시할 가정 (결과 문서·JSON에 반드시 기록)

1. **픽셀↔미터 보정**: 카메라 높이·FOV 미상 → **프레임 가로 = 6.0m 가정**(우리 셀 가로 ~6.1m에 맞춤). 좌표는 `x_m = cx * 6.0`, `z_m = cy * (6.0 * H/W)`. 정지/감속 반경(m)은 이 가정 위에서만 의미.
2. **프레임 시간간격**: 모델 스텝(0.4s)에 맞춰 **0.4s/frame(2.5fps) 가정**. Roboflow 프레임이 등간격이라는 가정 포함.
3. **로봇 위치**: 기본 = 전 GT 트랙 중심(통행 밀집). 민감도용으로 2~3개 대안 위치도 기록.
4. **상대 해석**: 절대 위험진입 시각엔 "가정 기반" 꼬리표, 핵심 결론은 GT vs 검출 **차이**.

## 6. 지표

- **ADE/FDE**([evaluator.ade/fde](../../trajectory/evaluator.py), ×640px): GT-트랙 입력 vs 검출-트랙 입력, 예측기별. 전체 + "움직인 것만"(spike `MOVE_PX`).
- **Δ위험진입**: 같은 가상 로봇에서 `track_risk`의 `tEntryStop`/`tEntrySlow`를 GT-트랙 예측 vs 검출-트랙 예측으로 비교(놓친 진입·늦은 진입·헛 진입).
- **실패모드별 하락**: miss/fragmentation/ID switch가 있는 윈도우 vs 없는 윈도우의 ADE/FDE·Δ위험진입 차이.

## 7. 안전 불변식 / 에러처리

- 검출기 가중치·네트워크·의존성 부재 → GT-트랙 경로는 그대로 산출하고 검출-트랙은 명시적 `log` 후 건너뜀(silent 실패·조용한 축소 금지 — 감사 원칙).
- 트랙 매칭 실패(GT에 대응 검출 없음)는 "완전 미검출"로 집계에 포함(누락 아님).
- 결정성: 매칭·집계 경로에 난수·시계 의존 없음. YOLO/ByteTrack 추론은 고정 가중치·고정 임계값.

## 8. 테스트 (TDD seam)

- **pytest `tests/test_dettrack.py`**(신규): `match_tracks`(명확한 1:1·ID switch·미매칭), `classify_failures`(miss/fragment/switch 합성 입력), `virtual_robot_risk`(risk 어댑터 == 직접 track_risk), `aggregate`(소규모 합성 레코드). 순수 함수부터 red→green.
- 파이프라인(`eval_traj_dettrack.py`)은 데이터·가중치 의존이라 pytest 비대상 — 실행 명령 + 산출 JSON으로 검증.
- 전체 회귀: `uv run --group serve --with pytest python -m pytest tests/ -q`.

## 9. 파일 · 완료 정의

**건드릴 파일**: `trajectory/dettrack.py`(신규 순수함수) · `train/eval_traj_dettrack.py`(신규 파이프라인) · `tests/test_dettrack.py`(신규) · `docs/chanwoo/detection-track-eval.md`(결과·가정·표) · 결과 JSON.

**완료 정의**
- 실제 클립 단위 **재현 명령 + 결과 JSON** 커밋.
- **GT-트랙 / 검출-트랙 성능 차이 표**(ADE/FDE + Δ위험진입, 실패모드별) 제공.
- 최소 1개 실제(오버헤드 사람) 클립 포함 — 주방 특정 아님을 §2대로 명시.
- 순수 함수 pytest 통과, 전체 스위트 green.
- §5 가정이 결과 문서·JSON에 기록됨.

## 10. 후속 (이 스펙 밖)

- **P0-3(라이브 안전 성능 계측)**: 위험진입 TP/FP/FN·lead time·stop duration은 로봇이 있는 **시뮬 seed 시나리오**에서 측정 — P0-2의 가상 로봇 위험진입은 "검출 노이즈의 상대 영향"까지다.
- 실제 **주방/조리실 클립** 확보 시 같은 파이프라인 재실행.
- P0-1(train/val/test split) 완료 후 LSTM 수치를 "독립 test 값"으로 갱신.
