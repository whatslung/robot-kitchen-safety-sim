# 발표 증거 에셋

캡스톤 발표 개편에 사용할 모델 결과 이미지다. 그래프는 저장소의 평가 결과와 팀 화재 저장소의 누수 통제 결과를 사용했다.

## 화재

- `fire/fire_model_before_after.png`
  - 동일 Indoor Fire Smoke grouped test, YOLOv8s, conf 0.25, frame-level.
  - 합성-only: recall 0.237, precision 0.842.
  - 실사-only: recall 0.899, precision 0.979.
  - 출처: `K-H-MOON/kitchen-fire-noise-poc` commit `b0c9d726`, `docs/AFTER_meeting.md` §5B.
- `fire/fire_synthesis_before_after.jpg`
  - C0 알파 합성과 C3 발광 합성의 시각 비교. 검출 결과가 아니라 합성 방식 비교다.
- `fire/FIRE_BOX_ASSET_STATUS.txt`
  - 실제 화재 true-positive 박스 출력의 미확보 사유와 Colab 재현 경로.

주의: 화재 박스 이미지는 임의로 만들지 않았다. 권위 체크포인트가 다른 Google Drive 계정에 있어 현재 팩에는 포함되지 않는다.

## 천장 사람 검출

- `person/overhead_person_before_after.png`
  - Roboflow overhead-person v3 동일 iid test 427장.
  - stock YOLO11s: recall 0.442, precision 0.627.
  - overhead fine-tuning: recall 0.980, precision 0.969.
- `person/overhead_person_detection_boxes_montage.png`
  - 실제 오버헤드 test 이미지의 fine-tuned YOLO11s 추론 박스.
- `person/overhead_person_detection_boxes_raw.jpg`
  - 발표 재편집용 원본 샘플.

모델 클래스는 `head`가 아니라 `person`이다. 0.98은 동일 분포 iid test 결과이며 새 조리실 cross-site 성능으로 해석하면 안 된다.

## 움직임 예측

- `motion/trajectory_accuracy_1p6s.png`
  - 독립 test 18 scenes, 5,895 windows.
  - CV, Kalman, LSTM, Transformer의 ADE/FDE@1.6s와 scene-bootstrap 95% CI.
  - 출처: `docs/chanwoo/results/traj-split-eval.json`.
- `motion/danger_entry_before_after.png`
  - 현재 정지 링 밖에서 다음 1.6초 내 3.1m 진입을 예측하는 안전 지표.
  - CV와 Transformer K=3 mode union 비교.
  - 출처: `docs/chanwoo/prediction-safety-eval.md`.

## 다음 위치 예측

- `future/lstm_three_future_prediction.png`
  - 체크포인트를 실제 실행해 만든 LSTM K=3 추론 결과.
  - 관측 8스텝, 세 가지 예측 경로, 실제 미래 GT, 정지·감속 링을 함께 표시한다.
  - seed 11의 학습 예시 장면이며 held-out 검증 사례가 아니다.
- `future/lstm_three_future_prediction.json`
  - 장면, 윈도우, 모드 확률, 정지 링 진입 시점, 체크포인트 경로 메타데이터.
- `future/sim_live_prediction_overlay.png`
  - 시뮬레이터의 실제 다중 미래 예측 오버레이.

`contact_sheet.png`에서 전체 에셋을 한 번에 확인할 수 있다.
